#include "simd.hpp"

#include <hwy/auto_tune.h>
#include <hwy/timer.h>

// Consecutively call `f(0)`, `f(1)`, ..., `f(15)`.
#define CALL16(f)                                                                      \
    do {                                                                               \
        f(0);                                                                          \
        f(1);                                                                          \
        f(2);                                                                          \
        f(3);                                                                          \
        f(4);                                                                          \
        f(5);                                                                          \
        f(6);                                                                          \
        f(7);                                                                          \
        f(8);                                                                          \
        f(9);                                                                          \
        f(10);                                                                         \
        f(11);                                                                         \
        f(12);                                                                         \
        f(13);                                                                         \
        f(14);                                                                         \
        f(15);                                                                         \
    } while (false)

// Highway recursive include machinery. See `hwy/examples/skeleton.cc` for details.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd.cpp"
#include <hwy/foreach_target.h>

// Include the main Highway header first.
#include <hwy/highway.h>

// Include all other target-dependent headers after the main Highway header.
#include <hwy/contrib/math/math-inl.h>

// All multi-target code must be placed in this namespace.
HWY_BEFORE_NAMESPACE();
namespace fastfilters2::simd::HWY_NAMESPACE {

// Namespace alias for the Highway ops.
namespace hn = hwy::HWY_NAMESPACE;

using D = hn::CappedTag<float, max_lane_bytes / sizeof(float)>;

// Copy `src` into `dst`, mirroring `radius` pixels on each side, except the first and
// the last pixel. For example, if `radius == 2`, then `[012345] -> 21[012345]43`.
static HWY_INLINE void mirror_copy(
        const float *HWY_RESTRICT src,
        float *HWY_RESTRICT dst,
        size_t size,
        size_t radius) {
    for (size_t i = 0; i < radius; ++i) {
        dst[i] = src[radius - i];
    }
    std::memcpy(dst + radius, src, sizeof(float) * size);
    for (size_t i = 0; i < radius; ++i) {
        dst[radius + size + i] = src[size - 2 - i];
    }
}

// Convolve `unroll` lanes from `src` with `kernel`, and store the result in `dst`.
// If `contiguous` is true, the source data and destination data are contiguous in
// memory, and all other parameters are ignored. Otherwise, the source data and
// destination data are strided, and the parameters describe the stride and image
// boundaries.
template <bool contiguous, bool symmetric, size_t unroll>
static HWY_INLINE void conv_lanes(
        const float *src,
        const float *kernel,
        size_t ksize,
        float *dst,
        size_t index,
        size_t size,
        size_t stride) {

    using V = hn::VFromD<D>;

    D d;
    HWY_LANES_CONSTEXPR auto lanes = hn::Lanes(d);

    // The current kernel value, replicated across all vector lanes.
    auto kv = hn::Set(d, kernel[0]);

    // Left and right source pointers. In the contiguous case, the pointers are
    // incremented/decremented directly. In the strided case, the pointers move by
    // `stride`, either in the direct or the reverse direction. For the left
    // pointer, direct/reverse is backward/forward. For the right pointer,
    // direct/reverse is forward/backward. The change in the direction occurs when a
    // pointer reaches an image boundary, which is necessary for the correct mirror
    // border handling.
    auto lsrc = src;
    auto rsrc = src;

    // On architectures with `HWY_HAVE_SCALABLE`, vector types are sizeless and
    // cannot be stored in arrays. Therefore, manually unroll accumulators with
    // macros. Also, some compilers generate unnecessary loads/stores or have other
    // difficulties when unrolling loops over arrays of SIMD vectors.
    V v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;
    static_assert(unroll <= 16);

// Initialize an accumulator with `src[0] * kernel[0]` (central element of the kernel).
#define INIT(i)                                                                        \
    if constexpr (i < unroll) {                                                        \
        v##i = hn::Mul(hn::Load(d, src + i * lanes), kv);                              \
    }

// Add `(src[k] ? src[-k]) * kernel[k]` to accumulator, where `?` is either `+` or `-`.
// This is the same as `(src[k] * kernel[k]) ? (src[-k] * kernel[k])`. On architectures
// without FMA (fused multiply-accumulate) units, the former method is faster because it
// saves 1 op. If FMA is available, the latter method should have the same performance,
// but is more accurate. Use the former method for backwards compatibility with the old
// code.
#define STEP(i)                                                                        \
    if constexpr (i < unroll) {                                                        \
        V lv, rv;                                                                      \
        if constexpr (contiguous) {                                                    \
            lv = hn::LoadU(d, lsrc + i * lanes);                                       \
            rv = hn::LoadU(d, rsrc + i * lanes);                                       \
        } else {                                                                       \
            lv = hn::Load(d, lsrc + i * lanes);                                        \
            rv = hn::Load(d, rsrc + i * lanes);                                        \
        }                                                                              \
        if constexpr (symmetric) {                                                     \
            v##i = hn::MulAdd(hn::Add(rv, lv), kv, v##i);                              \
        } else {                                                                       \
            v##i = hn::MulAdd(hn::Sub(rv, lv), kv, v##i);                              \
        }                                                                              \
    }

// Store the accumulated result.
#define STORE(i)                                                                       \
    if constexpr (i < unroll) {                                                        \
        hn::Store(v##i, d, dst + i * lanes);                                           \
    }

    CALL16(INIT);

    HWY_ASSUME(ksize >= 2);
    for (size_t k = 1; k < ksize; ++k) {
        kv = hn::Set(d, kernel[k]);

        if constexpr (contiguous) {
            // Simple pointer arithmetic is faster than branches or conditional moves.
            --lsrc;
            ++rsrc;
        } else {
            // Usually compiles to CMOV (conditional move) instructions on x86.
            lsrc = k <= index ? lsrc - stride : lsrc + stride;
            rsrc = index + k < size ? rsrc + stride : rsrc - stride;
        }

        CALL16(STEP);
    }

    CALL16(STORE);

#undef INIT
#undef STEP
#undef STORE
}

// Convolve across the contiguous dimension. `row_size` is the size of the
// contiguous dimension, and `total_size` is the total size of the image.
template <bool symmetric, size_t unroll>
static void conv_contiguous(
        const float *src,
        size_t row_size,
        size_t row_count,
        KernelView kernel,
        float *dst,
        float *row_buf) {

    D d;
    HWY_LANES_CONSTEXPR auto lanes = hn::Lanes(d);
    HWY_LANES_CONSTEXPR auto unrolled_lanes = unroll * lanes;

    // The source row size might be different from the buffer row size due to the
    // padding of the destination buffer.
    HWY_LANES_CONSTEXPR auto mask = lanes - 1;
    auto dst_row_size = (row_size + mask) & ~mask;

    auto radius = kernel.size - 1;
    const auto *src_row = row_buf + radius;

    // Loop over the rows.
    for (size_t row_idx = 0; row_idx < row_count;
         ++row_idx, src += row_size, dst += dst_row_size) {

        // Because the row is contiguous in memory, we need to physically mirror the
        // row edges so that `conv_lanes` can access the mirrored pixels. It might
        // be beneficial to shuffle the pixels instead of copying, but shuffle
        // instructions are usually slower than direct loads. Also, copying
        // automatically pads the input up to the batch size.
        mirror_copy(src, row_buf, row_size, radius);

        // Process the row in batches.
        size_t i = 0;
        if constexpr (unroll > 1) {
            for (; i + unrolled_lanes <= row_size; i += unrolled_lanes) {
                conv_lanes<true, symmetric, unroll>(
                        src_row + i, kernel.data, kernel.size, dst + i, 0, 0, 0);
            }
        }
        for (; i < row_size; i += lanes) {
            conv_lanes<true, symmetric, 1>(
                    src_row + i, kernel.data, kernel.size, dst + i, 0, 0, 0);
        }
    }
}

// Convolve across the non-contiguous dimension. `row_size` is the size of the
// contiguous dimension. Inner and outer sizes are the sizes of the main
// (convolution) and all other axes. Inner and outer strides are defined
// accordingly.
template <bool symmetric, size_t unroll>
static void conv_strided(
        const float *src,
        size_t row_size,
        size_t inner_size,
        size_t outer_size,
        size_t inner_stride,
        size_t outer_stride,
        KernelView kernel,
        float *dst) {

    D d;
    HWY_LANES_CONSTEXPR auto lanes = hn::Lanes(d);
    HWY_LANES_CONSTEXPR auto unrolled_lanes = unroll * lanes;

    // Loop over the outer dimension.
    for (size_t outer = 0; outer < outer_size; ++outer) {
        // Planes are not necessarily contiguous in memory (e.g. XZ planes), but the
        // X axis is always contiguous.
        auto src_plane = src + outer * outer_stride;
        auto dst_plane = dst + outer * outer_stride;

        // Process the plane in batches of strips, moving along the X dimension.
        size_t i = 0;

        if constexpr (unroll > 1) {
            for (; i + unrolled_lanes <= row_size; i += unrolled_lanes) {
                // Strips are sub-planes of the plane defined above. They start at `i`
                // and end at `i + batch_size` along the contiguous dimension, and span
                // the entire inner dimension.
                auto src_strip = src_plane + i;
                auto dst_strip = dst_plane + i;

                // Process the strip, moving along the inner (non-contiguous) dimension.
                // This maximizes cache utilization: each iteration reuses cache lines
                // in the current strip.
                for (size_t inner = 0; inner < inner_size; ++inner) {
                    conv_lanes<false, symmetric, unroll>(
                            src_strip + inner * inner_stride,
                            kernel.data,
                            kernel.size,
                            dst_strip + inner * inner_stride,
                            inner,
                            inner_size,
                            inner_stride);
                }
            }
        }

        for (; i < row_size; i += lanes) {
            auto src_strip = src_plane + i;
            auto dst_strip = dst_plane + i;

            for (size_t inner = 0; inner < inner_size; ++inner) {
                conv_lanes<false, symmetric, 1>(
                        src_strip + inner * inner_stride,
                        kernel.data,
                        kernel.size,
                        dst_strip + inner * inner_stride,
                        inner,
                        inner_size,
                        inner_stride);
            }
        }
    }
}

// Convolve across the specified axis, unrolled by the `unroll` lanes.
template <size_t unroll>
static void
conv_unroll(int axis, DataView3D src, KernelView kernel, float *dst, float *row_buf) {
    auto symmetric = kernel.order % 2 == 0;
    auto row_size = src.shape[2];

    if (axis == 2) {
        // Contiguous case, operate on rows.
        auto row_count = src.shape[0] * src.shape[1];
        if (symmetric) {
            conv_contiguous<true, unroll>(
                    src.data, row_size, row_count, kernel, dst, row_buf);
        } else {
            conv_contiguous<false, unroll>(
                    src.data, row_size, row_count, kernel, dst, row_buf);
        }

    } else {
        // Strided case, operate on planes.
        size_t inner_size, outer_size, inner_stride, outer_stride;
        if (axis == 1) {
            // XY planes.
            inner_size = src.shape[1];
            outer_size = src.shape[0];
            inner_stride = src.shape[2];
            outer_stride = src.shape[1] * src.shape[2];
        } else {
            // XZ planes.
            inner_size = src.shape[0];
            outer_size = src.shape[1];
            inner_stride = src.shape[1] * src.shape[2];
            outer_stride = src.shape[2];
        }

        if (symmetric) {
            conv_strided<true, unroll>(
                    src.data,
                    row_size,
                    inner_size,
                    outer_size,
                    inner_stride,
                    outer_stride,
                    kernel,
                    dst);
        } else {
            conv_strided<false, unroll>(
                    src.data,
                    row_size,
                    inner_size,
                    outer_size,
                    inner_stride,
                    outer_stride,
                    kernel,
                    dst);
        }
    }
}

// Common template context for all ufuncs.
template <int Count> struct Ufuncs {
    using V = hn::VFromD<D>;

    static void l2norm(MultiDataView<3> src, float *dst) {
        D d;
        HWY_LANES_CONSTEXPR auto lanes = hn::Lanes(d);
        V x, y, z;

        for (size_t i = 0; i < src.size; i += lanes) {
            x = hn::Load(d, src.data[0] + i);
            y = hn::Load(d, src.data[1] + i);
            if constexpr (Count > 2) {
                z = hn::Load(d, src.data[2] + i);
            }

            auto v = hn::MulAdd(y, y, hn::Mul(x, x));
            if constexpr (Count > 2) {
                v = hn::MulAdd(z, z, v);
            }
            v = hn::Sqrt(v);

            hn::Store(v, d, dst + i);
        }
    }

    static void add(MultiDataView<3> src, float *dst) {
        D d;
        HWY_LANES_CONSTEXPR auto lanes = hn::Lanes(d);
        V x, y, z;

        for (size_t i = 0; i < src.size; i += lanes) {
            x = hn::Load(d, src.data[0] + i);
            y = hn::Load(d, src.data[1] + i);
            if constexpr (Count > 2) {
                z = hn::Load(d, src.data[2] + i);
            }

            auto v = hn::Add(x, y);
            if constexpr (Count > 2) {
                v = hn::Add(z, v);
            }

            hn::Store(v, d, dst + i);
        }
    }

    static void mul_pairs(MultiDataView<3> src, MultiOutputView<6> dst) {
        D d;
        HWY_LANES_CONSTEXPR auto lanes = hn::Lanes(d);
        V x, y, z;
        V xx, xy, yy, xz, yz, zz;

        for (size_t i = 0; i < src.size; i += lanes) {
            x = hn::Load(d, src.data[0] + i);
            y = hn::Load(d, src.data[1] + i);
            if constexpr (Count > 2) {
                z = hn::Load(d, src.data[2] + i);
            }

            xx = hn::Mul(x, x);
            xy = hn::Mul(x, y);
            yy = hn::Mul(y, y);
            if constexpr (Count > 2) {
                xz = hn::Mul(x, z);
                yz = hn::Mul(y, z);
                zz = hn::Mul(z, z);
            }

            hn::Store(xx, d, dst[0] + i);
            hn::Store(xy, d, dst[1] + i);
            hn::Store(yy, d, dst[2] + i);
            if constexpr (Count > 2) {
                hn::Store(xz, d, dst[3] + i);
                hn::Store(yz, d, dst[4] + i);
                hn::Store(zz, d, dst[5] + i);
            }
        }
    }
};

static void eigenvalues2d(MultiDataView<6> src, MultiOutputView<3> dst) {
    // The code below has been adapted from the old reference implementation:
    // https://github.com/ilastik/fastfilters/blob/38e606fa5aacd571b07e2e34fda1fb2eeb6ca128/src/library/linalg_avx.c#L29

    D d;
    HWY_LANES_CONSTEXPR auto lanes = hn::Lanes(d);
    auto half = hn::Set(d, 0.5);

    for (size_t i = 0; i < src.size; i += lanes) {
        auto v_xx = hn::Load(d, src.data[0] + i);
        auto v_xy = hn::Load(d, src.data[1] + i);
        auto v_yy = hn::Load(d, src.data[2] + i);

        auto tmp0 = hn::Mul(hn::Add(v_xx, v_yy), half);
        auto tmp1 = hn::Mul(hn::Sub(v_xx, v_yy), half);
        tmp1 = hn::Mul(tmp1, tmp1);

        // Could be replaced with `hn::MulAdd`, but it is numerically different from the
        // old reference implementation.
        auto det = hn::Sqrt(hn::Add(tmp1, hn::Mul(v_xy, v_xy)));

        auto ev0 = hn::Add(tmp0, det);
        auto ev1 = hn::Sub(tmp0, det);

        // Determinant is always non-negative, therefore `ev0 >= ev1`.
        hn::Store(ev0, d, dst[0] + i);
        hn::Store(ev1, d, dst[1] + i);
    }
}

static void eigenvalues3d(MultiDataView<6> src, MultiOutputView<3> dst) {
    // The code below has been adapted from the old reference implementation:
    // https://github.com/ilastik/fastfilters/blob/38e606fa5aacd571b07e2e34fda1fb2eeb6ca128/src/library/linalg_avx2.c#L63

    D d;
    HWY_LANES_CONSTEXPR auto lanes = hn::Lanes(d);
    auto v_inv3 = hn::Set(d, 1.0 / 3.0);
    auto v_root3 = hn::Sqrt(hn::Set(d, 3.0));
    auto two = hn::Set(d, 2.0);
    auto one = hn::Set(d, 1.0);
    auto half = hn::Set(d, 0.5);
    auto zero = hn::Zero(d);

    for (size_t i = 0; i < src.size; i += lanes) {
        // Even though these variable names suggest a particular element ordering of the
        // input matrix, the actual ordering might be different (see the upstream code).
        // Names have not been updated in order to keep the code consistent with the old
        // reference implementation.
        auto v_a00 = hn::Load(d, src.data[0] + i);
        auto v_a01 = hn::Load(d, src.data[1] + i);
        auto v_a02 = hn::Load(d, src.data[2] + i);
        auto v_a11 = hn::Load(d, src.data[3] + i);
        auto v_a12 = hn::Load(d, src.data[4] + i);
        auto v_a22 = hn::Load(d, src.data[5] + i);

        // clang-format off

        // guard against float overflows
        auto v_max0 = hn::Max(hn::Abs(v_a00), hn::Abs(v_a01));
        auto v_max1 = hn::Max(hn::Abs(v_a02), hn::Abs(v_a11));
        auto v_max2 = hn::Max(hn::Abs(v_a12), hn::Abs(v_a22));
        auto v_max_element = hn::Max(hn::Max(v_max0, v_max1), v_max2);

        // replace zeros with ones to avoid NaNs
        v_max_element = hn::IfThenElse(hn::Eq(v_max_element, zero), one, v_max_element);

        // Divisions could be replaced with multiplications by the reciprocal, but this
        // would lead to numerical differences from the old reference implementation.
        v_a00 = hn::Div(v_a00, v_max_element);
        v_a01 = hn::Div(v_a01, v_max_element);
        v_a02 = hn::Div(v_a02, v_max_element);
        v_a11 = hn::Div(v_a11, v_max_element);
        v_a12 = hn::Div(v_a12, v_max_element);
        v_a22 = hn::Div(v_a22, v_max_element);

        auto c0 = hn::Sub(hn::Sub(hn::Sub(hn::Add(hn::Mul(hn::Mul(v_a00, v_a11), v_a22),
            hn::Mul(hn::Mul(hn::Mul(two, v_a01), v_a02), v_a12)),
            hn::Mul(hn::Mul(v_a00, v_a12), v_a12)),
            hn::Mul(hn::Mul(v_a11, v_a02), v_a02)),
            hn::Mul(hn::Mul(v_a22, v_a01), v_a01));
        auto c1 = hn::Sub(hn::Add(hn::Sub(hn::Add(hn::Sub(hn::Mul(v_a00, v_a11),
            hn::Mul(v_a01, v_a01)),
            hn::Mul(v_a00, v_a22)),
            hn::Mul(v_a02, v_a02)),
            hn::Mul(v_a11, v_a22)),
            hn::Mul(v_a12, v_a12));
        auto c2 = hn::Add(hn::Add(v_a00, v_a11), v_a22);
        auto c2Div3 = hn::Mul(c2, v_inv3);
        auto aDiv3 = hn::Mul(hn::Sub(c1, hn::Mul(c2, c2Div3)), v_inv3);

        aDiv3 = hn::Min(aDiv3, zero);

        auto mbDiv2 = hn::Mul(half, hn::Add(c0, hn::Mul(c2Div3, hn::Sub(hn::Mul(hn::Mul(two, c2Div3), c2Div3), c1))));
        auto q = hn::Add(hn::Mul(mbDiv2, mbDiv2), hn::Mul(hn::Mul(aDiv3, aDiv3), aDiv3));

        q = hn::Min(q, zero);

        auto magnitude = hn::Sqrt(hn::Neg(aDiv3));
        auto angle = hn::Mul(hn::Atan2(d, hn::Sqrt(hn::Neg(q)), mbDiv2), v_inv3);
        hn::VFromD<decltype(d)> cs, sn;

        hn::SinCos(d, angle, sn, cs);

        auto r0 = hn::Add(c2Div3, hn::Mul(hn::Mul(two, magnitude), cs));
        auto r1 = hn::Sub(c2Div3, hn::Mul(magnitude, hn::Add(cs, hn::Mul(v_root3, sn))));
        auto r2 = hn::Sub(c2Div3, hn::Mul(magnitude, hn::Sub(cs, hn::Mul(v_root3, sn))));

        auto v_r0_tmp = hn::Min(r0, r1);
        auto v_r1_tmp = hn::Max(r0, r1);

        auto v_r0 = hn::Min(v_r0_tmp, r2);
        auto v_r2_tmp = hn::Max(v_r0_tmp, r2);

        auto v_r1 = hn::Min(v_r1_tmp, v_r2_tmp);
        auto v_r2 = hn::Max(v_r1_tmp, v_r2_tmp);

        v_r0 = hn::Mul(v_r0, v_max_element);
        v_r1 = hn::Mul(v_r1, v_max_element);
        v_r2 = hn::Mul(v_r2, v_max_element);

        // clang-format on

        // Store the resulting eigenvalues in the descending order.
        hn::Store(v_r2, d, dst[0] + i);
        hn::Store(v_r1, d, dst[1] + i);
        hn::Store(v_r0, d, dst[2] + i);
    }
}

void l2norm(MultiDataView<3> src, float *dst) {
    if (src.count == 2) {
        Ufuncs<2>::l2norm(src, dst);
    } else if (src.count == 3) {
        Ufuncs<3>::l2norm(src, dst);
    }
}

void add(MultiDataView<3> src, float *dst) {
    if (src.count == 2) {
        Ufuncs<2>::add(src, dst);
    } else if (src.count == 3) {
        Ufuncs<3>::add(src, dst);
    }
}

void mul_pairs(MultiDataView<3> src, MultiOutputView<6> dst) {
    if (src.count == 2) {
        Ufuncs<2>::mul_pairs(src, dst);
    } else if (src.count == 3) {
        Ufuncs<3>::mul_pairs(src, dst);
    }
}

void eigenvalues(MultiDataView<6> src, MultiOutputView<3> dst) {
    if (src.count == 3) {
        eigenvalues2d(src, dst);
    } else if (src.count == 6) {
        eigenvalues3d(src, dst);
    }
}

size_t lane_count() { return hn::Lanes(D{}); }

// For the given unroll factor, define the corresponding convolve and auto-tuning
// functions.
#define DEF_UNROLLED(unroll)                                                           \
    void autotune##unroll(                                                             \
            size_t reps,                                                               \
            const float *src,                                                          \
            const float *kernel,                                                       \
            size_t ksize,                                                              \
            float *dst) {                                                              \
        for (size_t i = 0; i < reps; ++i) {                                            \
            conv_lanes<true, true, unroll>(src, kernel, ksize, dst, 0, 0, 0);          \
        }                                                                              \
    }                                                                                  \
    void convolve##unroll(                                                             \
            int axis, DataView3D src, KernelView kernel, float *dst, float *row_buf) { \
        conv_unroll<unroll>(axis, src, kernel, dst, row_buf);                          \
    }                                                                                  \
    static_assert(true, "For requiring trailing semicolon")

DEF_UNROLLED(1);
DEF_UNROLLED(2);
DEF_UNROLLED(4);
DEF_UNROLLED(8);
DEF_UNROLLED(16);

#undef DEF_UNROLLED

} // namespace fastfilters2::simd::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

// Function definitions for the header file. Each function just dispatches to the
// appropriate implementation from the corresponding function table built by Highway via
// the `HWY_EXPORT` macro.
#if HWY_ONCE
namespace fastfilters2::simd {

HWY_EXPORT(autotune1);
HWY_EXPORT(autotune2);
HWY_EXPORT(autotune4);
HWY_EXPORT(autotune8);
HWY_EXPORT(autotune16);

HWY_EXPORT(convolve1);
HWY_EXPORT(convolve2);
HWY_EXPORT(convolve4);
HWY_EXPORT(convolve8);
HWY_EXPORT(convolve16);

HWY_EXPORT(lane_count);

HWY_EXPORT(l2norm);
HWY_EXPORT(add);
HWY_EXPORT(mul_pairs);
HWY_EXPORT(eigenvalues);

static size_t g_lane_count;
static decltype(convolve) *g_convolve;

HWY_DLLEXPORT void initialize(int debug) {
    // Skip auto-tuning if it has already been done.
    if (g_convolve) {
        return;
    }

    // Large kernel with scale == 10 and order == 2.
    constexpr size_t radius = 4 * 10;
    constexpr size_t ksize = radius + 1;

    // Enough to cover the largest unroll factor.
    constexpr size_t size = 16 * (max_lane_bytes / sizeof(float)) + 2 * radius;

    alignas(max_lane_bytes) std::array<float, size> src{};
    alignas(max_lane_bytes) std::array<float, size> dst{};
    alignas(max_lane_bytes) std::array<float, ksize> kernel{};

    // Ensure that we are not consuming too much stack space.
    static_assert(sizeof src + sizeof dst + sizeof kernel <= 10240);

    auto autotune_funcs = std::array{
        HWY_DYNAMIC_POINTER(autotune1),
        HWY_DYNAMIC_POINTER(autotune2),
        HWY_DYNAMIC_POINTER(autotune4),
        HWY_DYNAMIC_POINTER(autotune8),
        HWY_DYNAMIC_POINTER(autotune16),
    };

    auto convolve_funcs = std::array{
        HWY_DYNAMIC_POINTER(convolve1),
        HWY_DYNAMIC_POINTER(convolve2),
        HWY_DYNAMIC_POINTER(convolve4),
        HWY_DYNAMIC_POINTER(convolve8),
        HWY_DYNAMIC_POINTER(convolve16),
    };

    auto unroll_from_index = [](size_t i) constexpr { return size_t{1} << i; };

    static_assert(autotune_funcs.size() == convolve_funcs.size());
    constexpr auto N = autotune_funcs.size();

    hwy::AutoTune<size_t, 12> tuner;
    std::vector<size_t> candidates;
    for (size_t i = 0; i < N; ++i) {
        candidates.push_back(i);
    }
    tuner.SetCandidates(candidates);

    while (!tuner.Best()) {
        auto i = tuner.NextConfig();
        auto func = autotune_funcs[i];
        auto t0 = hwy::timer::Start();

        // Run the function with a large number of repetitions in order to get a precise
        // estimate of the cost.
        func(1000, src.data() + radius, kernel.data(), ksize, dst.data());

        // `hwy::timer::Stop()` is more precise, but can fail on x86 CPUs without the
        // RDTSCP instruction. It is possible to check for the availability of RDTSCP,
        // but the added precision doesn't make a noticeable difference in this case,
        // and is not worth the added complexity.
        auto t1 = hwy::timer::Start();

        tuner.NotifyCost((t1 - t0) / unroll_from_index(i));
    }

    auto best = *tuner.Best();

    // Print the auto-tuning results. This is helpful for observing the actual selected
    // function variants.
    if (debug > 0) {
        std::fprintf(
                stderr,
                "[fastfilters2:initialize] arch=%s cost=<",
                hwy::TargetName(hwy::DispatchedTarget()));

        size_t i = 0;
        for (auto &cd : tuner.Costs()) {
            if (i > 0) {
                std::fprintf(stderr, " ");
            }
            std::fprintf(stderr, "%zu:%.0f", unroll_from_index(i++), cd.EstimateCost());
        }

        std::fprintf(stderr, "> unroll=%zu\n", unroll_from_index(best));
    }

    g_lane_count = HWY_DYNAMIC_DISPATCH(lane_count)();
    g_convolve = convolve_funcs[best];
}

HWY_DLLEXPORT size_t lane_count() { return g_lane_count; }

HWY_DLLEXPORT void
convolve(int axis, DataView3D src, KernelView kernel, float *dst, float *row_buf) {
    g_convolve(axis, src, kernel, dst, row_buf);
}

HWY_DLLEXPORT void l2norm(MultiDataView<3> src, float *dst) {
    HWY_DYNAMIC_DISPATCH(l2norm)(src, dst);
}

HWY_DLLEXPORT void add(MultiDataView<3> src, float *dst) {
    HWY_DYNAMIC_DISPATCH(add)(src, dst);
}

HWY_DLLEXPORT void mul_pairs(MultiDataView<3> src, MultiOutputView<6> dst) {
    HWY_DYNAMIC_DISPATCH(mul_pairs)(src, dst);
}

HWY_DLLEXPORT void eigenvalues(MultiDataView<6> src, MultiOutputView<3> dst) {
    HWY_DYNAMIC_DISPATCH(eigenvalues)(src, dst);
}

} // namespace fastfilters2::simd
#endif
