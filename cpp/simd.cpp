#include "simd.hpp"

#include <hwy/base.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd.cpp"
#include <hwy/foreach_target.h>

#include <hwy/highway.h>

#include <hwy/contrib/math/math-inl.h>

HWY_BEFORE_NAMESPACE();
namespace fastfilters2::simd::HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

static float *mirror_copy(const float *HWY_RESTRICT src, float *HWY_RESTRICT dst,
                          size_t size, size_t radius) {
    for (size_t i = 0; i < radius; ++i) {
        dst[i] = src[radius - i];
    }
    for (size_t i = 0; i < size; ++i) {
        dst[radius + i] = src[i];
    }
    for (size_t i = 0; i < radius; ++i) {
        dst[radius + size + i] = src[size - 2 - i];
    }
    return dst + radius;
}

template <bool contiguous, bool symmetric, size_t unroll, typename D>
static void convolve_lanes(D d, const float *src, const float *kernel, size_t ksize,
                           float *dst, size_t index, size_t size, size_t stride) {
    static_assert(1 <= unroll && unroll <= 8);
    hn::VFromD<D> acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;

    auto lanes = hn::Lanes(d);
    auto kv = hn::Set(d, kernel[0]);
    auto left = src;
    auto right = src;

#define _INIT(i)                                                                       \
    if constexpr (i < unroll) {                                                        \
        acc##i = hn::Mul(hn::LoadU(d, src + i * lanes), kv);                           \
    }

#define _STEP(i)                                                                       \
    if constexpr (i < unroll) {                                                        \
        auto lv = hn::LoadU(d, left + i * lanes);                                      \
        auto rv = hn::LoadU(d, right + i * lanes);                                     \
        if constexpr (symmetric) {                                                     \
            acc##i = hn::MulAdd(hn::Add(rv, lv), kv, acc##i);                          \
        } else {                                                                       \
            acc##i = hn::MulAdd(hn::Sub(rv, lv), kv, acc##i);                          \
        }                                                                              \
    }

#define _STORE(i)                                                                      \
    if constexpr (i < unroll) {                                                        \
        hn::StoreU(acc##i, d, dst + i * lanes);                                        \
    }

    // clang-format off
    _INIT(0); _INIT(1); _INIT(2); _INIT(3); _INIT(4); _INIT(5); _INIT(6); _INIT(7);
    // clang-format on

    for (size_t k = 1; k < ksize; ++k) {
        kv = hn::Set(d, kernel[k]);

        if constexpr (contiguous) {
            --left;
            ++right;
        } else {
            left = k <= index ? left - stride : left + stride;
            right = k <= size - 1 - index ? right + stride : right - stride;
        }

        // clang-format off
        _STEP(0); _STEP(1); _STEP(2); _STEP(3); _STEP(4); _STEP(5); _STEP(6); _STEP(7);
        // clang-format on
    }

    // clang-format off
    _STORE(0); _STORE(1); _STORE(2); _STORE(3); _STORE(4); _STORE(5); _STORE(6); _STORE(7);
    // clang-format on

#undef _INIT
#undef _STEP
#undef _STORE
}

template <bool symmetric, size_t unroll, typename D>
static void convolve_contiguous(D d, const float *src, const float *kernel,
                                size_t ksize, float *dst, float *scratch,
                                size_t row_size, size_t outer_size) {
    auto step = unroll * hn::Lanes(d);

    for (size_t outer = 0; outer < outer_size; ++outer) {
        auto row = mirror_copy(src, scratch, row_size, ksize - 1);

        for (size_t x = 0; x < row_size; x += step) {
            x = HWY_MIN(x, row_size - step);

            convolve_lanes<true, symmetric, unroll>(d, row + x, kernel, ksize, dst + x,
                                                    0, 0, 0);
        }

        src += row_size;
        dst += row_size;
    }
}

template <bool symmetric, size_t unroll, typename D>
static void convolve_strided(D d, const float *src, const float *kernel, size_t ksize,
                             float *dst, size_t row_size, size_t inner_size,
                             size_t inner_stride, size_t outer_size,
                             size_t outer_stride) {
    auto step = unroll * hn::Lanes(d);

    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t x = 0; x < row_size; x += step) {
            x = HWY_MIN(x, row_size - step);
            auto xsrc = src + x;
            auto xdst = dst + x;

            for (size_t inner = 0; inner < inner_size; ++inner) {
                convolve_lanes<false, symmetric, unroll>(
                        d, xsrc, kernel, ksize, xdst, inner, inner_size, inner_stride);

                xsrc += inner_stride;
                xdst += inner_stride;
            }
        }

        src += outer_stride;
        dst += outer_stride;
    }
}

template <Op op, size_t nargs, typename D>
static void simple_op(D d, const float **srcs, float *dst, size_t size) {
    static_assert(2 <= nargs && nargs <= 3);
    auto lanes = hn::Lanes(d);

    for (size_t i = 0; i < size; i += lanes) {
        i = HWY_MIN(i, size - lanes);

        hn::VFromD<D> a, b, c, v;
        a = hn::LoadU(d, srcs[0] + i);
        b = hn::LoadU(d, srcs[1] + i);
        if constexpr (nargs == 3) {
            c = hn::LoadU(d, srcs[2] + i);
        }

        if constexpr (op == Op::add) {
            v = hn::Add(a, b);
            if constexpr (nargs == 3) {
                v = hn::Add(c, v);
            }

        } else if constexpr (op == Op::mul) {
            v = hn::Mul(a, b);
            if constexpr (nargs == 3) {
                v = hn::Mul(c, v);
            }

        } else if constexpr (op == Op::l2norm) {
            v = hn::Mul(a, a);
            v = hn::MulAdd(b, b, v);
            if constexpr (nargs == 3) {
                v = hn::MulAdd(c, c, v);
            }
            v = hn::Sqrt(v);
        }

        hn::StoreU(v, d, dst + i);
    }
}

template <typename D>
static void eigenvalues_2d(D d, const float **srcs, float *dst, size_t size) {
    auto ev0 = dst;
    auto ev1 = dst + size;

    auto lanes = hn::Lanes(d);
    auto half = hn::Set(d, 0.5);

    for (size_t i = 0; i < size; i += lanes) {
        i = HWY_MIN(i, size - lanes);

        auto xx = hn::LoadU(d, srcs[0] + i);
        auto xy = hn::LoadU(d, srcs[1] + i);
        auto yy = hn::LoadU(d, srcs[2] + i);

        auto half_trace = hn::Mul(half, hn::Add(xx, yy));
        auto tmp = hn::Mul(half, hn::Sub(xx, yy));
        auto half_sqrt_discr = hn::Sqrt(hn::MulAdd(tmp, tmp, hn::Mul(xy, xy)));

        hn::StoreU(hn::Add(half_trace, half_sqrt_discr), d, ev0 + i);
        hn::StoreU(hn::Sub(half_trace, half_sqrt_discr), d, ev1 + i);
    }
}

template <typename D>
static void eigenvalues_3d(D d, const float **srcs, float *dst, size_t size) {
    auto ev0 = dst;
    auto ev1 = dst + size;
    auto ev2 = dst + 2 * size;

    auto lanes = hn::Lanes(d);
    auto zero = hn::Zero(d);
    auto third = hn::Set(d, 0.3333333333333333);
    auto half = hn::Set(d, 0.5);
    auto one = hn::Set(d, 1);
    auto root3 = hn::Set(d, 1.7320508075688772);
    auto two = hn::Set(d, 2);

    for (size_t i = 0; i < size; i += lanes) {
        i = HWY_MIN(i, size - lanes);

        // clang-format off

        auto v_a00 = hn::LoadU(d, srcs[0] + i);
        auto v_a01 = hn::LoadU(d, srcs[1] + i);
        auto v_a02 = hn::LoadU(d, srcs[2] + i);
        auto v_a11 = hn::LoadU(d, srcs[3] + i);
        auto v_a12 = hn::LoadU(d, srcs[4] + i);
        auto v_a22 = hn::LoadU(d, srcs[5] + i);

        // guard against float overflows
        auto v_max0 = hn::Max(hn::Abs(v_a00), hn::Abs(v_a01));
        auto v_max1 = hn::Max(hn::Abs(v_a02), hn::Abs(v_a11));
        auto v_max2 = hn::Max(hn::Abs(v_a12), hn::Abs(v_a22));
        auto v_max_element = hn::Max(hn::Max(v_max0, v_max1), v_max2);

        // replace zeros with ones to avoid NaNs
        // v_max_element = hn::Or(v_max_element, hn::And(one, hn::Eq(v_max_element, zero)));
        v_max_element = hn::IfThenElse(hn::Eq(v_max_element, zero), one, v_max_element);

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
        auto c2Div3 = hn::Mul(c2, third);
        auto aDiv3 = hn::Mul(hn::Sub(c1, hn::Mul(c2, c2Div3)), third);

        aDiv3 = hn::Min(aDiv3, zero);

        auto mbDiv2 = hn::Mul(half, hn::Add(c0, hn::Mul(c2Div3, hn::Sub(hn::Mul(hn::Mul(two, c2Div3), c2Div3), c1))));
        auto q = hn::Add(hn::Mul(mbDiv2, mbDiv2), hn::Mul(hn::Mul(aDiv3, aDiv3), aDiv3));

        q = hn::Min(q, zero);

        auto magnitude = hn::Sqrt(hn::Neg(aDiv3));
        auto angle = hn::Mul(hn::Atan2(d, hn::Sqrt(hn::Neg(q)), mbDiv2), third);
        hn::VFromD<D> cs, sn;

        hn::SinCos(d, angle, sn, cs);

        auto r0 = hn::Add(c2Div3, hn::Mul(hn::Mul(two, magnitude), cs));
        auto r1 = hn::Sub(c2Div3, hn::Mul(magnitude, hn::Add(cs, hn::Mul(root3, sn))));
        auto r2 = hn::Sub(c2Div3, hn::Mul(magnitude, hn::Sub(cs, hn::Mul(root3, sn))));

        auto v_r0_tmp = hn::Min(r0, r1);
        auto v_r1_tmp = hn::Max(r0, r1);

        auto v_r0 = hn::Min(v_r0_tmp, r2);
        auto v_r2_tmp = hn::Max(v_r0_tmp, r2);

        auto v_r1 = hn::Min(v_r1_tmp, v_r2_tmp);
        auto v_r2 = hn::Max(v_r1_tmp, v_r2_tmp);

        v_r0 = hn::Mul(v_r0, v_max_element);
        v_r1 = hn::Mul(v_r1, v_max_element);
        v_r2 = hn::Mul(v_r2, v_max_element);

        hn::StoreU(v_r0, d, ev2 + i);
        hn::StoreU(v_r1, d, ev1 + i);
        hn::StoreU(v_r2, d, ev0 + i);

        // clang-format on
    }
}

using SimdTag = hn::CappedTag<float, 64 / sizeof(float)>;

#if HWY_TARGET == HWY_AVX2
static constexpr size_t max_unroll = 8;
#else
static constexpr size_t max_unroll = 2;
#endif

size_t convolve_width() { return max_unroll * hn::Lanes(SimdTag{}); }

void convolve(int dim, Image data, Kernel kernel, float *dst, float *scratch) {
    SimdTag d;
    auto symmetric = kernel.order % 2 == 0;
    auto row_size = data.shape[0];

    if (dim == 0) {
        auto outer_size = data.shape[1] * data.shape[2];

        // clang-format off
        if (symmetric) {
            convolve_contiguous<true, max_unroll>(
                d, data.data, kernel.data, kernel.size, dst,
                scratch, row_size, outer_size);
        } else {
            convolve_contiguous<false, max_unroll>(
                d, data.data, kernel.data, kernel.size, dst,
                scratch, row_size, outer_size);
        }
        // clang-format on

    } else {
        size_t outer_size, outer_stride, inner_size, inner_stride;
        if (dim == 1) {
            outer_size = data.shape[2];
            outer_stride = data.shape[0] * data.shape[1];
            inner_size = data.shape[1];
            inner_stride = data.shape[0];
        } else {
            outer_size = data.shape[1];
            outer_stride = data.shape[0];
            inner_size = data.shape[2];
            inner_stride = data.shape[0] * data.shape[1];
        }

        // clang-format off
        if (symmetric) {
            convolve_strided<true, max_unroll>(
                d, data.data, kernel.data, kernel.size, dst,
                row_size, inner_size, inner_stride, outer_size, outer_stride);
        } else {
            convolve_strided<false, max_unroll>(
                d, data.data, kernel.data, kernel.size, dst,
                row_size, inner_size, inner_stride, outer_size, outer_stride);
        }
        // clang-format on
    }
}

void ufunc(Op op, const float **srcs, size_t nargs, float *dst, size_t size) {
    SimdTag d;

    if (op == Op::add) {
        if (nargs == 2) {
            simple_op<Op::add, 2>(d, srcs, dst, size);
        } else if (nargs == 3) {
            simple_op<Op::add, 3>(d, srcs, dst, size);
        }
    } else if (op == Op::mul) {
        if (nargs == 2) {
            simple_op<Op::mul, 2>(d, srcs, dst, size);
        } else if (nargs == 3) {
            simple_op<Op::mul, 3>(d, srcs, dst, size);
        }
    } else if (op == Op::l2norm) {
        if (nargs == 2) {
            simple_op<Op::l2norm, 2>(d, srcs, dst, size);
        } else if (nargs == 3) {
            simple_op<Op::l2norm, 3>(d, srcs, dst, size);
        }
    } else if (op == Op::eigenvalues) {
        if (nargs == 3) {
            eigenvalues_2d(d, srcs, dst, size);
        } else if (nargs == 6) {
            eigenvalues_3d(d, srcs, dst, size);
        }
    }
}

}; // namespace fastfilters2::simd::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace fastfilters2::simd {

HWY_EXPORT(convolve_width);
size_t convolve_width() { return HWY_DYNAMIC_DISPATCH(convolve_width)(); }

HWY_EXPORT(convolve);
void convolve(int dim, Image data, Kernel kernel, float *dst, float *scratch) {
    HWY_DYNAMIC_DISPATCH(convolve)(dim, data, kernel, dst, scratch);
}

HWY_EXPORT(ufunc);
void ufunc(Op op, const float **srcs, size_t nargs, float *dst, size_t size) {
    HWY_DYNAMIC_DISPATCH(ufunc)(op, srcs, nargs, dst, size);
}

} // namespace fastfilters2::simd
#endif
