#include <fastfilters2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>
#include <unordered_map>

#include <hwy/aligned_allocator.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fastfilters2.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <hwy/contrib/math/math-inl.h>

#ifndef VARGS
#define VARGS v00, v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11, v12, v13, v14, v15
#endif

HWY_BEFORE_NAMESPACE();
namespace fastfilters2::HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename D> using pointer = hn::TFromD<D> *HWY_RESTRICT;
template <typename D> using const_pointer = const hn::TFromD<D> *HWY_RESTRICT;

// Need a helper in order to capture Is in a template parameter.
// This could be replaced with a template lambda in C++20.
template <size_t N, typename Func, typename... Args, size_t... Is>
HWY_INLINE void static_for_helper(std::index_sequence<Is...>, Func func, Args &&...args) {
    auto predicated_func = [func](auto i, auto &&arg) {
        if constexpr (i < N) {
            func(i, std::forward<decltype(arg)>(arg));
        }
    };
    (predicated_func(std::integral_constant<size_t, Is>{}, std::forward<Args>(args)), ...);
}

/**
 * Apply function to the first N arguments, with guaranteed compile-time unrolling.
 * The function recieves an integral_constant index and the corresponding argument.
 */
template <size_t N, typename Func, typename... Args> HWY_INLINE void static_for(Func func, Args &&...args) {
    static_assert(N <= sizeof...(Args), "N must be less than or equal to the number of arguments");
    static_for_helper<N>(std::index_sequence_for<Args...>{}, func, std::forward<Args>(args)...);
}

/**
 * Copy src to dst, reflecting the first and last radius elements except for the first and last element.
 * Return pointer to the first unreflected element in dst.
 */
template <typename T> T *reflect_copy(T *HWY_RESTRICT dst, const T *HWY_RESTRICT src, size_t size, size_t radius) {
    std::reverse_copy(src + 1, src + 1 + radius, dst);
    std::copy(src, src + size, dst);
    std::reverse_copy(src + size - 1 - radius, src + size - 1, dst + radius + size);
    return dst + radius;
}

template <typename D, size_t Unroll, bool Symmetric>
void dot_product(pointer<D> dst,
                 const_pointer<D> src,
                 const_pointer<D> kernel,
                 size_t radius,
                 size_t stride,
                 size_t lborder,
                 size_t rborder) {

    HWY_ASSUME(radius > 0);

    D d;
    auto lanes = hn::Lanes(d);
    auto kvec = hn::Set(d, kernel[0]);
    hn::VFromD<D> VARGS;

    static_for<Unroll>([&](auto i, auto &v) { v = hn::Mul(hn::LoadU(d, src + i * lanes), kvec); }, VARGS);

    auto lsrc = src;
    auto rsrc = src;

    for (size_t k = 1; k <= radius; ++k) {
        kvec = hn::Set(d, kernel[k]);
        lsrc = k <= lborder ? lsrc - stride : lsrc + stride;
        rsrc = k <= rborder ? rsrc + stride : rsrc - stride;

        static_for<Unroll>(
                [&](auto i, auto &v) {
                    auto lvec = hn::LoadU(d, lsrc + i * lanes);
                    auto rvec = hn::LoadU(d, rsrc + i * lanes);
                    if constexpr (Symmetric) {
                        v = hn::MulAdd(hn::Add(rvec, lvec), kvec, v);
                    } else {
                        v = hn::MulAdd(hn::Sub(rvec, lvec), kvec, v);
                    }
                },
                VARGS);
    }

    static_for<Unroll>([&](auto i, const auto &v) { hn::StoreU(v, d, dst + i * lanes); }, VARGS);
}

template <typename D, size_t Unroll, bool Symmetric> struct conv_context {
    using ptr = const hn::TFromD<D> *HWY_RESTRICT;
    using mut_ptr = hn::TFromD<D> *HWY_RESTRICT;

    shape_type shape;
    ptr kernel;
    size_t radius;
    mut_ptr buf;

    D d;
    size_t lanes = hn::Lanes(d);
    size_t step = Unroll * lanes;

    /**
     * Convolve a group of consecutive lanes.
     */
    void conv_lane_batch(mut_ptr dst, ptr src, size_t stride, size_t lborder, size_t rborder) {
        HWY_ASSUME(radius > 0);

        auto kvec = hn::Set(d, kernel[0]);
        hn::VFromD<D> VARGS;

        static_for<Unroll>([&](auto i, auto &v) { v = hn::Mul(hn::LoadU(d, src + i * lanes), kvec); }, VARGS);

        auto lsrc = src;
        auto rsrc = src;

        for (size_t k = 1; k <= radius; ++k) {
            kvec = hn::Set(d, kernel[k]);
            lsrc = k <= lborder ? lsrc - stride : lsrc + stride;
            rsrc = k <= rborder ? rsrc + stride : rsrc - stride;

            static_for<Unroll>(
                    [&](auto i, auto &v) {
                        auto lvec = hn::LoadU(d, lsrc + i * lanes);
                        auto rvec = hn::LoadU(d, rsrc + i * lanes);
                        if constexpr (Symmetric) {
                            v = hn::MulAdd(hn::Add(rvec, lvec), kvec, v);
                        } else {
                            v = hn::MulAdd(hn::Sub(rvec, lvec), kvec, v);
                        }
                    },
                    VARGS);
        }

        static_for<Unroll>([&](auto i, const auto &v) { hn::StoreU(v, d, dst + i * lanes); }, VARGS);
    }

    /**
     * Convolve single memory-contiguous line.
     */
    void conv_line_contig(mut_ptr dst, ptr src, size_t size) {
        src = reflect_copy(buf, src, size, radius);
        for (size_t i = 0; i < size; i += step) {
            i = HWY_MIN(i, size - step);
            conv_lane_batch(dst + i, src + i, 1, radius, radius);
        }
    }

    /**
     * Convolve multiple strided lines.
     */
    void conv_line_strided(mut_ptr dst, ptr src, size_t size, size_t stride) {
        for (size_t i = 0; i < size; ++i) {
            conv_lane_batch(dst + i, src + i, stride, i, size - 1 - i);
        }
    }

    /**
     * Convolve a plane of multiple strided lines.
     */
    void conv_plane(mut_ptr dst, ptr src, size_t size_contig, size_t size_strided, size_t stride) {
        for (size_t i = 0; i < size_contig; i += step) {
            i = HWY_MIN(i, size_contig - step);
            conv_line_strided(dst + i, src + i, size_strided, stride);
        }
    }

    void conv_x(mut_ptr dst, ptr src) {
        for (size_t i = 0; i < shape[2] * shape[1]; ++i) {
            conv_line_contig(dst, src, shape[0]);
            dst += shape[0];
            src += shape[0];
        }
    }

    void conv_y(mut_ptr dst, ptr src) {
        for (size_t i = 0; i < shape[2]; ++i) {
            conv_plane(dst, src, shape[0], shape[1], shape[0]);
            dst += shape[0] * shape[1];
            src += shape[0] * shape[1];
        }
    }

    void conv_z(mut_ptr dst, ptr src) {
        for (size_t i = 0; i < shape[1]; ++i) {
            conv_plane(dst, src, shape[0], shape[2], shape[0] * shape[1]);
            dst += shape[0];
            src += shape[0];
        }
    }
};

template <typename D> void mirror_copy(pointer<D> dst, const_pointer<D> src, size_t size, size_t radius) {
    std::reverse_copy(src + 1, src + 1 + radius, dst);
    std::copy(src, src + size, dst + radius);
    std::reverse_copy(src + size - 1 - radius, src + size - 1, dst + radius + size);
}

template <typename D, size_t Unroll, bool Symmetric>
void convolve_x(pointer<D> dst,
                const_pointer<D> src,
                shape_type shape,
                const_pointer<D> kernel,
                size_t radius,
                pointer<D> buf) {

    auto step = Unroll * hn::Lanes(D{});
    auto ostride = shape[0];
    constexpr size_t istride = 1;

    for (size_t zy = 0; zy < shape[2] * shape[1]; ++zy) {
        mirror_copy<D>(buf, src, shape[0], radius);
        for (size_t x = 0; x < shape[0]; x += step) {
            x = HWY_MIN(x, shape[0] - step);
            dot_product<D, Unroll, Symmetric>(dst + x, buf + radius + x, kernel, radius, istride, radius, radius);
        }
        dst += ostride;
        src += ostride;
    }
}

template <typename D, size_t Unroll, bool Symmetric>
void convolve_y(pointer<D> dst,
                const_pointer<D> src,
                shape_type shape,
                const_pointer<D> kernel,
                size_t radius,
                pointer<D> /* buf */) {

    auto step = Unroll * hn::Lanes(D{});
    auto ostride = shape[1] * shape[0];
    auto istride = shape[0];

    for (size_t z = 0; z < shape[2]; ++z) {
        for (size_t x = 0; x < shape[0]; x += step) {
            x = HWY_MIN(x, shape[0] - step);
            auto i = x;
            for (size_t y = 0; y < shape[1]; ++y) {
                dot_product<D, Unroll, Symmetric>(dst + i, src + i, kernel, radius, istride, y, shape[1] - 1 - y);
                i += istride;
            }
        }
        dst += ostride;
        src += ostride;
    }
}

template <typename D, size_t Unroll, bool Symmetric>
void convolve_z(pointer<D> dst,
                const_pointer<D> src,
                shape_type shape,
                const_pointer<D> kernel,
                size_t radius,
                pointer<D> /* buf */) {

    auto step = Unroll * hn::Lanes(D{});
    auto ostride = shape[0];
    auto istride = shape[1] * shape[0];

    for (size_t y = 0; y < shape[1]; ++y) {
        for (size_t x = 0; x < shape[0]; x += step) {
            x = HWY_MIN(x, shape[0] - step);
            auto i = x;
            for (size_t z = 0; z < shape[2]; ++z) {
                dot_product<D, Unroll, Symmetric>(dst + i, src + i, kernel, radius, istride, z, shape[2] - 1 - z);
                i += istride;
            }
        }
        dst += ostride;
        src += ostride;
    }
}

// Implementation adapted from https://mazzo.li/posts/vectorized-atan2.html
template <typename D, typename V> HWY_INLINE V Atan2(D d, V y, V x) {
    auto pi = hn::Set(d, 3.141592653589793);
    auto halfpi = hn::Set(d, 1.5707963267948966);

    auto swapmask = hn::Gt(hn::Abs(y), hn::Abs(x));
    auto input = hn::Div(hn::IfThenElse(swapmask, x, y), hn::IfThenElse(swapmask, y, x));
    auto result = hn::Atan(d, input);
    result = hn::IfThenElse(swapmask, hn::Sub(hn::CopySignToAbs(halfpi, input), result), result);
    result = hn::Add(hn::IfNegativeThenElse(x, hn::CopySignToAbs(pi, y), hn::Zero(d)), result);

    return result;
}

using D = hn::CappedTag<float, 16>;
constexpr size_t Unroll = 8;

size_t min_size() { return Unroll * hn::Lanes(D{}); }

pointer<D> convolve(size_t dim,
                    size_t order,
                    pointer<D> dst,
                    const_pointer<D> src,
                    shape_type shape,
                    const_pointer<D> kernel,
                    size_t radius,
                    pointer<D> buf) {
    if (dim == 0) {
        if (order % 2 == 0) {
            convolve_x<D, Unroll, true>(dst, src, shape, kernel, radius, buf);
        } else {
            convolve_x<D, Unroll, false>(dst, src, shape, kernel, radius, buf);
        }
    } else if (dim == 1) {
        if (order % 2 == 0) {
            convolve_y<D, Unroll, true>(dst, src, shape, kernel, radius, buf);
        } else {
            convolve_y<D, Unroll, false>(dst, src, shape, kernel, radius, buf);
        }
    } else if (dim == 2) {
        if (order % 2 == 0) {
            convolve_z<D, Unroll, true>(dst, src, shape, kernel, radius, buf);
        } else {
            convolve_z<D, Unroll, false>(dst, src, shape, kernel, radius, buf);
        }
    }
    return dst;
}

pointer<D> add(pointer<D> dst, const_pointer<D> src1, const_pointer<D> src2, const_pointer<D> src3, size_t size) {
    D d;
    auto lanes = hn::Lanes(d);
    if (src3 == nullptr) {
        for (size_t i = 0; i < size; i += lanes) {
            hn::Store(hn::Add(hn::Load(d, src1 + i), hn::Load(d, src2 + i)), d, dst + i);
        }
    } else {
        for (size_t i = 0; i < size; i += lanes) {
            hn::Store(
                    hn::Add(hn::Add(hn::Load(d, src1 + i), hn::Load(d, src2 + i)), hn::Load(d, src3 + i)), d, dst + i);
        }
    }
    return dst;
}

pointer<D> mul(pointer<D> dst, const_pointer<D> src1, const_pointer<D> src2, const_pointer<D> src3, size_t size) {
    D d;
    auto lanes = hn::Lanes(d);
    if (src3 == nullptr) {
        for (size_t i = 0; i < size; i += lanes) {
            hn::Store(hn::Mul(hn::Load(d, src1 + i), hn::Load(d, src2 + i)), d, dst + i);
        }
    } else {
        for (size_t i = 0; i < size; i += lanes) {
            hn::Store(
                    hn::Mul(hn::Mul(hn::Load(d, src1 + i), hn::Load(d, src2 + i)), hn::Load(d, src3 + i)), d, dst + i);
        }
    }
    return dst;
}

pointer<D> l2norm(pointer<D> dst, const_pointer<D> src1, const_pointer<D> src2, const_pointer<D> src3, size_t size) {
    D d;
    auto lanes = hn::Lanes(d);
    if (src3 == nullptr) {
        for (size_t i = 0; i < size; i += lanes) {
            auto v1 = hn::Load(d, src1 + i);
            auto v2 = hn::Load(d, src2 + i);
            v1 = hn::Mul(v1, v1);
            v2 = hn::Mul(v2, v2);
            hn::Store(hn::Sqrt(hn::Add(v1, v2)), d, dst + i);
        }
    } else {
        for (size_t i = 0; i < size; i += lanes) {
            auto v1 = hn::Load(d, src1 + i);
            auto v2 = hn::Load(d, src2 + i);
            auto v3 = hn::Load(d, src3 + i);
            v1 = hn::Mul(v1, v1);
            v2 = hn::Mul(v2, v2);
            v3 = hn::Mul(v3, v3);
            hn::Store(hn::Sqrt(hn::Add(hn::Add(v1, v2), v3)), d, dst + i);
        }
    }
    return dst;
}

void eigenvalues2(pointer<D> dst0,
                  pointer<D> dst1,
                  const_pointer<D> src00,
                  const_pointer<D> src01,
                  const_pointer<D> src11,
                  size_t size) {
    D d;
    auto lanes = hn::Lanes(d);
    auto onehalf = hn::Set(d, 0.5);

    for (size_t i = 0; i < size; i += lanes) {
        auto vec00 = hn::Load(d, src00 + i);
        auto vec01 = hn::Load(d, src01 + i);
        auto vec11 = hn::Load(d, src11 + i);

        auto tmp0 = hn::Mul(hn::Add(vec00, vec11), onehalf);

        auto tmp1 = hn::Mul(hn::Sub(vec00, vec11), onehalf);
        tmp1 = hn::Mul(tmp1, tmp1);

        auto det = hn::Sqrt(hn::Add(tmp1, hn::Mul(vec01, vec01)));

        hn::Store(hn::Add(tmp0, det), d, dst0 + i);
        hn::Store(hn::Sub(tmp0, det), d, dst1 + i);
    }
}

void eigenvalues3(pointer<D> ev0,
                  pointer<D> ev1,
                  pointer<D> ev2,
                  const_pointer<D> a00,
                  const_pointer<D> a01,
                  const_pointer<D> a02,
                  const_pointer<D> a11,
                  const_pointer<D> a12,
                  const_pointer<D> a22,
                  size_t size) {

    D d;
    auto lanes = hn::Lanes(d);

    auto v_inv3 = hn::Set(d, 1.0 / 3);
    auto v_root3 = hn::Sqrt(hn::Set(d, 3));
    auto two = hn::Set(d, 2);
    auto one = hn::Set(d, 1);
    auto half = hn::Set(d, 0.5);
    auto zero = hn::Zero(d);

    for (size_t i = 0; i < size; i += lanes) {
        auto v_a00 = hn::Load(d, a00 + i);
        auto v_a01 = hn::Load(d, a01 + i);
        auto v_a02 = hn::Load(d, a02 + i);
        auto v_a11 = hn::Load(d, a11 + i);
        auto v_a12 = hn::Load(d, a12 + i);
        auto v_a22 = hn::Load(d, a22 + i);

        // guard against float overflows
        auto v_max0 = hn::Max(hn::Abs(v_a00), hn::Abs(v_a01));
        auto v_max1 = hn::Max(hn::Abs(v_a02), hn::Abs(v_a11));
        auto v_max2 = hn::Max(hn::Abs(v_a12), hn::Abs(v_a22));
        auto v_max_element = hn::Max(hn::Max(v_max0, v_max1), v_max2);

        // replace zeros with ones to avoid NaNs
        v_max_element = hn::IfThenElse(hn::Eq(v_max_element, zero), one, v_max_element);

        v_a00 = hn::Div(v_a00, v_max_element);
        v_a01 = hn::Div(v_a01, v_max_element);
        v_a02 = hn::Div(v_a02, v_max_element);
        v_a11 = hn::Div(v_a11, v_max_element);
        v_a12 = hn::Div(v_a12, v_max_element);
        v_a22 = hn::Div(v_a22, v_max_element);

        // clang-format off
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
        // clang-format on

        aDiv3 = hn::Min(aDiv3, zero);

        auto mbDiv2 = hn::Mul(half, hn::Add(c0, hn::Mul(c2Div3, hn::Sub(hn::Mul(hn::Mul(two, c2Div3), c2Div3), c1))));
        auto q = hn::Add(hn::Mul(mbDiv2, mbDiv2), hn::Mul(hn::Mul(aDiv3, aDiv3), aDiv3));

        q = hn::Min(q, zero);

        auto magnitude = hn::Sqrt(hn::Neg(aDiv3));
        auto angle = hn::Mul(Atan2(d, hn::Sqrt(hn::Neg(q)), mbDiv2), v_inv3);

        // TODO: Compute cos and sin in one step.
        auto cs = hn::Cos(d, angle);
        auto sn = hn::Sin(d, angle);

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

        hn::Store(v_r0, d, ev2 + i);
        hn::Store(v_r1, d, ev1 + i);
        hn::Store(v_r2, d, ev0 + i);
    }
}

}; // namespace fastfilters2::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace fastfilters2 {
HWY_EXPORT(min_size);
HWY_EXPORT(convolve);
HWY_EXPORT(add);
HWY_EXPORT(mul);
HWY_EXPORT(l2norm);
HWY_EXPORT(eigenvalues2);
HWY_EXPORT(eigenvalues3);

size_t kernel_radius(double scale, size_t order) { return std::ceil((3 + 0.5 * order) * scale); }

void gaussian_kernel(float *kernel, size_t radius, double scale, size_t order) {
    using namespace std::placeholders;

    auto begin = kernel;
    auto end = kernel + radius + 1;

    double inv_sigma = 1 / scale;
    double inv_sigma2 = inv_sigma * inv_sigma;

    // inv_sqrt_tau = 1 / sqrt(2 * pi)
    constexpr double inv_sqrt_tau = 0.3989422804014327;
    double norm = inv_sqrt_tau * inv_sigma;
    if (order > 0) {
        norm = -norm * inv_sigma2;
    }

    for (size_t x = 0; x <= radius; ++x) {
        double x2_over_sigma2 = x * x * inv_sigma2;
        double g = norm * std::exp(-0.5 * x2_over_sigma2);
        if (order == 0) {
            kernel[x] = g;
        } else if (order == 1) {
            kernel[x] = g * x;
        } else if (order == 2) {
            kernel[x] = g * (1 - x2_over_sigma2);
        }
    }

    if (order == 2) {
        double halfsum = std::reduce(begin + 1, end, 0.0);
        double mean = (kernel[0] + 2 * halfsum) / (2 * radius + 1);
        std::transform(begin, end, begin, std::bind(std::minus{}, _1, mean));
    }

    double sum = 0.0;
    for (size_t x = 1; x <= radius; ++x) {
        if (order == 0) {
            sum += kernel[x];
        } else if (order == 1) {
            sum += kernel[x] * x;
        } else if (order == 2) {
            sum += kernel[x] * x * x;
        }
    }
    if (order != 2) {
        sum = kernel[0] + 2 * sum;
    }
    std::transform(begin, end, begin, std::bind(std::multiplies{}, _1, 1 / sum));
}

struct context {
    const float *src_;
    shape_type shape;
    size_t total;
    size_t radius[3];
    hwy::AlignedFreeUniquePtr<float[]> kernel[3];
    hwy::AlignedFreeUniquePtr<float[]> buf;
    std::vector<hwy::AlignedFreeUniquePtr<float[]>> tmp;

    context(const float *src, shape_type shape, double scale)
            : src_{src}, shape{shape}, total{shape[0] * shape[1] * shape[2]} {
        for (size_t order = 0; order < 3; ++order) {
            radius[order] = kernel_radius(scale, order);
            kernel[order] = hwy::AllocateAligned<float>(radius[order] + 1);
            gaussian_kernel(kernel[order].get(), radius[order], scale, order);
        }
        buf = hwy::AllocateAligned<float>(shape[0] + 2 * radius[2]);
    }

    HWY_INLINE const float *src() { return src_; }

    HWY_INLINE float *allocate() {
        tmp.push_back(hwy::AllocateAligned<float>(total));
        return tmp.back().get();
    }

    template <size_t DIM, size_t ORDER> HWY_INLINE void convolve(float *dst, const float *src) {
        static_assert(DIM < 3);
        static_assert(ORDER < 3);
        HWY_DYNAMIC_DISPATCH(convolve)(
                    DIM,
                    ORDER,
                    dst,
                    src,
                    shape,
                    kernel[ORDER].get(),
                    radius[ORDER],
                    buf.get());
    }
};

void gaussian_smoothing(float *out, const float *data, shape_type shape, size_t ndim, double scale) {
    context ctx{data, shape, scale};
    auto x = ctx.allocate();
    ctx.convolve<0, 0>(x, data);
    if (ndim == 2) {
        ctx.convolve<1, 0>(out, x);
    } else {
        auto y = ctx.allocate();
        ctx.convolve<1, 0>(y, x);
        ctx.convolve<2, 0>(out, y);
    }
}

void gaussian_gradient_magnitude(float *out, const float *data, shape_type shape, size_t ndim, double scale) {
    context ctx{data, shape, scale};
    auto x = ctx.allocate();
    auto y = ctx.allocate();
    ctx.convolve<0, 1>(x, ctx.src());
    ctx.convolve<1, 0>(y, x);
    if (ndim == 2) {
        HWY_DYNAMIC_DISPATCH(l2norm)(out, y, x, nullptr, ctx.total);
    } else {
        auto z = ctx.allocate();
    }
}

void laplacian_of_gaussian(float *out, const float *data, shape_type shape, size_t ndim, double scale) {
    static_cast<void>(out);
    static_cast<void>(data);
    static_cast<void>(shape);
    static_cast<void>(ndim);
    static_cast<void>(scale);
}

void hessian_of_gaussian_eigenvalues(float *out, const float *data, shape_type shape, size_t ndim, double scale) {
    static_cast<void>(out);
    static_cast<void>(data);
    static_cast<void>(shape);
    static_cast<void>(ndim);
    static_cast<void>(scale);
}

void structure_tensor_eigenvalues(float *out, const float *data, shape_type shape, size_t ndim, double scale) {
    static_cast<void>(out);
    static_cast<void>(data);
    static_cast<void>(shape);
    static_cast<void>(ndim);
    static_cast<void>(scale);
}

}; // namespace fastfilters2
#endif
