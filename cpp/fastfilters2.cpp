#include <fastfilters2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <unordered_map>

#include <hwy/aligned_allocator.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fastfilters2.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>

#ifndef VARGS
#define VARGS                                                                                                          \
    v00, v01, v02, v03, v04, v05, v06, v07, v08, v09, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, \
            v23, v24, v25, v26, v27, v28, v29, v30, v31
#endif

HWY_BEFORE_NAMESPACE();
namespace fastfilters2::HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename D> using pointer = hn::TFromD<D> *HWY_RESTRICT;
template <typename D> using const_pointer = const hn::TFromD<D> *HWY_RESTRICT;

template <size_t N, typename Func, typename... Args> void static_for(Func func, Args &&...args) {
    auto call = [func](auto idx, auto &&arg) {
        if constexpr (idx < N) {
            func(idx, std::forward<decltype(arg)>(arg));
        }
    };
    auto fold = [call]<size_t... Idx>(std::index_sequence<Idx...>, auto &&...args) {
        (call(std::integral_constant<size_t, Idx>{}, std::forward<decltype(args)>(args)), ...);
    };
    fold(std::index_sequence_for<Args...>{}, std::forward<Args>(args)...);
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
    auto istride = 1;

    for (size_t zy = 0; zy < shape[2] * shape[1]; ++zy) {
        mirror_copy<D>(buf, src, shape[0], radius);
        for (size_t x = 0; x < shape[0]; x += step) {
            // x = HWY_MIN(x, shape[0] - step);
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
            // x = HWY_MIN(x, shape[0] - step);
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
            // x = HWY_MIN(x, shape[0] - step);
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

template <typename D, typename V> HWY_INLINE V Atan2(D d, V y, V x) {
    // Implementation adapted from https://mazzo.li/posts/vectorized-atan2.html

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

size_t batch_size() { return Unroll * hn::Lanes(D{}); }

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
HWY_EXPORT(batch_size);
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

struct filters::impl {
    impl(const float *data, shape_type shape, size_t ndim, double scale) : shape{shape}, ndim{ndim}, scale{scale} {
        // Round up line_size to a multiple of batch size, which is always a non-zero power of 2.
        auto batch_size = HWY_DYNAMIC_DISPATCH(batch_size)();
        line_size = (shape[0] + batch_size - 1) & -batch_size;

        size = line_size * shape[1] * shape[2];

        for (size_t order = 0; order < 3; ++order) {
            radius[order] = kernel_radius(scale, order);
            kernel[order] = hwy::AllocateAligned<float>(radius[order] + 1);
            gaussian_kernel(kernel[order].get(), radius[order], scale, order);
        }

        buf = hwy::AllocateAligned<float>(HWY_MAX(size, line_size + 2 * radius[2]));

        auto dst = conv<>();
        for (size_t zy = 0; zy < shape[2] * shape[1]; ++zy) {
            std::copy_n(data, shape[0], dst);
            std::fill(dst + shape[0], dst + line_size, 0);
            data += shape[0];
            dst += line_size;
        }
    };

    void gaussian_smoothing(float *out) {
        if (ndim == 2) {
            pack(out, conv<0, 0>());
        } else if (ndim == 3) {
            pack(out, conv<0, 0, 0>());
        }
    }

    void gaussian_gradient_magnitude(float *out) {
        auto l2norm = HWY_DYNAMIC_DISPATCH(l2norm);
        if (ndim == 2) {
            pack(out, l2norm(buf.get(), conv<1, 0>(), conv<0, 1>(), nullptr, size));
        } else if (ndim == 3) {
            pack(out, l2norm(buf.get(), conv<1, 0, 0>(), conv<0, 1, 0>(), conv<0, 0, 1>(), size));
        }
    }

    void laplacian_of_gaussian(float *out) {
        auto add = HWY_DYNAMIC_DISPATCH(add);
        if (ndim == 2) {
            pack(out, add(buf.get(), conv<2, 0>(), conv<0, 2>(), nullptr, size));
        } else if (ndim == 3) {
            pack(out, add(buf.get(), conv<2, 0, 0>(), conv<0, 2, 0>(), conv<0, 0, 2>(), size));
        }
    }

    void hessian_of_gaussian_eigenvalues(float *out) {
        if (ndim == 2) {

            auto slot1 = slot<1>();
            auto slot2 = slot<2>();

            auto ev = HWY_DYNAMIC_DISPATCH(eigenvalues2);
            ev(slot1, slot2, conv<2, 0>(), conv<1, 1>(), conv<0, 2>(), size);

            out = pack(out, slot1);
            out = pack(out, slot2);

        } else if (ndim == 3) {

            auto slot1 = slot<1>();
            auto slot2 = slot<2>();
            auto slot3 = slot<3>();

            auto ev = HWY_DYNAMIC_DISPATCH(eigenvalues3);
            ev(slot1,
               slot2,
               slot3,
               conv<2, 0, 0>(),
               conv<1, 1, 0>(),
               conv<1, 0, 1>(),
               conv<0, 2, 0>(),
               conv<0, 1, 1>(),
               conv<0, 0, 2>(),
               size);

            out = pack(out, slot1);
            out = pack(out, slot2);
            out = pack(out, slot3);
        }
    }

    void structure_tensor_eigenvalues(float *out) {
        auto f = HWY_DYNAMIC_DISPATCH(convolve);

        auto scale_ex = 0.5 * scale;

        auto radius0_ex = kernel_radius(scale_ex, 0);
        auto radius1_ex = kernel_radius(scale_ex, 1);

        auto kernel0_ex = hwy::AllocateAligned<float>(radius0_ex + 1);
        auto kernel1_ex = hwy::AllocateAligned<float>(radius1_ex + 1);

        gaussian_kernel(kernel0_ex.get(), radius0_ex, scale_ex, 0);
        gaussian_kernel(kernel1_ex.get(), radius1_ex, scale_ex, 1);

        if (ndim == 2) {

            auto slot1 = slot<1>();
            auto tmp0 = slot<2>();
            auto tmp1 = slot<3>();

            f(0, 1, slot1, conv<>(), shape, kernel1_ex.get(), radius1_ex, buf.get());
            f(1, 0, tmp0, slot1, shape, kernel0_ex.get(), radius0_ex, buf.get());

            f(0, 0, slot1, conv<>(), shape, kernel0_ex.get(), radius0_ex, buf.get());
            f(1, 1, tmp1, slot1, shape, kernel1_ex.get(), radius1_ex, buf.get());

            auto src00 = ste_gaussian_smoothing<1, 4>(tmp0, tmp0);
            auto src01 = ste_gaussian_smoothing<1, 5>(tmp0, tmp1);
            auto src11 = ste_gaussian_smoothing<1, 6>(tmp1, tmp1);

            auto dst0 = slot<7>();
            auto dst1 = slot<8>();

            auto ev = HWY_DYNAMIC_DISPATCH(eigenvalues2);
            ev(dst0, dst1, src00, src01, src11, size);

            out = pack(out, dst0);
            out = pack(out, dst1);

        } else if (ndim == 3) {

            auto slot1 = slot<1>();

            auto tmp0 = slot<2>();
            auto tmp1 = slot<3>();
            auto tmp2 = slot<4>();

            f(0, 1, tmp0, conv<>(), shape, kernel1_ex.get(), radius1_ex, buf.get());
            f(1, 0, slot1, tmp0, shape, kernel0_ex.get(), radius0_ex, buf.get());
            f(2, 0, tmp0, slot1, shape, kernel0_ex.get(), radius0_ex, buf.get());

            f(0, 0, tmp1, conv<>(), shape, kernel0_ex.get(), radius0_ex, buf.get());
            f(1, 1, slot1, tmp1, shape, kernel1_ex.get(), radius1_ex, buf.get());
            f(2, 0, tmp1, slot1, shape, kernel0_ex.get(), radius0_ex, buf.get());

            f(0, 0, tmp2, conv<>(), shape, kernel0_ex.get(), radius0_ex, buf.get());
            f(1, 0, slot1, tmp2, shape, kernel0_ex.get(), radius0_ex, buf.get());
            f(2, 1, tmp2, slot1, shape, kernel1_ex.get(), radius1_ex, buf.get());

            auto src00 = ste_gaussian_smoothing<1, 5>(tmp0, tmp0);
            auto src01 = ste_gaussian_smoothing<1, 6>(tmp0, tmp1);
            auto src02 = ste_gaussian_smoothing<1, 7>(tmp0, tmp2);
            auto src11 = ste_gaussian_smoothing<1, 8>(tmp1, tmp1);
            auto src12 = ste_gaussian_smoothing<1, 9>(tmp1, tmp2);
            auto src22 = ste_gaussian_smoothing<1, 10>(tmp2, tmp2);

            auto dst0 = slot<11>();
            auto dst1 = slot<12>();
            auto dst2 = slot<13>();

            auto ev = HWY_DYNAMIC_DISPATCH(eigenvalues3);
            ev(dst0, dst1, dst2, src00, src01, src02, src11, src12, src22, size);

            out = pack(out, dst0);
            out = pack(out, dst1);
            out = pack(out, dst2);

        }
    }

    template <int tmp_idx, int dst_idx> float *ste_gaussian_smoothing(const float *HWY_RESTRICT src1, const float *HWY_RESTRICT src2) {
        auto f = HWY_DYNAMIC_DISPATCH(convolve);
        auto mul = HWY_DYNAMIC_DISPATCH(mul);

        auto tmp = slot<tmp_idx>();
        auto dst = slot<dst_idx>();

        if (ndim == 2) {
            mul(dst, src1, src2, nullptr, size);
            f(0, 0, tmp, dst, shape, kernel[0].get(), radius[0], buf.get());
            f(1, 0, dst, tmp, shape, kernel[0].get(), radius[0], buf.get());
        } else if (ndim == 3) {
            mul(tmp, src1, src2, nullptr, size);
            f(0, 0, dst, tmp, shape, kernel[0].get(), radius[0], buf.get());
            f(1, 0, tmp, dst, shape, kernel[0].get(), radius[0], buf.get());
            f(2, 0, dst, tmp, shape, kernel[0].get(), radius[0], buf.get());
        }

        return dst;
    }

    float *pack(float *HWY_RESTRICT dst, const float *HWY_RESTRICT src) {
        for (size_t zy = 0; zy < shape[2] * shape[1]; ++zy) {
            std::copy_n(src, shape[0], dst);
            src += line_size;
            dst += shape[0];
        }
        return dst;
    }

    template <int idx> float *slot() {
        static_assert(idx > 0);
        constexpr auto key = -idx;
        auto it = cache.find(key);
        if (it == cache.end()) {
            it = cache.emplace(key, hwy::AllocateAligned<float>(size)).first;
        }
        return it->second.get();
    }

    template <int order0 = -1, int order1 = -1, int order2 = -1> float *conv() {
        return iconv<order0, order1, order2>();
    }

    template <int order0 = -1, int order1 = -1, int order2 = -1> HWY_INLINE float *iconv() {
        static_assert(-1 <= order0 && order0 < 3);
        static_assert(-1 <= order1 && order1 < 3);
        static_assert(-1 <= order2 && order2 < 3);
        constexpr auto key = (order0 + 1) + 4 * (order1 + 1) + 16 * (order2 + 1);

        if (auto it = cache.find(key); it != cache.end()) {
            return it->second.get();
        }

        auto dst = cache.emplace(key, hwy::AllocateAligned<float>(size)).first->second.get();

        auto f = HWY_DYNAMIC_DISPATCH(convolve);
        if constexpr (order2 >= 0) {
            f(2, order2, dst, iconv<order0, order1>(), shape, kernel[order2].get(), radius[order2], buf.get());
        } else if constexpr (order1 >= 0) {
            f(1, order1, dst, iconv<order0>(), shape, kernel[order1].get(), radius[order1], buf.get());
        } else if constexpr (order0 >= 0) {
            f(0, order0, dst, iconv<>(), shape, kernel[order0].get(), radius[order0], buf.get());
        }

        return dst;
    }

    shape_type shape;
    size_t ndim;
    double scale;

    size_t line_size;
    size_t size;
    std::array<size_t, 3> radius;
    std::array<hwy::AlignedFreeUniquePtr<float[]>, 3> kernel;
    hwy::AlignedFreeUniquePtr<float[]> buf;

    std::unordered_map<int, hwy::AlignedFreeUniquePtr<float[]>> cache;

    size_t ste_radius;
    hwy::AlignedFreeUniquePtr<float[]> ste_kernel;
};

filters::filters(const float *data, shape_type shape, size_t ndim, double scale)
        : pimpl{std::make_unique<impl>(data, shape, ndim, scale)} {}

filters::filters() = default;
filters::filters(filters &&other) noexcept = default;
filters &filters::operator=(filters &&) noexcept = default;
filters::~filters() = default;

void filters::gaussian_smoothing(float *out) { pimpl->gaussian_smoothing(out); }
void filters::gaussian_gradient_magnitude(float *out) { pimpl->gaussian_gradient_magnitude(out); }
void filters::laplacian_of_gaussian(float *out) { pimpl->laplacian_of_gaussian(out); }
void filters::hessian_of_gaussian_eigenvalues(float *out) { pimpl->hessian_of_gaussian_eigenvalues(out); }
void filters::structure_tensor_eigenvalues(float *out) { pimpl->structure_tensor_eigenvalues(out); }

}; // namespace fastfilters2
#endif
