#include <fastfilters2.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include <hwy/aligned_allocator.h>

#if defined(_MSC_VER)
#define FASTFILTERS2_ASSUME(cond) __assume(cond)
#elif defined(__clang__)
#define FASTFILTERS2_ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUC__)
#define FASTFILTERS2_ASSUME(cond)                                              \
    ((cond) ? static_cast<void>(0) : __builtin_unreachable())
#else
#define FASTFILTERS2_ASSUME(cond) static_cast<void>((cond) ? 0 : 0)
#endif

#if defined(__clang__) || defined(__GNUC__)
#define FASTFILTERS2_LIKELY(x) __builtin_expect(!!(x), 1)
#define FASTFILTERS2_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define FASTFILTERS2_LIKELY(x) (!!(x))
#define FASTFILTERS2_UNLIKELY(x) (!!(x))
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fastfilters2.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace fastfilters2::HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <typename D, bool SYMMETRIC, bool STRIDED, int_t UNROLL>
HWY_INLINE void convolve_step(hn::TFromD<D> *HWY_RESTRICT dst,
                              const hn::TFromD<D> *HWY_RESTRICT src,
                              const hn::TFromD<D> *HWY_RESTRICT kernel,
                              int_t radius, int_t llimit, int_t rlimit,
                              int_t stride) {

    FASTFILTERS2_ASSUME(radius >= 1);
    if constexpr (!STRIDED) {
        static_cast<void>(llimit);
        static_cast<void>(rlimit);
        static_cast<void>(stride);
    }

    D d;
    auto right = src;
    auto left = src;
    auto k = hn::Set(d, kernel[0]);

#if HWY_HAVE_SCALABLE
    static_assert(UNROLL == 1);
    hn::VFromD<D> v = hn::Mul(hn::LoadU(d, src), k);
#else
    static_assert(UNROLL >= 1);
    hn::VFromD<D> v[UNROLL];
    for (int_t i = 0; i < UNROLL; ++i) {
        v[i] = hn::Mul(hn::LoadU(d, src + i * hn::Lanes(d)), k);
    }
#endif

    for (int_t r = 1; r <= radius; ++r) {
        if constexpr (STRIDED) {
            right += r <= rlimit ? stride : -stride;
            left -= r <= llimit ? stride : -stride;
        } else {
            ++right;
            --left;
        }
        k = hn::Set(d, kernel[r]);

#if HWY_HAVE_SCALABLE
        {
            auto vright = hn::LoadU(d, right);
            auto vleft = hn::LoadU(d, left);
            if constexpr (SYMMETRIC) {
                v = hn::MulAdd(hn::Add(vright, vleft), k, v);
            } else {
                v = hn::MulAdd(hn::Sub(vright, vleft), k, v);
            }
        }
#else
        for (int_t i = 0; i < UNROLL; ++i) {
            auto vright = hn::LoadU(d, right + i * hn::Lanes(d));
            auto vleft = hn::LoadU(d, left + i * hn::Lanes(d));
            if constexpr (SYMMETRIC) {
                v[i] = hn::MulAdd(hn::Add(vright, vleft), k, v[i]);
            } else {
                v[i] = hn::MulAdd(hn::Sub(vright, vleft), k, v[i]);
            }
        }
#endif
    }

#if HWY_HAVE_SCALABLE
    hn::StoreU(v, d, dst);
#else
    for (int_t i = 0; i < UNROLL; ++i) {
        hn::StoreU(v[i], d, dst + i * hn::Lanes(d));
    }
#endif
}

template <typename D>
HWY_INLINE void mirrored_edges_copy(hn::TFromD<D> *HWY_RESTRICT dst,
                                    const hn::TFromD<D> *HWY_RESTRICT src,
                                    int_t size, int_t radius) {
    std::reverse_copy(src + 1, src + radius + 1, dst - radius);
    std::copy(src, src + size, dst);
    std::reverse_copy(src + size - radius - 1, src + size - 1, dst + size);
}

template <typename D, bool SYMMETRIC, bool STRIDED, int_t UNROLL = 1>
HWY_INLINE void convolve_impl(hn::TFromD<D> *HWY_RESTRICT dst,
                              const hn::TFromD<D> *HWY_RESTRICT src,
                              const int_t *HWY_RESTRICT size,
                              const hn::TFromD<D> *HWY_RESTRICT kernel,
                              int_t radius, hn::TFromD<D> *HWY_RESTRICT buf) {

    D d;
    int_t step = UNROLL * hn::Lanes(d);

    buf += radius;

    for (int_t y = 0; y < size[0]; ++y) {
        auto curr_src = src + size[1] * y;
        auto curr_dst = dst + size[1] * y;

        if constexpr (STRIDED) {
            for (int_t x = 0; x < size[1]; x += step) {
                x = std::min(size[1] - step, x);
                convolve_step<D, SYMMETRIC, STRIDED, UNROLL>(
                    curr_dst + x, curr_src + x, kernel, radius, y,
                    size[0] - y - 1, size[1]);
            }

        } else {
            mirrored_edges_copy<D>(buf, curr_src, size[1], radius);
            for (int_t x = 0; x < size[1]; x += step) {
                x = std::min(size[1] - step, x);
                convolve_step<D, SYMMETRIC, STRIDED, UNROLL>(
                    curr_dst + x, buf + x, kernel, radius, 0, 0, size[1]);
            }
        }
    }
}

template <typename D>
HWY_INLINE void l2norm_impl(hn::TFromD<D> *HWY_RESTRICT dst,
                            const hn::TFromD<D> *HWY_RESTRICT src1,
                            const hn::TFromD<D> *HWY_RESTRICT src2, int_t n) {
    D d;
    int_t step = hn::Lanes(d);

    for (int_t x = 0; x < n; x += step) {
        x = std::min(n - step, x);
        auto v1 = hn::LoadU(d, src1 + x);
        auto v2 = hn::LoadU(d, src2 + x);
        v1 = hn::Mul(v1, v1);
        v2 = hn::Mul(v2, v2);
        hn::StoreU(hn::Sqrt(hn::Add(v1, v2)), d, dst + x);
    }
}

template <typename D>
HWY_INLINE void add_impl(hn::TFromD<D> *HWY_RESTRICT dst,
                         const hn::TFromD<D> *HWY_RESTRICT src1,
                         const hn::TFromD<D> *HWY_RESTRICT src2, int_t n) {
    D d;
    int_t step = hn::Lanes(d);

    for (int_t x = 0; x < n; x += step) {
        x = std::min(n - step, x);
        auto v1 = hn::LoadU(d, src1 + x);
        auto v2 = hn::LoadU(d, src2 + x);
        hn::StoreU(hn::Add(v1, v2), d, dst + x);
    }
}

template <typename D>
HWY_INLINE void mul_impl(hn::TFromD<D> *HWY_RESTRICT dst,
                         const hn::TFromD<D> *HWY_RESTRICT src1,
                         const hn::TFromD<D> *HWY_RESTRICT src2, int_t n) {
    D d;
    int_t step = hn::Lanes(d);

    for (int_t x = 0; x < n; x += step) {
        x = std::min(n - step, x);
        auto v1 = hn::LoadU(d, src1 + x);
        auto v2 = hn::LoadU(d, src2 + x);
        hn::StoreU(hn::Mul(v1, v2), d, dst + x);
    }
}

template <typename D>
HWY_INLINE void eigenvalues2_impl(hn::TFromD<D> *HWY_RESTRICT dst_ev1,
                                  hn::TFromD<D> *HWY_RESTRICT dst_ev2,
                                  const hn::TFromD<D> *HWY_RESTRICT src_xx,
                                  const hn::TFromD<D> *HWY_RESTRICT src_xy,
                                  const hn::TFromD<D> *HWY_RESTRICT src_yy,
                                  int_t n) {

    D d;
    int_t step = hn::Lanes(d);
    auto inv_two = hn::Set(d, static_cast<hn::TFromD<D>>(0.5));

    for (int_t x = 0; x < n; x += step) {
        x = std::min(n - step, x);

        auto xx = hn::LoadU(d, src_xx + x);
        auto xy = hn::LoadU(d, src_xy + x);
        auto yy = hn::LoadU(d, src_yy + x);

        auto halftrace = hn::Mul(hn::Add(xx, yy), inv_two);
        auto temp = hn::Mul(hn::Sub(xx, yy), inv_two);
        auto halfdist = hn::Sqrt(hn::Add(hn::Mul(temp, temp), hn::Mul(xy, xy)));

        hn::StoreU(hn::Add(halftrace, halfdist), d, dst_ev1 + x);
        hn::StoreU(hn::Sub(halftrace, halfdist), d, dst_ev2 + x);
    }
}

void convolve(val_t *dst, const val_t *src, const int_t *size,
              const val_t *kernel, int_t radius, val_t *buf, int_t order) {
    using D = hn::CappedTag<val_t, 64>;
    if (order % 2 == 0) {
        convolve_impl<D, true, false, 8>(dst, src, size, kernel, radius, buf);
    } else {
        convolve_impl<D, false, false, 8>(dst, src, size, kernel, radius, buf);
    }
}

void convolve_strided(val_t *dst, const val_t *src, const int_t *size,
                      const val_t *kernel, int_t radius, int_t order) {
    using D = hn::CappedTag<val_t, 64>;
    if (order % 2 == 0) {
        convolve_impl<D, true, true, 8>(dst, src, size, kernel, radius,
                                        nullptr);
    } else {
        convolve_impl<D, false, true, 8>(dst, src, size, kernel, radius,
                                         nullptr);
    }
}

void l2norm(val_t *dst, const val_t *src1, const val_t *src2, int_t n) {
    l2norm_impl<hn::ScalableTag<val_t>>(dst, src1, src2, n);
}

void add(val_t *dst, const val_t *src1, const val_t *src2, int_t n) {
    add_impl<hn::ScalableTag<val_t>>(dst, src1, src2, n);
}

void mul(val_t *dst, const val_t *src1, const val_t *src2, int_t n) {
    mul_impl<hn::ScalableTag<val_t>>(dst, src1, src2, n);
}

void eigenvalues2(val_t *ev1, val_t *ev2, const val_t *xx, const val_t *xy,
                  const val_t *yy, int_t n) {
    eigenvalues2_impl<hn::ScalableTag<val_t>>(ev1, ev2, xx, xy, yy, n);
}

}; // namespace fastfilters2::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace fastfilters2 {
int_t kernel_radius(double scale, int_t order) {
    return std::ceil((3 + 0.5 * order) * scale);
}

void gaussian_kernel(val_t *kernel, int_t radius, double scale, int_t order) {
    double inv_sigma = 1 / scale;
    double inv_sigma2 = inv_sigma * inv_sigma;

    // inv_sqrt_tau = 1 / sqrt(2 * pi)
    constexpr double inv_sqrt_tau = 0.3989422804014327;
    double norm = inv_sqrt_tau * inv_sigma;
    if (order > 0) {
        norm = -norm * inv_sigma2;
    }

    for (int_t x = 0; x <= radius; ++x) {
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
        double halfsum = 0.0;
        for (int_t x = 1; x <= radius; ++x) {
            halfsum += kernel[x];
        }
        double mean = (kernel[0] + 2 * halfsum) / (2 * radius + 1);
        for (int_t x = 0; x <= radius; ++x) {
            kernel[x] -= mean;
        }
    }

    double sum = 0.0;
    for (int_t x = 1; x <= radius; ++x) {
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

    double inv_sum = 1 / sum;
    for (int_t x = 0; x <= radius; ++x) {
        kernel[x] *= inv_sum;
    }
}

HWY_EXPORT(convolve);
HWY_EXPORT(convolve_strided);
HWY_EXPORT(l2norm);
HWY_EXPORT(add);
HWY_EXPORT(mul);
HWY_EXPORT(eigenvalues2);

void compute_filters(val_t *HWY_RESTRICT dst, const val_t *HWY_RESTRICT src,
                     const int_t *HWY_RESTRICT size, int_t ndim, double scale) {
    auto size_max = *std::max_element(size, size + ndim);
    auto size_total = std::reduce(
        size, size + ndim, static_cast<decltype(*size)>(1), std::multiplies{});

    int_t radius[3];
    hwy::AlignedFreeUniquePtr<val_t[]> kernel[3];
    for (int order = 0; order < 3; ++order) {
        radius[order] = kernel_radius(scale, order);
        kernel[order] = hwy::AllocateAligned<val_t>(radius[order] + 1);
        gaussian_kernel(kernel[order].get(), radius[order], scale, order);
    }

    hwy::AlignedFreeUniquePtr<val_t[]> x[3];
    for (int order = 0; order < 3; ++order) {
        x[order] = hwy::AllocateAligned<val_t>(size_total);
    }

    auto buf = hwy::AllocateAligned<val_t>(size_max + 2 * radius[2]);
    for (int order = 0; order < 3; ++order) {
        HWY_DYNAMIC_DISPATCH(convolve)
        (x[order].get(), src, size, kernel[order].get(), radius[order],
         buf.get(), order);
    }

    // Gaussian Smoothing.

    HWY_DYNAMIC_DISPATCH(convolve_strided)
    (dst, x[0].get(), size, kernel[0].get(), radius[0], 0);
    dst += size_total;

    // Gaussian Gradient Magnitude.

    auto x1y0 = hwy::AllocateAligned<val_t>(size_total);
    auto x0y1 = hwy::AllocateAligned<val_t>(size_total);
    HWY_DYNAMIC_DISPATCH(convolve_strided)
    (x1y0.get(), x[1].get(), size, kernel[0].get(), radius[0], 0);
    HWY_DYNAMIC_DISPATCH(convolve_strided)
    (x0y1.get(), x[0].get(), size, kernel[1].get(), radius[1], 1);

    HWY_DYNAMIC_DISPATCH(l2norm)(dst, x1y0.get(), x0y1.get(), size_total);
    dst += size_total;

    // Laplacian of Gaussian.

    auto x2y0 = hwy::AllocateAligned<val_t>(size_total);
    auto x0y2 = hwy::AllocateAligned<val_t>(size_total);
    HWY_DYNAMIC_DISPATCH(convolve_strided)
    (x2y0.get(), x[2].get(), size, kernel[0].get(), radius[0], 0);
    HWY_DYNAMIC_DISPATCH(convolve_strided)
    (x0y2.get(), x[0].get(), size, kernel[2].get(), radius[2], 2);

    HWY_DYNAMIC_DISPATCH(add)(dst, x2y0.get(), x0y2.get(), size_total);
    dst += size_total;

    // Hessian of Gaussian Eigenvalues.

    auto x1y1 = hwy::AllocateAligned<val_t>(size_total);
    HWY_DYNAMIC_DISPATCH(convolve_strided)
    (x1y1.get(), x[1].get(), size, kernel[1].get(), radius[1], 1);

    HWY_DYNAMIC_DISPATCH(eigenvalues2)
    (dst, dst + size_total, x2y0.get(), x1y1.get(), x0y2.get(), size_total);
    dst += 2 * size_total;

    // Structure Tensor Eigenvalues.

    auto ste_xx = hwy::AllocateAligned<val_t>(size_total);
    auto ste_xy = hwy::AllocateAligned<val_t>(size_total);
    auto ste_yy = hwy::AllocateAligned<val_t>(size_total);
    auto tmp = hwy::AllocateAligned<val_t>(size_total);

    auto ste_smooth_scale = 0.5 * scale;
    auto ste_smooth_radius = kernel_radius(ste_smooth_scale, 0);
    auto ste_kernel = hwy::AllocateAligned<val_t>(ste_smooth_radius + 1);
    gaussian_kernel(ste_kernel.get(), ste_smooth_radius, ste_smooth_scale, 0);

    HWY_DYNAMIC_DISPATCH(mul)(ste_xx.get(), x1y0.get(), x1y0.get(), size_total);
    HWY_DYNAMIC_DISPATCH(mul)(ste_xy.get(), x1y0.get(), x0y1.get(), size_total);
    HWY_DYNAMIC_DISPATCH(mul)(ste_yy.get(), x0y1.get(), x0y1.get(), size_total);

    HWY_DYNAMIC_DISPATCH(convolve)
    (tmp.get(), ste_xx.get(), size, ste_kernel.get(), ste_smooth_radius,
     buf.get(), 0);
    HWY_DYNAMIC_DISPATCH(convolve_strided)
    (ste_xx.get(), tmp.get(), size, ste_kernel.get(), ste_smooth_radius, 0);

    HWY_DYNAMIC_DISPATCH(convolve)
    (tmp.get(), ste_xy.get(), size, ste_kernel.get(), ste_smooth_radius,
     buf.get(), 0);
    HWY_DYNAMIC_DISPATCH(convolve_strided)
    (ste_xy.get(), tmp.get(), size, ste_kernel.get(), ste_smooth_radius, 0);

    HWY_DYNAMIC_DISPATCH(convolve)
    (tmp.get(), ste_yy.get(), size, ste_kernel.get(), ste_smooth_radius,
     buf.get(), 0);
    HWY_DYNAMIC_DISPATCH(convolve_strided)
    (ste_yy.get(), tmp.get(), size, ste_kernel.get(), ste_smooth_radius, 0);

    HWY_DYNAMIC_DISPATCH(eigenvalues2)
    (dst, dst + size_total, ste_xx.get(), ste_xy.get(), ste_yy.get(),
     size_total);
}

}; // namespace fastfilters2
#endif
