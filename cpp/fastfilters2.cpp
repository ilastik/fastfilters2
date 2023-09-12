#include "fastfilters2.h"

#include <hwy/aligned_allocator.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#include <iostream>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fastfilters2.cpp"
#include <hwy/foreach_target.h>

#include <hwy/highway.h>

#include <hwy/contrib/math/math-inl.h>

HWY_BEFORE_NAMESPACE();
namespace fastfilters2::HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

void gaussian_kernel(float *kernel, size_t radius, double scale, size_t order) {
    static constexpr auto inv_sqrt_tau = 0.3989422804014327029;
    auto inv_scale = 1 / scale;
    auto inv_scale2 = inv_scale * inv_scale;
    auto norm = inv_sqrt_tau * inv_scale;
    if (order > 0) {
        norm = -norm * inv_scale2;
    }

    for (size_t x = 0; x <= radius; ++x) {
        auto x2_over_scale2 = x * x * inv_scale2;
        auto g = norm * std::exp(-0.5 * x2_over_scale2);
        if (order == 0) {
            kernel[x] = g;
        } else if (order == 1) {
            kernel[x] = g * x;
        } else if (order == 2) {
            kernel[x] = g * (1 - x2_over_scale2);
        }
    }

    if (order == 2) {
        auto sum = 0.0;
        for (size_t x = 1; x <= radius; ++x) {
            sum += kernel[x];
        }
        sum = kernel[0] + 2 * sum;

        auto mean = sum / (2 * radius + 1);
        for (size_t x = 0; x <= radius; ++x) {
            kernel[x] -= mean;
        }
    }

    auto sum = 0.0;
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

    auto inv_sum = 1 / sum;
    for (size_t x = 0; x <= radius; ++x) {
        kernel[x] *= inv_sum;
    }
}

template <typename T>
void copy_mirrored(T *HWY_RESTRICT dst,
                   const T *HWY_RESTRICT src,
                   size_t size,
                   size_t radius) {
    HWY_ASSUME(radius >= 1);
    for (size_t i = 0; i < radius; ++i) {
        dst[i] = src[radius - i];
    }
    for (size_t i = 0; i < size; ++i) {
        dst[radius + i] = src[i];
    }
    for (size_t i = 0; i < radius; ++i) {
        dst[radius + size + i] = src[size - 2 - i];
    }
}

template <typename D,
          bool symmetric,
          size_t unroll,
          bool contiguous,
          typename T = hn::TFromD<D>>
void convolve_lanes(T *dst,
                    const T *src,
                    const T *kernel,
                    size_t radius,
                    size_t stride,
                    size_t lborder,
                    size_t rborder) {

    HWY_ASSUME(radius >= 1);
    D d;
    auto n = hn::Lanes(d);
    auto k = hn::Set(d, kernel[0]);

    static_assert(1 <= unroll && unroll <= 16);
    hn::VFromD<D> v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, va, vb, vc, vd, ve, vf;

    // clang-format off
    if constexpr (unroll > 0x0) { v0 = hn::Mul(hn::LoadU(d, src + 0x0 * n), k); }
    if constexpr (unroll > 0x1) { v1 = hn::Mul(hn::LoadU(d, src + 0x1 * n), k); }
    if constexpr (unroll > 0x2) { v2 = hn::Mul(hn::LoadU(d, src + 0x2 * n), k); }
    if constexpr (unroll > 0x3) { v3 = hn::Mul(hn::LoadU(d, src + 0x3 * n), k); }
    if constexpr (unroll > 0x4) { v4 = hn::Mul(hn::LoadU(d, src + 0x4 * n), k); }
    if constexpr (unroll > 0x5) { v5 = hn::Mul(hn::LoadU(d, src + 0x5 * n), k); }
    if constexpr (unroll > 0x6) { v6 = hn::Mul(hn::LoadU(d, src + 0x6 * n), k); }
    if constexpr (unroll > 0x7) { v7 = hn::Mul(hn::LoadU(d, src + 0x7 * n), k); }
    if constexpr (unroll > 0x8) { v8 = hn::Mul(hn::LoadU(d, src + 0x8 * n), k); }
    if constexpr (unroll > 0x9) { v9 = hn::Mul(hn::LoadU(d, src + 0x9 * n), k); }
    if constexpr (unroll > 0xa) { va = hn::Mul(hn::LoadU(d, src + 0xa * n), k); }
    if constexpr (unroll > 0xb) { vb = hn::Mul(hn::LoadU(d, src + 0xb * n), k); }
    if constexpr (unroll > 0xc) { vc = hn::Mul(hn::LoadU(d, src + 0xc * n), k); }
    if constexpr (unroll > 0xd) { vd = hn::Mul(hn::LoadU(d, src + 0xd * n), k); }
    if constexpr (unroll > 0xe) { ve = hn::Mul(hn::LoadU(d, src + 0xe * n), k); }
    if constexpr (unroll > 0xf) { vf = hn::Mul(hn::LoadU(d, src + 0xf * n), k); }
    // clang-format on

    auto pl = src;
    auto pr = src;

    auto combine = [d](auto pl, auto pr, auto k, auto v) {
        if constexpr (symmetric) {
            return hn::MulAdd(hn::Add(hn::LoadU(d, pr), hn::LoadU(d, pl)), k, v);
        } else {
            return hn::MulAdd(hn::Sub(hn::LoadU(d, pr), hn::LoadU(d, pl)), k, v);
        }
    };

    for (size_t i = 1; i <= radius; ++i) {
        k = hn::Set(d, kernel[i]);

        if constexpr (contiguous) {
            --pl;
            ++pr;
        } else {
            pl = (i <= lborder) ? pl - stride : pl + stride;
            pr = (i <= rborder) ? pr + stride : pr - stride;
        }

        // clang-format off
        if constexpr (unroll > 0x0) { v0 = combine(pl + 0x0 * n, pr + 0x0 * n, k, v0); }
        if constexpr (unroll > 0x1) { v1 = combine(pl + 0x1 * n, pr + 0x1 * n, k, v1); }
        if constexpr (unroll > 0x2) { v2 = combine(pl + 0x2 * n, pr + 0x2 * n, k, v2); }
        if constexpr (unroll > 0x3) { v3 = combine(pl + 0x3 * n, pr + 0x3 * n, k, v3); }
        if constexpr (unroll > 0x4) { v4 = combine(pl + 0x4 * n, pr + 0x4 * n, k, v4); }
        if constexpr (unroll > 0x5) { v5 = combine(pl + 0x5 * n, pr + 0x5 * n, k, v5); }
        if constexpr (unroll > 0x6) { v6 = combine(pl + 0x6 * n, pr + 0x6 * n, k, v6); }
        if constexpr (unroll > 0x7) { v7 = combine(pl + 0x7 * n, pr + 0x7 * n, k, v7); }
        if constexpr (unroll > 0x8) { v8 = combine(pl + 0x8 * n, pr + 0x8 * n, k, v8); }
        if constexpr (unroll > 0x9) { v9 = combine(pl + 0x9 * n, pr + 0x9 * n, k, v9); }
        if constexpr (unroll > 0xa) { va = combine(pl + 0xa * n, pr + 0xa * n, k, va); }
        if constexpr (unroll > 0xb) { vb = combine(pl + 0xb * n, pr + 0xb * n, k, vb); }
        if constexpr (unroll > 0xc) { vc = combine(pl + 0xc * n, pr + 0xc * n, k, vc); }
        if constexpr (unroll > 0xd) { vd = combine(pl + 0xd * n, pr + 0xd * n, k, vd); }
        if constexpr (unroll > 0xe) { ve = combine(pl + 0xe * n, pr + 0xe * n, k, ve); }
        if constexpr (unroll > 0xf) { vf = combine(pl + 0xf * n, pr + 0xf * n, k, vf); }
        // clang-format on
    }

    // clang-format off
    if constexpr (unroll > 0x0) { hn::StoreU(v0, d, dst + 0x0 * n); }
    if constexpr (unroll > 0x1) { hn::StoreU(v1, d, dst + 0x1 * n); }
    if constexpr (unroll > 0x2) { hn::StoreU(v2, d, dst + 0x2 * n); }
    if constexpr (unroll > 0x3) { hn::StoreU(v3, d, dst + 0x3 * n); }
    if constexpr (unroll > 0x4) { hn::StoreU(v4, d, dst + 0x4 * n); }
    if constexpr (unroll > 0x5) { hn::StoreU(v5, d, dst + 0x5 * n); }
    if constexpr (unroll > 0x6) { hn::StoreU(v6, d, dst + 0x6 * n); }
    if constexpr (unroll > 0x7) { hn::StoreU(v7, d, dst + 0x7 * n); }
    if constexpr (unroll > 0x8) { hn::StoreU(v8, d, dst + 0x8 * n); }
    if constexpr (unroll > 0x9) { hn::StoreU(v9, d, dst + 0x9 * n); }
    if constexpr (unroll > 0xa) { hn::StoreU(va, d, dst + 0xa * n); }
    if constexpr (unroll > 0xb) { hn::StoreU(vb, d, dst + 0xb * n); }
    if constexpr (unroll > 0xc) { hn::StoreU(vc, d, dst + 0xc * n); }
    if constexpr (unroll > 0xd) { hn::StoreU(vd, d, dst + 0xd * n); }
    if constexpr (unroll > 0xe) { hn::StoreU(ve, d, dst + 0xe * n); }
    if constexpr (unroll > 0xf) { hn::StoreU(vf, d, dst + 0xf * n); }
    // clang-format on
}

template <size_t dim,
          typename D,
          bool symmetric,
          size_t unroll,
          typename T = hn::TFromD<D>>
void convolve(T *dst,
              const T *src,
              size_t size_x,
              size_t size_y,
              size_t size_z,
              const T *kernel,
              size_t radius,
              T *buffer) {

    static_assert(dim < 3);
    constexpr auto contiguous = dim == 0;

    auto step = unroll * hn::Lanes(D{});
    auto size_row = std::max(step, size_x);

    size_t size_inner, size_outer, stride_inner, stride_outer_src, stride_outer_dst;
    if constexpr (dim == 0) {
        size_inner = 1;
        size_outer = size_y * size_z;
        stride_inner = 0;
        stride_outer_src = size_x;
        stride_outer_dst = size_row;
    } else if constexpr (dim == 1) {
        size_inner = size_y;
        size_outer = size_z;
        stride_inner = size_x;
        stride_outer_src = size_x * size_y;
        stride_outer_dst = stride_outer_src;
    } else if constexpr (dim == 2) {
        size_inner = size_z;
        size_outer = size_y;
        stride_inner = size_x * size_y;
        stride_outer_src = size_x;
        stride_outer_dst = stride_outer_src;
    }

    for (size_t outer = 0; outer < size_outer; ++outer) {
        if constexpr (contiguous) {
            copy_mirrored(buffer, src, size_x, radius);
        }

        for (size_t x = 0; x < size_row; x += step) {
            x = HWY_MIN(size_row - step, x);

            decltype(src) psrc;
            if constexpr (contiguous) {
                psrc = buffer + radius + x;
            } else {
                psrc = src + x;
            }
            auto pdst = dst + x;

            for (size_t inner = 0; inner < size_inner; ++inner) {
                auto lborder = inner;
                auto rborder = size_inner - 1 - inner;
                convolve_lanes<D, symmetric, unroll, contiguous>(
                        pdst, psrc, kernel, radius, stride_inner, lborder, rborder);
                psrc += stride_inner;
                pdst += stride_inner;
            }
        }

        src += stride_outer_src;
        dst += stride_outer_dst;
    }
}

template <typename D, typename T = hn::TFromD<D>>
void eigenvalues2(
        T *dst0, T *dst1, const T *src00, const T *src01, const T *src11, size_t size) {
    D d;
    auto half = hn::Set(d, 0.5);

    for (size_t i = 0; i < size; i += hn::Lanes(d)) {
        auto v00 = hn::Load(d, src00 + i);
        auto v01 = hn::Load(d, src01 + i);
        auto v11 = hn::Load(d, src11 + i);

        auto tmp0 = hn::Mul(hn::Add(v00, v11), half);
        auto tmp1 = hn::Mul(hn::Sub(v00, v11), half);
        tmp1 = hn::Mul(tmp1, tmp1);

        auto det = hn::Sqrt(hn::Add(tmp1, hn::Mul(v01, v01)));

        hn::Store(hn::Add(tmp0, det), d, dst0 + i);
        hn::Store(hn::Sub(tmp0, det), d, dst1 + i);
    }
}

template <typename D, typename T = hn::TFromD<D>>
void eigenvalues3(T *dst0,
                  T *dst1,
                  T *dst2,
                  const T *src00,
                  const T *src01,
                  const T *src02,
                  const T *src11,
                  const T *src12,
                  const T *src22,
                  size_t size) {
    D d;
    auto v_inv3 = hn::Set(d, 1 / 3.0);
    auto v_root3 = hn::Sqrt(hn::Set(d, 3));
    auto two = hn::Set(d, 2);
    auto one = hn::Set(d, 1);
    auto half = hn::Set(d, 0.5);
    auto zero = hn::Zero(d);

    for (size_t i = 0; i < size; i += hn::Lanes(d)) {
        // clang-format off

        auto v_a00 = hn::Load(d, src00 + i);
        auto v_a01 = hn::Load(d, src01 + i);
        auto v_a02 = hn::Load(d, src02 + i);
        auto v_a11 = hn::Load(d, src11 + i);
        auto v_a12 = hn::Load(d, src12 + i);
        auto v_a22 = hn::Load(d, src22 + i);

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
        hn::VFromD<D> cs, sn;

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

        hn::Store(v_r0, d, dst2 + i);
        hn::Store(v_r1, d, dst1 + i);
        hn::Store(v_r2, d, dst0 + i);

        // clang-format on
    }
}

template <typename D, size_t unroll, typename T = hn::TFromD<D>> class filters {
private:
    D d;
    T *dst;
    const T *src;
    size_t size[3];
    size_t ndim;
    double scale;
    double window_ratio;
    size_t size_row;
    size_t size_total;
    hwy::AlignedFreeUniquePtr<T[]> row_buffer;
    hwy::AlignedFreeUniquePtr<T[]> kernels[3];
    size_t radii[3];
    hwy::AlignedFreeUniquePtr<T[]> tmp_buffer;

public:
    explicit filters(params params)
            : d{},
              dst{params.dst},
              src{params.src},
              size{params.size[0], params.size[1], params.size[2]},
              ndim{params.ndim},
              scale{params.scale},
              window_ratio{params.window_ratio},
              size_row{std::max(unroll * hn::Lanes(d), size[0])},
              size_total{size_row * size[1] * size[2]},
              row_buffer{hwy::AllocateAligned<T>(3 * size_row - 2)} {
        std::fill_n(row_buffer.get(), 3 * size_row - 2, 0);
    }

private:
    template <size_t order> void init_kernel() {
        radii[order] = kernel_radius(scale, order, window_ratio);
        kernels[order] = hwy::AllocateAligned<T>(radii[order] + 1);
        gaussian_kernel(kernels[order].get(), radii[order], scale, order);
    }

    template <size_t count> std::array<T*, count> new_buffers() {
        tmp_buffer = hwy::AllocateAligned<T>(count * size_total);
        std::array<T*, count> ptrs;
        T *p = tmp_buffer.get();
        for (size_t i = 0; i < count; ++i, p += size_total) {
            ptrs[i] = p;
        }
        return ptrs;
    }

    template <size_t dim, size_t order> T *conv(T *dst, const T *src) {
        constexpr bool symmetric = order % 2 == 0;
        auto size0 = dim == 0 ? size[0] : size_row;
        auto kernel = kernels[order].get();
        auto radius = radii[order];
        auto buffer = row_buffer.get();
        convolve<dim, D, symmetric, unroll>(
                dst, src, size0, size[1], size[2], kernel, radius, buffer);
        return dst;
    }

   template <typename F> void vmap2(T *dst, const T *src1, const T *src2, F func) {
        D d;
        for (size_t i = 0; i < size_total; i += hn::Lanes(d)) {
            auto v1 = hn::Load(d, src1 + i);
            auto v2 = hn::Load(d, src2 + i);
            hn::Store(func(v1, v2), d, dst + i);
        }
    }

    template <typename F>
    void vmap3(T *dst, const T *src1, const T *src2, const T *src3, F func) {
        D d;
        for (size_t i = 0; i < size_total; i += hn::Lanes(d)) {
            auto v1 = hn::Load(d, src1 + i);
            auto v2 = hn::Load(d, src2 + i);
            auto v3 = hn::Load(d, src3 + i);
            hn::Store(func(v1, v2, v3), d, dst + i);
        }
    }

    T *copy_packed(T *HWY_RESTRICT dst, const T *HWY_RESTRICT src) {
        for (size_t i = 0; i < size_total; i += size_row, dst += size[0]) {
            std::copy_n(src + i, size[0], dst);
        }
        return dst;
    }

public:
    void gaussian_smoothing() {
        init_kernel<0>();
        auto [t1, t2] = new_buffers<2>();
        auto res = conv<1, 0>(t2, conv<0, 0>(t1, src));
        if (ndim == 3) {
            res = conv<2, 0>(t1, res);
        }
        copy_packed(dst, res);
    }

    void gaussian_gradient_magnitude() {
        init_kernel<0>();
        init_kernel<1>();

        if (ndim == 2) {
            auto [t1, t2, t3] = new_buffers<3>();

            conv<1, 0>(t1, conv<0, 1>(t3, src));
            conv<1, 1>(t2, conv<0, 0>(t3, src));

            vmap2(t3, t1, t2, [](auto v1, auto v2) {
                return hn::Sqrt(hn::Add(hn::Mul(v1, v1), hn::Mul(v2, v2)));
            });
            copy_packed(dst, t3);

        } else {
            auto [t1, t2, t3, t4] = new_buffers<4>();

            conv<2, 0>(t1, conv<1, 0>(t4, conv<0, 1>(t3, src)));
            conv<2, 0>(t2, conv<1, 1>(t4, conv<0, 0>(t3, src)));
            conv<2, 1>(t3, conv<1, 0>(t4, t3));

            vmap3(t4, t1, t2, t3, [](auto v1, auto v2, auto v3) {
                return hn::Sqrt(hn::Add(hn::Add(hn::Mul(v1, v1), hn::Mul(v2, v2)),
                                        hn::Mul(v3, v3)));
            });
            copy_packed(dst, t4);
        }
    }

    void laplacian_of_gaussian() {
        init_kernel<0>();
        init_kernel<2>();

        if (ndim == 2) {
            auto [t1, t2, t3] = new_buffers<3>();

            conv<1, 0>(t1, conv<0, 2>(t3, src));
            conv<1, 2>(t2, conv<0, 0>(t3, src));

            vmap2(t3, t1, t2, [](auto v1, auto v2) { return hn::Add(v1, v2); });
            copy_packed(dst, t3);

        } else {
            auto [t1, t2, t3, t4] = new_buffers<4>();

            conv<2, 0>(t1, conv<1, 0>(t4, conv<0, 2>(t3, src)));
            conv<2, 0>(t2, conv<1, 2>(t4, conv<0, 0>(t3, src)));
            conv<2, 2>(t3, conv<1, 0>(t4, t3));

            vmap3(t4, t1, t2, t3, [](auto v1, auto v2, auto v3) {
                return hn::Add(hn::Add(v1, v2), v3);
            });
            copy_packed(dst, t4);
        }
    }

    void hessian_of_gaussian_eigenvalues() {
        init_kernel<0>();
        init_kernel<1>();
        init_kernel<2>();

        if (ndim == 2) {
            auto [t0, t1, xx, xy, yy] = new_buffers<5>();

            conv<1, 0>(xx, conv<0, 2>(t0, src));
            conv<1, 1>(xy, conv<0, 1>(t0, src));
            conv<1, 2>(yy, conv<0, 0>(t0, src));

            eigenvalues2<D>(t0, t1, xx, xy, yy, size_total);
            copy_packed(copy_packed(dst, t0), t1);

        } else {
            auto [t0, t1, t2, xx, xy, xz, yy, yz, zz] = new_buffers<9>();

            conv<2, 0>(xx, conv<1, 0>(t1, conv<0, 2>(t0, src)));
            conv<2, 0>(xy, conv<1, 1>(t1, conv<0, 1>(t0, src)));
            conv<2, 1>(xz, conv<1, 0>(t1, t0));
            conv<2, 0>(yy, conv<1, 2>(t1, conv<0, 0>(t0, src)));
            conv<2, 1>(yz, conv<1, 1>(t1, t0));
            conv<2, 2>(zz, conv<1, 0>(t1, t0));

            eigenvalues3<D>(t0, t1, t2, zz, yz, xz, yy, xy, xx, size_total);
            copy_packed(copy_packed(copy_packed(dst, t0), t1), t2);
        }
    }

    void structure_tensor_eigenvalues(double st_scale) {
        auto smooth_scale = scale;
        scale = st_scale;
        init_kernel<0>();
        init_kernel<1>();

        if (ndim == 2) {
            auto [x, y, xx, xy, yy] = new_buffers<5>();

            conv<1, 0>(x, conv<0, 1>(xx, src));
            conv<1, 1>(y, conv<0, 0>(xx, src));

            for (size_t i = 0; i < size_total; i += hn::Lanes(d)) {
                auto vx = hn::Load(d, x + i);
                auto vy = hn::Load(d, y + i);
                hn::Store(hn::Mul(vx, vx), d, xx + i);
                hn::Store(hn::Mul(vx, vy), d, xy + i);
                hn::Store(hn::Mul(vy, vy), d, yy + i);
            }

            scale = smooth_scale;
            init_kernel<0>();
            conv<1, 0>(xx, conv<0, 0>(x, xx));
            conv<1, 0>(xy, conv<0, 0>(x, xy));
            conv<1, 0>(yy, conv<0, 0>(x, yy));

            eigenvalues2<D>(x, y, xx, xy, yy, size_total);
            copy_packed(copy_packed(dst, x), y);

        } else {
            auto [x, y, z, xx, xy, xz, yy, yz, zz] = new_buffers<9>();

            conv<2, 0>(x, conv<1, 0>(yy, conv<0, 1>(xx, src)));
            conv<2, 0>(y, conv<1, 1>(yy, conv<0, 0>(xx, src)));
            conv<2, 1>(z, conv<1, 0>(yy, xx));

            for (size_t i = 0; i < size_total; i += hn::Lanes(d)) {
                auto vx = hn::Load(d, x + i);
                auto vy = hn::Load(d, y + i);
                auto vz = hn::Load(d, z + i);
                hn::Store(hn::Mul(vx, vx), d, xx + i);
                hn::Store(hn::Mul(vx, vy), d, xy + i);
                hn::Store(hn::Mul(vx, vz), d, xz + i);
                hn::Store(hn::Mul(vy, vy), d, yy + i);
                hn::Store(hn::Mul(vy, vz), d, yz + i);
                hn::Store(hn::Mul(vz, vz), d, zz + i);
            }

            scale = smooth_scale;
            init_kernel<0>();
            conv<2, 0>(xx, conv<1, 0>(y, conv<0, 0>(x, xx)));
            conv<2, 0>(xy, conv<1, 0>(y, conv<0, 0>(x, xy)));
            conv<2, 0>(xz, conv<1, 0>(y, conv<0, 0>(x, xz)));
            conv<2, 0>(yy, conv<1, 0>(y, conv<0, 0>(x, yy)));
            conv<2, 0>(yz, conv<1, 0>(y, conv<0, 0>(x, yz)));
            conv<2, 0>(zz, conv<1, 0>(y, conv<0, 0>(x, zz)));

            eigenvalues3<D>(x, y, z, zz, yz, xz, yy, xy, xx, size_total);
            copy_packed(copy_packed(copy_packed(dst, x), y), z);
        }
    }
};

using D = hn::CappedTag<float, 64 / sizeof(float)>;
static constexpr size_t unroll = 8;

void gaussian_smoothing(params params) {
    filters<D, unroll>{params}.gaussian_smoothing();
}

void gaussian_gradient_magnitude(params params) {
    filters<D, unroll>{params}.gaussian_gradient_magnitude();
}

void laplacian_of_gaussian(params params) {
    filters<D, unroll>{params}.laplacian_of_gaussian();
}

void hessian_of_gaussian_eigenvalues(params params) {
    filters<D, unroll>{params}.hessian_of_gaussian_eigenvalues();
}

void structure_tensor_eigenvalues(params params, double st_scale) {
    filters<D, unroll>{params}.structure_tensor_eigenvalues(st_scale);
}

} // namespace fastfilters2::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace fastfilters2 {

HWY_EXPORT(gaussian_kernel);
void gaussian_kernel(float *kernel, size_t radius, double scale, size_t order) {
    HWY_DYNAMIC_DISPATCH(gaussian_kernel)(kernel, radius, scale, order);
}

HWY_EXPORT(gaussian_smoothing);
void gaussian_smoothing(params params) {
    HWY_DYNAMIC_DISPATCH(gaussian_smoothing)(params);
}

HWY_EXPORT(gaussian_gradient_magnitude);
void gaussian_gradient_magnitude(params params) {
    HWY_DYNAMIC_DISPATCH(gaussian_gradient_magnitude)(params);
}

HWY_EXPORT(laplacian_of_gaussian);
void laplacian_of_gaussian(params params) {
    HWY_DYNAMIC_DISPATCH(laplacian_of_gaussian)(params);
}

HWY_EXPORT(hessian_of_gaussian_eigenvalues);
void hessian_of_gaussian_eigenvalues(params params) {
    HWY_DYNAMIC_DISPATCH(hessian_of_gaussian_eigenvalues)(params);
}

HWY_EXPORT(structure_tensor_eigenvalues);
void structure_tensor_eigenvalues(params params, double st_scale) {
    HWY_DYNAMIC_DISPATCH(structure_tensor_eigenvalues)(params, st_scale);
}

} // namespace fastfilters2
#endif
