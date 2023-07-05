#include "fastfilters2.h"
#include "util.h"

#include <hwy/aligned_allocator.h>

#include <algorithm>

// Comma-separated list of names.
// Used for declare multiple variables and passing them to varidic functions.
#ifndef ARGLIST
#define ARGLIST                                                                                                        \
    _arg00, _arg01, _arg02, _arg03, _arg04, _arg05, _arg06, _arg07, _arg08, _arg09, _arg10, _arg11, _arg12, _arg13,    \
            _arg14, _arg15, _arg16, _arg17, _arg18, _arg19, _arg20, _arg21, _arg22, _arg23, _arg24, _arg25, _arg26,    \
            _arg27, _arg28, _arg29, _arg30, _arg31
#endif

// Recursive include machinery that re-includes this file for each target.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fastfilters2.cpp"
#include <hwy/foreach_target.h>

// Target-dependent definitions should be included after foreach_target.h.
#include <hwy/contrib/math/math-inl.h>
#include <hwy/highway.h>

// Macros HWY_BEFORE_NAMESPACE/HWY_AFTER_NAMESPACE enable/disable compiler-specific function
// attributes that allow the compiler to use target-specific instructions.
// HWY_NAMESPACE is defined to be the name of each target in each recusive include pass.
HWY_BEFORE_NAMESPACE();
namespace fastfilters2::HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

namespace util = fastfilters2::util;

// Maximum lane size is capped to 64 bytes: this is the most common cache line size.
// It is also the actual lane size in AVX-512.
using D = hn::CappedTag<val_t, 64 / sizeof(val_t)>;
// Unroll is set to 8 because recent x86 CPUs have 2 FMA execution units with 4-cycle FMA latency.
// However, this value probably needs some tuning on other architectures.
constexpr ssize_t Unroll = 8;

/**
 * Return the number of elements in a consecutive group of lanes.
 */
ssize_t batch_size() { return Unroll * hn::Lanes(D{}); }

/**
 * Compute the central element and the right half of a Gaussian kernel.
 * The left half is not explicitly stored.
 * However, is equal to the reversed right half (symmetric case, when the order is even),
 * or the negated reversed right half (antisymmetric case, when the order is odd).
 */
void gaussian_kernel(mut_ptr kernel, ssize_t radius, double scale, ssize_t order) {
    HWY_ASSUME(radius > 0);
    HWY_ASSUME(scale > 0);
    HWY_ASSUME(order >= 0);
    HWY_ASSUME(order <= 2);

    // Compute constants for the subsequent computations.
    // Divisions are usually expensive and not pipelined,
    // so use multiplications by an inverse instead.
    double inv_sigma = 1 / scale;
    double inv_sigma2 = inv_sigma * inv_sigma;
    constexpr double inv_sqrt_tau = 0.3989422804014327; // 1 / sqrt(2 * pi)
    double norm = inv_sqrt_tau * inv_sigma;
    if (order > 0) {
        norm = -norm * inv_sigma2;
    }

    // Compute the initial coefficients from the Gaussian or Gaussian derivative formulae.
    // Even though the kernel is numerically normalized later anyway, still multiply by the norm
    // for numerical compatibility with the old version of this library.
    for (ssize_t x = 0; x <= radius; ++x) {
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

    // Remove the DC component for the second derivative.
    if (order == 2) {
        double sum = 0;
        for (ssize_t x = 1; x <= radius; ++x) {
            sum += kernel[x];
        }
        sum = kernel[0] + 2 * sum;
        double mean = sum / (2 * radius + 1);
        for (ssize_t x = 0; x <= radius; ++x) {
            kernel[x] -= mean;
        }
    }

    // Numerically normalize the kernel.
    // For the second derivative, even though the actual formula is 0.5 * kernel[x] * x * x,
    // skip multiplying by 0.5 when summing, but also don't multiply by 2 at the end
    // (kernel[0] should be zero because of the DC removal).
    double sum = 0;
    for (ssize_t x = 1; x <= radius; ++x) {
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
    for (ssize_t x = 0; x <= radius; ++x) {
        kernel[x] *= inv_sum;
    }
}

/**
 * Vertically convolve a batch of consecutive lanes with a kernel.
 * Stride is the distance between individual elements of the input.
 * Left (right) border is an index of the last kernel element that should not be reflected.
 * After that index, values from the input will be taken in the opposite direction.
 * For example, suppose that X is the input pointer, radius = 5, stride = 1, rborder = 2.
 * Then the input values used for computations are accesed as follows:
 * X[0], X[1], X[2], X[1], X[0], X[-1] (multiply indices by stride if stride > 1).
 */
template <ssize_t Order, bool Contiguous = false>
void conv_batch(ptr src, mut_ptr dst, ptr kernel, ssize_t radius, ssize_t stride, ssize_t lborder, ssize_t rborder) {

    HWY_ASSUME(radius > 0);

    D d;
    auto lanes = hn::Lanes(d);
    auto kvec = hn::Set(d, kernel[0]);

    // Initialize accumulators with central elements: src[0] * kernel[0].
    hn::VFromD<D> ARGLIST;
    util::static_for<Unroll>([&](auto idx, auto &vec) { vec = hn::Mul(hn::LoadU(d, src + idx * lanes), kvec); },
                             ARGLIST);

    // Pointers for the left and right sides.
    auto lsrc = src;
    auto rsrc = src;

    // Process each kernel element and left/right sides simultaneously for all lanes in the batch.
    for (ssize_t k = 1; k <= radius; ++k) {
        kvec = hn::Set(d, kernel[k]);

        if constexpr (Contiguous) {
            // In the contiguous case, avoiding CMOV gives some speedup.
            --lsrc;
            ++rsrc;
        } else {
            // Move left and right pointers by stride, but go in the reverse direction after border.
            lsrc = k <= lborder ? lsrc - stride : lsrc + stride;
            rsrc = k <= rborder ? rsrc + stride : rsrc - stride;
        }

        // Add contributions from src[k] * kernel[k] and src[-k] * kernel[k].
        // Use 1 ADD/SUB and 1 FMA instead of 2 FMA: (src[k] +/- src[-k]) * kernel[k].
        // This approach is more compatible with the old implementation, and also faster
        // on hardware without FMA units (2 ADD/SUB and 1 MUL instead of 2 ADD/SUB and 2 MUL).
        util::static_for<Unroll>(
                [&](auto idx, auto &vec) {
                    auto lvec = hn::LoadU(d, lsrc + idx * lanes);
                    auto rvec = hn::LoadU(d, rsrc + idx * lanes);
                    if constexpr (Order % 2 == 0) {
                        vec = hn::MulAdd(hn::Add(rvec, lvec), kvec, vec);
                    } else {
                        vec = hn::MulAdd(hn::Sub(rvec, lvec), kvec, vec);
                    }
                },
                ARGLIST);
    }

    // Write accumulators to the destination.
    util::static_for<Unroll>([&](auto idx, const auto &vec) { hn::StoreU(vec, d, dst + idx * lanes); }, ARGLIST);
}

void add2(ptr src1, ptr src2, mut_ptr dst, ssize_t size) {
    D d;
    ssize_t step = hn::Lanes(d);

    for (ssize_t i = 0; i < size; i += step) {
        i = std::min(i, size - step);

        auto vec1 = hn::LoadU(d, src1 + i);
        auto vec2 = hn::LoadU(d, src2 + i);

        auto out = hn::Add(vec1, vec2);
        hn::StoreU(out, d, dst + i);
    }
}

void add3(ptr src1, ptr src2, ptr src3, mut_ptr dst, ssize_t size) {
    D d;
    ssize_t step = hn::Lanes(d);

    for (ssize_t i = 0; i < size; i += step) {
        i = std::min(i, size - step);

        auto vec1 = hn::LoadU(d, src1 + i);
        auto vec2 = hn::LoadU(d, src2 + i);
        auto vec3 = hn::LoadU(d, src3 + i);

        auto out = hn::Add(hn::Add(vec1, vec2), vec3);
        hn::StoreU(out, d, dst + i);
    }
}

void mul2(ptr src1, ptr src2, mut_ptr dst, ssize_t size) {
    D d;
    ssize_t step = hn::Lanes(d);

    for (ssize_t i = 0; i < size; i += step) {
        i = std::min(i, size - step);

        auto vec1 = hn::LoadU(d, src1 + i);
        auto vec2 = hn::LoadU(d, src2 + i);

        auto out = hn::Mul(vec1, vec2);
        hn::StoreU(out, d, dst + i);
    }
}

void l2norm2(ptr src1, ptr src2, mut_ptr dst, ssize_t size) {
    D d;
    ssize_t step = hn::Lanes(d);

    for (ssize_t i = 0; i < size; i += step) {
        i = std::min(i, size - step);

        auto vec1 = hn::LoadU(d, src1 + i);
        auto vec2 = hn::LoadU(d, src2 + i);

        vec1 = hn::Mul(vec1, vec1);
        vec2 = hn::Mul(vec2, vec2);

        auto out = hn::Sqrt(hn::Add(vec1, vec2));
        hn::StoreU(out, d, dst + i);
    }
}

void l2norm3(ptr src1, ptr src2, ptr src3, mut_ptr dst, ssize_t size) {
    D d;
    ssize_t step = hn::Lanes(d);

    for (ssize_t i = 0; i < size; i += step) {
        i = std::min(i, size - step);

        auto vec1 = hn::LoadU(d, src1 + i);
        auto vec2 = hn::LoadU(d, src2 + i);
        auto vec3 = hn::LoadU(d, src3 + i);

        vec1 = hn::Mul(vec1, vec1);
        vec2 = hn::Mul(vec2, vec2);
        vec3 = hn::Mul(vec3, vec3);

        auto out = hn::Sqrt(hn::Add(hn::Add(vec1, vec2), vec3));
        hn::StoreU(out, d, dst + i);
    }
}

void eigenvalues2(ptr src00, ptr src01, ptr src11, mut_ptr dst, ssize_t size) {
    D d;
    ssize_t step = hn::Lanes(d);

    mut_ptr dst0 = dst;
    mut_ptr dst1 = dst + size;

    for (ssize_t i = 0; i < size; i += step) {
        i = std::min(i, size - step);

        auto vec00 = hn::LoadU(d, src00 + i);
        auto vec01 = hn::LoadU(d, src01 + i);
        auto vec11 = hn::LoadU(d, src11 + i);

        auto tmp0 = hn::Mul(hn::Add(vec00, vec11), hn::Set(d, 0.5));

        auto tmp1 = hn::Mul(hn::Sub(vec00, vec11), hn::Set(d, 0.5));
        tmp1 = hn::Mul(tmp1, tmp1);

        auto det = hn::Sqrt(hn::Add(tmp1, hn::Mul(vec01, vec01)));

        hn::StoreU(hn::Add(tmp0, det), d, dst0 + i);
        hn::StoreU(hn::Sub(tmp0, det), d, dst1 + i);
    }
}

/**
 * Implementation adapted from https://mazzo.li/posts/vectorized-atan2.html
 */
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

void eigenvalues3(ptr src00, ptr src01, ptr src02, ptr src11, ptr src12, ptr src22, mut_ptr dst, ssize_t size) {
    mut_ptr dst0 = dst;
    mut_ptr dst1 = dst + size;
    mut_ptr dst2 = dst + 2 * size;

    D d;
    ssize_t step = hn::Lanes(d);

    auto v_inv3 = hn::Set(d, 1.0 / 3.0);
    auto v_root3 = hn::Sqrt(hn::Set(d, 3.0));
    auto two = hn::Set(d, 2.0);
    auto one = hn::Set(d, 1.0);
    auto half = hn::Set(d, 0.5);
    auto zero = hn::Zero(d);

    for (ssize_t i = 0; i < size; i += step) {
        i = std::min(i, size - step);

        // clang-format off

        auto v_a00 = hn::LoadU(d, src00 + i);
        auto v_a01 = hn::LoadU(d, src01 + i);
        auto v_a02 = hn::LoadU(d, src02 + i);
        auto v_a11 = hn::LoadU(d, src11 + i);
        auto v_a12 = hn::LoadU(d, src12 + i);
        auto v_a22 = hn::LoadU(d, src22 + i);

        // guard against float overflows
        auto v_max0 = hn::Max(hn::Abs(v_a00), hn::Abs(v_a01));
        auto v_max1 = hn::Max(hn::Abs(v_a02), hn::Abs(v_a11));
        auto v_max2 = hn::Max(hn::Abs(v_a12), hn::Abs(v_a22));
        auto v_max_element = hn::Max(hn::Max(v_max0, v_max1), v_max2);

        // replace zeros with ones to avoid NaNs
        // OLD:
        // v_max_element = _mm256_or_ps(v_max_element, _mm256_and_ps(one, _mm256_cmp_ps(v_max_element, zero, _CMP_EQ_UQ)));
        // NEW:
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
        // OLD:
        // hn::VFromD<D> angle = hn::Mul(atan2_256_ps(hn::Sqrt(hn::Neg(q)), mbDiv2), v_inv3);
        // NEW:
        auto angle = hn::Mul(Atan2(d, hn::Sqrt(hn::Neg(q)), mbDiv2), v_inv3);

        hn::VFromD<D> cs, sn;
        // OLD:
        // sincos256_ps(angle, &sn, &cs);
        // NEW:
        sn = hn::Sin(d, angle);
        cs = hn::Cos(d, angle);

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

        hn::StoreU(v_r0, d, dst2 + i);
        hn::StoreU(v_r1, d, dst1 + i);
        hn::StoreU(v_r2, d, dst0 + i);

        // clang-format on
    }
}

template <ssize_t Order>
void conv_x(ptr src, mut_ptr dst, ssize_ptr shape, ptr kernel, ssize_t radius, mut_ptr buf) {
    ssize_t step = batch_size();
    auto contig_size = std::max(step, shape[0]);
    auto outer_size = shape[1] * shape[2];

    for (ssize_t i = 0; i < outer_size; ++i, src += shape[0], dst += contig_size) {
        util::reflect_copy(src, buf, shape[0], step, radius);
        for (ssize_t x = 0; x < contig_size; x += step) {
            x = std::min(x, contig_size - step);
            conv_batch<Order, true>(buf + radius + x, dst + x, kernel, radius, 0, 0, 0);
        }
    }
}

template <ssize_t Order>
void conv_y(ptr src, mut_ptr dst, ssize_ptr shape, ptr kernel, ssize_t radius, mut_ptr buf) {
    // Buffer is not needed for strided convolution, but accept it as a parameter anyway
    // in order to keep the function signature identical to conv_contig.
    static_cast<void>(buf);

    ssize_t step = batch_size();
    auto contig_size = std::max(step, shape[0]);
    auto inner_size = shape[1];
    auto outer_size = shape[2];
    auto outer_step = inner_size * contig_size;

    for (ssize_t i = 0; i < outer_size; ++i, src += outer_step, dst += outer_step) {
        for (ssize_t x = 0; x < contig_size; x += step) {
            x = std::min(x, contig_size - step);
            auto p = src + x;
            auto q = dst + x;
            for (ssize_t y = 0; y < inner_size; ++y, p += contig_size, q += contig_size) {
                conv_batch<Order>(p, q, kernel, radius, contig_size, y, inner_size - 1 - y);
            }
        }
    }
}

template <ssize_t Order>
void conv_z(ptr src, mut_ptr dst, ssize_ptr shape, ptr kernel, ssize_t radius, mut_ptr buf) {
    // Buffer is not needed for strided convolution, but accept it as a parameter anyway
    // in order to keep the function signature identical to conv_contig.
    static_cast<void>(buf);

    ssize_t step = batch_size();
    auto contig_size = std::max(step, shape[0]);
    auto outer_size = shape[1];
    auto inner_size = shape[2];
    auto inner_step = contig_size * outer_size;

    for (ssize_t i = 0; i < outer_size; ++i, src += contig_size, dst += contig_size) {
        for (ssize_t x = 0; x < contig_size; x += step) {
            x = std::min(x, contig_size - step);
            auto p = src + x;
            auto q = dst + x;
            for (ssize_t z = 0; z < inner_size; ++z, p += inner_step, q += inner_step) {
                conv_batch<Order>(p, q, kernel, radius, inner_step, z, inner_size - 1 - z);
            }
        }
    }
}

struct context {
    static constexpr ssize_t N = 3;

    ptr src_;
    mut_ptr dst_;
    ssize_ptr shape;
    ssize_t ndim;
    ssize_t size;
    ssize_t temp_size;
    ssize_t radius[N];
    std::vector<hwy::AlignedFreeUniquePtr<val_t[]>> cache;
    mut_ptr tmp_dst;

    context(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale)
            : src_{src}, dst_{dst}, shape{shape}, ndim{ndim}, size{1}, temp_size{1}, cache(N + 1) {

        auto step = batch_size();
        auto min_size = std::max(step, shape[0]);

        size = shape[0];
        temp_size = min_size;
        for (ssize_t i = 1; i < ndim; ++i) {
            size *= shape[i];
            temp_size *= shape[i];
        }

        for (ssize_t i = 0; i < N; ++i) {
            radius[i] = fastfilters2::kernel_radius(scale, i);
            cache[i] = hwy::AllocateAligned<val_t>(radius[i] + 1);
            gaussian_kernel(cache[i].get(), radius[i], scale, i);
        }
        cache[N] = hwy::AllocateAligned<val_t>(min_size + 2 * radius[N - 1]);
        tmp_dst = shape[0] < min_size ? allocate() : dst_;
    }

    ~context() {
        if (dst_ == tmp_dst) {
            return;
        }
        auto min_size = batch_size();
        auto q = dst_;
        for (mut_ptr p = tmp_dst; p != tmp_dst + temp_size; p += min_size) {
            q = std::copy_n(p, shape[0], q);
        }
    }

    ptr src() { return src_; }
    mut_ptr dst() { return tmp_dst; }

    mut_ptr allocate() {
        cache.push_back(hwy::AllocateAligned<val_t>(temp_size));
        return cache.back().get();
    }

    template <ssize_t Dim, ssize_t Order> void conv(ptr src, mut_ptr dst) {
        static_assert(0 <= Dim && Dim < N);
        static_assert(0 <= Order && Order < N);

        auto kernel = this->cache[Order].get();
        auto radius = this->radius[Order];
        auto buf = this->cache[N].get();

        if constexpr (Dim == 0) {
            conv_x<Order>(src, dst, shape, kernel, radius, buf);
        } else if constexpr (Dim == 1) {
            conv_y<Order>(src, dst, shape, kernel, radius, buf);
        } else if constexpr (Dim == 2) {
            conv_z<Order>(src, dst, shape, kernel, radius, buf);
        }
    }
};

void gaussian_smoothing(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale) {
    context ctx{src, dst, shape, ndim, scale};

    if (ndim == 2) {
        mut_ptr tmp = ctx.allocate();

        ctx.conv<0, 0>(ctx.src(), tmp);
        ctx.conv<1, 0>(tmp, ctx.dst());

    } else {
        mut_ptr tmp1 = ctx.allocate();
        mut_ptr tmp2 = ctx.allocate();

        ctx.conv<0, 0>(ctx.src(), tmp1);
        ctx.conv<1, 0>(tmp1, tmp2);
        ctx.conv<2, 0>(tmp2, ctx.dst());
    }
}

void gaussian_gradient_magnitude(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale) {
    context ctx{src, dst, shape, ndim, scale};

    if (ndim == 2) {
        mut_ptr tmp = ctx.allocate();
        mut_ptr x1y0 = ctx.allocate();
        mut_ptr x0y1 = ctx.allocate();

        ctx.conv<0, 1>(ctx.src(), tmp);
        ctx.conv<1, 0>(tmp, x1y0);

        ctx.conv<0, 0>(ctx.src(), tmp);
        ctx.conv<1, 1>(tmp, x0y1);

        l2norm2(x1y0, x0y1, ctx.dst(), ctx.temp_size);

    } else {
        mut_ptr tmp1 = ctx.allocate();
        mut_ptr tmp2 = ctx.allocate();
        mut_ptr x1y0z0 = ctx.allocate();
        mut_ptr x0y1z0 = ctx.allocate();
        mut_ptr x0y0z1 = ctx.allocate();

        ctx.conv<0, 1>(ctx.src(), tmp1);
        ctx.conv<1, 0>(tmp1, tmp2);
        ctx.conv<2, 0>(tmp2, x1y0z0);

        ctx.conv<0, 0>(ctx.src(), tmp1);
        ctx.conv<1, 1>(tmp1, tmp2);
        ctx.conv<2, 0>(tmp2, x0y1z0);

        ctx.conv<0, 0>(ctx.src(), tmp1);
        ctx.conv<1, 0>(tmp1, tmp2);
        ctx.conv<2, 1>(tmp2, x0y0z1);

        l2norm3(x1y0z0, x0y1z0, x0y0z1, ctx.dst(), ctx.temp_size);
    }
}

void laplacian_of_gaussian(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale) {
    context ctx{src, dst, shape, ndim, scale};

    if (ndim == 2) {
        mut_ptr tmp = ctx.allocate();
        mut_ptr x2y0 = ctx.allocate();
        mut_ptr x0y2 = ctx.allocate();

        ctx.conv<0, 2>(ctx.src(), tmp);
        ctx.conv<1, 0>(tmp, x2y0);

        ctx.conv<0, 0>(ctx.src(), tmp);
        ctx.conv<1, 2>(tmp, x0y2);

        add2(x2y0, x0y2, ctx.dst(), ctx.temp_size);

    } else {
        mut_ptr tmp1 = ctx.allocate();
        mut_ptr tmp2 = ctx.allocate();
        mut_ptr x2y0z0 = ctx.allocate();
        mut_ptr x0y2z0 = ctx.allocate();
        mut_ptr x0y0z2 = ctx.allocate();

        ctx.conv<0, 2>(ctx.src(), tmp1);
        ctx.conv<1, 0>(tmp1, tmp2);
        ctx.conv<2, 0>(tmp2, x2y0z0);

        ctx.conv<0, 0>(ctx.src(), tmp1);
        ctx.conv<1, 2>(tmp1, tmp2);
        ctx.conv<2, 0>(tmp2, x0y2z0);

        ctx.conv<0, 0>(ctx.src(), tmp1);
        ctx.conv<1, 0>(tmp1, tmp2);
        ctx.conv<2, 2>(tmp2, x0y0z2);

        add3(x2y0z0, x0y2z0, x0y0z2, ctx.dst(), ctx.temp_size);
    }
}

void hessian_of_gaussian_eigenvalues(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale) {
    context ctx{src, dst, shape, ndim, scale};

    if (ndim == 2) {
        mut_ptr tmp = ctx.allocate();
        mut_ptr xx = ctx.allocate();
        mut_ptr xy = ctx.allocate();
        mut_ptr yy = ctx.allocate();

        ctx.conv<0, 2>(ctx.src(), tmp);
        ctx.conv<1, 0>(tmp, xx);

        ctx.conv<0, 1>(ctx.src(), tmp);
        ctx.conv<1, 1>(tmp, xy);

        ctx.conv<0, 0>(ctx.src(), tmp);
        ctx.conv<1, 2>(tmp, yy);

        eigenvalues2(xx, xy, yy, ctx.dst(), ctx.temp_size);

    } else {
        mut_ptr tmp1 = ctx.allocate();
        mut_ptr tmp2 = ctx.allocate();
        mut_ptr xx = ctx.allocate();
        mut_ptr xy = ctx.allocate();
        mut_ptr xz = ctx.allocate();
        mut_ptr yy = ctx.allocate();
        mut_ptr yz = ctx.allocate();
        mut_ptr zz = ctx.allocate();

        ctx.conv<0, 2>(ctx.src(), tmp1);
        ctx.conv<1, 0>(tmp1, tmp2);
        ctx.conv<2, 0>(tmp2, xx);

        ctx.conv<0, 1>(ctx.src(), tmp1);
        ctx.conv<1, 1>(tmp1, tmp2);
        ctx.conv<2, 0>(tmp2, xy);

        ctx.conv<0, 1>(ctx.src(), tmp1);
        ctx.conv<1, 0>(tmp1, tmp2);
        ctx.conv<2, 1>(tmp2, xz);

        ctx.conv<0, 0>(ctx.src(), tmp1);
        ctx.conv<1, 2>(tmp1, tmp2);
        ctx.conv<2, 0>(tmp2, yy);

        ctx.conv<0, 0>(ctx.src(), tmp1);
        ctx.conv<1, 1>(tmp1, tmp2);
        ctx.conv<2, 1>(tmp2, yz);

        ctx.conv<0, 0>(ctx.src(), tmp1);
        ctx.conv<1, 0>(tmp1, tmp2);
        ctx.conv<2, 2>(tmp2, zz);

        // eigenvalues3(xx, xy, xz, yy, yz, zz, ctx.dst(), ctx.temp_size);

        // The old library used the following parameter order:
        eigenvalues3(zz, yz, xz, yy, xy, xx, ctx.dst(), ctx.temp_size);
    }
}

void structure_tensor_eigenvalues(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale) {
    context ctx{src, dst, shape, ndim, scale / 2};

    if (ndim == 2) {
        mut_ptr tmp = ctx.allocate();
        mut_ptr x1y0 = ctx.allocate();
        mut_ptr x0y1 = ctx.allocate();
        mut_ptr xx = ctx.allocate();
        mut_ptr xy = ctx.allocate();
        mut_ptr yy = ctx.allocate();

        ctx.conv<0, 1>(ctx.src(), tmp);
        ctx.conv<1, 0>(tmp, x1y0);

        ctx.conv<0, 0>(ctx.src(), tmp);
        ctx.conv<1, 1>(tmp, x0y1);

        mul2(x1y0, x1y0, tmp, ctx.temp_size);
        gaussian_smoothing(tmp, xx, shape, ndim, scale);

        mul2(x1y0, x0y1, tmp, ctx.temp_size);
        gaussian_smoothing(tmp, xy, shape, ndim, scale);

        mul2(x0y1, x0y1, tmp, ctx.temp_size);
        gaussian_smoothing(tmp, yy, shape, ndim, scale);

        eigenvalues2(xx, xy, yy, ctx.dst(), ctx.temp_size);

    } else {
        mut_ptr tmp1 = ctx.allocate();
        mut_ptr tmp2 = ctx.allocate();
        mut_ptr x1y0z0 = ctx.allocate();
        mut_ptr x0y1z0 = ctx.allocate();
        mut_ptr x0y0z1 = ctx.allocate();
        mut_ptr xx = ctx.allocate();
        mut_ptr xy = ctx.allocate();
        mut_ptr xz = ctx.allocate();
        mut_ptr yy = ctx.allocate();
        mut_ptr yz = ctx.allocate();
        mut_ptr zz = ctx.allocate();

        ctx.conv<0, 1>(ctx.src(), tmp1);
        ctx.conv<1, 0>(tmp1, tmp2);
        ctx.conv<2, 0>(tmp2, x1y0z0);

        ctx.conv<0, 0>(ctx.src(), tmp1);
        ctx.conv<1, 1>(tmp1, tmp2);
        ctx.conv<2, 0>(tmp2, x0y1z0);

        ctx.conv<0, 0>(ctx.src(), tmp1);
        ctx.conv<1, 0>(tmp1, tmp2);
        ctx.conv<2, 1>(tmp2, x0y0z1);

        mul2(x1y0z0, x1y0z0, tmp1, ctx.temp_size);
        gaussian_smoothing(tmp1, xx, shape, ndim, scale);

        mul2(x1y0z0, x0y1z0, tmp1, ctx.temp_size);
        gaussian_smoothing(tmp1, xy, shape, ndim, scale);

        mul2(x1y0z0, x0y0z1, tmp1, ctx.temp_size);
        gaussian_smoothing(tmp1, xz, shape, ndim, scale);

        mul2(x0y1z0, x0y1z0, tmp1, ctx.temp_size);
        gaussian_smoothing(tmp1, yy, shape, ndim, scale);

        mul2(x0y1z0, x0y0z1, tmp1, ctx.temp_size);
        gaussian_smoothing(tmp1, yz, shape, ndim, scale);

        mul2(x0y0z1, x0y0z1, tmp1, ctx.temp_size);
        gaussian_smoothing(tmp1, zz, shape, ndim, scale);

        // eigenvalues3(xx, xy, xz, yy, yz, zz, ctx.dst(), ctx.temp_size);

        // The old library used the following parameter order:
        eigenvalues3(zz, yz, xz, yy, xy, xx, ctx.dst(), ctx.temp_size);
    }
}
}; // namespace fastfilters2::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

// Everything within the HWY_ONCE block is excluded from recursive includes.
#if HWY_ONCE
namespace fastfilters2 {

// HWY_EXPORT creates a table for the corresponding function that stores implementations
// for all compiled targets from recusive includes.
// The best implementation could be selected at runtime via HWY_DYNAMIC_DISPATCH.

HWY_EXPORT(gaussian_kernel);
void gaussian_kernel(mut_ptr kernel, ssize_t radius, double scale, ssize_t order) {
    HWY_DYNAMIC_DISPATCH(gaussian_kernel)(kernel, radius, scale, order);
}

HWY_EXPORT(gaussian_smoothing);
void gaussian_smoothing(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale) {
    HWY_DYNAMIC_DISPATCH(gaussian_smoothing)(src, dst, shape, ndim, scale);
}

HWY_EXPORT(gaussian_gradient_magnitude);
void gaussian_gradient_magnitude(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale) {
    HWY_DYNAMIC_DISPATCH(gaussian_gradient_magnitude)(src, dst, shape, ndim, scale);
}

HWY_EXPORT(laplacian_of_gaussian);
void laplacian_of_gaussian(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale) {
    HWY_DYNAMIC_DISPATCH(laplacian_of_gaussian)(src, dst, shape, ndim, scale);
}

HWY_EXPORT(hessian_of_gaussian_eigenvalues);
void hessian_of_gaussian_eigenvalues(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale) {
    HWY_DYNAMIC_DISPATCH(hessian_of_gaussian_eigenvalues)(src, dst, shape, ndim, scale);
}

HWY_EXPORT(structure_tensor_eigenvalues);
void structure_tensor_eigenvalues(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale) {
    HWY_DYNAMIC_DISPATCH(structure_tensor_eigenvalues)(src, dst, shape, ndim, scale);
}
}; // namespace fastfilters2
#endif
