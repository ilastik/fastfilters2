#pragma once

#include <cmath>

namespace fastfilters2 {

inline size_t kernel_size(double scale, double truncate, int order) {
    // Kernel size is computed according to the old implementation:
    // https://github.com/ilastik/fastfilters/blob/38e606fa5aacd571b07e2e34fda1fb2eeb6ca128/src/library/fir_kernel.c#L74
    auto fradius = truncate > 0 ? std::round(truncate * scale)
                                : std::ceil((3 + 0.5 * order) * scale);
    // Note the the old code computed `len` which is actually the radius, so we need to
    // add 1 to get the size. Also, the old code had an issue: if `truncate` and `scale`
    // are small, the radius might be 0, which produces a useless kernel. To fix this,
    // ensure that the kernel size is at least 2.
    return fradius > 0 ? static_cast<size_t>(fradius) + 1 : 2;
}

inline void gaussian_kernel(float *kernel, size_t ksize, double scale, int order) {
    // This is a simplified version of the old implementation:
    // https://github.com/ilastik/fastfilters/blob/38e606fa5aacd571b07e2e34fda1fb2eeb6ca128/src/library/fir_kernel.c#L58

    auto inv_scale = 1 / scale;
    auto inv_scale2 = inv_scale * inv_scale;

    constexpr double inv_sqrt_tau = 0.3989422804014327;
    auto norm = inv_sqrt_tau * inv_scale;
    if (order > 0) {
        norm *= -inv_scale2;
    }

    auto factor = -0.5 * inv_scale2;

    for (size_t x = 0; x < ksize; ++x) {
        auto x2 = x * x;
        auto g = norm * std::exp(factor * x2);
        if (order == 1) {
            g *= x;
        } else if (order == 2) {
            g *= 1 - x2 * inv_scale2;
        }
        kernel[x] = static_cast<float>(g);
    }

    if (order == 2) {
        double sum = kernel[0];
        for (size_t x = 1; x < ksize; ++x) {
            sum += 2 * kernel[x];
        }
        auto dc = static_cast<float>(sum / (2 * ksize - 1));
        for (size_t x = 0; x < ksize; ++x) {
            kernel[x] -= dc;
        }
    }

    double sum = order == 0 ? kernel[0] : 0;
    for (size_t x = 1; x < ksize; ++x) {
        if (order == 0) {
            sum += 2 * kernel[x];
        } else if (order == 1) {
            sum += 2 * x * kernel[x];
        } else {
            sum += x * x * kernel[x];
        }
    }

    auto inv_sum = static_cast<float>(1 / sum);
    for (size_t x = 0; x < ksize; ++x) {
        kernel[x] *= inv_sum;
    }
}

} // namespace fastfilters2
