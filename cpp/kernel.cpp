#include "kernel.hpp"

namespace fastfilters2 {

void gaussian_kernel(float *kernel, size_t size, double scale, int order) {
    constexpr double inv_sqrt_tau = 0.3989422804014327;

    auto inv_scale = 1 / scale;
    auto inv_scale2 = inv_scale * inv_scale;

    auto factor = -0.5 * inv_scale2;

    auto norm = inv_sqrt_tau * inv_scale;
    if (order > 0) {
        norm *= -inv_scale2;
    }

    for (size_t i = 0; i < size; ++i) {
        auto x = static_cast<double>(i);
        auto x2 = x * x;

        auto g = norm * std::exp(factor * x2);
        if (order == 1) {
            g *= x;
        } else if (order == 2) {
            g *= 1 - x2 * inv_scale2;
        }

        kernel[i] = g;
    }

    if (order == 2) {
        auto sum = static_cast<double>(kernel[0]);
        for (size_t i = 1; i < size; ++i) {
            sum += 2 * kernel[i];
        }
        auto dc = static_cast<float>(sum / (2 * size - 1));
        for (size_t i = 0; i < size; ++i) {
            kernel[i] -= dc;
        }
    }

    auto sum = order == 0 ? static_cast<double>(kernel[0]) : 0;
    for (size_t i = 1; i < size; ++i) {
        auto k = static_cast<double>(kernel[i]);
        if (order == 0) {
            sum += 2 * k;
        } else if (order == 1) {
            auto x = static_cast<double>(i);
            sum += 2 * x * k;
        } else {
            auto x2 = static_cast<double>(i * i);
            sum += x2 * k;
        }
    }

    auto inv_sum = 1 / sum;
    for (size_t i = 0; i < size; ++i) {
        kernel[i] *= inv_sum;
    }
}

} // namespace fastfilters2
