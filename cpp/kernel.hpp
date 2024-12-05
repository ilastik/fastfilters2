#pragma once

#include <cmath>
#include <cstddef>

namespace fastfilters2 {

inline size_t kernel_size(double scale, double truncate, int order) {
    if (truncate > 0) {
        auto size = static_cast<size_t>(std::round(truncate * scale)) + 1;
        return size + (size == 1);
    }
    return static_cast<size_t>(std::ceil((3 + 0.5 * order) * scale)) + 1;
}

void gaussian_kernel(float *kernel, size_t size, double scale, int order);

} // namespace fastfilters2
