#ifndef FASTFILTERS2_H_
#define FASTFILTERS2_H_

#include <cstddef>
#include <type_traits>

namespace fastfilters2 {
using std::size_t;

size_t kernel_radius(double scale, size_t order);
void gaussian_kernel(float *kernel, size_t radius, double scale, size_t order);
void compute_filters(float *dst, const float *src, const size_t *size,
                     size_t ndim, double scale);
} // namespace fastfilters2

#endif
