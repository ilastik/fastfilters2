#ifndef FASTFILTERS2_H_
#define FASTFILTERS2_H_

#include <cstddef>
#include <type_traits>

namespace fastfilters2 {
using int_t = std::make_signed_t<std::size_t>;
using val_t = float;

int_t kernel_radius(double scale, int_t order);
void gaussian_kernel(val_t *kernel, int_t radius, double scale, int_t order);
void compute_filters(val_t *dst, const val_t *src, const int_t *size,
                     int_t ndim, double scale);
} // namespace fastfilters2

#endif
