#ifndef FASTFILTERS2_H_
#define FASTFILTERS2_H_

#include <hwy/base.h>

#include <cstddef>
#include <type_traits>

namespace fastfilters2 {
using val_t = float;
using ptr = const val_t *HWY_RESTRICT;
using mut_ptr = val_t *HWY_RESTRICT;

// Same as decltype(0z) in C++23.
using ssize_t = std::make_signed_t<std::size_t>;
using ssize_ptr = const ssize_t *HWY_RESTRICT;

ssize_t batch_size();
void gaussian_smoothing(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale);

ssize_t kernel_radius(double scale, ssize_t order);
void gaussian_kernel(mut_ptr kernel, ssize_t size, double scale, ssize_t order);

} // namespace fastfilters2

#endif
