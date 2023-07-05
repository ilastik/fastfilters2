#ifndef FASTFILTERS2_H_
#define FASTFILTERS2_H_

#include <hwy/base.h>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace fastfilters2 {
using val_t = float;
using ptr = const val_t *HWY_RESTRICT;
using mut_ptr = val_t *HWY_RESTRICT;

// Same as decltype(0z) in C++23.
using ssize_t = std::make_signed_t<std::size_t>;
using ssize_ptr = const ssize_t *HWY_RESTRICT;

/**
 * Return radius of a Gaussian kernel for given scale (sigma) and order.
 * Order is the kernel derivative order (0, 1 or 2).
 * Total (virtual) size of a kernel: 2 * radius + 1.
 * The actual number of elements to be allocated: radius + 1 (central element and the right half).
 */
inline ssize_t kernel_radius(double scale, ssize_t order) {
    return std::ceil((3 + 0.5 * order) * scale);
}

void gaussian_kernel(mut_ptr kernel, ssize_t size, double scale, ssize_t order);

void gaussian_smoothing(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale);
void gaussian_gradient_magnitude(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale);
void laplacian_of_gaussian(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale);
void hessian_of_gaussian_eigenvalues(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale);
void structure_tensor_eigenvalues(ptr src, mut_ptr dst, ssize_ptr shape, ssize_t ndim, double scale);
} // namespace fastfilters2

#endif
