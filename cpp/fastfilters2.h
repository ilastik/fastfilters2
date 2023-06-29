#ifndef FASTFILTERS2_H_
#define FASTFILTERS2_H_

#include <array>
#include <cstddef>

namespace fastfilters2 {
using std::size_t;
using shape_type = std::array<size_t, 3>;

size_t kernel_radius(double sigma, size_t order);
void gaussian_kernel(float *kernel, size_t radius, double scale, size_t order);

void gaussian_smoothing(float *out, const float *data, shape_type shape, size_t ndim, double scale);
void gaussian_gradient_magnitude(float *out, const float *data, shape_type shape, size_t ndim, double scale);
void laplacian_of_gaussian(float *out, const float *data, shape_type shape, size_t ndim, double scale);
void hessian_of_gaussian_eigenvalues(float *out, const float *data, shape_type shape, size_t ndim, double scale);
void structure_tensor_eigenvalues(float *out, const float *data, shape_type shape, size_t ndim, double scale);

} // namespace fastfilters2

#endif
