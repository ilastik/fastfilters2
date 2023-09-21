#ifndef FASTFILTERS2_H_
#define FASTFILTERS2_H_

#include <cmath>
#include <cstddef>

namespace fastfilters2 {
using std::size_t;

void gaussian_kernel(float *kernel, size_t radius, double scale, size_t order);

struct params {
    float *dst;
    const float *src;
    size_t size[3];
    size_t ndim;
    double scale;
    double window_ratio;
};

void gaussian_smoothing(params params);
void gaussian_gradient_magnitude(params params);
void laplacian_of_gaussian(params params);
void hessian_of_gaussian_eigenvalues(params params);
void structure_tensor_eigenvalues(params params, double st_scale);

inline size_t kernel_radius(double scale, size_t order, double window_ratio) {
    return window_ratio > 0 ? std::floor(window_ratio * scale + 0.5)
                            : std::ceil((3 + 0.5 * order) * scale);
}
} // namespace fastfilters2

#endif
