#ifndef FASTFILTERS2_H_
#define FASTFILTERS2_H_

#include <array>
#include <cstddef>
#include <experimental/propagate_const>
#include <memory>

namespace fastfilters2 {
using std::size_t;
using shape_type = std::array<size_t, 3>;

size_t kernel_radius(double sigma, size_t order);
void gaussian_kernel(float *kernel, size_t radius, double scale, size_t order);

class filters {
public:
    filters(const float *data, shape_type shape, size_t ndim, double scale);

    filters();
    filters(const filters &other) = delete;
    filters &operator=(filters &other) = delete;
    filters(filters &&other) noexcept;
    filters &operator=(filters &&) noexcept;
    ~filters();

    void gaussian_smoothing(float *out);
    void gaussian_gradient_magnitude(float *out);
    void laplacian_of_gaussian(float *out);
    void hessian_of_gaussian_eigenvalues(float *out);
    void structure_tensor_eigenvalues(float *out);

private:
    struct impl;
    std::experimental::propagate_const<std::unique_ptr<impl>> pimpl;
};
} // namespace fastfilters2

#endif
