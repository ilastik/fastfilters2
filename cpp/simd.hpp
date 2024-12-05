#pragma once

#include <array>
#include <cstddef>

namespace fastfilters2::simd {

using Shape = std::array<size_t, 3>;

struct Image {
    const float *data;
    Shape shape;
};

struct Kernel {
    const float *data;
    size_t size;
    int order;
};

enum class Op { add, mul, l2norm, eigenvalues };

size_t convolve_width();
void convolve(int dim, Image data, Kernel kernel, float *dst, float *scratch);
void ufunc(Op op, const float **srcs, size_t nargs, float *dst, size_t size);

} // namespace fastfilters2
