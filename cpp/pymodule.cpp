#include "fastfilters2.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace ff = fastfilters2;
namespace py = pybind11;

static_assert(std::is_same_v<ff::ssize_t, py::ssize_t>);
using py::ssize_t;

/**
 * Take a fastfilters function and wrap it in a pybind11 function.
 */
template <ssize_t MaxOrder, bool Eigenvalues = false, typename Func> auto filter(Func func) {
    return [func](py::array_t<ff::val_t, py::array::c_style | py::array::forcecast> src,
                  double scale) -> py::array_t<ff::val_t> {
        auto shape = src.shape();
        auto ndim = src.ndim();

        if (ndim < 2 || ndim > 3) {
            throw std::invalid_argument("src must be 2D or 3D");
        }
        if (scale <= 0) {
            throw std::invalid_argument("scale must be positive");
        }

        auto radius = ff::kernel_radius(scale, MaxOrder);
        for (ssize_t i = 0; i < ndim; ++i) {
            if (shape[i] <= radius) {
                throw std::invalid_argument("src is too small for the given scale");
            }
        }

        std::vector<ssize_t> pyshape;
        if constexpr (Eigenvalues) {
            pyshape.push_back(ndim);
        }
        for (ssize_t i = 0; i < ndim; ++i) {
            pyshape.push_back(shape[i]);
        }

        py::array_t<ff::val_t> dst(pyshape);
        func(src.data(), dst.mutable_data(), shape, ndim, scale);
        return dst;
    };
}

PYBIND11_MODULE(_core, m) {
    using namespace py::literals;

    m.def(
            "gaussian_kernel",
            [](double scale, ssize_t order) {
                if (scale <= 0) {
                    throw std::invalid_argument("scale must be positive");
                }
                if (order < 0 || order > 2) {
                    throw std::invalid_argument("order must be 0, 1, or 2");
                }

                auto radius = ff::kernel_radius(scale, order);
                py::array_t<ff::val_t> kernel{radius + 1};
                ff::gaussian_kernel(kernel.mutable_data(), radius, scale, order);
                return kernel;
            },
            "scale"_a,
            "order"_a = 0);

    m.def("gaussian_smoothing", filter<0>(ff::gaussian_smoothing), "src"_a, "scale"_a);
    m.def("gaussian_gradient_magnitude", filter<1>(ff::gaussian_gradient_magnitude), "src"_a, "scale"_a);
    m.def("laplacian_of_gaussian", filter<2>(ff::laplacian_of_gaussian), "src"_a, "scale"_a);
    m.def("hessian_of_gaussian_eigenvalues", filter<2, true>(ff::hessian_of_gaussian_eigenvalues), "src"_a, "scale"_a);
    m.def("structure_tensor_eigenvalues", filter<2, true>(ff::structure_tensor_eigenvalues), "src"_a, "scale"_a);
};
