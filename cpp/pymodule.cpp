#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fastfilters2.h"

namespace ff = fastfilters2;
namespace py = pybind11;

using array_type = py::array_t<float>;
using input_array_type = py::array_t<float, py::array::c_style | py::array::forcecast>;

template <typename Func>
array_type apply_filter(const array_type &data, double sigma, size_t order, py::ssize_t channels, Func func) {
    auto ndim = static_cast<size_t>(data.ndim());
    if (ndim != 2 && ndim != 3) {
        throw std::invalid_argument("only 2D and 3D inputs are supported");
    }
    if (sigma <= 0) {
        throw std::invalid_argument("sigma should be positive");
    }

    auto radius = ff::kernel_radius(sigma, order);
    for (size_t i = 0; i < ndim; ++i) {
        if (static_cast<size_t>(data.shape(i)) <= radius) {
            throw std::invalid_argument("input is too small for the given sigma");
        }
    }

    std::vector<py::ssize_t> shape(ndim + static_cast<bool>(channels));
    for (size_t i = 0; i < shape.size(); ++i) {
        shape[i] = i < ndim ? data.shape(ndim - i - 1) : channels;
    }
    array_type result{shape};

    ff::shape_type ff_shape;
    for (size_t i = 0; i < ff_shape.size(); ++i) {
        ff_shape[i] = i < ndim ? data.shape(ndim - i - 1) : 1;
    }
    func(result.mutable_data(), data.data(), ff_shape, ndim, sigma);
    return result;
}

PYBIND11_MODULE(_core, m) {
    using namespace py::literals;

    m.def(
            "gaussian_kernel",
            [](double sigma, py::ssize_t order) {
                if (sigma <= 0) {
                    throw std::invalid_argument("sigma should be positive");
                }
                if (order < 0 || order > 2) {
                    throw std::invalid_argument("order should be 0, 1 or 2");
                }
                auto radius = ff::kernel_radius(sigma, order);
                array_type kernel{static_cast<py::ssize_t>(radius) + 1};
                ff::gaussian_kernel(kernel.mutable_data(), radius, sigma, order);
                return kernel;
            },
            "sigma"_a,
            py::kw_only(),
            "order"_a = 0);

    m.def(
            "gaussian_smoothing",
            [](input_array_type data, double sigma) { return apply_filter(data, sigma, 0, 0, ff::gaussian_smoothing); },
            "data"_a,
            "sigma"_a);
};
