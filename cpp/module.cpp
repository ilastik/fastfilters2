#include "fastfilters2.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace ff = fastfilters2;
namespace py = pybind11;

using py::array_t;
using py::ssize_t;
using std::size_t;

size_t get_radius(double scale, ssize_t order, double window_ratio) {
    if (scale <= 0) {
        throw std::invalid_argument("scale must be positive");
    }
    if (order < 0 || order > 2) {
        throw std::invalid_argument("order must be 0, 1, or 2");
    }
    if (window_ratio < 0) {
        throw std::invalid_argument("window_ratio must be non-negative");
    }

    auto radius = ff::kernel_radius(scale, order, window_ratio);
    if (radius == 0) {
        throw std::invalid_argument("scale and/or window_ratio is too small");
    }
    if (radius >= std::numeric_limits<ssize_t>::max()) {
        throw std::invalid_argument("scale and/or window_ratio is too large");
    }
    return radius;
}

void validate_shape(const ssize_t *shape, size_t ndim, size_t radius) {
    for (size_t i = 0; i < ndim; ++i) {
        if (static_cast<size_t>(shape[i]) <= radius) {
            throw std::invalid_argument("data is too small");
        }
    }
}

template <size_t max_order, bool eigenvalues>
auto get_params(const array_t<float> &data, double scale, double window_ratio) {
    auto ndim = data.ndim();
    if (ndim < 2 || ndim > 3) {
        throw std::invalid_argument("data must be 2D or 3D");
    }

    auto shape = data.shape();
    validate_shape(shape, ndim, get_radius(scale, max_order, window_ratio));

    std::vector<ssize_t> out_shape;
    if constexpr (eigenvalues) {
        out_shape.push_back(ndim);
    }
    std::copy_n(shape, ndim, std::back_inserter(out_shape));
    array_t<float> out{out_shape};

    ff::params params;
    params.dst = out.mutable_data();
    params.src = data.data();
    std::fill_n(params.size, 3, 1);
    std::reverse_copy(shape, shape + ndim, params.size);
    params.ndim = ndim;
    params.scale = scale;
    params.window_ratio = window_ratio;

    return std::tuple{params, out};
}

template <size_t max_order, bool eigenvalues = false, typename F>
auto wrap_filter(F func) {
    return [func](array_t<float, py::array::c_style | py::array::forcecast> data,
                  double scale,
                  double window_ratio) {
        auto [params, out] =
                get_params<max_order, eigenvalues>(data, scale, window_ratio);
        py::gil_scoped_release release;
        func(params);
        return out;
    };
}

PYBIND11_MODULE(_core, m) {
    using namespace py::literals;

    m.def(
            "gaussian_kernel",
            [](double scale, ssize_t order, double window_ratio) {
                auto radius = get_radius(scale, order, window_ratio);
                array_t<float> kernel{static_cast<ssize_t>(radius) + 1};
                ff::gaussian_kernel(kernel.mutable_data(), radius, scale, order);
                return kernel;
            },
            "Compute Gaussian kernel or Gaussian derivative.",
            "scale"_a,
            py::kw_only(),
            "order"_a = 0,
            "window_ratio"_a = 0.0);

    m.def("gaussian_smoothing",
          wrap_filter<0>(ff::gaussian_smoothing),
          "Compute Gaussian smoothing.",
          "data"_a,
          "scale"_a,
          py::kw_only(),
          "window_ratio"_a = 0.0);

    m.def("gaussian_gradient_magnitude",
          wrap_filter<1>(ff::gaussian_gradient_magnitude),
          "Compute Gaussian gradient magnitude.",
          "data"_a,
          "scale"_a,
          py::kw_only(),
          "window_ratio"_a = 0.0);

    m.def("laplacian_of_gaussian",
          wrap_filter<2>(ff::laplacian_of_gaussian),
          "Compute Laplacian of Gaussian.",
          "data"_a,
          "scale"_a,
          py::kw_only(),
          "window_ratio"_a = 0.0);

    m.def("hessian_of_gaussian_eigenvalues",
          wrap_filter<2, true>(ff::hessian_of_gaussian_eigenvalues),
          "Compute eigenvalues of Hessian of Gaussian.",
          "data"_a,
          "scale"_a,
          py::kw_only(),
          "window_ratio"_a = 0.0);

    m.def(
            "structure_tensor_eigenvalues",
            [](array_t<float, py::array::c_style | py::array::forcecast> data,
               double scale,
               double st_scale,
               double window_ratio) {
                auto [params, out] = get_params<0, true>(data, scale, window_ratio);
                if (st_scale < 0) {
                    throw std::invalid_argument("st_scale must be non-negative");
                }
                if (st_scale == 0) {
                    st_scale = 0.5 * scale;
                }
                validate_shape(data.shape(),
                               data.ndim(),
                               get_radius(st_scale, 1, window_ratio));
                py::gil_scoped_release release;
                ff::structure_tensor_eigenvalues(params, st_scale);
                return out;
            },
            "Compute eigenvalues of structure tensor.",
            "data"_a,
            "scale"_a,
            py::kw_only(),
            "st_scale"_a = 0.0,
            "window_ratio"_a = 0.0);
}
