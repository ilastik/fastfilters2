#include "fastfilters2.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace ff = fastfilters2;
namespace py = pybind11;
static_assert(std::is_same_v<ff::ssize_t, py::ssize_t>);

PYBIND11_MODULE(_core, m) {
    using namespace py::literals;
    constexpr auto c_contig = py::array::c_style | py::array::forcecast;

    m.def(
            "gaussian_kernel",
            [](double scale, py::ssize_t order) {
                if (scale <= 0) {
                    throw std::invalid_argument("scale must be positive");
                }
                if (order < 0 || order > 2) {
                    throw std::invalid_argument("order must be 0, 1, or 2");
                }

                auto size = ff::kernel_radius(scale, order) + 1;
                py::array_t<ff::val_t> kernel{size};
                ff::gaussian_kernel(kernel.mutable_data(), size, scale, order);
                return kernel;
            },
            "scale"_a,
            "order"_a = 0);

    m.def(
            "gaussian_smoothing",
            [](py::array_t<ff::val_t, c_contig> src, double scale) {
                auto shape = src.shape();
                auto ndim = src.ndim();

                if (ndim < 2 || ndim > 3) {
                    throw std::invalid_argument("src must be 2D or 3D");
                }
                if (scale <= 0) {
                    throw std::invalid_argument("scale must be positive");
                }

                auto radius = ff::kernel_radius(scale, 0);
                for (auto i = 0; i < ndim; ++i) {
                    if (shape[i] <= radius) {
                        throw std::invalid_argument("src is too small");
                    }
                }

                py::array_t<ff::val_t> dst{{shape, shape + ndim}};
                ff::gaussian_smoothing(src.data(), dst.mutable_data(), shape, ndim, scale);
                return dst;
            },
            "src"_a,
            "scale"_a);
};
