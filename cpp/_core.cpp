#include <stdexcept>
#include <type_traits>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <fastfilters2.h>

namespace ff = fastfilters2;
namespace py = pybind11;
using std::size_t;

py::array_t<float> gaussian_kernel(double scale, size_t order) {
  if (scale <= 0) {
    throw std::invalid_argument("scale should be greater than 0");
  }
  if (order > 2) {
    throw std::invalid_argument("orders greater than 2 are not supported");
  }

  auto radius = ff::kernel_radius(scale, order);
  py::array_t<float> kernel{static_cast<py::ssize_t>(radius) + 1};
  ff::gaussian_kernel(kernel.mutable_data(), radius, scale, order);
  return kernel;
}

py::array_t<float> compute_filters(
    py::array_t<float, py::array::c_style | py::array::forcecast> data,
    double scale) {

  if (scale <= 0) {
    throw std::invalid_argument("scale should be greater than 0");
  }
  if (data.ndim() != 2) {
    throw std::invalid_argument("only 2D arrays are supported");
  }

  py::array_t<float> result{{py::ssize_t{7}, data.shape(0), data.shape(1)}};

  size_t shape[] = {7, static_cast<size_t>(data.shape(0)),
                    static_cast<size_t>(data.shape(1))};

  ff::compute_filters(result.mutable_data(), data.data(), shape, 2, scale);
  return result;
}

PYBIND11_MODULE(_core, m) {
  using namespace py::literals;

  py::options options;
  options.disable_function_signatures();

  m.def("gaussian_kernel", &gaussian_kernel, "scale"_a, "order"_a = 0);

  m.def("compute_filters", &compute_filters, "data"_a, "scale"_a);
};
