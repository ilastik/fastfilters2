#include <stdexcept>
#include <type_traits>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <fastfilters2.h>

namespace ff = fastfilters2;
namespace py = pybind11;
static_assert(std::is_same_v<ff::int_t, py::ssize_t>);

py::array_t<ff::val_t> gaussian_kernel(double scale, py::ssize_t order) {
  if (scale <= 0) {
    throw std::invalid_argument("scale should be greater than 0");
  }
  if (order < 0) {
    throw std::invalid_argument("order cannot be negative");
  }
  if (order > 2) {
    throw std::invalid_argument("orders greater than 2 are not supported");
  }

  auto radius = ff::kernel_radius(scale, order);
  py::array_t<ff::val_t> kernel{radius + 1};
  ff::gaussian_kernel(kernel.mutable_data(), radius, scale, order);
  return kernel;
}

py::array_t<ff::val_t> compute_filters(
    py::array_t<ff::val_t, py::array::c_style | py::array::forcecast> data,
    double scale) {

  if (scale <= 0) {
    throw std::invalid_argument("scale should be greater than 0");
  }
  if (data.ndim() != 2) {
    throw std::invalid_argument("only 2D arrays are supported");
  }

  py::ssize_t size[3];
  size[0] = 7;
  size[1] = data.shape(0);
  size[2] = data.shape(1);
  py::array_t<ff::val_t> result{{size, size + 3}};

  ff::compute_filters(result.mutable_data(), data.data(), data.shape(),
                      data.ndim(), scale);
  return result;
}

PYBIND11_MODULE(_core, m) {
  using namespace py::literals;

  py::options options;
  options.disable_function_signatures();

  m.def("gaussian_kernel", &gaussian_kernel, "scale"_a, "order"_a = 0);

  m.def("compute_filters", &compute_filters, "data"_a, "scale"_a);
};
