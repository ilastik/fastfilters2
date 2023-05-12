#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <fastfilters2.h>

namespace ff = fastfilters2;
namespace py = pybind11;

py::array_t<float> gaussian_kernel(double sigma, size_t order, size_t radius, bool twosided) {
    // TODO: change size_t to ssize_t for better error messages when arguments are negative.
    if (!(sigma > 0)) {
        throw std::invalid_argument("sigma should be positive");
    }
    if (!(0 <= order && order < 3)) {
        throw std::invalid_argument("order should be 0, 1 or 2");
    }
    if (radius == 0) {
        radius = ff::kernel_radius(sigma, order);
    }

    py::array_t<float> kernel{static_cast<py::ssize_t>(twosided ? 2 * radius + 1 : radius + 1)};

    auto start = kernel.mutable_data();
    if (twosided) {
        start += radius;
    }
    ff::gaussian_kernel(start, radius, sigma, order);

    if (twosided) {
        auto lhs = kernel.mutable_data();
        auto rhs = start + 1;
        std::reverse_copy(rhs, rhs + radius, lhs);
        if (order % 2 == 1) {
            std::transform(lhs, lhs + radius, lhs, std::negate{});
        }
    }

    return kernel;
}

class Filters {
public:
    Filters(py::array_t<float, py::array::c_style | py::array::forcecast> data, double scale)
            : shape{data.shape(), data.shape() + data.ndim()}, evshape(data.ndim() + 1) {

        auto ndim = shape.size();
        if (!(ndim == 2 || ndim == 3)) {
            throw std::invalid_argument("only 2D and 3D arrays are supported");
        }
        if (!(scale > 0)) {
            throw std::invalid_argument("scale should be positive");
        }
        // TODO: ensure that data dimensions are sufficient for the individual filters.

        evshape[0] = ndim;
        std::copy(shape.begin(), shape.end(), evshape.begin() + 1);

        ff::shape_type ffshape;
        std::fill(ffshape.begin(), ffshape.end(), 1);
        std::reverse_copy(shape.begin(), shape.end(), ffshape.begin());
        f = ff::filters{data.data(), ffshape, ndim, scale};
    }

    py::array_t<float> gaussian_smoothing() {
        py::array_t<float> out{shape};
        f.gaussian_smoothing(out.mutable_data());
        return out;
    }

    py::array_t<float> gaussian_gradient_magnitude() {
        py::array_t<float> out{shape};
        f.gaussian_gradient_magnitude(out.mutable_data());
        return out;
    }

    py::array_t<float> laplacian_of_gaussian() {
        py::array_t<float> out{shape};
        f.laplacian_of_gaussian(out.mutable_data());
        return out;
    }

    py::array_t<float> hessian_of_gaussian_eigenvalues() {
        py::array_t<float> out{evshape};
        f.hessian_of_gaussian_eigenvalues(out.mutable_data());
        return out;
    }

    py::array_t<float> structure_tensor_eigenvalues() {
        py::array_t<float> out{evshape};
        f.structure_tensor_eigenvalues(out.mutable_data());
        return out;
    }

private:
    std::vector<py::ssize_t> shape;
    std::vector<py::ssize_t> evshape;
    ff::filters f;
};

PYBIND11_MODULE(_core, m) {
    using namespace py::literals;

    py::options options;
    options.disable_function_signatures();

    m.def("gaussian_kernel",
          &gaussian_kernel,
          "sigma"_a,
          py::kw_only{},
          "order"_a = 0,
          "radius"_a = 0,
          "twosided"_a = false);

    py::class_<Filters>(m, "Filters")
            .def(py::init<py::array_t<float, py::array::c_style | py::array::forcecast>, double>())
            .def("gaussian_smoothing", &Filters::gaussian_smoothing)
            .def("gaussian_gradient_magnitude", &Filters::gaussian_gradient_magnitude)
            .def("laplacian_of_gaussian", &Filters::laplacian_of_gaussian)
            .def("hessian_of_gaussian_eigenvalues", &Filters::hessian_of_gaussian_eigenvalues)
            .def("structure_tensor_eigenvalues", &Filters::structure_tensor_eigenvalues);
};
