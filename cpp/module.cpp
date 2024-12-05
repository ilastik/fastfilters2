#include "kernel.hpp"
#include "simd.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ff = fastfilters2;
namespace nb = nanobind;

using ff::simd::Op;
using ndarray_in = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using ndarray_out = nb::ndarray<float, nb::numpy>;

static ndarray_out allocate_output(size_t ndim, std::array<size_t, 4> shape) {
    size_t size = 1;
    for (size_t i = 0; i < ndim; ++i) {
        size *= shape[i];
    }
    std::unique_ptr<float[]> data{new float[size]};
    auto cleanup = [](void *ptr) noexcept { delete[] static_cast<float *>(ptr); };
    nb::capsule owner{data.get(), cleanup};
    return {data.release(), ndim, shape.data(), owner};
}

class Buffers {
    std::vector<std::unique_ptr<float[]>> data_;

public:
    float *allocate(size_t size) { return data_.emplace_back(new float[size]).get(); }

    template <size_t N> std::array<float *, N> allocate(size_t size) {
        auto begin = allocate(N * size);
        std::array<float *, N> ptrs;
        for (size_t i = 0; i < N; ++i) {
            ptrs[i] = begin + i * size;
        }
        return ptrs;
    }
};

class Convolver {
    Buffers &buffers_;
    ff::simd::Shape shape_;
    double scale_;
    double truncate_;
    std::array<std::tuple<float *, size_t>, 3> kernels_;
    float *scratch_;

public:
    Convolver(Buffers &buffers, ff::simd::Shape shape, double scale, double truncate)
            : buffers_{buffers}, shape_{shape}, scale_{scale}, truncate_{truncate},
              kernels_{} {
        auto max_radius = ff::kernel_size(scale, truncate, 2) - 1;
        auto scratch_size = shape_[0] + 2 * max_radius;
        scratch_ = buffers_.allocate(scratch_size);
        std::fill_n(scratch_, scratch_size, 0);
    }

    void convolve(int dim, const float *src, int order, float *dst) {
        auto [kernel, ksize] = kernels_[order];
        if (!kernel) {
            ksize = ff::kernel_size(scale_, truncate_, order);
            for (auto d : shape_) {
                if (d > 1 && d < ksize) {
                    throw std::invalid_argument{"data is too small"};
                }
            }
            kernel = buffers_.allocate(ksize);
            kernels_[order] = {kernel, ksize};
            ff::gaussian_kernel(kernel, ksize, scale_, order);
        }
        ff::simd::convolve(dim, {src, shape_}, {kernel, ksize, order}, dst, scratch_);
    }

    void sequence(const float *src, int order_x, float *dst_x, int order_y,
                  float *dst_y) {
        sequence(src, order_x, dst_x, order_y, dst_y, 0, nullptr);
    }

    void sequence(const float *src, int order_x, float *dst_x, int order_y,
                  float *dst_y, int order_z, float *dst_z) {
        if (dst_x) {
            convolve(0, src, order_x, dst_x);
            src = dst_x;
        }
        if (dst_y) {
            convolve(1, src, order_y, dst_y);
            src = dst_y;
        }
        if (dst_z) {
            convolve(2, src, order_z, dst_z);
        }
    }

    void derivative(int order, const float *src, float *tmp, float *dst_x,
                    float *dst_y) {
        sequence(src, order, tmp, 0, dst_x);
        sequence(src, 0, tmp, order, dst_y);
    }

    void derivative(int order, const float *src, float *tmp, float *dst_x, float *dst_y,
                    float *dst_z) {
        sequence(src, order, dst_x, 0, tmp, 0, dst_x);
        sequence(src, 0, dst_z, order, tmp, 0, dst_y);
        sequence(dst_z, 0, nullptr, 0, tmp, order, dst_z);
    }

    void hessian(const float *src, float *tmp, float *dst_xx, float *dst_xy,
                 float *dst_yy) {
        sequence(src, 2, tmp, 0, dst_xx);
        sequence(src, 1, tmp, 1, dst_xy);
        sequence(src, 0, tmp, 2, dst_yy);
    }

    void hessian(const float *src, float *tmp, float *dst_xx, float *dst_xy,
                 float *dst_xz, float *dst_yy, float *dst_yz, float *dst_zz) {
        sequence(src, 2, dst_xx, 0, tmp, 0, dst_xx);
        sequence(src, 1, dst_xz, 1, tmp, 0, dst_xy);
        sequence(dst_xz, 1, nullptr, 0, tmp, 1, dst_xz);
        sequence(src, 0, dst_zz, 2, tmp, 0, dst_yy);
        sequence(dst_zz, 0, nullptr, 1, tmp, 1, dst_yz);
        sequence(dst_zz, 0, nullptr, 0, tmp, 2, dst_zz);
    }
};

class Context {
    ndarray_in src_;
    ff::simd::Shape shape_;
    size_t size;
    ndarray_out dst_;
    Buffers buffers_;

public:
    Context(ndarray_in src, size_t channels = 1) : src_{src} {
        shape_[0] = std::max(ff::simd::convolve_width(), src.shape(src.ndim() - 1));
        shape_[1] = src.shape(src.ndim() - 2);
        shape_[2] = src.ndim() >= 3 ? src.shape(src.ndim() - 3) : 1;
        size = shape_[0] * shape_[1] * shape_[2];

        std::array<size_t, 4> dst_shape;
        size_t dst_ndim = 0;
        if (channels > 1) {
            dst_shape[dst_ndim++] = channels;
        }
        for (size_t i = 0; i < src.ndim() - 1; ++i) {
            dst_shape[dst_ndim++] = src.shape(i);
        }
        dst_shape[dst_ndim++] = shape_[0];

        dst_ = allocate_output(dst_ndim, dst_shape);
    }

    const float *src() const { return src_.data(); }

    float *dst() { return dst_.data(); }

    float *allocate() { return buffers_.allocate(size); }

    template <size_t N> std::array<float *, N> allocate() {
        return buffers_.allocate<N>(size);
    }

    Convolver convolver(double scale, double truncate) {
        return {buffers_, shape_, scale, truncate};
    }

    template <Op op, typename... Args> void ufunc(float *dst, Args... args) {
        auto srcs = std::array{static_cast<const float *>(args)...};
        ff::simd::ufunc(op, srcs.data(), srcs.size(), dst, size);
    }

    ndarray_out output() {
        auto width = src_.shape(src_.ndim() - 1);
        auto padded_width = dst_.shape(dst_.ndim() - 1);
        if (width == padded_width) {
            return dst_;
        }

        std::array<size_t, 4> trimmed_shape;
        for (size_t i = 0; i < dst_.ndim() - 1; ++i) {
            trimmed_shape[i] = dst_.shape(i);
        }
        trimmed_shape[dst_.ndim() - 1] = width;
        auto trimmed = allocate_output(dst_.ndim(), trimmed_shape);

        auto p = dst_.data();
        auto q = trimmed.data();
        for (size_t i = 0; i < trimmed.size(); i += width) {
            std::copy_n(p, width, q);
            p += padded_width;
            q += width;
        }

        return trimmed;
    }
};

namespace validate {
static void scale(double scale) {
    if (scale <= 0) {
        throw std::invalid_argument{"scale must be positive"};
    }
}

static void truncate(double truncate) {
    if (truncate < 0) {
        throw std::invalid_argument{"truncate must be positive or zero"};
    }
}

static void order(int order) {
    if (order < 0 || order > 2) {
        throw std::invalid_argument{"order must be between 0 and 2"};
    }
}

static void filter_args(ndarray_in data, double scale, double truncate) {
    if (data.ndim() < 2 || data.ndim() > 3) {
        throw std::invalid_argument{"data must be 2D or 3D"};
    }
    for (size_t i = 0; i < data.ndim(); ++i) {
        if (data.shape(i) == 0) {
            throw std::invalid_argument{"data cannot be empty"};
        }
        if (data.shape(i) == 1) {
            throw std::invalid_argument{"data cannot have singleton dimensions"};
        }
    }
    validate::scale(scale);
    validate::truncate(truncate);
}
} // namespace validate

namespace py {
static ndarray_out gaussian_kernel(double scale, double truncate, int order) {
    validate::scale(scale);
    validate::truncate(truncate);
    validate::order(order);

    auto kernel = allocate_output(1, {ff::kernel_size(scale, truncate, order)});
    ff::gaussian_kernel(kernel.data(), kernel.shape(0), scale, order);
    return kernel;
}

static ndarray_out gaussian_smoothing(ndarray_in data, double scale, double truncate,
                                      int order) {
    validate::filter_args(data, scale, truncate);
    validate::order(order);

    Context ctx{data};
    auto conv = ctx.convolver(scale, truncate);
    auto dst = ctx.dst();
    auto tmp = ctx.allocate();

    if (data.ndim() == 2) {
        conv.sequence(ctx.src(), order, tmp, order, dst);
    } else {
        conv.sequence(ctx.src(), order, dst, order, tmp, order, dst);
    }

    return ctx.output();
}

static ndarray_out gaussian_gradient_magnitude(ndarray_in data, double scale,
                                               double truncate) {
    validate::filter_args(data, scale, truncate);

    Context ctx{data};
    auto conv = ctx.convolver(scale, truncate);
    auto dst = ctx.dst();

    if (data.ndim() == 2) {
        auto [x1, y1] = ctx.allocate<2>();
        conv.derivative(1, ctx.src(), dst, x1, y1);
        ctx.ufunc<Op::l2norm>(dst, x1, y1);

    } else {
        auto [x1, y1, z1] = ctx.allocate<3>();
        conv.derivative(1, ctx.src(), dst, x1, y1, z1);
        ctx.ufunc<Op::l2norm>(dst, x1, y1, z1);
    }

    return ctx.output();
}

static ndarray_out laplacian_of_gaussian(ndarray_in data, double scale,
                                         double truncate) {
    validate::filter_args(data, scale, truncate);

    Context ctx{data};
    auto conv = ctx.convolver(scale, truncate);
    auto dst = ctx.dst();

    if (data.ndim() == 2) {
        auto [x2, y2] = ctx.allocate<2>();
        conv.derivative(2, ctx.src(), dst, x2, y2);
        ctx.ufunc<Op::add>(dst, x2, y2);

    } else {
        auto [x2, y2, z2] = ctx.allocate<3>();
        conv.derivative(2, ctx.src(), dst, x2, y2, z2);
        ctx.ufunc<Op::add>(dst, x2, y2, z2);
    }

    return ctx.output();
}

static ndarray_out hessian_of_gaussian_eigenvalues(ndarray_in data, double scale,
                                                   double truncate) {
    validate::filter_args(data, scale, truncate);

    Context ctx{data, data.ndim()};
    auto conv = ctx.convolver(scale, truncate);
    auto dst = ctx.dst();

    if (data.ndim() == 2) {
        auto [xx, xy, yy] = ctx.allocate<3>();
        conv.hessian(ctx.src(), dst, xx, xy, yy);
        ctx.ufunc<Op::eigenvalues>(dst, xx, xy, yy);

    } else {
        auto [xx, xy, xz, yy, yz, zz] = ctx.allocate<6>();
        conv.hessian(ctx.src(), dst, xx, xy, xz, yy, yz, zz);
        ctx.ufunc<Op::eigenvalues>(dst, xx, xy, xz, yy, yz, zz);
    }

    return ctx.output();
}

static ndarray_out structure_tensor_eigenvalues(ndarray_in data, double scale,
                                                double truncate, double smooth_scale) {
    validate::filter_args(data, scale, truncate);

    Context ctx{data, data.ndim()};
    if (smooth_scale < 0) {
        throw std::invalid_argument{"smooth_scale must be positive or zero"};
    }
    if (smooth_scale == 0) {
        smooth_scale = 0.5 * scale;
    }

    auto conv = ctx.convolver(scale, truncate);
    auto smooth = ctx.convolver(smooth_scale, truncate);
    auto dst = ctx.dst();

    auto mul_smooth = [&](const float *lhs, const float *rhs, float *tmp, float *dst) {
        if (data.ndim() == 2) {
            ctx.ufunc<Op::mul>(dst, lhs, rhs);
            smooth.sequence(dst, 0, tmp, 0, dst);
        } else {
            ctx.ufunc<Op::mul>(tmp, lhs, rhs);
            smooth.sequence(tmp, 0, dst, 0, tmp, 0, dst);
        }
    };

    if (data.ndim() == 2) {
        auto [x1, y1, xx, xy, yy] = ctx.allocate<5>();
        conv.derivative(1, ctx.src(), dst, x1, y1);
        mul_smooth(x1, x1, dst, xx);
        mul_smooth(x1, y1, dst, xy);
        mul_smooth(y1, y1, dst, yy);
        ctx.ufunc<Op::eigenvalues>(dst, xx, xy, yy);

    } else {
        auto [x1, y1, z1, xx, xy, xz, yy, yz, zz] = ctx.allocate<9>();
        conv.derivative(1, ctx.src(), dst, x1, y1, z1);
        mul_smooth(x1, x1, dst, xx);
        mul_smooth(x1, y1, dst, xy);
        mul_smooth(x1, z1, dst, xz);
        mul_smooth(y1, y1, dst, yy);
        mul_smooth(y1, z1, dst, yz);
        mul_smooth(z1, z1, dst, zz);
        ctx.ufunc<Op::eigenvalues>(dst, xx, xy, xz, yy, yz, zz);
    }

    return ctx.output();
}
} // namespace py

NB_MODULE(_internal, m) {
    using namespace nb::literals;

    nb::kw_only kw_only;
    nb::call_guard<nb::gil_scoped_release> no_gil;

    // clang-format off

    m.def("gaussian_kernel", &py::gaussian_kernel,
        "scale"_a, kw_only, "truncate"_a = 0, "order"_a = 0);

    m.def("gaussian_smoothing", &py::gaussian_smoothing,
        "data"_a, "scale"_a, kw_only, "truncate"_a = 0, "order"_a = 0,
        no_gil);

    m.def("gaussian_gradient_magnitude", &py::gaussian_gradient_magnitude,
        "data"_a, "scale"_a, kw_only, "truncate"_a = 0,
        no_gil);

    m.def("laplacian_of_gaussian", &py::laplacian_of_gaussian,
        "data"_a, "scale"_a, kw_only, "truncate"_a = 0,
        no_gil);

    m.def("hessian_of_gaussian_eigenvalues", &py::hessian_of_gaussian_eigenvalues,
        "data"_a, "scale"_a, kw_only, "truncate"_a = 0,
        no_gil);

    m.def("structure_tensor_eigenvalues", &py::structure_tensor_eigenvalues,
        "data"_a, "scale"_a, kw_only, "truncate"_a = 0, "smooth_scale"_a = 0,
        no_gil);

    // clang-format on
}
