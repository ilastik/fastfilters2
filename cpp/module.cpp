#include "kernel.hpp"
#include "simd.hpp"

#include <hwy/aligned_allocator.h>
#include <hwy/base.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>

namespace ff = fastfilters2;
namespace nb = nanobind;

using NDArrayIn = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using NDArray = nb::ndarray<float, nb::numpy>;

// Stores `NDArrayIn` and provides a convenient interface for `Filters`.
class NDArrayWrapper {
    NDArrayIn data;

public:
    NDArrayWrapper(NDArrayIn data) : data{data} {
        if (data.ndim() < 2 || data.ndim() > 3) {
            throw std::invalid_argument{
                "number of dimensions must be 2 or 3, got " +
                std::to_string(data.ndim())};
        }

        for (size_t i = 0; i < data.ndim(); ++i) {
            if (data.shape(i) == 0) {
                throw std::invalid_argument{"data cannot be empty"};
            }
            if (data.shape(i) == 1) {
                throw std::invalid_argument{"data cannot have singleton dimensions"};
            }
        }
    }

    size_t shape(size_t i) const noexcept { return data.shape(i); }
    size_t ndim() const noexcept { return data.ndim(); }
    const float *get() const noexcept { return data.data(); }
    operator const float *() const noexcept { return data.data(); }
};

// `std::unique_ptr` with a custom allocator, and the ability to decay to a raw pointer.
class Buffer {
    struct Deleter {
        void operator()(void *ptr) const noexcept { deallocate(ptr); }
    };

    std::unique_ptr<float[], Deleter> ptr;

public:
    static void deallocate(void *ptr) noexcept {
        hwy::FreeAlignedBytes(ptr, nullptr, nullptr);
    }

    Buffer() = default;

    explicit Buffer(size_t size)
            : ptr{hwy::detail::AllocateAlignedItems<float>(size, nullptr, nullptr)} {
        if (!ptr) {
            throw std::bad_alloc{};
        }
    }

    float *get() noexcept { return ptr.get(); }
    const float *get() const noexcept { return ptr.get(); }

    operator float *() noexcept { return ptr.get(); }
    operator const float *() const noexcept { return ptr.get(); }

    float *release() noexcept { return ptr.release(); }
};

// Stores kernel data and some of kernel parameters.
class Kernel {
    size_t size_;
    Buffer data_;
    int order_;

public:
    explicit Kernel(double scale, double truncate, int order)
            : size_{ff::kernel_size(scale, truncate, order)},
              data_{size_},
              order_{order} {
        ff::gaussian_kernel(data_, size_, scale, order);
    }

    const float *data() const noexcept { return data_.get(); }
    size_t size() const noexcept { return size_; }
    int order() const noexcept { return order_; }

    operator ff::simd::KernelView() const noexcept {
        return {data_.get(), size_, order_};
    }
};

// Shared context for the filter operations.
class Filters {
    static constexpr size_t max_ndim = 3;
    using ConvShape = std::array<size_t, max_ndim>;

    static ConvShape conv_shape(NDArrayIn data) {
        ConvShape shape;
        for (size_t i = 0, j = 0; i < shape.size(); ++i) {
            // Set the missing leftmost dimensions to 1.
            shape[i] = data.ndim() + i < shape.size() ? 1 : data.shape(j++);
        }

        // Round the last dimension up to the nearest multiple of `lane_count`.
        auto mask = ff::simd::lane_count() - 1;
        shape[shape.size() - 1] = (shape[shape.size() - 1] + mask) & ~mask;

        return shape;
    }

    NDArrayWrapper data;
    double scale;
    double truncate;
    ConvShape buf_shape;
    size_t buf_size;
    Buffer row_buf;

    template <size_t N> std::array<Buffer, N> allocate_buffers() {
        std::array<Buffer, N> bufs;
        for (auto &buf : bufs) {
            buf = Buffer{buf_size};
        }
        return bufs;
    }

    Buffer allocate_output(size_t n_channels = 1) {
        return Buffer{n_channels * buf_size};
    }

    NDArray into_result(Buffer &src, size_t n_channels = 1) {
        size_t shape[max_ndim + 1];
        int64_t strides[max_ndim + 1];
        size_t ndim = 0;

        if (n_channels > 1) {
            shape[ndim++] = n_channels;
        }
        for (size_t i = 0; i < data.ndim(); ++i) {
            shape[ndim++] = data.shape(i);
        }

        // Stride of the second-to-last dimension might be different due to the padding
        // introduced in `conv_shape`.
        strides[ndim - 1] = 1;
        strides[ndim - 2] = buf_shape[buf_shape.size() - 1];
        for (size_t i = ndim - 2; i-- > 0;) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        auto raw = src.release();
        return {raw, ndim, shape, nb::capsule{raw, src.deallocate}, strides};
    }

    // Convolve `src` with `kernel` along `axis`, and store the result in `dst`. If
    // `src` is `nullptr`, just return the destination pointer (useful for skipping
    // redundant computations).
    const float *
    convolve_axis(int axis, const float *src, const Kernel &kernel, float *dst) {
        // Check against the input data dimensions because `shape` might be padded.
        if (data.shape(axis) < kernel.size()) {
            throw std::invalid_argument{
                "data dimensions are too small for the given kernel size"};
        }

        if (src == nullptr) {
            return dst;
        }

        // Lazy-allocate the row buffer. Do not allocate in the constructor, when the
        // GIL is not yet released (a small performance gain).
        if (!row_buf) {
            // Allocate more than the required size for the row buffer in order to avoid
            // reallocations on larger kernels. Kernel cannot be larger than the row
            // size, so assume that the maximum kernel size is equal to the row size.
            auto row_buf_size = 3 * buf_shape[buf_shape.size() - 1];
            row_buf = Buffer{row_buf_size};
            // Need to zero-initialize this buffer because the downstream code might
            // read uninitialized floats. This is safe by itself, because these values
            // are unused and needed just for the SIMD padding. However, some of these
            // values might happen to be subnormal. On some CPUs, including x86,
            // operations on subnormals trigger an extremely slow microcode path. It is
            // possible to avoid this on x86 by setting FTZ/DAZ CPU flags, but this
            // approach requires saving and restoring these flags (we don't want the
            // outside code to be influenced by this), and it is x86-only.
            hwy::ZeroBytes(row_buf, sizeof(float) * row_buf_size);
        }

        // If source is the input data, the contiguous dimension might be different from
        // the buffer dimension due to the padding.
        auto shape = buf_shape;
        if (src == data.get()) {
            shape[buf_shape.size() - 1] = data.shape(data.ndim() - 1);
        }

        ff::simd::convolve(axis, {src, shape}, kernel, dst, row_buf);
        return dst;
    }

    const float *convolve(
            const float *src,
            const Kernel &kernel_x,
            const Kernel &kernel_y,
            float *dst_x,
            float *dst_y) {
        auto ptr = src;
        ptr = convolve_axis(2, ptr, kernel_x, dst_x);
        ptr = convolve_axis(1, ptr, kernel_y, dst_y);
        return ptr;
    }

    const float *convolve(
            const float *src,
            const Kernel &kernel_x,
            const Kernel &kernel_y,
            const Kernel &kernel_z,
            float *dst_x,
            float *dst_y,
            float *dst_z) {
        auto ptr = src;
        ptr = convolve_axis(2, ptr, kernel_x, dst_x);
        ptr = convolve_axis(1, ptr, kernel_y, dst_y);
        ptr = convolve_axis(0, ptr, kernel_z, dst_z);
        return ptr;
    }

    void l2norm(const float *x, const float *y, float *dst) {
        ff::simd::l2norm({{x, y}, buf_size, 2}, dst);
    }

    void l2norm(const float *x, const float *y, const float *z, float *dst) {
        ff::simd::l2norm({{x, y, z}, buf_size, 3}, dst);
    }

    void add(const float *x, const float *y, float *dst) {
        ff::simd::add({{x, y}, buf_size, 2}, dst);
    }

    void add(const float *x, const float *y, const float *z, float *dst) {
        ff::simd::add({{x, y, z}, buf_size, 3}, dst);
    }

    void mul_pairs(
            const float *x,
            const float *y,
            float *dst_xx,
            float *dst_xy,
            float *dst_yy) {
        ff::simd::mul_pairs({{x, y}, buf_size, 2}, {{dst_xx, dst_xy, dst_yy}});
    }

    void mul_pairs(
            const float *x,
            const float *y,
            const float *z,
            float *dst_xx,
            float *dst_xy,
            float *dst_yy,
            float *dst_xz,
            float *dst_yz,
            float *dst_zz) {
        ff::simd::mul_pairs(
                {{x, y, z}, buf_size, 3},
                {{dst_xx, dst_xy, dst_yy, dst_xz, dst_yz, dst_zz}});
    }

    void eigenvalues(const float *xx, const float *xy, const float *yy, float *dst) {
        ff::simd::eigenvalues({{xx, xy, yy}, buf_size, 3}, {dst, dst + buf_size});
    }

    void eigenvalues(
            const float *xx,
            const float *xy,
            const float *yy,
            const float *xz,
            const float *yz,
            const float *zz,
            float *dst) {
        // Note the different input order. This particular order matches the old
        // implementation.
        ff::simd::eigenvalues(
                {{zz, yz, xz, yy, xy, xx}, buf_size, 6},
                {dst, dst + buf_size, dst + 2 * buf_size});
    }

public:
    Filters(NDArrayIn data, double scale, double truncate)
            : data{data},
              scale{scale},
              truncate{truncate},
              buf_shape{conv_shape(data)},
              buf_size{buf_shape[0] * buf_shape[1] * buf_shape[2]} {

        if (scale <= 0) {
            throw std::invalid_argument{
                "scale must be positive, got " + std::to_string(scale)};
        }
        if (truncate < 0) {
            throw std::invalid_argument{
                "truncate must be positive or zero, got " + std::to_string(truncate)};
        }
    }

    NDArray gaussian_derivative(int order) {
        if (order < 0 || order > 2) {
            throw std::invalid_argument{
                "order must be between 0 and 2, got " + std::to_string(order)};
        }

        Buffer out;
        {
            nb::gil_scoped_release release;
            out = allocate_output();

            Kernel k{scale, truncate, order};

            auto [tmp] = allocate_buffers<1>();
            if (data.ndim() == 2) {
                convolve(data, k, k, tmp, out);
            } else if (data.ndim() == 3) {
                convolve(data, k, k, k, out, tmp, out);
            }
        }

        return into_result(out);
    }

    NDArray gaussian_gradient_magnitude() {
        Buffer out;
        {
            nb::gil_scoped_release release;
            out = allocate_output();

            Kernel k0{scale, truncate, 0};
            Kernel k1{scale, truncate, 1};

            if (data.ndim() == 2) {
                auto [x, y] = allocate_buffers<2>();
                convolve(data, k1, k0, out, x);
                convolve(data, k0, k1, out, y);
                l2norm(x, y, out);

            } else if (data.ndim() == 3) {
                auto [x, y, z] = allocate_buffers<3>();
                convolve(data, k1, k0, k0, x, out, x);
                convolve(data, k0, k1, k0, z, out, y);
                convolve(nullptr, k0, k0, k1, z, out, z);
                l2norm(x, y, z, out);
            }
        }

        return into_result(out);
    }

    NDArray laplacian_of_gaussian() {
        Buffer out;
        {
            nb::gil_scoped_release release;
            out = allocate_output();

            Kernel k0{scale, truncate, 0};
            Kernel k2{scale, truncate, 2};

            if (data.ndim() == 2) {
                auto [x, y] = allocate_buffers<2>();
                convolve(data, k2, k0, out, x);
                convolve(data, k0, k2, out, y);
                add(x, y, out);

            } else if (data.ndim() == 3) {
                auto [x, y, z] = allocate_buffers<3>();
                convolve(data, k2, k0, k0, x, out, x);
                convolve(data, k0, k2, k0, z, out, y);
                convolve(nullptr, k0, k0, k2, z, out, z);
                add(x, y, z, out);
            }
        }

        return into_result(out);
    }

    NDArray hessian_of_gaussian_eigenvalues() {
        Buffer out;
        {
            nb::gil_scoped_release release;
            out = allocate_output(data.ndim());

            Kernel k0{scale, truncate, 0};
            Kernel k1{scale, truncate, 1};
            Kernel k2{scale, truncate, 2};

            if (data.ndim() == 2) {
                auto [xx, xy, yy] = allocate_buffers<3>();
                convolve(data, k2, k0, out, xx);
                convolve(data, k1, k1, out, xy);
                convolve(data, k0, k2, out, yy);
                eigenvalues(xx, xy, yy, out);

            } else if (data.ndim() == 3) {
                auto [xx, xy, yy, xz, yz, zz] = allocate_buffers<6>();
                convolve(data, k2, k0, k0, xx, out, xx);
                convolve(data, k1, k1, k0, xz, out, xy);
                convolve(nullptr, k1, k0, k1, xz, out, xz);
                convolve(data, k0, k2, k0, zz, out, yy);
                convolve(nullptr, k0, k1, k1, zz, out, yz);
                convolve(nullptr, k0, k0, k2, zz, out, zz);
                eigenvalues(xx, xy, yy, xz, yz, zz, out);
            }
        }

        return into_result(out, data.ndim());
    }

    NDArray structure_tensor_eigenvalues(double derivative_scale) {
        if (derivative_scale < 0) {
            throw std::invalid_argument{
                "derivative_scale must be positive or 0, got " +
                std::to_string(derivative_scale)};
        }
        if (derivative_scale == 0) {
            derivative_scale = 0.5 * scale;
        }

        Buffer out;
        {
            nb::gil_scoped_release release;
            out = allocate_output(data.ndim());

            Kernel k0{derivative_scale, truncate, 0};
            Kernel k1{derivative_scale, truncate, 1};
            Kernel ks{scale, truncate, 0};

            if (data.ndim() == 2) {
                auto [x, y, xx, xy, yy] = allocate_buffers<5>();
                convolve(data, k1, k0, out, x);
                convolve(data, k0, k1, out, y);
                mul_pairs(x, y, xx, xy, yy);
                convolve(xx, ks, ks, out, xx);
                convolve(xy, ks, ks, out, xy);
                convolve(yy, ks, ks, out, yy);
                eigenvalues(xx, xy, yy, out);

            } else if (data.ndim() == 3) {
                auto [x, y, z, xx, xy, yy, xz, yz, zz] = allocate_buffers<9>();
                convolve(data, k1, k0, k0, x, out, x);
                convolve(data, k0, k1, k0, z, out, y);
                convolve(nullptr, k0, k0, k1, z, out, z);
                mul_pairs(x, y, z, xx, xy, yy, xz, yz, zz);
                convolve(xx, ks, ks, ks, xx, out, xx);
                convolve(xy, ks, ks, ks, xy, out, xy);
                convolve(yy, ks, ks, ks, yy, out, yy);
                convolve(xz, ks, ks, ks, xz, out, xz);
                convolve(yz, ks, ks, ks, yz, out, yz);
                convolve(zz, ks, ks, ks, zz, out, zz);
                eigenvalues(xx, xy, yy, xz, yz, zz, out);
            }
        }

        return into_result(out, data.ndim());
    }
};

NB_MODULE(_internal, m) {
    using namespace nb::literals;

    int debug = 0;
    if (auto begin = std::getenv("FASTFILTERS2_DEBUG"); begin != nullptr) {
        auto end = begin + std::strlen(begin);
        auto result = std::from_chars(begin, end, debug);
        // `std::from_chars` parses valid numeric prefixes, but we want the entire
        // string to be a valid integer.
        if (!(result.ptr == end && result.ec == std::errc{})) {
            // Don't report the error, just disable the debug mode: module import must
            // not break if a debug environment variable is malformed.
            debug = 0;
        }
    }
    ff::simd::initialize(debug);

    m.def(
            "gaussian_smoothing",
            [](NDArrayIn data, double scale, double truncate) {
                return Filters{data, scale, truncate}.gaussian_derivative(0);
            },
            "data"_a,
            "scale"_a,
            nb::kw_only{},
            "truncate"_a = 0);

    m.def(
            "gaussian_gradient_magnitude",
            [](NDArrayIn data, double scale, double truncate) {
                return Filters{data, scale, truncate}.gaussian_gradient_magnitude();
            },
            "data"_a,
            "scale"_a,
            nb::kw_only{},
            "truncate"_a = 0);

    m.def(
            "laplacian_of_gaussian",
            [](NDArrayIn data, double scale, double truncate) {
                return Filters{data, scale, truncate}.laplacian_of_gaussian();
            },
            "data"_a,
            "scale"_a,
            nb::kw_only{},
            "truncate"_a = 0);

    m.def(
            "hessian_of_gaussian_eigenvalues",
            [](NDArrayIn data, double scale, double truncate) {
                return Filters{data, scale, truncate}.hessian_of_gaussian_eigenvalues();
            },
            "data"_a,
            "scale"_a,
            nb::kw_only{},
            "truncate"_a = 0);

    m.def(
            "structure_tensor_eigenvalues",
            [](NDArrayIn data, double scale, double derivative_scale, double truncate) {
                return Filters{data, scale, truncate}.structure_tensor_eigenvalues(
                        derivative_scale);
            },
            "data"_a,
            "scale"_a,
            nb::kw_only{},
            "derivative_scale"_a = 0,
            "truncate"_a = 0);

    m.def(
            "gaussian_derivative",
            [](NDArrayIn data, double scale, int order, double truncate) {
                return Filters{data, scale, truncate}.gaussian_derivative(order);
            },
            "data"_a,
            "scale"_a,
            nb::kw_only{},
            "order"_a,
            "truncate"_a = 0);
}
