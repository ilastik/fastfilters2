#pragma once

#include <array>
#include <cstddef>

// SIMD-accelerated functions. Dispatch to the best available implementation in runtime.
// Before calling any function in this namespace, clients must call `initialize()`.
namespace fastfilters2::simd {

// A static upper bound on the number of bytes per vector lane. This value is large
// enough in order to accomodate the full size of SVE and RVV vectors.
constexpr size_t max_lane_bytes = 256;

// Non-owning view of a 3-dimensional array.
struct DataView3D {
    // Pointer to the first element of the input data.
    const float *data;
    // Row-major shape of the input data.
    std::array<size_t, 3> shape;
};

// Non-owning view of a kernel.
struct KernelView {
    // Pointer to the first element of the kernel.
    const float *data;
    // Number of physically stored pixels (kernel radius + 1).
    size_t size;
    // Kernel order.
    int order;
};

// Non-owning view of a multiple 1-dimensional input arrays. `N` is the maximum number
// of input arrays (an upper bound for `count`).
template <size_t N> struct MultiDataView {
    // Pointers to the first elements of the input arrays.
    std::array<const float *, N> data;
    // The total size of each input array.
    size_t size;
    // The actual number of input arrays. Must not exceed `N`. Valid values are
    // specified by each function that accepts this type as an argument.
    int count;
};

// Non-owning view of multiple 1-dimensional output arrays. `N` is the maximum number of
// output arrays. For each particular function, the size of each output array is the
// same as `MultiDataView::size`, and `MultiDataView::count` is used to infer the
// corresponding count output array count.
template <size_t N> using MultiOutputView = std::array<float *, N>;

// Auto-tune the library for the current CPU. Must be called before any other functions
// in this namespace. Not thread-safe. If `debug` is positive, print some diagnostic
// information to stderr.
void initialize(int debug = 0);

// Return the number of (float32) lanes per SIMD register. All SIMD functions in this
// namespace work only with inputs that are a multiple of this size.
size_t lane_count();

// Convolve `src` with `kernel` along the specified `axis`, and write results to `dst`.
// `row_buf` is a temporary buffer that has at least `row_size + 2 * radius` elements,
// where `row_size` is the last dimension of `src` and `radius` is the radius of the
// `kernel`. `axis` must be a valid index into the shape of `src`. The size of `kernel`
// must not exceed the target dimension. `dst` should point to a buffer large enough to
// hold the output. As a special ad-hoc optimization, if `axis == 2`, then the source
// shape might be smaller than `Config::min_row_size`; in this case, `row_buf` must be
// at least `Config::min_row_size + 2 * radius` elements, and the output shape should be
// `(*src.shape[:-1], Config::min_row_size)`.
void convolve(int axis, DataView3D src, KernelView kernel, float *dst, float *row_buf);

// Compute the L2 norm of the input data for 2 or 3 input arrays.
void l2norm(MultiDataView<3> src, float *dst);

// Add the input data to the output data for 2 or 3 input arrays.
void add(MultiDataView<3> src, float *dst);

// Compute pairwise products of the input data for 2 or 3 input arrays, in the
// column-major order (xx, xy, yy, xz, yz, zz).
void mul_pairs(MultiDataView<3> src, MultiOutputView<6> dst);

// Compute eigenvalues of many symmetric positive-definite 2D or 3D matrices, that are
// stored element-wise in `src`. Matrices are upper-triangular, and passed in the
// column-major order (xx, xy, yy, xz, yz, zz). The eigenvalues are returned in the
// descending order.
void eigenvalues(MultiDataView<6> src, MultiOutputView<3> dst);

} // namespace fastfilters2::simd
