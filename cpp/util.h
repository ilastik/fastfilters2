#ifndef FASTFILTERS2_UTIL_H_
#define FASTFILTERS2_UTIL_H_

#include "fastfilters2.h"

#include <algorithm>
#include <utility>

namespace fastfilters2::util {
template <ssize_t N, ssize_t... Indices, typename Func, typename... Args>
void static_for_helper(std::integer_sequence<ssize_t, Indices...>, Func func, Args &&...args) {
    // Call the real func only for the N first indices and arguments at compile-time.
    auto predicated_func = [func](auto idx, auto &&arg) {
        if constexpr (idx < N) {
            func(idx, std::forward<decltype(arg)>(arg));
        }
    };
    // Folding over the comma operator calls the function sequentially on each index and argument.
    (predicated_func(std::integral_constant<ssize_t, Indices>{}, std::forward<Args>(args)), ...);
}

/**
 * Apply function to the first N arguments, with guaranteed compile-time unrolling.
 * The function recieves an integral_constant index and the corresponding argument.
 */
template <ssize_t N, typename Func, typename... Args> void static_for(Func func, Args &&...args) {
    constexpr ssize_t Size = sizeof...(Args);
    static_assert(N <= Size, "Not enough arguments for the given N");
    // Need a helper in order to capture compile-time indices in a template parameter.
    // This could be replaced with a template lambda in C++20.
    constexpr auto indices = std::make_integer_sequence<ssize_t, Size>{};
    static_for_helper<N>(indices, func, std::forward<Args>(args)...);
}

/**
 * Copy src to dst with the reflected borders and zero padding.
 * dst should point to a memory of at least min_size + 2 * radius elements.
 * 1. Copy the first radius elements in reverse order, without the first element.
 * 2. Copy all elements from src to dst, from first to last, in the same order.
 * 3. Copy the last radius elements in reverse order, without the last element.
 * 4. If min_size > size, fill the rest of dst with zeros.
 */
template <typename T>
void reflect_copy(const T *HWY_RESTRICT src, T *HWY_RESTRICT dst, ssize_t size, ssize_t min_size, ssize_t radius) {
    HWY_ASSUME(size > 0);
    // HWY_ASSUME(min_size > 0);
    // HWY_ASSUME(min_size % 2 == 0);
    HWY_ASSUME(radius > 0);
    // HWY_ASSUME(size > radius);

    dst = std::reverse_copy(src + 1, src + 1 + radius, dst);
    dst = std::copy_n(src, size, dst);
    dst = std::reverse_copy(src + size - 1 - radius, src + size - 1, dst);
    std::fill_n(dst, std::max(ssize_t{0}, min_size - size), 0);
}
} // namespace fastfilters2::util

#endif // FASTFILTERS2_UTIL_H_
