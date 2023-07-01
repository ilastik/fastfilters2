#ifndef FASTFILTERS2_UTIL_H_
#define FASTFILTERS2_UTIL_H_

#include "fastfilters2.h"

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
 * Copy src to dst, reflecting the first and the last radius elements.
 * The first and the last element are not included in the reflection.
 * Example for size = 6 and radius = 2: [012345] -> 21[012345]43.
 */
template <typename T>
void reflect_copy(const T *HWY_RESTRICT src, T *HWY_RESTRICT dst, ssize_t size, ssize_t radius) {
    HWY_ASSUME(size > 0);
    HWY_ASSUME(radius > 0);
    HWY_ASSUME(size > radius);

    for (ssize_t i = 0; i < radius; ++i) {
        dst[i] = src[radius - i];
    }
    for (ssize_t i = 0; i < size; ++i) {
        dst[radius + i] = src[i];
    }
    for (ssize_t i = 0; i < radius; ++i) {
        dst[radius + size + i] = src[size - 2 - i];
    }
}
} // namespace fastfilters2::util

#endif // FASTFILTERS2_UTIL_H_
