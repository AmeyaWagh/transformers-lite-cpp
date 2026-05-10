#pragma once
#include <algorithm>
#include <cstring>
#include <memory>

#include "xpu.hpp"

namespace transformers_lite {

/** @brief CPU SIMD accelerators, auto-detected at compile time. */
enum class CPUAccelerator { NONE, AVX512 };

#if defined(__AVX512F__)
inline constexpr CPUAccelerator kCPUAccelerator = CPUAccelerator::AVX512;
#else
inline constexpr CPUAccelerator kCPUAccelerator = CPUAccelerator::NONE;
#endif

/**
 * @brief CPU compute backend providing host-side memory operations.
 *
 * @tparam T element data type
 */
template <class T> struct CPU : public XPU {
    static constexpr CPUAccelerator accelerator = kCPUAccelerator;
    using allocator_type = std::allocator<T>;

    /**
     * @brief Fill memory with a constant value.
     *
     * @param begin_ pointer to the start of the memory region
     * @param num_elements number of elements to fill
     * @param val value to fill with
     */
    static void fill(T *begin_, size_t num_elements, T val) { std::fill(begin_, begin_ + num_elements, val); }

    /**
     * @brief Copy elements from source to destination.
     *
     * @param src_ pointer to source data
     * @param dest_ pointer to destination data
     * @param num_elements number of elements to copy
     */
    static void copy(const T *src_, T *dest_, size_t num_elements) { std::memcpy(dest_, src_, num_elements * sizeof(T)); }

    /**
     * @brief Access an element at the given index.
     *
     * @param data pointer to the memory region
     * @param index element index
     * @return T& reference to the element
     */
    static auto get(T *data, size_t index) -> T & { return *(data + index); }
};

} // namespace transformers_lite
