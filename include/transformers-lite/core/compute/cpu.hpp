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
 * @brief CPU backend tag: carries allocator and low-level memory primitives.
 * Compute ops live in Ops<CPU, T> (see core/ops/cpu_ops.hpp).
 *
 * @tparam T element data type
 */
template <class T> struct CPU : public XPU {
    static constexpr CPUAccelerator accelerator = kCPUAccelerator;
    using allocator_type = std::allocator<T>;

    static void fill(T *begin_, size_t num_elements, T val) { std::fill(begin_, begin_ + num_elements, val); }
    static void copy(const T *src_, T *dest_, size_t num_elements) { std::memcpy(dest_, src_, num_elements * sizeof(T)); }
    static auto get(T *data, size_t index) -> T & { return *(data + index); }
};

} // namespace transformers_lite
