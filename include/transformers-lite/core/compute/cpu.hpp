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

    static void fill(T *data, size_t numElements, T val) { std::fill(data, data + numElements, val); }
    static void copy(const T *src, T *dest, size_t numElements) { std::memcpy(dest, src, numElements * sizeof(T)); }
    static auto get(T *data, size_t index) -> T & { return *(data + index); }
};

} // namespace transformers_lite
