#pragma once
#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>

#include "avx512.hpp"
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

    static void add(T *out, const T *a, const T *b, size_t n) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>)
            addAVX512(out, a, b, n);
        else
            for (size_t i = 0; i < n; ++i)
                out[i] = a[i] + b[i];
    }

    static void sub(T *out, const T *a, const T *b, size_t n) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>)
            subAVX512(out, a, b, n);
        else
            for (size_t i = 0; i < n; ++i)
                out[i] = a[i] - b[i];
    }

    static void mul(T *out, const T *a, const T *b, size_t n) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>)
            mulAVX512(out, a, b, n);
        else
            for (size_t i = 0; i < n; ++i)
                out[i] = a[i] * b[i];
    }

    static void div(T *out, const T *a, const T *b, size_t n) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>)
            divAVX512(out, a, b, n);
        else
            for (size_t i = 0; i < n; ++i)
                out[i] = a[i] / b[i];
    }

    static void matmul(T *out, const T *x, const T *w, int n, int d) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            matmulAVX512(out, x, w, n, d);
        } else {
            int i;
#pragma omp parallel for private(i)
            for (i = 0; i < d; i++) {
                T val = T(0);
                for (int j = 0; j < n; j++)
                    val += w[i * n + j] * x[j];
                out[i] = val;
            }
        }
    }
};

} // namespace transformers_lite
