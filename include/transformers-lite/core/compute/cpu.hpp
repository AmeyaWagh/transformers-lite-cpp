#pragma once
#include <algorithm>
#include <cmath>
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

    static void scale(T *out, const T *a, T scalar, size_t n) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>)
            scaleAVX512(out, a, scalar, n);
        else
            for (size_t i = 0; i < n; ++i)
                out[i] = a[i] * scalar;
    }

    [[nodiscard]] static T dot(const T *a, const T *b, size_t n) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>)
            return static_cast<T>(dotAVX512(a, b, n));
        else {
            T val = T(0);
            for (size_t i = 0; i < n; ++i)
                val += a[i] * b[i];
            return val;
        }
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

    static void silu(T *out, const T *x, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            T val = x[i];
            out[i] = val / (static_cast<T>(1) + std::exp(-val));
        }
    }

    static void rope(T *q, T *k, int pos, size_t head_size, size_t dim, size_t kv_dim) {
        for (size_t i = 0; i < dim; i += 2) {
            size_t head_dim = i % head_size;
            T freq = static_cast<T>(1) / std::pow(static_cast<T>(10000), head_dim / static_cast<T>(head_size));
            T theta = static_cast<T>(pos) * freq;
            T fcr = std::cos(theta);
            T fci = std::sin(theta);
            size_t rotn = i < kv_dim ? 2 : 1;
            for (size_t r = 0; r < rotn; r++) {
                T *vec = r == 0 ? q : k;
                T v0 = vec[i];
                T v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }

    /** @brief In-place softmax over the first n elements of x. */
    static void softmax(T *x, size_t n) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            softmaxAVX512(x, n);
        } else {
            T max_val = x[0];
            for (size_t i = 1; i < n; ++i)
                if (x[i] > max_val)
                    max_val = x[i];
            T sum = T(0);
            for (size_t i = 0; i < n; ++i) {
                x[i] = std::exp(x[i] - max_val);
                sum += x[i];
            }
            for (size_t i = 0; i < n; ++i)
                x[i] /= sum;
        }
    }

    /**
     * @brief Causal scaled dot-product attention with GQA support.
     *
     * For each query head h:
     *   att[t] = dot(q_h, k_t) / sqrt(head_size)   for t in [0, pos]
     *   att     = softmax(att)
     *   out_h   = sum_t( att[t] * v_t )
     *
     * @param out        output buffer (n_heads * head_size,)
     * @param q          query buffer  (n_heads * head_size,)
     * @param key_cache  flat KV cache (seq_len * kv_dim,)
     * @param val_cache  flat V  cache (seq_len * kv_dim,)
     * @param att_buf    scratch attention scores (n_heads * seq_len,) — written in place
     * @param pos        current sequence position (attend to [0, pos])
     * @param n_heads    number of query heads
     * @param kv_heads   number of key/value heads
     * @param head_size  elements per head
     * @param kv_dim     kv_heads * head_size
     * @param seq_len    maximum sequence length (stride of the KV cache)
     */
    static void scaledDotProductAttention(T *out, const T *q, const T *key_cache, const T *val_cache, T *att_buf, int pos, size_t n_heads, size_t kv_heads,
                                          size_t head_size, size_t kv_dim, size_t seq_len) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            scaledDotProductAttentionAVX512(out, q, key_cache, val_cache, att_buf, pos, n_heads, kv_heads, head_size, kv_dim, seq_len);
            return;
        }
        const size_t kv_mul = n_heads / kv_heads;
        const T scale = static_cast<T>(1) / std::sqrt(static_cast<T>(head_size));
        size_t h;
#pragma omp parallel for private(h)
        for (h = 0; h < n_heads; h++) {
            const T *q_h = q + h * head_size;
            T *att_h = att_buf + h * seq_len;

            for (size_t t = 0; t <= static_cast<size_t>(pos); t++) {
                const T *k_t = key_cache + t * kv_dim + (h / kv_mul) * head_size;
                att_h[t] = dot(q_h, k_t, head_size) * scale;
            }

            softmax(att_h, static_cast<size_t>(pos) + 1);

            T *out_h = out + h * head_size;
            for (size_t i = 0; i < head_size; i++)
                out_h[i] = T(0);
            for (size_t t = 0; t <= static_cast<size_t>(pos); t++) {
                const T *v_t = val_cache + t * kv_dim + (h / kv_mul) * head_size;
                T a = att_h[t];
                for (size_t i = 0; i < head_size; i++)
                    out_h[i] += a * v_t[i];
            }
        }
    }
};

} // namespace transformers_lite
