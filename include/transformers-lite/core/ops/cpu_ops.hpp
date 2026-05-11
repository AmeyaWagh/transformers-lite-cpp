#pragma once
#include <cmath>
#include <type_traits>

#include "../compute/cpu.hpp"
#include "ops_dispatch.hpp"

#ifdef __AVX512F__
#include "avx512_ops.hpp"
#endif

namespace transformers_lite {

template <typename T> struct Ops<CPU, T> {
    static void add(T *out, const T *lhs, const T *rhs, size_t len) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            addAVX512(out, lhs, rhs, len);
        } else {
            for (size_t i = 0; i < len; ++i) {
                out[i] = lhs[i] + rhs[i];
            }
        }
    }

    static void sub(T *out, const T *lhs, const T *rhs, size_t len) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            subAVX512(out, lhs, rhs, len);
        } else {
            for (size_t i = 0; i < len; ++i) {
                out[i] = lhs[i] - rhs[i];
            }
        }
    }

    static void mul(T *out, const T *lhs, const T *rhs, size_t len) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            mulAVX512(out, lhs, rhs, len);
        } else {
            for (size_t i = 0; i < len; ++i) {
                out[i] = lhs[i] * rhs[i];
            }
        }
    }

    static void div(T *out, const T *lhs, const T *rhs, size_t len) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            divAVX512(out, lhs, rhs, len);
        } else {
            for (size_t i = 0; i < len; ++i) {
                out[i] = lhs[i] / rhs[i];
            }
        }
    }

    static void scale(T *out, const T *src, T scalar, size_t len) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            scaleAVX512(out, src, scalar, len);
        } else {
            for (size_t i = 0; i < len; ++i) {
                out[i] = src[i] * scalar;
            }
        }
    }

    [[nodiscard]] static T dot(const T *lhs, const T *rhs, size_t len) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            return static_cast<T>(dotAVX512(lhs, rhs, len));
        } else {
            T val = T(0);
            for (size_t i = 0; i < len; ++i) {
                val += lhs[i] * rhs[i];
            }
            return val;
        }
    }

    static void matmul(T *out, const T *input, const T *weights, int inDim, int outDim) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            matmulAVX512(out, input, weights, inDim, outDim);
        } else {
            int i;
#pragma omp parallel for private(i)
            for (i = 0; i < outDim; i++) {
                T val = T(0);
                for (int j = 0; j < inDim; j++) {
                    val += weights[(i * inDim) + j] * input[j];
                }
                out[i] = val;
            }
        }
    }

    static void silu(T *out, const T *src, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            T val = src[i];
            out[i] = val / (static_cast<T>(1) + std::exp(-val));
        }
    }

    static void rope(T *query, T *keys, int pos, size_t headSize, size_t dim, size_t kvDim) {
        for (size_t i = 0; i < dim; i += 2) {
            size_t head_dim = i % headSize;
            T freq = static_cast<T>(1) / std::pow(static_cast<T>(10000), head_dim / static_cast<T>(headSize));
            T theta = static_cast<T>(pos) * freq;
            T fcr = std::cos(theta);
            T fci = std::sin(theta);
            size_t rotn = i < kvDim ? 2 : 1;
            for (size_t r = 0; r < rotn; r++) {
                T *vec = r == 0 ? query : keys;
                T real = vec[i];
                T imag = vec[i + 1];
                vec[i] = (real * fcr) - (imag * fci);
                vec[i + 1] = (real * fci) + (imag * fcr);
            }
        }
    }

    static void softmax(T *data, size_t len) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            softmaxAVX512(data, len);
        } else {
            T max_val = data[0];
            for (size_t i = 1; i < len; ++i) {
                if (data[i] > max_val) {
                    max_val = data[i];
                }
            }
            T sum = T(0);
            for (size_t i = 0; i < len; ++i) {
                data[i] = std::exp(data[i] - max_val);
                sum += data[i];
            }
            for (size_t i = 0; i < len; ++i) {
                data[i] /= sum;
            }
        }
    }

    static void scaledDotProductAttention(T *out, const T *query, const T *keyCache, const T *valCache, T *attBuf, int pos, size_t nHeads, size_t kvHeads,
                                          size_t headSize, size_t kvDim, size_t seqLen) {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            scaledDotProductAttentionAVX512(out, query, keyCache, valCache, attBuf, pos, nHeads, kvHeads, headSize, kvDim, seqLen);
            return;
        }
        const size_t kv_mul = nHeads / kvHeads;
        const T scale = static_cast<T>(1) / std::sqrt(static_cast<T>(headSize));
        size_t h;
#pragma omp parallel for private(h)
        for (h = 0; h < nHeads; h++) {
            const T *q_h = query + (h * headSize);
            T *att_h = attBuf + (h * seqLen);
            const size_t kv_off = (h / kv_mul) * headSize;

            for (size_t t = 0; t <= static_cast<size_t>(pos); t++) {
                const T *k_t = keyCache + (t * kvDim) + kv_off;
                att_h[t] = dot(q_h, k_t, headSize) * scale;
            }

            softmax(att_h, static_cast<size_t>(pos) + 1);

            T *out_h = out + (h * headSize);
            for (size_t i = 0; i < headSize; i++) {
                out_h[i] = T(0);
            }
            for (size_t t = 0; t <= static_cast<size_t>(pos); t++) {
                const T *v_t = valCache + (t * kvDim) + kv_off;
                T att = att_h[t];
                for (size_t i = 0; i < headSize; i++) {
                    out_h[i] += att * v_t[i];
                }
            }
        }
    }
};

} // namespace transformers_lite
