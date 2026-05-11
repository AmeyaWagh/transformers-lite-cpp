#pragma once

#ifdef __AVX512F__
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace transformers_lite {

/**
 * @brief Matrix-vector multiply using AVX-512 FMA intrinsics.
 *
 * weights (outDim,inDim) @ input (inDim,) -> out (outDim,)
 * Processes 16 floats per iteration with _mm512_fmadd_ps; scalar tail handles remainder.
 */
inline void matmulAVX512(float *out, const float *input, const float *weights, int inDim, int outDim) {
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < outDim; i++) {
        __m512 sum = _mm512_setzero_ps();
        int j = 0;
        for (; j <= inDim - 16; j += 16) {
            __m512 wi = _mm512_loadu_ps(weights + (i * inDim) + j);
            __m512 xi = _mm512_loadu_ps(input + j);
            sum = _mm512_fmadd_ps(wi, xi, sum);
        }
        float val = _mm512_reduce_add_ps(sum);
        for (; j < inDim; j++) {
            val += weights[(i * inDim) + j] * input[j];
        }
        out[i] = val;
    }
}

/** @brief Element-wise addition of two float arrays using AVX-512. */
inline void addAVX512(float *out, const float *lhs, const float *rhs, size_t len) {
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(lhs + i), _mm512_loadu_ps(rhs + i)));
    }
    for (; i < len; ++i) {
        out[i] = lhs[i] + rhs[i];
    }
}

/** @brief Element-wise subtraction of two float arrays using AVX-512. */
inline void subAVX512(float *out, const float *lhs, const float *rhs, size_t len) {
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_sub_ps(_mm512_loadu_ps(lhs + i), _mm512_loadu_ps(rhs + i)));
    }
    for (; i < len; ++i) {
        out[i] = lhs[i] - rhs[i];
    }
}

/** @brief Element-wise multiplication of two float arrays using AVX-512. */
inline void mulAVX512(float *out, const float *lhs, const float *rhs, size_t len) {
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(lhs + i), _mm512_loadu_ps(rhs + i)));
    }
    for (; i < len; ++i) {
        out[i] = lhs[i] * rhs[i];
    }
}

/** @brief Element-wise division of two float arrays using AVX-512. */
inline void divAVX512(float *out, const float *lhs, const float *rhs, size_t len) {
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_div_ps(_mm512_loadu_ps(lhs + i), _mm512_loadu_ps(rhs + i)));
    }
    for (; i < len; ++i) {
        out[i] = lhs[i] / rhs[i];
    }
}

/** @brief Scalar multiply: out[i] = src[i] * scalar, using AVX-512. */
inline void scaleAVX512(float *out, const float *src, float scalar, size_t len) {
    __m512 vecScale = _mm512_set1_ps(scalar);
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(src + i), vecScale));
    }
    for (; i < len; ++i) {
        out[i] = src[i] * scalar;
    }
}

/** @brief Dot product of two float arrays using AVX-512. */
inline float dotAVX512(const float *lhs, const float *rhs, size_t len) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(lhs + i), _mm512_loadu_ps(rhs + i), sum);
    }
    float val = _mm512_reduce_add_ps(sum);
    for (; i < len; ++i) {
        val += lhs[i] * rhs[i];
    }
    return val;
}

/** @brief In-place softmax over the first len elements of data using AVX-512. */
inline void softmaxAVX512(float *data, size_t len) {
    float max_val = data[0];
    size_t i = 0;
    if (len >= 16) {
        __m512 vmax = _mm512_loadu_ps(data);
        for (i = 16; i + 16 <= len; i += 16) {
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(data + i));
        }
        max_val = _mm512_reduce_max_ps(vmax);
    }
    for (; i < len; ++i) {
        max_val = std::max(max_val, data[i]);
    }

    float sum = 0.0F;
    for (i = 0; i < len; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    const float inv_sum = 1.0F / sum;
    __m512 vecInv = _mm512_set1_ps(inv_sum);
    i = 0;
    for (; i + 16 <= len; i += 16) {
        _mm512_storeu_ps(data + i, _mm512_mul_ps(_mm512_loadu_ps(data + i), vecInv));
    }
    for (; i < len; ++i) {
        data[i] *= inv_sum;
    }
}

/** @brief Causal scaled dot-product attention with GQA support using AVX-512. */
inline void scaledDotProductAttentionAVX512(float *out, const float *query, const float *keyCache, const float *valCache, float *attBuf, int pos, size_t nHeads,
                                            size_t kvHeads, size_t headSize, size_t kvDim, size_t seqLen) {
    const size_t kv_mul = nHeads / kvHeads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(headSize));
    size_t headIdx = 0;
#pragma omp parallel for private(headIdx)
    for (headIdx = 0; headIdx < nHeads; headIdx++) {
        const float *q_h = query + (headIdx * headSize);
        float *att_h = attBuf + (headIdx * seqLen);
        const size_t kv_off = (headIdx / kv_mul) * headSize;

        for (size_t t = 0; t <= static_cast<size_t>(pos); t++) {
            att_h[t] = dotAVX512(q_h, keyCache + (t * kvDim) + kv_off, headSize) * scale;
        }

        const size_t seqCount = static_cast<size_t>(pos) + 1;
        size_t i = 0;

        float max_val = att_h[0];
        if (seqCount >= 16) {
            __m512 vmax = _mm512_loadu_ps(att_h);
            for (i = 16; i + 16 <= seqCount; i += 16) {
                vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(att_h + i));
            }
            max_val = _mm512_reduce_max_ps(vmax);
        }
        for (; i < seqCount; ++i) {
            max_val = std::max(max_val, att_h[i]);
        }

        float sum = 0.0F;
        for (i = 0; i < seqCount; ++i) {
            att_h[i] = std::exp(att_h[i] - max_val);
            sum += att_h[i];
        }

        const float inv_sum = 1.0F / sum;
        __m512 vecInv = _mm512_set1_ps(inv_sum);
        i = 0;
        for (; i + 16 <= seqCount; i += 16) {
            _mm512_storeu_ps(att_h + i, _mm512_mul_ps(_mm512_loadu_ps(att_h + i), vecInv));
        }
        for (; i < seqCount; ++i) {
            att_h[i] *= inv_sum;
        }

        float *out_h = out + (headIdx * headSize);
        i = 0;
        for (; i + 16 <= headSize; i += 16) {
            _mm512_storeu_ps(out_h + i, _mm512_setzero_ps());
        }
        for (; i < headSize; ++i) {
            out_h[i] = 0.0F;
        }

        for (size_t t = 0; t <= static_cast<size_t>(pos); t++) {
            const float *v_t = valCache + (t * kvDim) + kv_off;
            __m512 vecAtt = _mm512_set1_ps(att_h[t]);
            i = 0;
            for (; i + 16 <= headSize; i += 16) {
                _mm512_storeu_ps(out_h + i, _mm512_fmadd_ps(vecAtt, _mm512_loadu_ps(v_t + i), _mm512_loadu_ps(out_h + i)));
            }
            for (; i < headSize; ++i) {
                out_h[i] += att_h[t] * v_t[i];
            }
        }
    }
}

} // namespace transformers_lite
#endif // __AVX512F__
