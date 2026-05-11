#pragma once

#ifdef __AVX512F__
#include <cmath>
#include <immintrin.h>

namespace transformers_lite {

/**
 * @brief Matrix-vector multiply using AVX-512 FMA intrinsics.
 *
 * W (d,n) @ x (n,) -> xout (d,)
 * Processes 16 floats per iteration with _mm512_fmadd_ps; scalar tail handles remainder.
 */
inline void matmulAVX512(float *xout, const float *x, const float *w, int n, int d) {
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        __m512 sum = _mm512_setzero_ps();
        int j = 0;
        for (; j <= n - 16; j += 16) {
            __m512 wi = _mm512_loadu_ps(w + i * n + j);
            __m512 xi = _mm512_loadu_ps(x + j);
            sum = _mm512_fmadd_ps(wi, xi, sum);
        }
        float val = _mm512_reduce_add_ps(sum);
        for (; j < n; j++)
            val += w[i * n + j] * x[j];
        xout[i] = val;
    }
}

/** @brief Element-wise addition of two float arrays using AVX-512. */
inline void addAVX512(float *out, const float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < n; ++i)
        out[i] = a[i] + b[i];
}

/** @brief Element-wise subtraction of two float arrays using AVX-512. */
inline void subAVX512(float *out, const float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(out + i, _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < n; ++i)
        out[i] = a[i] - b[i];
}

/** @brief Element-wise multiplication of two float arrays using AVX-512. */
inline void mulAVX512(float *out, const float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < n; ++i)
        out[i] = a[i] * b[i];
}

/** @brief Element-wise division of two float arrays using AVX-512. */
inline void divAVX512(float *out, const float *a, const float *b, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(out + i, _mm512_div_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    for (; i < n; ++i)
        out[i] = a[i] / b[i];
}

/** @brief Scalar multiply: out[i] = a[i] * scalar, using AVX-512. */
inline void scaleAVX512(float *out, const float *a, float scalar, size_t n) {
    __m512 vs = _mm512_set1_ps(scalar);
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), vs));
    for (; i < n; ++i)
        out[i] = a[i] * scalar;
}

/** @brief Dot product of two float arrays using AVX-512. */
inline float dotAVX512(const float *a, const float *b, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), sum);
    float val = _mm512_reduce_add_ps(sum);
    for (; i < n; ++i)
        val += a[i] * b[i];
    return val;
}

/** @brief In-place softmax over the first n elements of x using AVX-512. */
inline void softmaxAVX512(float *x, size_t n) {
    float max_val = x[0];
    size_t i = 0;
    if (n >= 16) {
        __m512 vmax = _mm512_loadu_ps(x);
        for (i = 16; i + 16 <= n; i += 16)
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(x + i));
        max_val = _mm512_reduce_max_ps(vmax);
    }
    for (; i < n; ++i)
        if (x[i] > max_val)
            max_val = x[i];

    float sum = 0.f;
    for (i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    const float inv_sum = 1.0f / sum;
    __m512 vinv = _mm512_set1_ps(inv_sum);
    i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), vinv));
    for (; i < n; ++i)
        x[i] *= inv_sum;
}

/** @brief Causal scaled dot-product attention with GQA support using AVX-512. */
inline void scaledDotProductAttentionAVX512(float *out, const float *q, const float *key_cache, const float *val_cache, float *att_buf, int pos, size_t n_heads,
                                            size_t kv_heads, size_t head_size, size_t kv_dim, size_t seq_len) {
    const size_t kv_mul = n_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    size_t h;
#pragma omp parallel for private(h)
    for (h = 0; h < n_heads; h++) {
        const float *q_h = q + h * head_size;
        float *att_h = att_buf + h * seq_len;
        const size_t kv_off = (h / kv_mul) * head_size;

        for (size_t t = 0; t <= static_cast<size_t>(pos); t++)
            att_h[t] = dotAVX512(q_h, key_cache + t * kv_dim + kv_off, head_size) * scale;

        const size_t n = static_cast<size_t>(pos) + 1;
        size_t i = 0;

        float max_val = att_h[0];
        if (n >= 16) {
            __m512 vmax = _mm512_loadu_ps(att_h);
            for (i = 16; i + 16 <= n; i += 16)
                vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(att_h + i));
            max_val = _mm512_reduce_max_ps(vmax);
        }
        for (; i < n; ++i)
            if (att_h[i] > max_val)
                max_val = att_h[i];

        float sum = 0.f;
        for (i = 0; i < n; ++i) {
            att_h[i] = std::exp(att_h[i] - max_val);
            sum += att_h[i];
        }

        const float inv_sum = 1.0f / sum;
        __m512 vinv = _mm512_set1_ps(inv_sum);
        i = 0;
        for (; i + 16 <= n; i += 16)
            _mm512_storeu_ps(att_h + i, _mm512_mul_ps(_mm512_loadu_ps(att_h + i), vinv));
        for (; i < n; ++i)
            att_h[i] *= inv_sum;

        float *out_h = out + h * head_size;
        i = 0;
        for (; i + 16 <= head_size; i += 16)
            _mm512_storeu_ps(out_h + i, _mm512_setzero_ps());
        for (; i < head_size; ++i)
            out_h[i] = 0.f;

        for (size_t t = 0; t <= static_cast<size_t>(pos); t++) {
            const float *v_t = val_cache + t * kv_dim + kv_off;
            __m512 va = _mm512_set1_ps(att_h[t]);
            i = 0;
            for (; i + 16 <= head_size; i += 16)
                _mm512_storeu_ps(out_h + i, _mm512_fmadd_ps(va, _mm512_loadu_ps(v_t + i), _mm512_loadu_ps(out_h + i)));
            for (; i < head_size; ++i)
                out_h[i] += att_h[t] * v_t[i];
        }
    }
}

} // namespace transformers_lite
#endif // __AVX512F__
