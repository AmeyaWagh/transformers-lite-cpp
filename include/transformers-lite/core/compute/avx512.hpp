#pragma once

#ifdef __AVX512F__
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

} // namespace transformers_lite
#endif // __AVX512F__
