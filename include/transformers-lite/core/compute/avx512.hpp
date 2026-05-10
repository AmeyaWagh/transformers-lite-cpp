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
        for (; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

} // namespace transformers_lite
#endif // __AVX512F__
