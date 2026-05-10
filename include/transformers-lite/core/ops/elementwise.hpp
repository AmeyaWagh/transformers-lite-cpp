#pragma once
#include <cassert>
#include <cmath>

#include "../tensor.hpp"

namespace transformers_lite {

/**
 * @brief RMS normalization on tensor views.
 *
 * out[j] = weight[j] * (x[j] / rms(x))
 */
template <typename T> void rmsnorm(TensorView<T> &out, TensorView<T> &x, const TensorView<T> &weight) {
    T ss = static_cast<T>(0);
    for (auto j = 0; j < x.size(); ++j)
        ss += x(j) * x(j);
    ss /= x.size();
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (auto j = 0; j < x.size(); ++j)
        out(j) = weight(j) * (ss * x(j));
}

/**
 * @brief SiLU activation in-place: x[i] = x[i] * sigmoid(x[i])
 */
template <typename T> void silu_inpl(TensorView<T> &x) {
    for (size_t i = 0; i < x.size(); i++) {
        auto val = x[i];
        val *= (1.0f / (1.0f + expf(-val)));
        x[i] = val;
    }
}

/**
 * @brief Dot product of two tensor views.
 */
template <typename T> [[nodiscard]] T dot_prod(TensorView<T> &a, TensorView<T> &b) {
    assert(a.size() == b.size());
    T value = 0;
    for (size_t i = 0; i < a.size(); ++i)
        value += a(i) * b(i);
    return value;
}

/**
 * @brief Softmax in-place over the first n elements (all elements if n == -1).
 */
template <typename T> void softmax(TensorView<T> &x, int n = -1) {
    if (n == -1)
        n = x.size();
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val)
            max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

/**
 * @brief Index of the maximum element.
 */
template <typename T> [[nodiscard]] size_t argmax(TensorView<T> &x) {
    size_t max_idx = 0;
    T max_val = x[0];
    for (auto i = 1; i < x.size(); i++) {
        if (x[i] > max_val) {
            max_idx = i;
            max_val = x[i];
        }
    }
    return max_idx;
}

/**
 * @brief Zero all elements in-place.
 */
template <typename T> void setZero(TensorView<T> &x) {
    memset(x.data(), 0, x.size() * sizeof(T));
}

/**
 * @brief RoPE (Rotary Position Embedding) applied in-place to q and k.
 */
template <typename T> void rope(TensorView<T> &q, TensorView<T> &k, int pos, size_t head_size) {
    size_t dim = q.size();
    size_t kv_dim = k.size();
    for (size_t i = 0; i < dim; i += 2) {
        size_t head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / static_cast<float>(head_size));
        float theta = pos * freq;
        float fcr = cosf(theta);
        float fci = sinf(theta);
        size_t rotn = i < kv_dim ? 2 : 1;
        for (size_t r = 0; r < rotn; r++) {
            TensorView<T> &vec = r == 0 ? q : k;
            T v0 = vec[i];
            T v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

} // namespace transformers_lite
