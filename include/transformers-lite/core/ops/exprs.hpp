#pragma once
#include <cmath>

#include "../tensor.hpp"

namespace transformers_lite {

// ── MatMulExpr ────────────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T> struct MatMulExpr {
    using value_type = T;
    TensorView<T> x; // input  (n,)
    TensorView<T> w; // weight (m, n)

    [[nodiscard]] Shape outputShape() const { return Shape{w.shape()[0]}; }

    void evalInto(TensorView<T> &out) const { COMPUTE<T>::matmul(out.data(), x.data(), w.data(), static_cast<int>(x.size()), static_cast<int>(out.size())); }
};

// ── RMSNormExpr ───────────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T> struct RMSNormExpr {
    using value_type = T;
    TensorView<T> x;
    TensorView<T> weight;

    [[nodiscard]] Shape outputShape() const { return x.shape(); }

    void evalInto(TensorView<T> &out) const {
        const size_t n = x.size();
        Tensor<COMPUTE, T> ss = COMPUTE<T>::dot(x.data(), x.data(), n);
        ss /= static_cast<T>(n);
        ss += static_cast<T>(1e-5);
        ss = static_cast<T>(1) / std::sqrt(ss.item());
        COMPUTE<T>::scale(out.data(), x.data(), ss.item(), n);
        COMPUTE<T>::mul(out.data(), out.data(), weight.data(), n);
    }
};

// ── Primitive element-wise expression types ───────────────────────────────────

template <template <class> class COMPUTE, typename T> struct ElemwiseAddExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    [[nodiscard]] Shape outputShape() const { return a.shape(); }

    void evalInto(TensorView<T> &out) const { COMPUTE<T>::add(out.data(), a.data(), b.data(), a.size()); }
};

template <template <class> class COMPUTE, typename T> struct ElemwiseSubExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    [[nodiscard]] Shape outputShape() const { return a.shape(); }

    void evalInto(TensorView<T> &out) const { COMPUTE<T>::sub(out.data(), a.data(), b.data(), a.size()); }
};

template <template <class> class COMPUTE, typename T> struct ElemwiseMulExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    [[nodiscard]] Shape outputShape() const { return a.shape(); }

    void evalInto(TensorView<T> &out) const { COMPUTE<T>::mul(out.data(), a.data(), b.data(), a.size()); }
};

template <template <class> class COMPUTE, typename T> struct ElemwiseDivExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    [[nodiscard]] Shape outputShape() const { return a.shape(); }

    void evalInto(TensorView<T> &out) const { COMPUTE<T>::div(out.data(), a.data(), b.data(), a.size()); }
};

// ── ScaledDotProductAttentionExpr ─────────────────────────────────────────────

template <template <class> class COMPUTE, typename T> struct ScaledDotProductAttentionExpr {
    using value_type = T;
    TensorView<T> q;               // (n_heads * head_size,)
    TensorView<T> key_cache;       // (seq_len * kv_dim,) flat
    TensorView<T> val_cache;       // (seq_len * kv_dim,) flat
    mutable TensorView<T> att_buf; // (n_heads * seq_len,) scratch written during eval
    int pos;
    size_t n_heads;
    size_t kv_heads;
    size_t head_size;
    size_t kv_dim;
    size_t seq_len;

    [[nodiscard]] Shape outputShape() const { return Shape(n_heads * head_size); }

    void evalInto(TensorView<T> &out) const {
        COMPUTE<T>::scaledDotProductAttention(out.data(), q.data(), key_cache.data(), val_cache.data(), att_buf.data(), pos, n_heads, kv_heads, head_size,
                                              kv_dim, seq_len);
    }
};

// ── Factory functions ──────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto matmul(const Tensor<COMPUTE, T> &x, const Tensor<COMPUTE, T> &w) -> MatMulExpr<COMPUTE, T> {
    return {TensorView<T>{x}, TensorView<T>{w}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto rmsnorm(const Tensor<COMPUTE, T> &x, const Tensor<COMPUTE, T> &weight) -> RMSNormExpr<COMPUTE, T> {
    return {TensorView<T>{x}, TensorView<T>{weight}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto scaledDotProductAttention(const Tensor<COMPUTE, T> &q, const Tensor<COMPUTE, T> &key_cache, const Tensor<COMPUTE, T> &val_cache,
                                             Tensor<COMPUTE, T> &att_buf, int pos, size_t n_heads, size_t kv_heads, size_t head_size, size_t kv_dim,
                                             size_t seq_len) -> ScaledDotProductAttentionExpr<COMPUTE, T> {
    return {TensorView<T>{q}, TensorView<T>{key_cache}, TensorView<T>{val_cache}, TensorView<T>{att_buf}, pos, n_heads, kv_heads, head_size, kv_dim, seq_len};
}

// ── Operator overloads ────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto operator+(const Tensor<COMPUTE, T> &a, const Tensor<COMPUTE, T> &b) -> ElemwiseAddExpr<COMPUTE, T> {
    return {TensorView<T>{a}, TensorView<T>{b}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto operator-(const Tensor<COMPUTE, T> &a, const Tensor<COMPUTE, T> &b) -> ElemwiseSubExpr<COMPUTE, T> {
    return {TensorView<T>{a}, TensorView<T>{b}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto operator*(const Tensor<COMPUTE, T> &a, const Tensor<COMPUTE, T> &b) -> ElemwiseMulExpr<COMPUTE, T> {
    return {TensorView<T>{a}, TensorView<T>{b}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto operator/(const Tensor<COMPUTE, T> &a, const Tensor<COMPUTE, T> &b) -> ElemwiseDivExpr<COMPUTE, T> {
    return {TensorView<T>{a}, TensorView<T>{b}};
}

} // namespace transformers_lite
