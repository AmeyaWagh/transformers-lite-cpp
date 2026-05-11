#pragma once
#include <cmath>

#include "../tensor.hpp"
#include "ops_dispatch.hpp"

namespace transformers_lite {

// ── MatMulExpr ────────────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T> struct MatMulExpr {
    using value_type = T;
    TensorView<T> x; // input  (n,)
    TensorView<T> w; // weight (m, n)

    [[nodiscard]] Shape outputShape() const { return Shape{w.shape()[0]}; }

    void evalInto(TensorView<T> &out) const {
        Ops<COMPUTE, T>::matmul(out.data(), x.data(), w.data(), static_cast<int>(x.size()), static_cast<int>(out.size()));
    }
};

// ── RMSNormExpr ───────────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T> struct RMSNormExpr {
    using value_type = T;
    TensorView<T> x;
    TensorView<T> weight;

    [[nodiscard]] Shape outputShape() const { return x.shape(); }

    void evalInto(TensorView<T> &out) const {
        const size_t size = x.size();
        T scale = Ops<COMPUTE, T>::dot(x.data(), x.data(), size);
        scale /= static_cast<T>(size);
        scale += static_cast<T>(1e-5);
        scale = static_cast<T>(1) / std::sqrt(scale);
        Ops<COMPUTE, T>::scale(out.data(), x.data(), scale, size);
        Ops<COMPUTE, T>::mul(out.data(), out.data(), weight.data(), size);
    }
};

// ── Primitive element-wise expression types ───────────────────────────────────

template <template <class> class COMPUTE, typename T> struct ElemwiseAddExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    [[nodiscard]] Shape outputShape() const { return a.shape(); }

    void evalInto(TensorView<T> &out) const { Ops<COMPUTE, T>::add(out.data(), a.data(), b.data(), a.size()); }
};

template <template <class> class COMPUTE, typename T> struct ElemwiseSubExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    [[nodiscard]] Shape outputShape() const { return a.shape(); }

    void evalInto(TensorView<T> &out) const { Ops<COMPUTE, T>::sub(out.data(), a.data(), b.data(), a.size()); }
};

template <template <class> class COMPUTE, typename T> struct ElemwiseMulExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    [[nodiscard]] Shape outputShape() const { return a.shape(); }

    void evalInto(TensorView<T> &out) const { Ops<COMPUTE, T>::mul(out.data(), a.data(), b.data(), a.size()); }
};

template <template <class> class COMPUTE, typename T> struct ElemwiseDivExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    [[nodiscard]] Shape outputShape() const { return a.shape(); }

    void evalInto(TensorView<T> &out) const { Ops<COMPUTE, T>::div(out.data(), a.data(), b.data(), a.size()); }
};

// ── SiluExpr ──────────────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T> struct SiluExpr {
    using value_type = T;
    TensorView<T> x;

    [[nodiscard]] Shape outputShape() const { return x.shape(); }

    void evalInto(TensorView<T> &out) const { Ops<COMPUTE, T>::silu(out.data(), x.data(), x.size()); }
};

// ── RopeExpr ──────────────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T> struct RopeExpr {
    using value_type = T;
    TensorView<T> q;
    mutable TensorView<T> k; // rotated in-place as a side effect
    int pos;
    size_t head_size;

    [[nodiscard]] Shape outputShape() const { return q.shape(); }

    void evalInto(TensorView<T> &out) const {
        if (out.data() != q.data()) {
            COMPUTE<T>::copy(q.data(), out.data(), q.size()); // memory op stays on COMPUTE
        }
        Ops<COMPUTE, T>::rope(out.data(), k.data(), pos, head_size, out.size(), k.size());
    }
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
        Ops<COMPUTE, T>::scaledDotProductAttention(out.data(), q.data(), key_cache.data(), val_cache.data(), att_buf.data(), pos, n_heads, kv_heads, head_size,
                                                   kv_dim, seq_len);
    }
};

// ── Factory functions ──────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto matmul(const Tensor<COMPUTE, T> &x, const Tensor<COMPUTE, T> &weights) -> MatMulExpr<COMPUTE, T> {
    return {TensorView<T>{x}, TensorView<T>{weights}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto rmsnorm(const Tensor<COMPUTE, T> &x, const Tensor<COMPUTE, T> &weight) -> RMSNormExpr<COMPUTE, T> {
    return {TensorView<T>{x}, TensorView<T>{weight}};
}

template <template <class> class COMPUTE, typename T> [[nodiscard]] auto silu(const Tensor<COMPUTE, T> &x) -> SiluExpr<COMPUTE, T> {
    return {TensorView<T>{x}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto rope(const Tensor<COMPUTE, T> &query, Tensor<COMPUTE, T> &keys, int pos, size_t headSize) -> RopeExpr<COMPUTE, T> {
    return {TensorView<T>{query}, TensorView<T>{keys}, pos, headSize};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto scaledDotProductAttention(const Tensor<COMPUTE, T> &query, const Tensor<COMPUTE, T> &keyCache, const Tensor<COMPUTE, T> &valCache,
                                             Tensor<COMPUTE, T> &attBuf, int pos, size_t nHeads, size_t kvHeads, size_t headSize, size_t kvDim, size_t seqLen)
    -> ScaledDotProductAttentionExpr<COMPUTE, T> {
    return {TensorView<T>{query}, TensorView<T>{keyCache}, TensorView<T>{valCache}, TensorView<T>{attBuf}, pos, nHeads, kvHeads, headSize, kvDim, seqLen};
}

// ── Operator overloads ────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto operator+(const Tensor<COMPUTE, T> &lhs, const Tensor<COMPUTE, T> &rhs) -> ElemwiseAddExpr<COMPUTE, T> {
    return {TensorView<T>{lhs}, TensorView<T>{rhs}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto operator-(const Tensor<COMPUTE, T> &lhs, const Tensor<COMPUTE, T> &rhs) -> ElemwiseSubExpr<COMPUTE, T> {
    return {TensorView<T>{lhs}, TensorView<T>{rhs}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto operator*(const Tensor<COMPUTE, T> &lhs, const Tensor<COMPUTE, T> &rhs) -> ElemwiseMulExpr<COMPUTE, T> {
    return {TensorView<T>{lhs}, TensorView<T>{rhs}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto operator/(const Tensor<COMPUTE, T> &lhs, const Tensor<COMPUTE, T> &rhs) -> ElemwiseDivExpr<COMPUTE, T> {
    return {TensorView<T>{lhs}, TensorView<T>{rhs}};
}

} // namespace transformers_lite
