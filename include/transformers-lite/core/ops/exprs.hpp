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

// ── Factory functions ──────────────────────────────────────────────────────────

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto matmul(const Tensor<COMPUTE, T> &x, const Tensor<COMPUTE, T> &w) -> MatMulExpr<COMPUTE, T> {
    return {TensorView<T>{x}, TensorView<T>{w}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto rmsnorm(const Tensor<COMPUTE, T> &x, const Tensor<COMPUTE, T> &weight) -> RMSNormExpr<COMPUTE, T> {
    return {TensorView<T>{x}, TensorView<T>{weight}};
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
