#pragma once
#include <cmath>

#include "ops.hpp"

namespace transformers_lite {

// ── Expression types ──────────────────────────────────────────────────────────
// All operands are captured as TensorView<T> (non-owning, cheap copy).
// COMPUTE is a template parameter so eval_into can dispatch to the right kernel.

template <template <class> class COMPUTE, typename T> struct MatMulExpr {
    using value_type = T;
    TensorView<T> x; // input  (n,)
    TensorView<T> w; // weight (m, n)

    Shape output_shape() const { return Shape{w.shape()[0]}; }

    void eval_into(Tensor<COMPUTE, T> &out) const { matmulCPU(out.data(), x.data(), w.data(), static_cast<int>(x.size()), static_cast<int>(out.size())); }
};

template <template <class> class COMPUTE, typename T> struct RMSNormExpr {
    using value_type = T;
    TensorView<T> x;
    TensorView<T> weight;

    Shape output_shape() const { return x.shape(); }

    void eval_into(Tensor<COMPUTE, T> &out) const {
        T ss = 0;
        const size_t n = x.size();
        for (size_t j = 0; j < n; ++j)
            ss += x[j] * x[j];
        ss /= static_cast<T>(n);
        ss += static_cast<T>(1e-5);
        ss = static_cast<T>(1) / std::sqrt(ss);
        for (size_t j = 0; j < n; ++j)
            out[j] = weight[j] * (ss * x[j]);
    }
};

template <template <class> class COMPUTE, typename T> struct AddExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    Shape output_shape() const { return a.shape(); }

    void eval_into(Tensor<COMPUTE, T> &out) const {
        const size_t n = a.size();
        for (size_t i = 0; i < n; ++i)
            out[i] = a[i] + b[i];
    }
};

template <template <class> class COMPUTE, typename T> struct HadamardExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    Shape output_shape() const { return a.shape(); }

    void eval_into(Tensor<COMPUTE, T> &out) const {
        const size_t n = a.size();
        for (size_t i = 0; i < n; ++i)
            out[i] = a[i] * b[i];
    }
};

// ── Factory functions ──────────────────────────────────────────────────────────
// All operands are Tensor<COMPUTE,T> so COMPUTE is always deduced unambiguously.
// TensorView captures are made inside the factory (non-owning pointer copy).
// These 2-arg overloads coexist with the 3-arg void versions in ops.hpp.

template <template <class> class COMPUTE, typename T> auto matmul(const Tensor<COMPUTE, T> &x, const Tensor<COMPUTE, T> &w) -> MatMulExpr<COMPUTE, T> {
    return {TensorView<T>{x}, TensorView<T>{w}};
}

template <template <class> class COMPUTE, typename T> auto rmsnorm(const Tensor<COMPUTE, T> &x, const Tensor<COMPUTE, T> &weight) -> RMSNormExpr<COMPUTE, T> {
    return {TensorView<T>{x}, TensorView<T>{weight}};
}

template <template <class> class COMPUTE, typename T> auto add(const Tensor<COMPUTE, T> &a, const Tensor<COMPUTE, T> &b) -> AddExpr<COMPUTE, T> {
    return {TensorView<T>{a}, TensorView<T>{b}};
}

template <template <class> class COMPUTE, typename T> auto hadamard(const Tensor<COMPUTE, T> &a, const Tensor<COMPUTE, T> &b) -> HadamardExpr<COMPUTE, T> {
    return {TensorView<T>{a}, TensorView<T>{b}};
}

} // namespace transformers_lite
