#pragma once
#include <cmath>
#include <type_traits>

#include "ops.hpp"

namespace transformers_lite {

// ── Expression types ──────────────────────────────────────────────────────────
// All operands are captured as TensorView<T> (non-owning, cheap copy).
// COMPUTE is a template parameter so evalInto can dispatch to the right kernel.

template <template <class> class COMPUTE, typename T> struct MatMulExpr {
    using value_type = T;
    TensorView<T> x; // input  (n,)
    TensorView<T> w; // weight (m, n)

    [[nodiscard]] Shape outputShape() const { return Shape{w.shape()[0]}; }

    void evalInto(TensorView<T> &out) const {
        if constexpr (kCPUAccelerator == CPUAccelerator::AVX512 && std::is_same_v<T, float>) {
            matmulAVX512(out.data(), x.data(), w.data(), static_cast<int>(x.size()), static_cast<int>(out.size()));
        } else {
            const int n = static_cast<int>(x.size());
            const int d = static_cast<int>(out.size());
            int i;
#pragma omp parallel for private(i)
            for (i = 0; i < d; i++) {
                T val = static_cast<T>(0);
                for (int j = 0; j < n; j++)
                    val += w[i * n + j] * x[j];
                out[i] = val;
            }
        }
    }
};

template <template <class> class COMPUTE, typename T> struct RMSNormExpr {
    using value_type = T;
    TensorView<T> x;
    TensorView<T> weight;

    [[nodiscard]] Shape outputShape() const { return x.shape(); }

    void evalInto(Tensor<COMPUTE, T> &out) const {
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

    [[nodiscard]] Shape outputShape() const { return a.shape(); }

    void evalInto(Tensor<COMPUTE, T> &out) const {
        const size_t n = a.size();
        for (size_t i = 0; i < n; ++i)
            out[i] = a[i] + b[i];
    }
};

template <template <class> class COMPUTE, typename T> struct HadamardExpr {
    using value_type = T;
    TensorView<T> a;
    TensorView<T> b;

    [[nodiscard]] Shape outputShape() const { return a.shape(); }

    void evalInto(Tensor<COMPUTE, T> &out) const {
        const size_t n = a.size();
        for (size_t i = 0; i < n; ++i)
            out[i] = a[i] * b[i];
    }
};

// ── Factory functions ──────────────────────────────────────────────────────────
// All operands are Tensor<COMPUTE,T> so COMPUTE is always deduced unambiguously.
// TensorView captures are made inside the factory (non-owning pointer copy).
// These 2-arg overloads coexist with the 3-arg void versions in ops.hpp.

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto matmul(const Tensor<COMPUTE, T> &x, const Tensor<COMPUTE, T> &w) -> MatMulExpr<COMPUTE, T> {
    return {TensorView<T>{x}, TensorView<T>{w}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto rmsnorm(const Tensor<COMPUTE, T> &x, const Tensor<COMPUTE, T> &weight) -> RMSNormExpr<COMPUTE, T> {
    return {TensorView<T>{x}, TensorView<T>{weight}};
}

template <template <class> class COMPUTE, typename T> [[nodiscard]] auto add(const Tensor<COMPUTE, T> &a, const Tensor<COMPUTE, T> &b) -> AddExpr<COMPUTE, T> {
    return {TensorView<T>{a}, TensorView<T>{b}};
}

template <template <class> class COMPUTE, typename T>
[[nodiscard]] auto hadamard(const Tensor<COMPUTE, T> &a, const Tensor<COMPUTE, T> &b) -> HadamardExpr<COMPUTE, T> {
    return {TensorView<T>{a}, TensorView<T>{b}};
}

} // namespace transformers_lite
