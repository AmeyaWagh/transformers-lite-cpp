#pragma once

#include "../core/ops.hpp"
#include "../core/tensor.hpp"
#include "layer.hpp"

namespace transformers_lite {

/**
 * @brief Linear (fully-connected) layer: out = W * x.
 *
 * @tparam COMPUTE compute backend
 * @tparam T datatype
 */
template <template <class> class COMPUTE, class T> class Linear : public LayerBase<COMPUTE, T, Linear<COMPUTE, T>> {
    using Base = LayerBase<COMPUTE, T, Linear<COMPUTE, T>>;

 public:
    using typename Base::compute;
    using typename Base::ptr;
    using typename Base::value_type;

    /**
     * @brief Construct a Linear layer.
     *
     * @param wcls weight matrix view (out_dim, in_dim)
     */
    explicit Linear(TensorView<value_type> &wcls) : m_wcls(wcls) {}

    /**
     * @brief Forward pass: out = wcls * x.
     *
     * @param x   input tensor (in_dim)
     * @param out output tensor (out_dim)
     */
    void forward(const Tensor<COMPUTE, value_type> &x, Tensor<COMPUTE, value_type> &out) { matmul(out, x, m_wcls); }

    /** @brief Output dimension (number of rows in the weight matrix). */
    auto outDim() const -> size_t { return m_wcls.shape().shapeVec()[0]; }

 private:
    TensorView<value_type> m_wcls; // (out_dim, in_dim)
};

} // namespace transformers_lite
