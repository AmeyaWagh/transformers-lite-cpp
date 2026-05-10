#pragma once
#include <string>
#include <unordered_map>

#include "../core/exprs.hpp"
#include "../core/state_dict.hpp"
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

    /** @brief Construct an unbound Linear layer; call initializeLayer before forward. */
    Linear() = default;

    /**
     * @brief Bind weight views from a state dict.
     *
     * Expected keys: "weight".
     *
     * @param sd map of weight name to tensor view
     */
    void initializeLayer(const StateDict<value_type> &sd) { m_wcls = sd.at("weight"); }

    /**
     * @brief Forward pass: out = wcls * x.
     *
     * Allocates the output buffer on the first call; subsequent calls reuse it.
     *
     * @param x input tensor (in_dim,)
     * @return reference to the layer-owned output buffer (out_dim,)
     */
    Tensor<COMPUTE, value_type> &forward(const Tensor<COMPUTE, value_type> &x) {
        m_out = matmul(x, m_wcls);
        return m_out;
    }

    /** @brief Output dimension (number of rows in the weight matrix). */
    auto outDim() const -> size_t { return m_wcls.shape().shapeVec()[0]; }

 private:
    Tensor<COMPUTE, value_type> m_wcls; // (out_dim, in_dim)
    Tensor<COMPUTE, value_type> m_out;  // (out_dim,) — allocated on first forward call
};

} // namespace transformers_lite
