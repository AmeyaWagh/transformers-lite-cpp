#pragma once

#include "../core/ops.hpp"
#include "../core/state_dict.hpp"
#include "../core/tensor.hpp"
#include "layer.hpp"

namespace transformers_lite {

/**
 * @brief Feed-forward network layer with SwiGLU activation.
 *
 * Implements: w2(silu(w1(x)) * w3(x))
 *
 * @tparam COMPUTE compute backend
 * @tparam T datatype
 */
template <template <class> class COMPUTE, class T> class FeedForward : public LayerBase<COMPUTE, T, FeedForward<COMPUTE, T>> {
    using Base = LayerBase<COMPUTE, T, FeedForward<COMPUTE, T>>;

 public:
    using typename Base::compute;
    using typename Base::ptr;
    using typename Base::value_type;

    /**
     * @brief Construct a FeedForward layer from dimensions only; call initializeLayer before forward.
     *
     * @param dim transformer model dimension
     * @param hidden_dim hidden layer dimension
     */
    FeedForward(size_t dim, size_t hiddenDim) : m_dim(dim), m_hidden_dim(hiddenDim), m_hb(Shape(hiddenDim)), m_hb2(Shape(hiddenDim)) {}

    /**
     * @brief Bind weight views from a state dict.
     *
     * Expected keys: "w1.weight", "w2.weight", "w3.weight".
     *
     * @param stateDict map of weight name to tensor view
     */
    void initializeLayer(const StateDict<value_type> &stateDict) {
        m_w1 = stateDict.at("w1.weight");
        m_w2 = stateDict.at("w2.weight");
        m_w3 = stateDict.at("w3.weight");
    }

    /**
     * @brief Forward pass: out = w2(silu(w1(x)) * w3(x))
     *
     * Allocates the output buffer on the first call; subsequent calls reuse it.
     *
     * @param input input tensor (dim,)
     * @return reference to the layer-owned output buffer (dim,)
     */
    Tensor<COMPUTE, value_type> &forward(const Tensor<COMPUTE, value_type> &input) {
        assert(input.shape() == Shape(m_dim));

        // Compute Linear projections: m_hb = w1 * input, m_hb2 = w3 * input
        m_hb = matmul(input, m_w1);
        m_hb2 = matmul(input, m_w3);

        // Apply activation and combine
        m_hb = silu(m_hb);

        // Element-wise Hadamard product: m_hb = m_hb * m_hb2
        m_hb = m_hb * m_hb2;
        m_out = matmul(m_hb, m_w2);
        return m_out;
    }

 private:
    size_t m_dim;
    size_t m_hidden_dim;
    Tensor<COMPUTE, value_type> m_w1;  // (hidden_dim, dim)
    Tensor<COMPUTE, value_type> m_w2;  // (dim, hidden_dim)
    Tensor<COMPUTE, value_type> m_w3;  // (hidden_dim, dim)
    Tensor<COMPUTE, value_type> m_hb;  // (hidden_dim,) — pre-allocated at construction
    Tensor<COMPUTE, value_type> m_hb2; // (hidden_dim,) — pre-allocated at construction
    Tensor<COMPUTE, value_type> m_out; // (dim,)        — allocated on first forward call
};

} // namespace transformers_lite
