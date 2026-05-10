#pragma once
#include <string>
#include <unordered_map>

#include "../core/ops.hpp"
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
    FeedForward(size_t dim, size_t hidden_dim) : m_dim(dim), m_hidden_dim(hidden_dim), m_hb(Shape(hidden_dim)), m_hb2(Shape(hidden_dim)) {}

    /**
     * @brief Bind weight views from a state dict.
     *
     * Expected keys: "w1", "w2", "w3".
     *
     * @param state_dict map of weight name to tensor view
     */
    void initializeLayer(const std::unordered_map<std::string, TensorView<value_type>> &state_dict) {
        m_w1 = state_dict.at("w1");
        m_w2 = state_dict.at("w2");
        m_w3 = state_dict.at("w3");
    }

    /**
     * @brief Forward pass: out = w2(silu(w1(x)) * w3(x))
     *
     * @param in input tensor (dim)
     * @param out output tensor (dim)
     */
    void forward(Tensor<COMPUTE, value_type> &in, Tensor<COMPUTE, value_type> &out) {
        matmul(m_hb, in, m_w1);
        matmul(m_hb2, in, m_w3);

        // SwiGLU non-linearity
        silu_inpl(m_hb);
        hadamard_prod(m_hb, m_hb, m_hb2);

        matmul(out, m_hb, m_w2);
    }

 private:
    size_t m_dim;                      // transformer dimension
    size_t m_hidden_dim;               // hidden layer dimension
    TensorView<value_type> m_w1;       // (hidden_dim, dim)
    TensorView<value_type> m_w2;       // (dim, hidden_dim)
    TensorView<value_type> m_w3;       // (hidden_dim, dim)
    Tensor<COMPUTE, value_type> m_hb;  // hidden buffer (hidden_dim,)
    Tensor<COMPUTE, value_type> m_hb2; // gate buffer (hidden_dim,)
};

} // namespace transformers_lite
