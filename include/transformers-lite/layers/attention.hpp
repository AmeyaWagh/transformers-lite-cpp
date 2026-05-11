#pragma once

#include "../core/ops.hpp"
#include "../core/state_dict.hpp"
#include "../core/tensor.hpp"
#include "layer.hpp"

namespace transformers_lite {

/**
 * @brief Multi-head attention layer with KV cache and RoPE positional encoding.
 *
 * @tparam COMPUTE compute backend
 * @tparam T datatype
 */
template <template <class> class COMPUTE, class T> class Attention : public LayerBase<COMPUTE, T, Attention<COMPUTE, T>> {
    using Base = LayerBase<COMPUTE, T, Attention<COMPUTE, T>>;

 public:
    using typename Base::compute;
    using typename Base::ptr;
    using typename Base::value_type;

    /**
     * @brief Construct an Attention layer from dimensions only; call initializeLayer before forward.
     *
     * @param kvDim key/value cache dimension per position
     * @param dim transformer model dimension
     * @param nHeads number of query heads
     * @param kvHeads number of key/value heads
     * @param seqLen maximum sequence length
     */
    explicit Attention(size_t kvDim, size_t dim, size_t nHeads, size_t kvHeads, size_t seqLen)
        : m_key_cache(Shape(seqLen * kvDim)), m_value_cache(Shape(seqLen * kvDim)), m_q(Shape(dim)), m_att(Shape(nHeads, seqLen)), m_out(Shape(dim)),
          m_kv_dim(kvDim), m_dim(dim), m_n_heads(nHeads), m_kv_heads(kvHeads), m_head_size(dim / nHeads), m_seq_len(seqLen) {}

    /**
     * @brief Bind weight views from a state dict.
     *
     * Expected keys: "wq.weight", "wk.weight", "wv.weight".
     *
     * @param stateDict map of weight name to tensor view
     */
    void initializeLayer(const StateDict<value_type> &stateDict) {
        m_wq = stateDict.at("wq.weight");
        m_wk = stateDict.at("wk.weight");
        m_wv = stateDict.at("wv.weight");
    }

    /**
     * @brief Run the attention forward pass for a single position.
     *
     * Computes QKV projections, applies RoPE, performs multi-head attention
     * with cached keys/values.
     *
     * @param input input tensor (dim,)
     * @param pos current sequence position
     * @return reference to the layer-owned output buffer (dim,)
     */
    Tensor<COMPUTE, value_type> &forward(const Tensor<COMPUTE, value_type> &input, int pos) {
        Tensor<COMPUTE, value_type> keyCur(m_key_cache.view(Shape(m_seq_len, m_kv_dim)).slice(pos));
        Tensor<COMPUTE, value_type> valCur(m_value_cache.view(Shape(m_seq_len, m_kv_dim)).slice(pos));

        m_q = matmul(input, m_wq);
        keyCur = matmul(input, m_wk);
        valCur = matmul(input, m_wv);

        m_q = rope(m_q, keyCur, pos, m_head_size);

        m_out = scaledDotProductAttention(m_q, m_key_cache, m_value_cache, m_att, pos, m_n_heads, m_kv_heads, m_head_size, m_kv_dim, m_seq_len);
        return m_out;
    }

 private:
    Tensor<COMPUTE, value_type> m_wq;          // (dim, n_heads * head_size)
    Tensor<COMPUTE, value_type> m_wk;          // (dim, kv_dim)
    Tensor<COMPUTE, value_type> m_wv;          // (dim, kv_dim)
    Tensor<COMPUTE, value_type> m_key_cache;   // (seq_len * kv_dim)
    Tensor<COMPUTE, value_type> m_value_cache; // (seq_len * kv_dim)
    Tensor<COMPUTE, value_type> m_q;           // (dim,)
    Tensor<COMPUTE, value_type> m_att;         // (n_heads, seq_len)
    Tensor<COMPUTE, value_type> m_out;         // (dim,) — pre-allocated at construction
    size_t m_kv_dim;
    size_t m_dim;
    size_t m_n_heads;
    size_t m_kv_heads;
    size_t m_head_size;
    size_t m_seq_len;
};

} // namespace transformers_lite
