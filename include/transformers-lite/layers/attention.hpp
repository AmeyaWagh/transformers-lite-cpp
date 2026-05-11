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
     * @param kv_dim key/value cache dimension per position
     * @param dim transformer model dimension
     * @param n_heads number of query heads
     * @param kv_heads number of key/value heads
     * @param seq_len maximum sequence length
     */
    explicit Attention(size_t kv_dim, size_t dim, size_t n_heads, size_t kv_heads, size_t seq_len)
        : m_key_cache(Shape(seq_len * kv_dim)), m_value_cache(Shape(seq_len * kv_dim)), m_q(Shape(dim)), m_att(Shape(n_heads, seq_len)), m_out(Shape(dim)),
          m_kv_dim(kv_dim), m_dim(dim), m_n_heads(n_heads), m_kv_heads(kv_heads), m_head_size(dim / n_heads), m_seq_len(seq_len) {}

    /**
     * @brief Bind weight views from a state dict.
     *
     * Expected keys: "wq.weight", "wk.weight", "wv.weight".
     *
     * @param sd map of weight name to tensor view
     */
    void initializeLayer(const StateDict<value_type> &sd) {
        m_wq = sd.at("wq.weight");
        m_wk = sd.at("wk.weight");
        m_wv = sd.at("wv.weight");
    }

    /**
     * @brief Run the attention forward pass for a single position.
     *
     * Computes QKV projections, applies RoPE, performs multi-head attention
     * with cached keys/values.
     *
     * @param in input tensor (dim,)
     * @param pos_ current sequence position
     * @return reference to the layer-owned output buffer (dim,)
     */
    Tensor<COMPUTE, value_type> &forward(const Tensor<COMPUTE, value_type> &in, int pos_) {
        Tensor<COMPUTE, value_type> k(m_key_cache.view(Shape(m_seq_len, m_kv_dim)).slice(pos_));
        Tensor<COMPUTE, value_type> v(m_value_cache.view(Shape(m_seq_len, m_kv_dim)).slice(pos_));

        m_q = matmul(in, m_wq);
        k = matmul(in, m_wk);
        v = matmul(in, m_wv);

        rope(m_q, k, pos_, m_head_size);

        m_out = scaledDotProductAttention(m_q, m_key_cache, m_value_cache, m_att, pos_, m_n_heads, m_kv_heads, m_head_size, m_kv_dim, m_seq_len);
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
