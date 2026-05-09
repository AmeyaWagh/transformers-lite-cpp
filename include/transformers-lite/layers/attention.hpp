#pragma once
#include <cmath>

#include "../core/ops.hpp"
#include "../core/tensor.hpp"
#include "../core/types.hpp"
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
     * @brief Construct an Attention layer.
     *
     * @param wq query weight matrix view
     * @param wk key weight matrix view
     * @param wv value weight matrix view
     * @param kv_dim key/value cache dimension per position
     * @param dim transformer model dimension
     * @param n_heads number of query heads
     * @param kv_heads number of key/value heads
     * @param seq_len maximum sequence length
     */
    explicit Attention(TensorView<value_type> &wq, TensorView<value_type> &wk, TensorView<value_type> &wv, size_t kv_dim, size_t dim, size_t n_heads,
                       size_t kv_heads, size_t seq_len)
        : m_wq(wq), m_wk(wk), m_wv(wv), m_key_cache(Shape(seq_len * kv_dim)), m_value_cache(Shape(seq_len * kv_dim)), m_q(Shape(dim)),
          m_att(Shape(n_heads, seq_len)), m_kv_dim(kv_dim), m_dim(dim), m_n_heads(n_heads), m_kv_heads(kv_heads), m_head_size(dim / n_heads),
          m_seq_len(seq_len) {}

    /**
     * @brief Run the attention forward pass for a single position.
     *
     * Computes QKV projections, applies RoPE, performs multi-head attention
     * with cached keys/values, and writes the result to xb.
     *
     * @param in input tensor (dim)
     * @param xb output tensor (dim)
     * @param pos_ current sequence position
     */
    void forward(Tensor<COMPUTE, value_type> &in, Tensor<COMPUTE, value_type> &xb, int pos_) {
        // key and value point to the kv cache
        // Note kv_cache is (seq_len*kv_dim) i.e (seq_len*(dim*n_kv_heads/n_heads))
        TensorView<value_type> k = m_key_cache.view(Shape(m_seq_len, m_kv_dim)).slice(pos_);
        TensorView<value_type> v = m_value_cache.view(Shape(m_seq_len, m_kv_dim)).slice(pos_);

        // qkv matmuls for this position
        matmul(m_q, in, m_wq);
        matmul(k, in, m_wk);
        matmul(v, in, m_wv);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        // Currently CPU only
        for (size_t i = 0; i < m_dim; i += 2) {
            size_t head_dim = i % m_head_size;
            float32_t freq = 1.0f / powf(10000.0f, head_dim / static_cast<float32_t>(m_head_size));
            float32_t val = pos_ * freq;
            float32_t fcr = cosf(val);
            float32_t fci = sinf(val);
            size_t rotn = i < m_kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (size_t v = 0; v < rotn; v++) {
                TensorView<value_type> vec = v == 0 ? m_q : k; // the vector to rotate (query or key)
                value_type v0 = vec[i];
                value_type v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        size_t kv_mul = m_n_heads / m_kv_heads;
        size_t h;
#pragma omp parallel for private(h)
        for (h = 0; h < m_n_heads; h++) {
            // get the query vector for this head
            TensorView<value_type> q_ = m_q.view(Shape(m_n_heads, m_head_size)).slice(h);

            // attention scores for this head
            TensorView<value_type> att_ = m_att.slice(h);

            // iterate over all timesteps, including the current one
            for (size_t t = 0; t <= static_cast<size_t>(pos_); t++) {
                // get the key vector for this head and at this timestep
                // head_size = (dim / n_heads)
                // kv_cache is (seq_len * kv_dim)
                //          -> (seq_len * (dim * n_kv_heads / n_heads))
                //          -> (seq_len*n_kv_heads, dim/n_heads)
                //          -> (seq_len*n_kv_heads, head_size)
                // offset = t * m_kv_dim + (h / kv_mul)
                //          -> t * (dim * n_kv_heads / n_heads) + h * n_kv_heads / n_heads
                //          -> (t * dim + h)*(n_kv_heads / n_heads)
                TensorView<value_type> k_(m_key_cache.data() + t * m_kv_dim + (h / kv_mul) * m_head_size, Shape(m_head_size));
                // calculate the attention score as the dot product of q and k
                value_type score = dot_prod(q_, k_);
                score /= sqrtf(m_head_size);
                // save the score to the attention buffer
                att_(t) = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att_, pos_ + 1);

            // weighted sum of the values, store back into xb
            // view xb of shape (dim) as (n_heads * head_size)
            TensorView<value_type> xb_ = xb.view(Shape(m_n_heads, m_head_size)).slice(h);

            setZero(xb_);
            for (size_t t = 0; t <= static_cast<size_t>(pos_); t++) {
                // get the value vector for this head and at this timestep
                TensorView<value_type> v_(m_value_cache.data() + t * m_kv_dim + (h / kv_mul) * m_head_size, Shape(m_head_size));
                // get the attention weight for this timestep
                value_type a = att_[t];
                // accumulate the weighted value into xb
                for (size_t i = 0; i < m_head_size; i++) {
                    xb_(i) += a * v_(i);
                }
            }
        }
    }

 private:
    TensorView<value_type> m_wq;               // query (dim, n_heads * head_size)
    TensorView<value_type> m_wk;               // key (dim, kv_dim * head_size)
    TensorView<value_type> m_wv;               // value (dim, kv_dim * head_size)
    Tensor<COMPUTE, value_type> m_key_cache;   // key cache (seq_len * kv_dim)
    Tensor<COMPUTE, value_type> m_value_cache; // value cache (seq_len * kv_dim)
    Tensor<COMPUTE, value_type> m_q;           // query tensor (dim)
    Tensor<COMPUTE, value_type> m_att;         // attention tensor (n_heads * seq_len)
    size_t m_kv_dim;                           // key value cache dimension ((dim * n_kv_heads) / n_heads)
    size_t m_dim;                              // transformer dimension
    size_t m_n_heads;                          // number of heads
    size_t m_kv_heads;                         // number of key/value heads (can be < query heads because of multiquery)
    size_t m_head_size;                        // (dim / n_heads)
    size_t m_seq_len;                          // max sequence length
};

} // namespace transformers_lite
