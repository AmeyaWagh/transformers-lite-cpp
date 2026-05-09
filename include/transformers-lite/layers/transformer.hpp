#pragma once
#include <vector>

#include "../core/ops.hpp"
#include "../core/tensor.hpp"
#include "attention.hpp"
#include "feedforward.hpp"
#include "linear.hpp"

namespace transformers_lite {

/**
 * @brief Transformer configuration
 *
 */
struct TransformerConfig {
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because
                    // of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len;    // max sequence length
};

/**
 * @brief Holds all weight tensors for a Transformer model.
 *
 * @tparam COMPUTE compute backend
 * @tparam T datatype
 */
template <template <class> class COMPUTE, class T> struct TransformerWeights {
    // token embedding table
    Tensor<COMPUTE, T> token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    Tensor<COMPUTE, T> rms_att_weight; // (layer, dim) rmsnorm weights
    Tensor<COMPUTE, T> rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    Tensor<COMPUTE, T> wq; // (layer, dim, n_heads * head_size)
    Tensor<COMPUTE, T> wk; // (layer, dim, n_kv_heads * head_size)
    Tensor<COMPUTE, T> wv; // (layer, dim, n_kv_heads * head_size)
    Tensor<COMPUTE, T> wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    Tensor<COMPUTE, T> w1; // (layer, hidden_dim, dim)
    Tensor<COMPUTE, T> w2; // (layer, dim, hidden_dim)
    Tensor<COMPUTE, T> w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    Tensor<COMPUTE, T> rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    Tensor<COMPUTE, T> wcls; // (vocab_size, dim)
};

/**
 * @brief A single transformer block consisting of attention, feed-forward, and residual connections.
 *
 * @tparam COMPUTE compute backend
 * @tparam T datatype
 */
template <template <class> class COMPUTE, class T> class TransformerBlock : public LayerBase<COMPUTE, T, TransformerBlock<COMPUTE, T>> {
    using Base = LayerBase<COMPUTE, T, TransformerBlock<COMPUTE, T>>;

 public:
    using typename Base::compute;
    using typename Base::ptr;
    using typename Base::value_type;

    /**
     * @brief Construct a TransformerBlock.
     *
     * @param attention attention layer (takes ownership)
     * @param feed_forward feed-forward layer (takes ownership)
     * @param rms_ffn_weight RMS normalization weights for the FFN sub-layer
     * @param wo output projection weight matrix view
     * @param w_rms_att RMS normalization weights for the attention sub-layer
     * @param dim transformer model dimension
     */
    explicit TransformerBlock(typename Attention<COMPUTE, value_type>::ptr attention, typename FeedForward<COMPUTE, value_type>::ptr feed_forward,
                              TensorView<value_type> &rms_ffn_weight, TensorView<value_type> &wo, TensorView<value_type> &w_rms_att, size_t dim)
        : m_attention(std::move(attention)), m_feedforward(std::move(feed_forward)), m_w_rms_ffn(rms_ffn_weight), m_xh(Shape(dim)), m_xh2(Shape(dim)), m_wo(wo),
          m_w_rms_att(w_rms_att) {}

    /**
     * @brief Forward pass through the transformer block.
     *
     * Applies attention with RMS normalization, residual connection, then FFN
     * with RMS normalization and a second residual connection.
     *
     * @param x input/output tensor (dim), modified in-place
     * @param pos_ current sequence position
     */
    void forward(Tensor<COMPUTE, value_type> &x, int pos_) {
        // attention rmsnorm
        rmsnorm(m_xh, x, m_w_rms_att);

        // forward attention
        m_attention->forward(m_xh, m_xh, pos_);

        // final matmul to get the output of the attention
        matmul(m_xh2, m_xh, m_wo);

        // residual connection back into x
        add(x, x, m_xh2);

        // ffn rmsnorm
        rmsnorm(m_xh, x, m_w_rms_ffn);

        // forward FFN
        m_feedforward->forward(m_xh, m_xh);

        // residual connection
        add(x, x, m_xh);
    }

 private:
    typename Attention<COMPUTE, value_type>::ptr m_attention;
    typename FeedForward<COMPUTE, value_type>::ptr m_feedforward;
    Tensor<COMPUTE, value_type> m_xh;   // pre-branch buffer (dim)
    Tensor<COMPUTE, value_type> m_xh2;  // post-attention buffer (dim)
    TensorView<value_type> m_wo;        // output projection (n_heads * head_size, dim)
    TensorView<value_type> m_w_rms_att; // attention RMSNorm weights (dim)
    TensorView<value_type> m_w_rms_ffn; // FFN RMSNorm weights (dim)
};

/**
 * @brief Complete Transformer model composed of stacked TransformerBlocks and a final linear layer.
 *
 * @tparam COMPUTE compute backend
 * @tparam T datatype
 */
template <template <class> class COMPUTE, class T> class Transformer {
 public:
    using ptr = std::unique_ptr<Transformer>;
    using value_type = T;
    using compute = COMPUTE<T>;

    /**
     * @brief Construct a Transformer from a config and pre-loaded weights.
     *
     * @param config transformer hyperparameters
     * @param weights pre-loaded model weights
     */
    Transformer(TransformerConfig &config, TransformerWeights<COMPUTE, value_type> &weights) : m_config(config), m_weights(weights), m_linear(nullptr) {
        initializeLayers();
    }

    /**
     * @brief Build attention, feed-forward, and linear layers from the loaded weights.
     */
    void initializeLayers() {
        size_t kv_dim = static_cast<size_t>((m_config.dim * m_config.n_kv_heads) / m_config.n_heads);
        size_t dim = static_cast<size_t>(m_config.dim);
        size_t n_heads = static_cast<size_t>(m_config.n_heads);
        size_t head_size = static_cast<size_t>(m_config.dim / m_config.n_heads);
        size_t hidden_dim = static_cast<size_t>(m_config.hidden_dim);
        size_t n_kv_heads = static_cast<size_t>(m_config.n_kv_heads);
        size_t seq_len = static_cast<size_t>(m_config.seq_len);

        // NOTE dim == n_heads * head_size
        (void)head_size;

        for (size_t layer_idx = 0; layer_idx < static_cast<size_t>(m_config.n_layers); layer_idx++) {
            TensorView<value_type> wq = m_weights.wq.slice(layer_idx); // (dim, n_heads * head_size)
            TensorView<value_type> wk = m_weights.wk.slice(layer_idx); // (dim, n_kv_heads * head_size)
            TensorView<value_type> wv = m_weights.wv.slice(layer_idx); // (dim, n_kv_heads * head_size)
            auto attention = std::make_unique<Attention<COMPUTE, value_type>>(wq, wk, wv, kv_dim, dim, n_heads, n_kv_heads, seq_len);

            TensorView<value_type> w1 = m_weights.w1.slice(layer_idx); // (hidden_dim, dim)
            TensorView<value_type> w2 = m_weights.w2.slice(layer_idx); // (dim, hidden_dim)
            TensorView<value_type> w3 = m_weights.w3.slice(layer_idx); // (hidden_dim, dim)
            auto feedforward = std::make_unique<FeedForward<COMPUTE, value_type>>(w1, w2, w3, dim, hidden_dim);

            TensorView<value_type> wo = m_weights.wo.slice(layer_idx);                    // (dim, dim)
            TensorView<value_type> w_rms_att = m_weights.rms_att_weight.slice(layer_idx); // (dim)
            TensorView<value_type> w_rms_ffn = m_weights.rms_ffn_weight.slice(layer_idx); // (dim)
            auto block = std::make_unique<TransformerBlock<COMPUTE, value_type>>(std::move(attention), std::move(feedforward), w_rms_ffn, wo, w_rms_att, dim);

            m_layers.push_back(std::move(block));
        }

        m_linear = std::make_unique<Linear<COMPUTE, value_type>>(m_weights.wcls);
        m_out_logits.reShape(Shape(m_linear->outDim()));
        m_x_in.reShape(Shape(dim));
    }

    ~Transformer() {}

    /**
     * @brief Run the full transformer forward pass for one token.
     *
     * @param token input token index (used to look up embedding)
     * @param pos current sequence position
     * @param logits output CPU tensor filled with logits over the vocabulary
     */
    void forward(int token, int pos, Tensor<CPU, value_type> &logits) {
        TensorView<value_type> content_row = m_weights.token_embedding_table.slice(token); // (dim)

        m_x_in.copyFrom(content_row);

        for (auto &layer : m_layers) {
            layer->forward(m_x_in, pos);
        }

        rmsnorm(m_x_in, m_x_in, m_weights.rms_final_weight);

        m_linear->forward(m_x_in, m_out_logits);

        logits.copyFrom(m_out_logits);
    }

    /** @brief Get the transformer configuration. */
    auto getConfig() const -> const TransformerConfig & { return m_config; }

 private:
    TransformerConfig m_config;
    TransformerWeights<COMPUTE, value_type> m_weights;
    typename Linear<COMPUTE, value_type>::ptr m_linear;
    Tensor<COMPUTE, value_type> m_x_in;
    Tensor<COMPUTE, value_type> m_out_logits;
    std::vector<typename TransformerBlock<COMPUTE, value_type>::ptr> m_layers;
};

} // namespace transformers_lite
