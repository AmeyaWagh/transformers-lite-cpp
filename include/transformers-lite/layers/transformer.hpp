#pragma once
#include <map>
#include <string>
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

    /**
     * @brief Return all weight tensors as a flat map of non-owning views.
     *
     * Layered tensors (e.g. wq, wk, ...) are returned unsliced; callers
     * should slice per-layer before passing to initializeLayer.
     */
    auto stateDict() const -> std::map<std::string, TensorView<T>> {
        return {
            {"token_embedding_table", token_embedding_table},
            {"rms_att_weight", rms_att_weight},
            {"rms_ffn_weight", rms_ffn_weight},
            {"wq", wq},
            {"wk", wk},
            {"wv", wv},
            {"wo", wo},
            {"w1", w1},
            {"w2", w2},
            {"w3", w3},
            {"rms_final_weight", rms_final_weight},
            {"wcls", wcls},
        };
    }
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
     * @brief Construct a TransformerBlock with pre-bound weight views.
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
     * @brief Construct a TransformerBlock from dimensions only; call initializeLayer before forward.
     *
     * @param attention attention layer (takes ownership)
     * @param feed_forward feed-forward layer (takes ownership)
     * @param dim transformer model dimension
     */
    explicit TransformerBlock(typename Attention<COMPUTE, value_type>::ptr attention, typename FeedForward<COMPUTE, value_type>::ptr feed_forward, size_t dim)
        : m_attention(std::move(attention)), m_feedforward(std::move(feed_forward)), m_xh(Shape(dim)), m_xh2(Shape(dim)) {}

    /**
     * @brief Bind weight views from a per-layer state dict and propagate to sub-layers.
     *
     * Expected keys: "wo", "rms_att", "rms_ffn", "wq", "wk", "wv", "w1", "w2", "w3".
     *
     * @param state_dict map of weight name to tensor view (already sliced for this layer)
     */
    void initializeLayer(const std::map<std::string, TensorView<value_type>> &state_dict) {
        m_wo = state_dict.at("wo");
        m_w_rms_att = state_dict.at("rms_att");
        m_w_rms_ffn = state_dict.at("rms_ffn");
        m_attention->initializeLayer({{"wq", state_dict.at("wq")}, {"wk", state_dict.at("wk")}, {"wv", state_dict.at("wv")}});
        m_feedforward->initializeLayer({{"w1", state_dict.at("w1")}, {"w2", state_dict.at("w2")}, {"w3", state_dict.at("w3")}});
    }

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
     * The weights object must outlive this Transformer — layers hold non-owning
     * views into its tensors.
     *
     * @param config transformer hyperparameters
     * @param weights pre-loaded model weights (not owned)
     */
    Transformer(TransformerConfig &config, TransformerWeights<COMPUTE, value_type> &weights) : m_config(config), m_weights(weights), m_linear(nullptr) {
        initializeLayers();
    }

    /**
     * @brief Build layers by slicing weights from the state dict and calling initializeLayer on each.
     */
    void initializeLayers() {
        const size_t kv_dim = static_cast<size_t>((m_config.dim * m_config.n_kv_heads) / m_config.n_heads);
        const size_t dim = static_cast<size_t>(m_config.dim);
        const size_t n_heads = static_cast<size_t>(m_config.n_heads);
        const size_t hidden_dim = static_cast<size_t>(m_config.hidden_dim);
        const size_t n_kv_heads = static_cast<size_t>(m_config.n_kv_heads);
        const size_t seq_len = static_cast<size_t>(m_config.seq_len);

        auto sd = m_weights.stateDict();

        for (size_t l = 0; l < static_cast<size_t>(m_config.n_layers); l++) {
            std::map<std::string, TensorView<value_type>> layer_sd = {
                {"wq", sd.at("wq").slice(l)},
                {"wk", sd.at("wk").slice(l)},
                {"wv", sd.at("wv").slice(l)},
                {"wo", sd.at("wo").slice(l)},
                {"w1", sd.at("w1").slice(l)},
                {"w2", sd.at("w2").slice(l)},
                {"w3", sd.at("w3").slice(l)},
                {"rms_att", sd.at("rms_att_weight").slice(l)},
                {"rms_ffn", sd.at("rms_ffn_weight").slice(l)},
            };

            auto attention = std::make_unique<Attention<COMPUTE, value_type>>(kv_dim, dim, n_heads, n_kv_heads, seq_len);
            auto feedforward = std::make_unique<FeedForward<COMPUTE, value_type>>(dim, hidden_dim);
            auto block = std::make_unique<TransformerBlock<COMPUTE, value_type>>(std::move(attention), std::move(feedforward), dim);
            block->initializeLayer(layer_sd);
            m_layers.push_back(std::move(block));
        }

        m_linear = std::make_unique<Linear<COMPUTE, value_type>>();
        m_linear->initializeLayer({{"wcls", sd.at("wcls")}});
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
    const TransformerWeights<COMPUTE, value_type> &m_weights;
    typename Linear<COMPUTE, value_type>::ptr m_linear;
    Tensor<COMPUTE, value_type> m_x_in;
    Tensor<COMPUTE, value_type> m_out_logits;
    std::vector<typename TransformerBlock<COMPUTE, value_type>::ptr> m_layers;
};

} // namespace transformers_lite
