#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "../core/exprs.hpp"
#include "../core/ops.hpp"
#include "../core/state_dict.hpp"
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
    Tensor<COMPUTE, T> token_embedding_table; // (vocab_size, dim)
    Tensor<COMPUTE, T> rms_att_weight;        // (layer, dim)
    Tensor<COMPUTE, T> rms_ffn_weight;        // (layer, dim)
    Tensor<COMPUTE, T> wq;                    // (layer, dim, n_heads * head_size)
    Tensor<COMPUTE, T> wk;                    // (layer, dim, n_kv_heads * head_size)
    Tensor<COMPUTE, T> wv;                    // (layer, dim, n_kv_heads * head_size)
    Tensor<COMPUTE, T> wo;                    // (layer, n_heads * head_size, dim)
    Tensor<COMPUTE, T> w1;                    // (layer, hidden_dim, dim)
    Tensor<COMPUTE, T> w2;                    // (layer, dim, hidden_dim)
    Tensor<COMPUTE, T> w3;                    // (layer, hidden_dim, dim)
    Tensor<COMPUTE, T> rms_final_weight;      // (dim,)
    Tensor<COMPUTE, T> wcls;                  // (vocab_size, dim)

    /**
     * @brief Return all weight tensors as a flat state dict with PyTorch-style keys.
     *
     * Per-layer weights are pre-sliced. All views point into this object,
     * which must outlive the returned map.
     *
     * @param n_layers number of transformer layers
     */
    auto stateDict(size_t n_layers) const -> StateDict<T> {
        StateDict<T> sd;
        sd["token_embedding_table.weight"] = token_embedding_table;
        sd["rms_final.weight"] = rms_final_weight;
        sd["output.weight"] = wcls;
        for (size_t i = 0; i < n_layers; ++i) {
            const std::string pfx = "layers." + std::to_string(i) + ".";
            sd[pfx + "attention.wq.weight"] = wq.slice(i);
            sd[pfx + "attention.wk.weight"] = wk.slice(i);
            sd[pfx + "attention.wv.weight"] = wv.slice(i);
            sd[pfx + "attention.wo.weight"] = wo.slice(i);
            sd[pfx + "attention_norm.weight"] = rms_att_weight.slice(i);
            sd[pfx + "ffn_norm.weight"] = rms_ffn_weight.slice(i);
            sd[pfx + "feedforward.w1.weight"] = w1.slice(i);
            sd[pfx + "feedforward.w2.weight"] = w2.slice(i);
            sd[pfx + "feedforward.w3.weight"] = w3.slice(i);
        }
        return sd;
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
     * @brief Construct a TransformerBlock, creating Attention and FeedForward sub-layers.
     *
     * @param kv_dim    key/value cache dimension per position
     * @param dim       transformer model dimension
     * @param n_heads   number of query heads
     * @param kv_heads  number of key/value heads
     * @param seq_len   maximum sequence length
     * @param hidden_dim FFN hidden dimension
     */
    explicit TransformerBlock(size_t kv_dim, size_t dim, size_t n_heads, size_t kv_heads, size_t seq_len, size_t hidden_dim)
        : m_attention(kv_dim, dim, n_heads, kv_heads, seq_len), m_feedforward(dim, hidden_dim), m_xh(Shape(dim)), m_xh2(Shape(dim)) {}

    /**
     * @brief Bind all weights from the per-layer state dict.
     *
     * @param sd per-layer StateDict (pre-sliced for this layer)
     */
    void initializeLayer(const StateDict<value_type> &sd) {
        m_wo = sd.at("attention.wo.weight");
        m_w_rms_att = sd.at("attention_norm.weight");
        m_w_rms_ffn = sd.at("ffn_norm.weight");
        m_attention.initializeLayer(sd.getLayerWeights("attention"));
        m_feedforward.initializeLayer(sd.getLayerWeights("feedforward"));
    }

    /**
     * @brief Forward pass through the transformer block.
     *
     * x is never modified. m_x is allocated on the first call and reused after.
     *
     * @param x input tensor (dim,)
     * @param pos_ current sequence position
     * @return reference to the layer-owned residual output buffer (dim,)
     */
    Tensor<COMPUTE, value_type> &forward(const Tensor<COMPUTE, value_type> &x, int pos_) {
        // attention branch
        m_xh = rmsnorm(x, m_w_rms_att);
        auto &attn = m_attention.forward(m_xh, pos_);
        m_xh2 = matmul(attn, m_wo);
        m_x = add(x, m_xh2); // first residual; m_x allocated on first call

        // FFN branch
        m_xh = rmsnorm(m_x, m_w_rms_ffn);
        auto &ffn = m_feedforward.forward(m_xh);
        m_x = add(m_x, ffn); // second residual; safe: element-wise self-assign

        return m_x;
    }

 private:
    Attention<COMPUTE, value_type> m_attention;
    FeedForward<COMPUTE, value_type> m_feedforward;
    Tensor<COMPUTE, value_type> m_xh;        // (dim,) — pre-allocated
    Tensor<COMPUTE, value_type> m_xh2;       // (dim,) — pre-allocated
    Tensor<COMPUTE, value_type> m_x;         // (dim,) — lazy allocated on first forward call
    Tensor<COMPUTE, value_type> m_wo;        // (n_heads * head_size, dim)
    Tensor<COMPUTE, value_type> m_w_rms_att; // (dim,)
    Tensor<COMPUTE, value_type> m_w_rms_ffn; // (dim,)
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
     * @brief Construct from a flat PyTorch-style state dict.
     *
     * @param config transformer hyperparameters
     * @param state_dict flat map of weight name → tensor view
     */
    Transformer(TransformerConfig &config, const StateDict<value_type> &state_dict) : m_config(config) { initializeLayers(state_dict); }

    /**
     * @brief Convenience constructor: builds the state dict from a TransformerWeights object.
     *
     * @param config transformer hyperparameters
     * @param weights pre-loaded model weights (not owned; must outlive this Transformer)
     */
    Transformer(TransformerConfig &config, TransformerWeights<COMPUTE, value_type> &weights) : m_config(config) {
        initializeLayers(weights.stateDict(static_cast<size_t>(config.n_layers)));
    }

    /**
     * @brief Build layers from a flat state dict.
     *
     * @param state_dict flat map of weight name → tensor view
     */
    void initializeLayers(const StateDict<value_type> &state_dict) {
        auto kv_dim = static_cast<size_t>((m_config.dim * m_config.n_kv_heads) / m_config.n_heads);
        auto dim = static_cast<size_t>(m_config.dim);
        auto n_heads = static_cast<size_t>(m_config.n_heads);
        auto hidden_dim = static_cast<size_t>(m_config.hidden_dim);
        auto n_kv_heads = static_cast<size_t>(m_config.n_kv_heads);
        auto seq_len = static_cast<size_t>(m_config.seq_len);

        m_token_embedding = state_dict.at("token_embedding_table.weight");
        m_rms_final = state_dict.at("rms_final.weight");

        m_layers.reserve(static_cast<size_t>(m_config.n_layers));
        for (size_t l = 0; l < static_cast<size_t>(m_config.n_layers); ++l) {
            m_layers.emplace_back(kv_dim, dim, n_heads, n_kv_heads, seq_len, hidden_dim);
            m_layers.back().initializeLayer(state_dict.getLayerWeights("layers." + std::to_string(l)));
        }

        m_linear.initializeLayer(state_dict.getLayerWeights("output"));

        m_x_in.reShape(Shape(dim));   // working buffer for token embedding copy
        m_x_norm.reShape(Shape(dim)); // output buffer for final RMSNorm
    }

    ~Transformer() = default;

    /**
     * @brief Run the full transformer forward pass for one token.
     *
     * No heap allocation after the first call. Returns a reference to the linear
     * layer's output buffer; valid until the next forward() call.
     *
     * @param token input token index
     * @param pos   current sequence position
     * @return reference to the logit tensor (vocab_size,)
     */
    Tensor<COMPUTE, value_type> &forward(int token, int pos) {
        // Copy embedding row into a Tensor so downstream blocks have COMPUTE info.
        m_x_in.copyFrom(m_token_embedding.slice(token));

        // Each block reads its input and returns a ref to its own output buffer.
        Tensor<COMPUTE, value_type> *x = &m_x_in;
        for (auto &layer : m_layers)
            x = &layer.forward(*x, pos);

        m_x_norm = rmsnorm(*x, m_rms_final);
        return m_linear.forward(m_x_norm);
    }

    /** @brief Get the transformer configuration. */
    [[nodiscard]] auto getConfig() const -> const TransformerConfig & { return m_config; }

 private:
    TransformerConfig m_config;
    Tensor<COMPUTE, value_type> m_token_embedding; // borrows from embedding table
    Tensor<COMPUTE, value_type> m_rms_final;       // borrows from final RMSNorm weights
    Linear<COMPUTE, value_type> m_linear;
    Tensor<COMPUTE, value_type> m_x_in;   // (dim,) — token embedding working copy
    Tensor<COMPUTE, value_type> m_x_norm; // (dim,) — final RMSNorm output
    std::vector<TransformerBlock<COMPUTE, value_type>> m_layers;
};

} // namespace transformers_lite
