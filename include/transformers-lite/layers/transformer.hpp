#pragma once
#include <string>
#include <unordered_map>
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
     * @brief Return all weight tensors as a flat state dict with PyTorch-style keys.
     *
     * Per-layer weights are pre-sliced: e.g. "layers.0.attention.wq.weight" is
     * already a view into layer 0's wq slice. All views point into this object,
     * which must outlive the returned map.
     *
     * @param n_layers number of transformer layers
     */
    auto stateDict(size_t n_layers) const -> std::unordered_map<std::string, TensorView<T>> {
        std::unordered_map<std::string, TensorView<T>> sd;
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
     * Call initializeLayer to bind weights before calling forward.
     *
     * @param kv_dim    key/value cache dimension per position
     * @param dim       transformer model dimension
     * @param n_heads   number of query heads
     * @param kv_heads  number of key/value heads
     * @param seq_len   maximum sequence length
     * @param hidden_dim FFN hidden dimension
     */
    explicit TransformerBlock(size_t kv_dim, size_t dim, size_t n_heads, size_t kv_heads, size_t seq_len, size_t hidden_dim)
        : m_attention(std::make_unique<Attention<COMPUTE, value_type>>(kv_dim, dim, n_heads, kv_heads, seq_len)),
          m_feedforward(std::make_unique<FeedForward<COMPUTE, value_type>>(dim, hidden_dim)), m_xh(Shape(dim)), m_xh2(Shape(dim)) {}

    /**
     * @brief Bind all weights from the per-layer state dict.
     *
     * Expected keys: "wq", "wk", "wv", "wo", "w1", "w2", "w3", "rms_att", "rms_ffn".
     *
     * @param state_dict per-layer weight map (pre-sliced for this layer)
     */
    void initializeLayer(const std::unordered_map<std::string, TensorView<value_type>> &state_dict) {
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
     * @brief Construct from a flat PyTorch-style state dict.
     *
     * All TensorViews in the dict must outlive this Transformer.
     *
     * @param config transformer hyperparameters
     * @param state_dict flat map of weight name → tensor view
     */
    Transformer(TransformerConfig &config, const std::unordered_map<std::string, TensorView<value_type>> &state_dict) : m_config(config), m_linear(nullptr) {
        initializeLayers(state_dict);
    }

    /**
     * @brief Convenience constructor: builds the state dict from a TransformerWeights object.
     *
     * @param config transformer hyperparameters
     * @param weights pre-loaded model weights (not owned; must outlive this Transformer)
     */
    Transformer(TransformerConfig &config, TransformerWeights<COMPUTE, value_type> &weights) : m_config(config), m_linear(nullptr) {
        initializeLayers(weights.stateDict(static_cast<size_t>(config.n_layers)));
    }

    /**
     * @brief Build layers from a flat state dict.
     *
     * Expected keys follow the PyTorch naming convention:
     *   "token_embedding_table.weight", "rms_final.weight", "output.weight",
     *   "layers.{i}.attention.{wq,wk,wv,wo}.weight",
     *   "layers.{i}.{attention,ffn}_norm.weight",
     *   "layers.{i}.feedforward.{w1,w2,w3}.weight"
     *
     * @param state_dict flat map of weight name → tensor view
     */
    void initializeLayers(const std::unordered_map<std::string, TensorView<value_type>> &state_dict) {
        const size_t kv_dim = static_cast<size_t>((m_config.dim * m_config.n_kv_heads) / m_config.n_heads);
        const size_t dim = static_cast<size_t>(m_config.dim);
        const size_t n_heads = static_cast<size_t>(m_config.n_heads);
        const size_t hidden_dim = static_cast<size_t>(m_config.hidden_dim);
        const size_t n_kv_heads = static_cast<size_t>(m_config.n_kv_heads);
        const size_t seq_len = static_cast<size_t>(m_config.seq_len);

        m_token_embedding = state_dict.at("token_embedding_table.weight");
        m_rms_final = state_dict.at("rms_final.weight");

        for (size_t l = 0; l < static_cast<size_t>(m_config.n_layers); ++l) {
            const std::string pfx = "layers." + std::to_string(l) + ".";
            const std::unordered_map<std::string, TensorView<value_type>> layer_sd = {
                {"wq", state_dict.at(pfx + "attention.wq.weight")},   {"wk", state_dict.at(pfx + "attention.wk.weight")},
                {"wv", state_dict.at(pfx + "attention.wv.weight")},   {"wo", state_dict.at(pfx + "attention.wo.weight")},
                {"w1", state_dict.at(pfx + "feedforward.w1.weight")}, {"w2", state_dict.at(pfx + "feedforward.w2.weight")},
                {"w3", state_dict.at(pfx + "feedforward.w3.weight")}, {"rms_att", state_dict.at(pfx + "attention_norm.weight")},
                {"rms_ffn", state_dict.at(pfx + "ffn_norm.weight")},
            };

            auto block = std::make_unique<TransformerBlock<COMPUTE, value_type>>(kv_dim, dim, n_heads, n_kv_heads, seq_len, hidden_dim);
            block->initializeLayer(layer_sd);
            m_layers.push_back(std::move(block));
        }

        m_linear = std::make_unique<Linear<COMPUTE, value_type>>();
        m_linear->initializeLayer({{"wcls", state_dict.at("output.weight")}});
        m_out_logits.reShape(Shape(m_linear->outDim()));
        m_x_in.reShape(Shape(dim));
    }

    ~Transformer() = default;

    /**
     * @brief Run the full transformer forward pass for one token.
     *
     * @param token input token index (used to look up embedding)
     * @param pos current sequence position
     * @param logits output CPU tensor filled with logits over the vocabulary
     */
    void forward(int token, int pos, Tensor<CPU, value_type> &logits) {
        TensorView<value_type> content_row = m_token_embedding.slice(token);
        m_x_in.copyFrom(content_row);

        for (auto &layer : m_layers) {
            layer->forward(m_x_in, pos);
        }

        rmsnorm(m_x_in, m_x_in, m_rms_final);
        m_linear->forward(m_x_in, m_out_logits);
        logits.copyFrom(m_out_logits);
    }

    /** @brief Get the transformer configuration. */
    auto getConfig() const -> const TransformerConfig & { return m_config; }

 private:
    TransformerConfig m_config;
    TensorView<value_type> m_token_embedding; // view into the embedding table
    TensorView<value_type> m_rms_final;       // view into the final RMSNorm weights
    typename Linear<COMPUTE, value_type>::ptr m_linear;
    Tensor<COMPUTE, value_type> m_x_in;
    Tensor<COMPUTE, value_type> m_out_logits;
    std::vector<typename TransformerBlock<COMPUTE, value_type>::ptr> m_layers;
};

} // namespace transformers_lite
