#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

#include <transformers-lite/core/tensor.hpp>
#include <transformers-lite/layers/transformer.hpp>

using namespace transformers_lite;

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

static std::string data_path(const char *name) {
    return std::string(TEST_DATA_DIR) + "/" + name;
}

// ── Fixture ───────────────────────────────────────────────────────────────────

class TransformerFixture : public ::testing::Test {
 protected:
    TransformerConfig config{};
    TransformerWeights<CPU, float> weights;

    // Load shape floats directly into an existing tensor member.
    // Uses reShape (which reallocates) then fread into the raw buffer.
    static void load(std::FILE *f, Tensor<CPU, float> &t, Shape shape) {
        t.reShape(shape);
        std::fread(t.data(), sizeof(float), t.size(), f);
    }

    void SetUp() override {
        std::FILE *f = std::fopen(data_path("weights.bin").c_str(), "rb");
        ASSERT_NE(f, nullptr) << "Cannot open weights.bin — build target 'transformer_test_data' first "
                                 "(cmake --build . --target transformer_test_data)";

        int cfg[7];
        std::fread(cfg, sizeof(int), 7, f);
        config.dim = cfg[0];
        config.hidden_dim = cfg[1];
        config.n_layers = cfg[2];
        config.n_heads = cfg[3];
        config.n_kv_heads = cfg[4];
        config.vocab_size = cfg[5];
        config.seq_len = cfg[6];

        const size_t L = static_cast<size_t>(config.n_layers);
        const size_t D = static_cast<size_t>(config.dim);
        const size_t H = static_cast<size_t>(config.hidden_dim);
        const size_t Nh = static_cast<size_t>(config.n_heads);
        const size_t Nk = static_cast<size_t>(config.n_kv_heads);
        const size_t V = static_cast<size_t>(config.vocab_size);
        const size_t hs = D / Nh;      // head_size
        const size_t kv = D * Nk / Nh; // kv_dim
        (void)kv;

        // Must match TransformerWeights declaration order and gen_weights.py
        load(f, weights.token_embedding_table, Shape(V, D));
        load(f, weights.rms_att_weight, Shape(L, D));
        load(f, weights.rms_ffn_weight, Shape(L, D));
        load(f, weights.wq, Shape(L, Nh * hs, D));
        load(f, weights.wk, Shape(L, Nk * hs, D));
        load(f, weights.wv, Shape(L, Nk * hs, D));
        load(f, weights.wo, Shape(L, D, D));
        load(f, weights.w1, Shape(L, H, D));
        load(f, weights.w2, Shape(L, D, H));
        load(f, weights.w3, Shape(L, H, D));
        load(f, weights.rms_final_weight, Shape(D));
        load(f, weights.wcls, Shape(V, D));
        std::fclose(f);
    }
};

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_F(TransformerFixture, MatchesPythonReference) {
    std::FILE *f = std::fopen(data_path("expected.bin").c_str(), "rb");
    ASSERT_NE(f, nullptr) << "Cannot open expected.bin";

    int n_cases = 0;
    std::fread(&n_cases, sizeof(int), 1, f);
    ASSERT_GT(n_cases, 0);

    // Single model instance: KV caches accumulate across sequential calls,
    // mirroring the Python script which shares cache arrays between test cases.
    Transformer<CPU, float> model(config, weights);
    Tensor<CPU, float> logits(Shape(static_cast<size_t>(config.vocab_size)));

    for (int c = 0; c < n_cases; ++c) {
        int token = 0, pos = 0;
        std::fread(&token, sizeof(int), 1, f);
        std::fread(&pos, sizeof(int), 1, f);

        std::vector<float> expected(static_cast<size_t>(config.vocab_size));
        std::fread(expected.data(), sizeof(float), expected.size(), f);

        model.forward(token, pos, logits);

        for (int i = 0; i < config.vocab_size; ++i) {
            EXPECT_NEAR(logits.data()[i], expected[i], 1e-4f) << "case=" << c << " token=" << token << " pos=" << pos << " vocab_idx=" << i;
        }
    }

    std::fclose(f);
}
