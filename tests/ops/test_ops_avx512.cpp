#include <gtest/gtest.h>

#include <transformers-lite/core/ops.hpp>
#include <transformers-lite/core/tensor.hpp>

using namespace transformers_lite;

#ifdef __AVX512F__
#include <cmath>
#include <transformers-lite/core/compute/avx512.hpp>

// ── helpers ──────────────────────────────────────────────────────────────────

static void refScaledDotProductAttention(float *out, const float *q, const float *key_cache, const float *val_cache, float *att_buf, int pos, size_t n_heads,
                                         size_t kv_heads, size_t head_size, size_t kv_dim, size_t seq_len) {
    const size_t kv_mul = n_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    for (size_t h = 0; h < n_heads; h++) {
        const float *q_h = q + h * head_size;
        float *att_h = att_buf + h * seq_len;
        const size_t kv_off = (h / kv_mul) * head_size;

        for (size_t t = 0; t <= static_cast<size_t>(pos); t++) {
            float dot = 0.f;
            for (size_t i = 0; i < head_size; i++)
                dot += q_h[i] * key_cache[t * kv_dim + kv_off + i];
            att_h[t] = dot * scale;
        }

        const size_t n = static_cast<size_t>(pos) + 1;
        float max_val = att_h[0];
        for (size_t i = 1; i < n; i++)
            if (att_h[i] > max_val)
                max_val = att_h[i];
        float sum = 0.f;
        for (size_t i = 0; i < n; i++) {
            att_h[i] = std::exp(att_h[i] - max_val);
            sum += att_h[i];
        }
        for (size_t i = 0; i < n; i++)
            att_h[i] /= sum;

        float *out_h = out + h * head_size;
        for (size_t i = 0; i < head_size; i++)
            out_h[i] = 0.f;
        for (size_t t = 0; t <= static_cast<size_t>(pos); t++) {
            const float *v_t = val_cache + t * kv_dim + kv_off;
            for (size_t i = 0; i < head_size; i++)
                out_h[i] += att_h[t] * v_t[i];
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

// Configuration: head_size=32 (2 AVX512 registers), n_heads=2, kv_heads=2
static constexpr size_t HEAD_SIZE = 32;
static constexpr size_t N_HEADS = 2;
static constexpr size_t KV_HEADS = 2;
static constexpr size_t KV_DIM = N_HEADS * HEAD_SIZE; // MHA: kv_heads == n_heads
static constexpr size_t SEQ_LEN = 64;

TEST(ScaledDotProductAttentionAVX512, ZeroQKGivesUniformAttentionOverV) {
    // q=0, k=0 => all scores=0 => softmax=[1/(pos+1), ...] => output = mean of v rows
    const size_t pos = 3;
    std::vector<float> q(N_HEADS * HEAD_SIZE, 0.f);
    std::vector<float> key_cache(SEQ_LEN * KV_DIM, 0.f);
    std::vector<float> val_cache(SEQ_LEN * KV_DIM, 0.f);
    std::vector<float> att_buf(N_HEADS * SEQ_LEN, 0.f);
    std::vector<float> out(N_HEADS * HEAD_SIZE, 0.f);

    // Fill v rows 0..pos with constant values per row: row t = (t+1)
    for (size_t t = 0; t <= pos; t++)
        for (size_t i = 0; i < KV_DIM; i++)
            val_cache[t * KV_DIM + i] = static_cast<float>(t + 1);

    scaledDotProductAttentionAVX512(out.data(), q.data(), key_cache.data(), val_cache.data(), att_buf.data(), static_cast<int>(pos), N_HEADS, KV_HEADS,
                                    HEAD_SIZE, KV_DIM, SEQ_LEN);

    // Uniform attention: output = mean(1, 2, 3, 4) = 2.5
    const float expected = (1.f + 2.f + 3.f + 4.f) / 4.f;
    for (size_t i = 0; i < N_HEADS * HEAD_SIZE; i++)
        EXPECT_NEAR(out[i], expected, 1e-4f);
}

TEST(ScaledDotProductAttentionAVX512, MatchesScalarReference) {
    // Random-ish inputs; verify AVX512 output matches scalar to float tolerance
    const size_t pos = 7;
    std::vector<float> q(N_HEADS * HEAD_SIZE);
    std::vector<float> key_cache(SEQ_LEN * KV_DIM);
    std::vector<float> val_cache(SEQ_LEN * KV_DIM);
    std::vector<float> att_avx(N_HEADS * SEQ_LEN, 0.f);
    std::vector<float> att_ref(N_HEADS * SEQ_LEN, 0.f);
    std::vector<float> out_avx(N_HEADS * HEAD_SIZE, 0.f);
    std::vector<float> out_ref(N_HEADS * HEAD_SIZE, 0.f);

    for (size_t i = 0; i < q.size(); i++)
        q[i] = static_cast<float>(i % 7) - 3.f;
    for (size_t i = 0; i < key_cache.size(); i++)
        key_cache[i] = static_cast<float>(i % 5) - 2.f;
    for (size_t i = 0; i < val_cache.size(); i++)
        val_cache[i] = static_cast<float>(i % 3) - 1.f;

    scaledDotProductAttentionAVX512(out_avx.data(), q.data(), key_cache.data(), val_cache.data(), att_avx.data(), static_cast<int>(pos), N_HEADS, KV_HEADS,
                                    HEAD_SIZE, KV_DIM, SEQ_LEN);
    refScaledDotProductAttention(out_ref.data(), q.data(), key_cache.data(), val_cache.data(), att_ref.data(), static_cast<int>(pos), N_HEADS, KV_HEADS,
                                 HEAD_SIZE, KV_DIM, SEQ_LEN);

    for (size_t i = 0; i < out_avx.size(); i++)
        EXPECT_NEAR(out_avx[i], out_ref[i], 1e-4f) << "mismatch at index " << i;
}

TEST(ScaledDotProductAttentionAVX512, GQAHalvedKVHeads) {
    // kv_heads = n_heads/2 — two query heads share one KV head
    const size_t n_heads = 4;
    const size_t kv_heads = 2;
    const size_t kv_dim = kv_heads * HEAD_SIZE;
    const size_t pos = 3;

    std::vector<float> q(n_heads * HEAD_SIZE);
    std::vector<float> key_cache(SEQ_LEN * kv_dim);
    std::vector<float> val_cache(SEQ_LEN * kv_dim);
    std::vector<float> att_avx(n_heads * SEQ_LEN, 0.f);
    std::vector<float> att_ref(n_heads * SEQ_LEN, 0.f);
    std::vector<float> out_avx(n_heads * HEAD_SIZE, 0.f);
    std::vector<float> out_ref(n_heads * HEAD_SIZE, 0.f);

    for (size_t i = 0; i < q.size(); i++)
        q[i] = static_cast<float>(i % 5) - 2.f;
    for (size_t i = 0; i < key_cache.size(); i++)
        key_cache[i] = static_cast<float>(i % 4) - 1.f;
    for (size_t i = 0; i < val_cache.size(); i++)
        val_cache[i] = static_cast<float>(i % 3);

    scaledDotProductAttentionAVX512(out_avx.data(), q.data(), key_cache.data(), val_cache.data(), att_avx.data(), static_cast<int>(pos), n_heads, kv_heads,
                                    HEAD_SIZE, kv_dim, SEQ_LEN);
    refScaledDotProductAttention(out_ref.data(), q.data(), key_cache.data(), val_cache.data(), att_ref.data(), static_cast<int>(pos), n_heads, kv_heads,
                                 HEAD_SIZE, kv_dim, SEQ_LEN);

    for (size_t i = 0; i < out_avx.size(); i++)
        EXPECT_NEAR(out_avx[i], out_ref[i], 1e-4f) << "GQA mismatch at index " << i;
}

TEST(ScaledDotProductAttentionAVX512, NonMultipleOf16HeadSize) {
    // head_size=12 (not a multiple of 16) exercises the scalar tail paths
    const size_t head_size = 12;
    const size_t n_heads = 2;
    const size_t kv_heads = 2;
    const size_t kv_dim = kv_heads * head_size;
    const size_t pos = 5;

    std::vector<float> q(n_heads * head_size);
    std::vector<float> key_cache(SEQ_LEN * kv_dim);
    std::vector<float> val_cache(SEQ_LEN * kv_dim);
    std::vector<float> att_avx(n_heads * SEQ_LEN, 0.f);
    std::vector<float> att_ref(n_heads * SEQ_LEN, 0.f);
    std::vector<float> out_avx(n_heads * head_size, 0.f);
    std::vector<float> out_ref(n_heads * head_size, 0.f);

    for (size_t i = 0; i < q.size(); i++)
        q[i] = static_cast<float>(i % 5) - 2.f;
    for (size_t i = 0; i < key_cache.size(); i++)
        key_cache[i] = static_cast<float>(i % 3) - 1.f;
    for (size_t i = 0; i < val_cache.size(); i++)
        val_cache[i] = static_cast<float>(i % 4) - 2.f;

    scaledDotProductAttentionAVX512(out_avx.data(), q.data(), key_cache.data(), val_cache.data(), att_avx.data(), static_cast<int>(pos), n_heads, kv_heads,
                                    head_size, kv_dim, SEQ_LEN);
    refScaledDotProductAttention(out_ref.data(), q.data(), key_cache.data(), val_cache.data(), att_ref.data(), static_cast<int>(pos), n_heads, kv_heads,
                                 head_size, kv_dim, SEQ_LEN);

    for (size_t i = 0; i < out_avx.size(); i++)
        EXPECT_NEAR(out_avx[i], out_ref[i], 1e-4f) << "tail mismatch at index " << i;
}

#endif // __AVX512F__
