#include <cmath>
#include <gtest/gtest.h>
#include <memory>

#include <transformers-lite/core/tensor.hpp>
#include <transformers-lite/layers/transformer.hpp>

using namespace transformers_lite;

static constexpr size_t DIM = 4;
static constexpr size_t N_HEADS = 2;
static constexpr size_t N_KV_HEADS = 2;
static constexpr size_t KV_DIM = 4; // DIM * N_KV_HEADS / N_HEADS
static constexpr size_t HIDDEN_DIM = 8;
static constexpr size_t SEQ_LEN = 8;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void fill_raw(Tensor<CPU, float> &t, float val) {
    for (size_t i = 0; i < t.size(); ++i)
        t.data()[i] = val;
}

static Tensor<CPU, float> zeros(Shape s) {
    Tensor<CPU, float> t(s);
    fill_raw(t, 0.f);
    return t;
}

static Tensor<CPU, float> filled(Shape s, float val) {
    Tensor<CPU, float> t(s);
    fill_raw(t, val);
    return t;
}

// rows×cols matrix with 1s on the leading diagonal, 0s elsewhere
static Tensor<CPU, float> eye(size_t rows, size_t cols) {
    Tensor<CPU, float> t(Shape(rows, cols));
    fill_raw(t, 0.f);
    size_t diag = std::min(rows, cols);
    for (size_t i = 0; i < diag; ++i)
        t.data()[i * cols + i] = 1.f;
    return t;
}

// ── Linear ────────────────────────────────────────────────────────────────────

TEST(LinearTest, ZeroWeightsGivesZeroOutput) {
    auto w = zeros(Shape(3, DIM));
    Linear<CPU, float> layer(w);

    auto x = filled(Shape(DIM), 2.f);
    auto out = zeros(Shape(3));
    layer.forward(x, out);

    for (size_t i = 0; i < 3; ++i)
        EXPECT_FLOAT_EQ(out.data()[i], 0.f);
}

TEST(LinearTest, IdentityWeightsPassthrough) {
    auto w = eye(DIM, DIM);
    Linear<CPU, float> layer(w);

    auto x = zeros(Shape(DIM));
    x.data()[0] = 1.f;
    x.data()[1] = 2.f;
    x.data()[2] = 3.f;
    x.data()[3] = 4.f;
    auto out = zeros(Shape(DIM));
    layer.forward(x, out);

    EXPECT_FLOAT_EQ(out.data()[0], 1.f);
    EXPECT_FLOAT_EQ(out.data()[1], 2.f);
    EXPECT_FLOAT_EQ(out.data()[2], 3.f);
    EXPECT_FLOAT_EQ(out.data()[3], 4.f);
}

TEST(LinearTest, ScaledIdentityDoublesOutput) {
    auto w = eye(DIM, DIM);
    for (size_t i = 0; i < DIM; ++i)
        w.data()[i * DIM + i] = 2.f;
    Linear<CPU, float> layer(w);

    auto x = zeros(Shape(DIM));
    x.data()[0] = 3.f;
    x.data()[1] = 5.f;
    auto out = zeros(Shape(DIM));
    layer.forward(x, out);

    EXPECT_FLOAT_EQ(out.data()[0], 6.f);
    EXPECT_FLOAT_EQ(out.data()[1], 10.f);
}

TEST(LinearTest, OutDimMatchesWeightRows) {
    auto w = zeros(Shape(5, DIM));
    Linear<CPU, float> layer(w);
    EXPECT_EQ(layer.outDim(), 5u);
}

// ── FeedForward ───────────────────────────────────────────────────────────────

TEST(FeedForwardTest, ZeroInputGivesZeroOutput) {
    auto w1 = zeros(Shape(HIDDEN_DIM, DIM));
    auto w2 = zeros(Shape(DIM, HIDDEN_DIM));
    auto w3 = zeros(Shape(HIDDEN_DIM, DIM));
    FeedForward<CPU, float> ff(w1, w2, w3, DIM, HIDDEN_DIM);

    auto in = zeros(Shape(DIM));
    auto out = zeros(Shape(DIM));
    ff.forward(in, out);

    for (size_t i = 0; i < DIM; ++i)
        EXPECT_FLOAT_EQ(out.data()[i], 0.f);
}

TEST(FeedForwardTest, ZeroGateWeightsGivesZeroOutput) {
    // w3=0 → hb2=0 → hadamard product = 0 → out = w2 @ 0 = 0
    auto w1 = eye(HIDDEN_DIM, DIM);
    auto w2 = eye(DIM, HIDDEN_DIM);
    auto w3 = zeros(Shape(HIDDEN_DIM, DIM));
    FeedForward<CPU, float> ff(w1, w2, w3, DIM, HIDDEN_DIM);

    auto in = filled(Shape(DIM), 1.f);
    auto out = zeros(Shape(DIM));
    ff.forward(in, out);

    for (size_t i = 0; i < DIM; ++i)
        EXPECT_FLOAT_EQ(out.data()[i], 0.f);
}

TEST(FeedForwardTest, ZeroW1GivesZeroOutput) {
    // w1=0 → hb=0 → silu(0)=0 → hadamard product = 0 → out = 0
    auto w1 = zeros(Shape(HIDDEN_DIM, DIM));
    auto w2 = eye(DIM, HIDDEN_DIM);
    auto w3 = filled(Shape(HIDDEN_DIM, DIM), 1.f);
    FeedForward<CPU, float> ff(w1, w2, w3, DIM, HIDDEN_DIM);

    auto in = filled(Shape(DIM), 2.f);
    auto out = zeros(Shape(DIM));
    ff.forward(in, out);

    for (size_t i = 0; i < DIM; ++i)
        EXPECT_FLOAT_EQ(out.data()[i], 0.f);
}

TEST(FeedForwardTest, KnownArithmetic) {
    // dim=2, hidden_dim=2, w1=w2=w3=I(2×2), in=[1,1]
    // hb  = I @ [1,1] = [1,1]
    // hb2 = I @ [1,1] = [1,1]
    // silu(hb) = [silu(1), silu(1)]
    // hb = silu(hb) * hb2 = [silu(1), silu(1)]
    // out = I @ hb → [silu(1), silu(1)]
    constexpr size_t D = 2, H = 2;
    auto w1 = eye(H, D);
    auto w2 = eye(D, H);
    auto w3 = eye(H, D);
    FeedForward<CPU, float> ff(w1, w2, w3, D, H);

    auto in = filled(Shape(D), 1.f);
    auto out = zeros(Shape(D));
    ff.forward(in, out);

    float expected = 1.f / (1.f + std::exp(-1.f)); // silu(1) = 1 * sigmoid(1)
    EXPECT_NEAR(out.data()[0], expected, 1e-5f);
    EXPECT_NEAR(out.data()[1], expected, 1e-5f);
}

// ── Attention ─────────────────────────────────────────────────────────────────

TEST(AttentionTest, ZeroWeightsGivesZeroOutput) {
    // wq=wk=wv=0 → q=k=0 (RoPE of 0 = 0), v=0
    // softmax([0,...]) = uniform, weighted sum of v=0 → xb=0
    auto wq = zeros(Shape(DIM, DIM));
    auto wk = zeros(Shape(KV_DIM, DIM));
    auto wv = zeros(Shape(KV_DIM, DIM));
    Attention<CPU, float> attn(wq, wk, wv, KV_DIM, DIM, N_HEADS, N_KV_HEADS, SEQ_LEN);

    auto in = filled(Shape(DIM), 2.f);
    auto xb = zeros(Shape(DIM));
    attn.forward(in, xb, 0);

    for (size_t i = 0; i < DIM; ++i)
        EXPECT_FLOAT_EQ(xb.data()[i], 0.f);
}

TEST(AttentionTest, OutputShapePreserved) {
    auto wq = zeros(Shape(DIM, DIM));
    auto wk = zeros(Shape(KV_DIM, DIM));
    auto wv = zeros(Shape(KV_DIM, DIM));
    Attention<CPU, float> attn(wq, wk, wv, KV_DIM, DIM, N_HEADS, N_KV_HEADS, SEQ_LEN);

    auto in = filled(Shape(DIM), 1.f);
    auto xb = zeros(Shape(DIM));
    attn.forward(in, xb, 0);

    EXPECT_EQ(xb.size(), DIM);
}

TEST(AttentionTest, IdentityValueWeightPassthroughAtPos0) {
    // wq=wk=0, wv=I → q=k=0, v=in
    // At pos=0: softmax([0]) = [1.0], xb = 1.0 * v[0] = in
    // RoPE at pos=0: cos(0)=1, sin(0)=0 → no rotation applied
    auto wq = zeros(Shape(DIM, DIM));
    auto wk = zeros(Shape(KV_DIM, DIM));
    auto wv = eye(KV_DIM, DIM);
    Attention<CPU, float> attn(wq, wk, wv, KV_DIM, DIM, N_HEADS, N_KV_HEADS, SEQ_LEN);

    auto in = zeros(Shape(DIM));
    in.data()[0] = 1.f;
    in.data()[1] = 2.f;
    in.data()[2] = 3.f;
    in.data()[3] = 4.f;
    auto xb = zeros(Shape(DIM));
    attn.forward(in, xb, 0);

    EXPECT_NEAR(xb.data()[0], 1.f, 1e-5f);
    EXPECT_NEAR(xb.data()[1], 2.f, 1e-5f);
    EXPECT_NEAR(xb.data()[2], 3.f, 1e-5f);
    EXPECT_NEAR(xb.data()[3], 4.f, 1e-5f);
}

TEST(AttentionTest, KVCacheAccumulation) {
    // wq=wk=0 → all attention scores = 0 → softmax = uniform
    // wv=I → v[t] = in_t
    // At pos=0: xb = in0
    // At pos=1: softmax([0,0]) = [0.5, 0.5], xb = 0.5*in0 + 0.5*in1
    auto wq = zeros(Shape(DIM, DIM));
    auto wk = zeros(Shape(KV_DIM, DIM));
    auto wv = eye(KV_DIM, DIM);
    Attention<CPU, float> attn(wq, wk, wv, KV_DIM, DIM, N_HEADS, N_KV_HEADS, SEQ_LEN);

    auto in0 = filled(Shape(DIM), 2.f);
    auto in1 = filled(Shape(DIM), 4.f);
    auto xb = zeros(Shape(DIM));

    attn.forward(in0, xb, 0); // populate KV cache at pos=0
    attn.forward(in1, xb, 1); // uniform attention over pos=0 and pos=1

    // Expected: 0.5 * 2 + 0.5 * 4 = 3
    for (size_t i = 0; i < DIM; ++i)
        EXPECT_NEAR(xb.data()[i], 3.f, 1e-5f);
}

// ── TransformerBlock ──────────────────────────────────────────────────────────

// Helper that builds a zero-weight TransformerBlock for reuse.
static auto makeZeroBlock() {
    // All-zero weights: rmsnorm outputs 0, so attn/FFN branches output 0,
    // and the residual connections preserve x unchanged.
    static auto wq = zeros(Shape(DIM, DIM));
    static auto wk = zeros(Shape(KV_DIM, DIM));
    static auto wv = zeros(Shape(KV_DIM, DIM));
    static auto wo = zeros(Shape(DIM, DIM));
    static auto w_rms_att = zeros(Shape(DIM));
    static auto w_rms_ffn = zeros(Shape(DIM));
    static auto w1 = zeros(Shape(HIDDEN_DIM, DIM));
    static auto w2 = zeros(Shape(DIM, HIDDEN_DIM));
    static auto w3 = zeros(Shape(HIDDEN_DIM, DIM));

    auto attn = std::make_unique<Attention<CPU, float>>(wq, wk, wv, KV_DIM, DIM, N_HEADS, N_KV_HEADS, SEQ_LEN);
    auto ff = std::make_unique<FeedForward<CPU, float>>(w1, w2, w3, DIM, HIDDEN_DIM);
    return TransformerBlock<CPU, float>(std::move(attn), std::move(ff), w_rms_ffn, wo, w_rms_att, DIM);
}

TEST(TransformerBlockTest, ZeroWeightsPreservesInputViaResidual) {
    // rmsnorm(x, 0) = 0 → attn branch = 0 → wo @ 0 = 0 → x += 0
    // rmsnorm(x, 0) = 0 → FFN branch = 0 → x += 0  → x unchanged
    auto block = makeZeroBlock();

    auto x = zeros(Shape(DIM));
    x.data()[0] = 1.f;
    x.data()[1] = 2.f;
    x.data()[2] = 3.f;
    x.data()[3] = 4.f;
    block.forward(x, 0);

    EXPECT_FLOAT_EQ(x.data()[0], 1.f);
    EXPECT_FLOAT_EQ(x.data()[1], 2.f);
    EXPECT_FLOAT_EQ(x.data()[2], 3.f);
    EXPECT_FLOAT_EQ(x.data()[3], 4.f);
}

TEST(TransformerBlockTest, MultiStepForwardDoesNotCrash) {
    auto block = makeZeroBlock();
    auto x = filled(Shape(DIM), 1.f);
    for (int pos = 0; pos < static_cast<int>(SEQ_LEN) - 1; ++pos)
        block.forward(x, pos);
    SUCCEED();
}
