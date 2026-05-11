#include <gtest/gtest.h>

#include <transformers-lite/core/ops.hpp>
#include <transformers-lite/core/tensor.hpp>

using namespace transformers_lite;

// ── helpers ──────────────────────────────────────────────────────────────────

static Tensor<CPU, float> makeTensor(std::initializer_list<float> vals) {
    std::vector<float> v(vals);
    return Tensor<CPU, float>(Shape(v.size()), v);
}

// ── matmul ───────────────────────────────────────────────────────────────────

TEST(MatmulCPU, SquareMatrix) {
    // W(2,2) @ x(2) -> out(2)
    // W = [[1,2],[3,4]], x = [1,1] -> out = [3, 7]
    Tensor<CPU, float> w(Shape(2, 2));
    w(0, 0) = 1.f;
    w(0, 1) = 2.f;
    w(1, 0) = 3.f;
    w(1, 1) = 4.f;
    auto x = makeTensor({1.f, 1.f});
    Tensor<CPU, float> out(Shape(2));

    out = matmul(x, w);

    EXPECT_FLOAT_EQ(out(0), 3.f);
    EXPECT_FLOAT_EQ(out(1), 7.f);
}

TEST(MatmulCPU, RectangularMatrix) {
    // W(2,3) @ x(3) -> out(2)
    // W = [[1,2,3],[4,5,6]], x = [1,1,1] -> out = [6, 15]
    Tensor<CPU, float> w(Shape(2, 3));
    w(0, 0) = 1.f;
    w(0, 1) = 2.f;
    w(0, 2) = 3.f;
    w(1, 0) = 4.f;
    w(1, 1) = 5.f;
    w(1, 2) = 6.f;
    auto x = makeTensor({1.f, 1.f, 1.f});
    Tensor<CPU, float> out(Shape(2));

    out = matmul(x, w);

    EXPECT_FLOAT_EQ(out(0), 6.f);
    EXPECT_FLOAT_EQ(out(1), 15.f);
}

TEST(MatmulCPU, ZeroVector) {
    Tensor<CPU, float> w(Shape(2, 2));
    w(0, 0) = 1.f;
    w(0, 1) = 2.f;
    w(1, 0) = 3.f;
    w(1, 1) = 4.f;
    auto x = makeTensor({0.f, 0.f});
    Tensor<CPU, float> out(Shape(2));

    out = matmul(x, w);

    EXPECT_FLOAT_EQ(out(0), 0.f);
    EXPECT_FLOAT_EQ(out(1), 0.f);
}

// ── add ───────────────────────────────────────────────────────────────────────

TEST(AddCPU, ElementWise) {
    auto a = makeTensor({1.f, 2.f, 3.f});
    auto b = makeTensor({4.f, 5.f, 6.f});
    Tensor<CPU, float> result(Shape(3));

    result = a + b;

    EXPECT_FLOAT_EQ(result(0), 5.f);
    EXPECT_FLOAT_EQ(result(1), 7.f);
    EXPECT_FLOAT_EQ(result(2), 9.f);
}

TEST(AddCPU, ZeroTensor) {
    auto a = makeTensor({1.f, 2.f, 3.f});
    auto b = makeTensor({0.f, 0.f, 0.f});
    Tensor<CPU, float> result(Shape(3));

    result = a + b;

    EXPECT_FLOAT_EQ(result(0), 1.f);
    EXPECT_FLOAT_EQ(result(1), 2.f);
    EXPECT_FLOAT_EQ(result(2), 3.f);
}

// ── hadamard_prod ─────────────────────────────────────────────────────────────

TEST(HadamardProdCPU, ElementWise) {
    auto a = makeTensor({1.f, 2.f, 3.f});
    auto b = makeTensor({4.f, 5.f, 6.f});
    Tensor<CPU, float> result(Shape(3));

    result = a * b;

    EXPECT_FLOAT_EQ(result(0), 4.f);
    EXPECT_FLOAT_EQ(result(1), 10.f);
    EXPECT_FLOAT_EQ(result(2), 18.f);
}

TEST(HadamardProdCPU, WithZero) {
    auto a = makeTensor({1.f, 2.f, 3.f});
    auto b = makeTensor({0.f, 0.f, 0.f});
    Tensor<CPU, float> result(Shape(3));

    result = a * b;

    EXPECT_FLOAT_EQ(result(0), 0.f);
    EXPECT_FLOAT_EQ(result(1), 0.f);
    EXPECT_FLOAT_EQ(result(2), 0.f);
}

// ── silu ─────────────────────────────────────────────────────────────────────

TEST(SiluCPU, ZeroIsZero) {
    auto x = makeTensor({0.f});
    x = silu(x);
    EXPECT_FLOAT_EQ(x[0], 0.f);
}

TEST(SiluCPU, PositiveInput) {
    // silu(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1))
    auto x = makeTensor({1.f});
    x = silu(x);
    EXPECT_NEAR(x[0], 1.f / (1.f + std::exp(-1.f)), 1e-6f);
}

TEST(SiluCPU, NegativeInput) {
    // silu(-1) = -1 * sigmoid(-1)
    auto x = makeTensor({-1.f});
    x = silu(x);
    EXPECT_NEAR(x[0], -1.f / (1.f + std::exp(1.f)), 1e-6f);
}

TEST(SiluCPU, MultiElement) {
    auto x = makeTensor({0.f, 1.f, -1.f, 2.f});
    x = silu(x);
    EXPECT_NEAR(x[0], 0.f, 1e-6f);
    EXPECT_NEAR(x[1], 1.f / (1.f + std::exp(-1.f)), 1e-6f);
    EXPECT_NEAR(x[2], -1.f / (1.f + std::exp(1.f)), 1e-6f);
    EXPECT_NEAR(x[3], 2.f / (1.f + std::exp(-2.f)), 1e-6f);
}

// ── rope ─────────────────────────────────────────────────────────────────────

TEST(RopeCPU, ZeroPosIsIdentity) {
    // At pos=0: theta=0 -> cos=1, sin=0 -> rotation is identity
    auto q = makeTensor({1.f, 2.f, 3.f, 4.f});
    auto k = makeTensor({5.f, 6.f, 7.f, 8.f});
    q = rope(q, k, 0, 2);
    EXPECT_NEAR(q[0], 1.f, 1e-6f);
    EXPECT_NEAR(q[1], 2.f, 1e-6f);
    EXPECT_NEAR(q[2], 3.f, 1e-6f);
    EXPECT_NEAR(q[3], 4.f, 1e-6f);
    EXPECT_NEAR(k[0], 5.f, 1e-6f);
    EXPECT_NEAR(k[1], 6.f, 1e-6f);
}

TEST(RopeCPU, MatchesScalarReference) {
    // Reference: apply rotation manually for pos=3, head_size=4
    const int pos = 3;
    const size_t head_size = 4;
    auto q = makeTensor({1.f, 0.f, 0.f, 1.f});
    auto k = makeTensor({1.f, 0.f, 0.f, 1.f});

    auto ref_rotate = [&](float v0, float v1, size_t head_dim) -> std::pair<float, float> {
        float freq = 1.f / std::pow(10000.f, head_dim / static_cast<float>(head_size));
        float theta = pos * freq;
        return {v0 * std::cos(theta) - v1 * std::sin(theta), v0 * std::sin(theta) + v1 * std::cos(theta)};
    };

    auto [q0, q1] = ref_rotate(1.f, 0.f, 0);
    auto [q2, q3] = ref_rotate(0.f, 1.f, 2);
    auto [k0, k1] = ref_rotate(1.f, 0.f, 0);

    q = rope(q, k, pos, head_size);

    EXPECT_NEAR(q[0], q0, 1e-5f);
    EXPECT_NEAR(q[1], q1, 1e-5f);
    EXPECT_NEAR(q[2], q2, 1e-5f);
    EXPECT_NEAR(q[3], q3, 1e-5f);
    EXPECT_NEAR(k[0], k0, 1e-5f);
    EXPECT_NEAR(k[1], k1, 1e-5f);
}

TEST(RopeCPU, KRotatedAsKVDimSideEffect) {
    // k is rotated in-place as a side effect; original q tensor also picks up the rotation
    const int pos = 1;
    const size_t head_size = 2;
    auto q = makeTensor({1.f, 0.f});
    auto k = makeTensor({0.f, 1.f});

    float freq = 1.f / std::pow(10000.f, 0.f / 2.f); // head_dim=0
    float theta = pos * freq;                        // = 1.0
    float expected_k0 = 0.f * std::cos(theta) - 1.f * std::sin(theta);
    float expected_k1 = 0.f * std::sin(theta) + 1.f * std::cos(theta);

    q = rope(q, k, pos, head_size);

    EXPECT_NEAR(k[0], expected_k0, 1e-5f);
    EXPECT_NEAR(k[1], expected_k1, 1e-5f);
}
