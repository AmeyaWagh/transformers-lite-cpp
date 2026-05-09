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

    TensorView<float> wv(w.data(), w.shape());
    TensorView<float> xv(x.data(), x.shape());
    TensorView<float> outv(out.data(), out.shape());
    matmul(outv, xv, wv);

    EXPECT_FLOAT_EQ(outv(0), 3.f);
    EXPECT_FLOAT_EQ(outv(1), 7.f);
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

    TensorView<float> wv(w.data(), w.shape());
    TensorView<float> xv(x.data(), x.shape());
    TensorView<float> outv(out.data(), out.shape());
    matmul(outv, xv, wv);

    EXPECT_FLOAT_EQ(outv(0), 6.f);
    EXPECT_FLOAT_EQ(outv(1), 15.f);
}

TEST(MatmulCPU, ZeroVector) {
    Tensor<CPU, float> w(Shape(2, 2));
    w(0, 0) = 1.f;
    w(0, 1) = 2.f;
    w(1, 0) = 3.f;
    w(1, 1) = 4.f;
    auto x = makeTensor({0.f, 0.f});
    Tensor<CPU, float> out(Shape(2));

    TensorView<float> wv(w.data(), w.shape());
    TensorView<float> xv(x.data(), x.shape());
    TensorView<float> outv(out.data(), out.shape());
    matmul(outv, xv, wv);

    EXPECT_FLOAT_EQ(outv(0), 0.f);
    EXPECT_FLOAT_EQ(outv(1), 0.f);
}

// ── rmsnorm ──────────────────────────────────────────────────────────────────

TEST(RmsNormCPU, UnitWeights) {
    // rms([1,2,3,4]) = sqrt((1+4+9+16)/4) = sqrt(7.5)
    // out[i] = x[i] / rms(x)  (weights all 1)
    auto x = makeTensor({1.f, 2.f, 3.f, 4.f});
    auto w = makeTensor({1.f, 1.f, 1.f, 1.f});
    Tensor<CPU, float> out(Shape(4));

    TensorView<float> xv(x.data(), x.shape());
    TensorView<float> wv(w.data(), w.shape());
    TensorView<float> outv(out.data(), out.shape());
    rmsnorm(outv, xv, wv);

    float rms = 1.f / sqrtf(7.5f + 1e-5f);
    EXPECT_NEAR(outv(0), 1.f * rms, 1e-5f);
    EXPECT_NEAR(outv(1), 2.f * rms, 1e-5f);
    EXPECT_NEAR(outv(2), 3.f * rms, 1e-5f);
    EXPECT_NEAR(outv(3), 4.f * rms, 1e-5f);
}

TEST(RmsNormCPU, ScaledWeights) {
    auto x = makeTensor({1.f, 1.f, 1.f, 1.f});
    auto w = makeTensor({2.f, 2.f, 2.f, 2.f});
    Tensor<CPU, float> out(Shape(4));

    TensorView<float> xv(x.data(), x.shape());
    TensorView<float> wv(w.data(), w.shape());
    TensorView<float> outv(out.data(), out.shape());
    rmsnorm(outv, xv, wv);

    // rms([1,1,1,1]) = 1, so out = 2 * 1 * x[i] = 2
    EXPECT_NEAR(outv(0), 2.f, 1e-4f);
    EXPECT_NEAR(outv(1), 2.f, 1e-4f);
}

// ── softmax ──────────────────────────────────────────────────────────────────

TEST(SoftmaxCPU, SumsToOne) {
    auto x = makeTensor({1.f, 2.f, 3.f});
    TensorView<float> xv(x.data(), x.shape());
    softmax(xv);

    float sum = xv(0) + xv(1) + xv(2);
    EXPECT_NEAR(sum, 1.f, 1e-6f);
}

TEST(SoftmaxCPU, MaxElementDominates) {
    auto x = makeTensor({0.f, 0.f, 100.f});
    TensorView<float> xv(x.data(), x.shape());
    softmax(xv);

    EXPECT_NEAR(xv(2), 1.f, 1e-5f);
}

TEST(SoftmaxCPU, PartialN) {
    // Apply softmax only to first 2 elements; third unchanged
    auto x = makeTensor({1.f, 2.f, 999.f});
    TensorView<float> xv(x.data(), x.shape());
    softmax(xv, 2);

    float sum = xv(0) + xv(1);
    EXPECT_NEAR(sum, 1.f, 1e-6f);
}

// ── silu ─────────────────────────────────────────────────────────────────────

TEST(SiluCPU, ZeroIsZero) {
    auto x = makeTensor({0.f});
    TensorView<float> xv(x.data(), x.shape());
    silu_inpl(xv);
    EXPECT_FLOAT_EQ(xv(0), 0.f);
}

TEST(SiluCPU, PositiveInput) {
    // silu(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1))
    auto x = makeTensor({1.f});
    TensorView<float> xv(x.data(), x.shape());
    silu_inpl(xv);
    float expected = 1.f / (1.f + expf(-1.f));
    EXPECT_NEAR(xv(0), expected, 1e-6f);
}

TEST(SiluCPU, NegativeInput) {
    // silu(-1) = -1 * sigmoid(-1)
    auto x = makeTensor({-1.f});
    TensorView<float> xv(x.data(), x.shape());
    silu_inpl(xv);
    float expected = -1.f * (1.f / (1.f + expf(1.f)));
    EXPECT_NEAR(xv(0), expected, 1e-6f);
}

// ── add ───────────────────────────────────────────────────────────────────────

TEST(AddCPU, ElementWise) {
    auto a = makeTensor({1.f, 2.f, 3.f});
    auto b = makeTensor({4.f, 5.f, 6.f});
    Tensor<CPU, float> result(Shape(3));

    TensorView<float> av(a.data(), a.shape());
    TensorView<float> bv(b.data(), b.shape());
    TensorView<float> rv(result.data(), result.shape());
    add(rv, av, bv);

    EXPECT_FLOAT_EQ(rv(0), 5.f);
    EXPECT_FLOAT_EQ(rv(1), 7.f);
    EXPECT_FLOAT_EQ(rv(2), 9.f);
}

TEST(AddCPU, ZeroTensor) {
    auto a = makeTensor({1.f, 2.f, 3.f});
    auto b = makeTensor({0.f, 0.f, 0.f});
    Tensor<CPU, float> result(Shape(3));

    TensorView<float> av(a.data(), a.shape());
    TensorView<float> bv(b.data(), b.shape());
    TensorView<float> rv(result.data(), result.shape());
    add(rv, av, bv);

    EXPECT_FLOAT_EQ(rv(0), 1.f);
    EXPECT_FLOAT_EQ(rv(1), 2.f);
    EXPECT_FLOAT_EQ(rv(2), 3.f);
}

// ── hadamard_prod ─────────────────────────────────────────────────────────────

TEST(HadamardProdCPU, ElementWise) {
    auto a = makeTensor({1.f, 2.f, 3.f});
    auto b = makeTensor({4.f, 5.f, 6.f});
    Tensor<CPU, float> result(Shape(3));

    TensorView<float> av(a.data(), a.shape());
    TensorView<float> bv(b.data(), b.shape());
    TensorView<float> rv(result.data(), result.shape());
    hadamard_prod(rv, av, bv);

    EXPECT_FLOAT_EQ(rv(0), 4.f);
    EXPECT_FLOAT_EQ(rv(1), 10.f);
    EXPECT_FLOAT_EQ(rv(2), 18.f);
}

TEST(HadamardProdCPU, WithZero) {
    auto a = makeTensor({1.f, 2.f, 3.f});
    auto b = makeTensor({0.f, 0.f, 0.f});
    Tensor<CPU, float> result(Shape(3));

    TensorView<float> av(a.data(), a.shape());
    TensorView<float> bv(b.data(), b.shape());
    TensorView<float> rv(result.data(), result.shape());
    hadamard_prod(rv, av, bv);

    EXPECT_FLOAT_EQ(rv(0), 0.f);
    EXPECT_FLOAT_EQ(rv(1), 0.f);
    EXPECT_FLOAT_EQ(rv(2), 0.f);
}

// ── dot_prod ──────────────────────────────────────────────────────────────────

TEST(DotProdCPU, Basic) {
    // [1,2,3] · [4,5,6] = 4 + 10 + 18 = 32
    auto a = makeTensor({1.f, 2.f, 3.f});
    auto b = makeTensor({4.f, 5.f, 6.f});
    TensorView<float> av(a.data(), a.shape());
    TensorView<float> bv(b.data(), b.shape());

    EXPECT_FLOAT_EQ(dot_prod(av, bv), 32.f);
}

TEST(DotProdCPU, Orthogonal) {
    auto a = makeTensor({1.f, 0.f});
    auto b = makeTensor({0.f, 1.f});
    TensorView<float> av(a.data(), a.shape());
    TensorView<float> bv(b.data(), b.shape());

    EXPECT_FLOAT_EQ(dot_prod(av, bv), 0.f);
}

// ── argmax ────────────────────────────────────────────────────────────────────

TEST(ArgmaxCPU, MaxAtEnd) {
    auto x = makeTensor({1.f, 2.f, 3.f, 4.f, 5.f});
    TensorView<float> xv(x.data(), x.shape());
    EXPECT_EQ(argmax(xv), 4UL);
}

TEST(ArgmaxCPU, MaxAtStart) {
    auto x = makeTensor({9.f, 1.f, 2.f, 3.f});
    TensorView<float> xv(x.data(), x.shape());
    EXPECT_EQ(argmax(xv), 0UL);
}

TEST(ArgmaxCPU, MaxInMiddle) {
    auto x = makeTensor({1.f, 5.f, 3.f, 2.f, 4.f});
    TensorView<float> xv(x.data(), x.shape());
    EXPECT_EQ(argmax(xv), 1UL);
}

// ── setZero ───────────────────────────────────────────────────────────────────

TEST(SetZeroCPU, ZeroesAllElements) {
    auto x = makeTensor({1.f, 2.f, 3.f, 4.f});
    TensorView<float> xv(x.data(), x.shape());
    setZero(xv);

    for (size_t i = 0; i < xv.size(); ++i) {
        EXPECT_FLOAT_EQ(xv[i], 0.f);
    }
}
