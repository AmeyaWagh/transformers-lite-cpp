#include <gtest/gtest.h>

#include <transformers-lite/ops.hpp>
#include <transformers-lite/tensor.hpp>

using namespace transformers_lite;

TEST(ShapeTest, OneDimensional) {
    Shape s(5);
    EXPECT_EQ(s.numDims(), 1);
    EXPECT_EQ(s[0], 5);
    EXPECT_EQ(s.size(), 5);
}

TEST(ShapeTest, TwoDimensional) {
    Shape s(3, 4);
    EXPECT_EQ(s.numDims(), 2);
    EXPECT_EQ(s[0], 3);
    EXPECT_EQ(s[1], 4);
    EXPECT_EQ(s.size(), 12);
}

TEST(TensorTest, CreateAndAccess) {
    Tensor<CPU, float> t(Shape(4));
    t(0) = 1.0f;
    t(1) = 2.0f;
    t(2) = 3.0f;
    t(3) = 4.0f;

    EXPECT_FLOAT_EQ(t(0), 1.0f);
    EXPECT_FLOAT_EQ(t(1), 2.0f);
    EXPECT_FLOAT_EQ(t(2), 3.0f);
    EXPECT_FLOAT_EQ(t(3), 4.0f);
}

TEST(TensorTest, BracketAccess) {
    Tensor<CPU, float> t(Shape(3));
    t[0] = 10.0f;
    t[1] = 20.0f;
    t[2] = 30.0f;

    EXPECT_FLOAT_EQ(t[0], 10.0f);
    EXPECT_FLOAT_EQ(t[1], 20.0f);
    EXPECT_FLOAT_EQ(t[2], 30.0f);
}

TEST(TensorTest, TwoDimensionalAccess) {
    Tensor<CPU, float> t(Shape(2, 3));
    // Fill with row-major values
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t(i, j) = static_cast<float>(i * 3 + j);
        }
    }

    EXPECT_FLOAT_EQ(t(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(t(0, 2), 2.0f);
    EXPECT_FLOAT_EQ(t(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(t(1, 2), 5.0f);
}

TEST(OpsTest, Matmul) {
    // W (2,3) @ x (3,) -> xout (2,)
    // W = [[1,2,3],[4,5,6]], x = [1,1,1]
    // xout = [6, 15]
    Tensor<CPU, float> w(Shape(2, 3));
    w(0, 0) = 1.0f;
    w(0, 1) = 2.0f;
    w(0, 2) = 3.0f;
    w(1, 0) = 4.0f;
    w(1, 1) = 5.0f;
    w(1, 2) = 6.0f;

    Tensor<CPU, float> x(Shape(3));
    x(0) = 1.0f;
    x(1) = 1.0f;
    x(2) = 1.0f;

    Tensor<CPU, float> xout(Shape(2));

    TensorView<float> xout_v(xout.data(), xout.shape());
    TensorView<float> x_v(x.data(), x.shape());
    TensorView<float> w_v(w.data(), w.shape());

    matmul(xout_v, x_v, w_v);

    EXPECT_FLOAT_EQ(xout_v(0), 6.0f);
    EXPECT_FLOAT_EQ(xout_v(1), 15.0f);
}
