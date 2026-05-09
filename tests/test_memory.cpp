#include <gtest/gtest.h>

#include <transformers-lite/core/tensor.hpp>

using namespace transformers_lite;

// ── Memory lifecycle ──────────────────────────────────────────────────────────

TEST(MemoryLifecycle, DefaultConstructAndDestroy) {
    // Default-constructed Tensor has m_data=nullptr; destructor must not
    // call deallocate(nullptr, ...).
    Tensor<CPU, float> t;
    (void)t;
}

TEST(MemoryLifecycle, MoveConstructLeavesValidState) {
    // After a move the source has m_data=nullptr; its destructor must be safe.
    Tensor<CPU, float> a(Shape(4));
    a(0) = 1.f;
    a(1) = 2.f;
    a(2) = 3.f;
    a(3) = 4.f;

    Tensor<CPU, float> b(std::move(a));

    EXPECT_FLOAT_EQ(b(0), 1.f);
    EXPECT_FLOAT_EQ(b(3), 4.f);
    // Both a (moved-from, m_data==nullptr) and b are destroyed at end of scope.
}

TEST(MemoryLifecycle, CopyConstructFromNonEmpty) {
    Tensor<CPU, float> a(Shape(3));
    a(0) = 10.f;
    a(1) = 20.f;
    a(2) = 30.f;

    Tensor<CPU, float> b(a); // const copy constructor

    EXPECT_FLOAT_EQ(b(0), 10.f);
    EXPECT_FLOAT_EQ(b(1), 20.f);
    EXPECT_FLOAT_EQ(b(2), 30.f);

    // Mutating b must not affect a (deep copy).
    b(0) = 99.f;
    EXPECT_FLOAT_EQ(a(0), 10.f);
}

TEST(MemoryLifecycle, CopyConstructFromEmpty) {
    // Copying an empty (default-constructed) Tensor must not memcpy into
    // a null pointer.
    Tensor<CPU, float> a;
    Tensor<CPU, float> b(a);
    (void)b;
}
