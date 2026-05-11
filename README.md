# transformers-lite

Transformer inference in C++. A header-only library providing tensor operations and transformer building blocks.

The library has **zero external dependencies** — it relies only on the C++ standard library. OpenMP `#pragma` directives are used in a few hot loops for optional parallelism but are silently ignored by compilers without OpenMP support.

## Requirements

- CMake 3.16+
- C++20 compiler (GCC 11+, Clang 14+)

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Build with tests

Tests use [GoogleTest](https://github.com/google/googletest) and are fetched automatically via CMake's `FetchContent` — no manual install needed.

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build
cd build && ctest --output-on-failure
```

## Usage

The library is header-only. Link against the `transformers_lite` CMake target:

```cmake
add_subdirectory(transformers-lite)
target_link_libraries(your_target PRIVATE transformers_lite)
```

Then include the headers you need:

```cpp
#include <transformers-lite/tensor.hpp>
#include <transformers-lite/ops.hpp>

using namespace transformers;

int main() {
    Tensor<CPU, float> x(Shape(4));
    x(0) = 1.0f;
    x(1) = 2.0f;
    x(2) = 3.0f;
    x(3) = 4.0f;
}
```

## Adding custom ops

Most custom ops only require a new **expression struct** — no changes to the library internals.

An expression must satisfy the `TensorExpr<E, COMPUTE, T>` concept:

| Method | Description |
|---|---|
| `Shape outputShape() const` | Returns the shape of the result |
| `void evalInto(TensorView<T>& out) const` | Writes the computed result into `out` |

`Tensor::operator=` detects this interface automatically, so the assignment syntax `result = myExpr(...)` just works.

### Example: element-wise clamp

```cpp
#include <transformers-lite/core/ops/exprs.hpp>

namespace transformers_lite {

template <template <class> class COMPUTE, typename T>
struct ClampExpr {
    using value_type = T;
    TensorView<T> x;
    T low, high;

    Shape outputShape() const { return x.shape(); }

    void evalInto(TensorView<T>& out) const {
        for (size_t i = 0; i < x.size(); ++i)
            out[i] = std::clamp(x[i], low, high);
    }
};

template <template <class> class COMPUTE, typename T>
auto clamp(const Tensor<COMPUTE, T>& x, T low, T high) -> ClampExpr<COMPUTE, T> {
    return {TensorView<T>{x}, low, high};
}

} // namespace transformers_lite
```

Usage:

```cpp
Tensor<CPU, float> x(Shape(4), {-2.f, 0.5f, 1.5f, 3.f});
Tensor<CPU, float> y;
y = clamp(x, 0.f, 1.f); // y == [0, 0.5, 1, 1]
```

### When you also need a new backend kernel

If the op is compute-intensive and needs an AVX-512 (or future CUDA) fast path, add the dispatch inside `evalInto` using the existing `Ops<COMPUTE, T>` pattern:

1. Add the method to `Ops<CPU, T>` in `include/transformers-lite/core/ops/cpu_ops.hpp`.
2. Add an AVX-512 implementation in `include/transformers-lite/core/ops/avx512_ops.hpp` under `#ifdef __AVX512F__`, then dispatch with `if constexpr` inside the `Ops<CPU, T>` method.
3. Call `Ops<COMPUTE, T>::yourMethod(...)` from `evalInto`.
