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
