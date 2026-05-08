# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build library only
cmake -S . -B build
cmake --build build

# Build with tests (GoogleTest fetched automatically via FetchContent)
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build

# Run all tests
cd build && ctest --output-on-failure

# Run a single test binary
./build/tests/test_tensor

# Run a specific test case
./build/tests/test_tensor --gtest_filter=TensorTest.CreateAndAccess
```

## Design philosophy

This library is intentionally modelled after **Eigen**: it is **header-only**. There must never be `.cpp` files compiled into a separate object or archive. All implementation — including full function bodies and template instantiations — lives in the `.hpp` files under `include/transformers-lite/`. The CMake target is `INTERFACE`-only; users get the library by adding the include path, with no link step against a compiled artifact.

Consequences of this constraint:
- Do not introduce `.cpp` source files for library code.
- Do not split declarations into `.hpp` and definitions into `.cpp` (or `.ipp` without a good reason).
- Optional backends (CUDA, AVX-512) are gated with `#ifdef` so a CPU-only user never pulls in backend-specific headers.

## Architecture

This is a **header-only C++20 library** (`namespace transformers_lite`) for transformer inference with zero external dependencies. All code lives in `include/transformers-lite/`.

### Layer hierarchy (bottom-up)

| Header | Purpose |
|---|---|
| `memory.hpp` | `Memory<COMPUTE, T>` — owns a heap buffer; `CPU<T>`, `CUDA<T>`, `AVX512<T>` backend tags |
| `tensor.hpp` | `Shape` (row-major strides, slice/offset), `TensorView<T>` (non-owning), `Tensor<COMPUTE,T>` (owns `Memory`) |
| `ops.hpp` | Free functions operating on `TensorView<T>`: `matmul`, `rmsnorm`, `softmax`, `silu_inpl`, `add`, `hadamard_prod`, `dot_prod`, `argmax`, `setZero` |
| `transformer.hpp` | `Attention`, `FeedForward`, `TransformerBlock`, `Linear`, `Transformer`; `TransformerConfig` and `TransformerWeights` structs |
| `types.hpp` | `float32_t`, `float64_t` aliases |

### Key design patterns

- **`Tensor<COMPUTE, T>` vs `TensorView<T>`**: `Tensor` owns memory (via `Memory<COMPUTE,T>`); `TensorView` is a non-owning pointer+shape pair. Ops take `TensorView&` so they work on slices of larger tensors without copies.
- **Backend template parameter**: `COMPUTE` is a template-template parameter (`CPU`, `CUDA`, `AVX512`). Only `CPU` is implemented. All ops are currently CPU-only.
- **`Shape` indexing**: `shape(i, j)` returns the flat memory offset for multi-dimensional access using pre-computed row-major strides. `shape.slice(i)` returns a lower-rank `Shape`; `shape.offset(i)` returns the byte offset for pointer arithmetic.
- **KV cache in `Attention`**: Stored as flat `Tensor<COMPUTE,T>` of size `seq_len * kv_dim`, reshaped to `(seq_len, kv_dim)` on access. RoPE encoding is applied in-place after QKV projections.
- **`TransformerBlock` forward**: RMSNorm → Attention → output projection → residual add → RMSNorm → FFN (SwiGLU: `silu(w1(x)) * w3(x)`) → residual add.
- **`Transformer::forward`**: Looks up token embedding, copies to device tensor, runs all `TransformerBlock` layers, applies final RMSNorm, runs `Linear` classifier, copies logits back to CPU.
- OpenMP `#pragma omp parallel for` is used in `matmulCPU` and the multi-head attention loop; silently ignored without OpenMP support.
