#pragma once
#include <concepts>
#include <memory>

namespace transformers_lite {

/**
 * @brief CRTP base for all layer types.
 *
 * Provides the three standard type aliases (value_type, compute, ptr) that
 * every layer must expose. Copy is deleted — layers own weights by reference
 * and should not be silently shallow-copied. Move is defaulted so layers can
 * be stored in containers via std::move.
 *
 * @tparam COMPUTE compute backend (e.g. CPU, CUDA, AVX512)
 * @tparam T       element type (e.g. float)
 * @tparam Derived the concrete layer class (CRTP parameter)
 */
template <template <class> class COMPUTE, class T, typename Derived> class LayerBase {
 public:
    using value_type = T;
    using compute = COMPUTE<T>;
    using ptr = std::unique_ptr<Derived>;

    LayerBase() = default;
    LayerBase(LayerBase &&) noexcept = default;
    LayerBase &operator=(LayerBase &&) noexcept = default;

    LayerBase(const LayerBase &) = delete;
    LayerBase &operator=(const LayerBase &) = delete;

 protected:
    ~LayerBase() = default; // non-virtual: never deleted through a base pointer
};

// ── Concepts ──────────────────────────────────────────────────────────────────

/**
 * @brief Satisfied by any type that exposes the standard layer type aliases.
 */
template <typename L>
concept AnyLayer = requires {
    typename L::value_type;
    typename L::compute;
    typename L::ptr;
};

/**
 * @brief Satisfied when L::forward is callable with the given argument types.
 *
 * Usage examples:
 *   LayerWith<Linear<CPU,float>, const Tensor<CPU,float>&, Tensor<CPU,float>&>
 *   LayerWith<TransformerBlock<CPU,float>, Tensor<CPU,float>&, int>
 */
template <typename L, typename... Args>
concept LayerWith = AnyLayer<L> && requires(L &l, Args... args) {
    { l.forward(args...) } -> std::same_as<void>;
};

} // namespace transformers_lite
