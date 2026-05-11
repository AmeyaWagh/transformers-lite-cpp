#pragma once
#include <concepts>
#include <cstddef>

namespace transformers_lite {

/** @brief Base tag type for compute backends. */
struct XPU {};

/**
 * @brief Concept constraining a compute backend instantiated for element type T.
 *
 * A conforming backend (e.g. CPU<T>) must expose:
 *   - allocator_type          — an allocator for T
 *   - fill(T*, size_t, T)     — fill a range with a value
 *   - copy(const T*, T*, size_t) — copy elements src → dest
 *   - get(T*, size_t) -> T&   — indexed element access
 */
template <class Backend, class T>
concept ComputeBackend = requires(T *ptr, const T *cptr, std::size_t len, T val, std::size_t idx) {
    typename Backend::allocator_type;
    { Backend::fill(ptr, len, val) } -> std::same_as<void>;
    { Backend::copy(cptr, ptr, len) } -> std::same_as<void>;
    { Backend::get(ptr, idx) } -> std::same_as<T &>;
};

} // namespace transformers_lite
