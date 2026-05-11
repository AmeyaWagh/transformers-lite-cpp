#pragma once

namespace transformers_lite {

/**
 * @brief Primary dispatch struct for compute ops.
 *
 * Specialize for each backend in the corresponding *_ops.hpp header.
 * Expression templates call Ops<COMPUTE, T>::method(...) so that op
 * implementations are fully decoupled from the memory-management backend.
 */
template <template <class> class COMPUTE, typename T> struct Ops;

} // namespace transformers_lite
