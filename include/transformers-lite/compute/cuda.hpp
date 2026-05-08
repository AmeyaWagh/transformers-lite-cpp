#pragma once
#include <cuda_runtime.h>

#include "xpu.hpp"

namespace transformers_lite {

/**
 * @brief CUDA compute backend.
 *
 * @tparam T element data type
 */
template <class T> struct CUDA : public XPU {
    // @TODO implement cuda allocator
};

} // namespace transformers_lite
