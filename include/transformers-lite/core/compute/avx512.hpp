#pragma once
#include <immintrin.h>

#include "xpu.hpp"

namespace transformers_lite {

/**
 * @brief AVX-512 compute backend.
 *
 * @tparam T element data type
 */
template <class T> struct AVX512 : public XPU {
    // @TODO avx512
};

} // namespace transformers_lite
