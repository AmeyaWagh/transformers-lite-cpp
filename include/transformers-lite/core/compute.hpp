#pragma once

#include "compute/cpu.hpp"

#ifdef TRANSFORMERS_CUDA_ENABLED
#include "compute/cuda.hpp"
#endif

#ifdef TRANSFORMERS_AVX512_ENABLED
#include "compute/avx512.hpp"
#endif
