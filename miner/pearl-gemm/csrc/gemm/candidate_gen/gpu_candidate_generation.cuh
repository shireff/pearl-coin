#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pearl {
namespace mining {

cudaError_t launch_gpu_candidate_generation(
    int8_t* d_a_raw,
    int8_t* d_b_raw_t,
    uint32_t batch_size,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint64_t base_seed,
    cudaStream_t stream);

} // namespace mining
} // namespace pearl