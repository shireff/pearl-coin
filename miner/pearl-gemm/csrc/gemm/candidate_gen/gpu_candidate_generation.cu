#include "gpu_candidate_generation.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

namespace pearl {
namespace mining {

__global__ void gpu_generate_raw_matrices_kernel(
    int8_t* __restrict__ d_a_raw,
    int8_t* __restrict__ d_b_raw_t,
    uint32_t batch_size,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint64_t base_seed) {

    uint32_t cid = blockIdx.x;
    if (cid >= batch_size) return;

    uint32_t tid = threadIdx.x;
    uint32_t block_size = blockDim.x;

    curandStatePhilox4x32_10_t rng_state;
    uint64_t candidate_seed = base_seed + static_cast<uint64_t>(cid);
    curand_init(candidate_seed, 0, 0, &rng_state);

    int8_t* my_a_raw = d_a_raw + cid * static_cast<size_t>(m) * k;
    int8_t* my_b_raw_t = d_b_raw_t + cid * static_cast<size_t>(n) * k;

    uint32_t a_elements = m * k;
    uint32_t b_t_elements = n * k;

    for (uint32_t idx = tid; idx < a_elements; idx += block_size) {
        uint32_t r = curand(&rng_state);
        int8_t raw_val = static_cast<int8_t>(static_cast<int32_t>(r % 129) - 64);
        my_a_raw[idx] = raw_val;
    }

    for (uint32_t idx = tid; idx < b_t_elements; idx += block_size) {
        uint32_t r = curand(&rng_state);
        int8_t raw_val = static_cast<int8_t>(static_cast<int32_t>(r % 129) - 64);
        my_b_raw_t[idx] = raw_val;
    }
}

cudaError_t launch_gpu_candidate_generation(
    int8_t* d_a_raw,
    int8_t* d_b_raw_t,
    uint32_t batch_size,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint64_t base_seed,
    cudaStream_t stream) {

    if (batch_size == 0 || m == 0 || n == 0 || k == 0) {
        return cudaErrorInvalidValue;
    }

    uint32_t threads_per_block = 256;
    uint32_t grid_size = batch_size;

    gpu_generate_raw_matrices_kernel<<<grid_size, threads_per_block, 0, stream>>>(
        d_a_raw, d_b_raw_t,
        batch_size, m, n, k, base_seed);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

} // namespace mining
} // namespace pearl