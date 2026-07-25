#include "gpu_mining_kernel.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace pearl {
namespace mining {

void launch_gpu_mining(
    const int32_t* a_noised,
    const int32_t* b_noised_t,
    const int32_t* a_rows_data,
    const int32_t* b_cols_data,
    const MiningJob* jobs,
    uint32_t* jackpots,
    uint32_t num_candidates,
    uint32_t P,
    uint32_t Q,
    uint32_t tile_h,
    uint32_t tile_w,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t rank,
    cudaStream_t stream) {

    const int tile_size = tile_h * tile_w;
    const size_t shared_mem_bytes = (tile_size + JACKPOT_SIZE) * sizeof(int32_t);

    dim3 block(128);
    dim3 grid(num_candidates);

    gpu_mining_kernel<<<grid, block, shared_mem_bytes, stream>>>(
        a_noised, b_noised_t, a_rows_data, b_cols_data,
        jobs, jackpots,
        num_candidates, P, Q, tile_h, tile_w,
        m, n, k, rank
    );
}

void launch_gpu_jackpot_hash(
    const uint32_t* jackpots,
    const uint32_t* keys,
    uint32_t* hashes,
    uint32_t num_candidates,
    uint32_t num_combos,
    cudaStream_t stream) {

    const uint32_t total = num_candidates * num_combos;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    gpu_jackpot_hash_kernel<<<blocks, threads, 0, stream>>>(
        jackpots, keys, hashes, num_candidates, num_combos
    );
}

} // namespace mining
} // namespace pearl
