#include "gpu_mining_wmma.cuh"
#include <cuda_runtime.h>

namespace pearl {
namespace mining {

void launch_gpu_mining_wmma(
    const int8_t*  d_a_noised,
    const int8_t*  d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols,
    const MiningJob* d_jobs,
    uint32_t* d_jackpots,
    uint32_t* d_hashes,
    uint8_t*  d_winner_flags,
    uint32_t num_candidates,
    uint32_t P,
    uint32_t Q,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t rank,
    cudaStream_t stream)
{
    // FIX #12/#18/#19: one warp per (candidate, combo); 256 threads = 8 warps.
    // grid covers all work so every SM is occupied.
    const uint32_t total_warps = num_candidates * P * Q;
    constexpr int THREADS = 256;
    const int WARPS_PER_BLOCK = THREADS / 32;
    const int grid = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    gpu_mining_wmma_kernel<<<grid, THREADS, 0, stream>>>(
        d_a_noised, d_b_noised_t,
        d_a_rows, d_b_cols,
        d_jobs, d_jackpots, d_hashes, d_winner_flags,
        num_candidates, P, Q, m, n, k, rank
    );
}

} // namespace mining
} // namespace pearl
