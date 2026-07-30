#include "gpu_mining_wmma.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace pearl {
namespace mining {

// Explicit instantiations + launch wrappers for each supported tile size.
// The template kernel is defined in gpu_mining_wmma.cuh.

template __global__ __launch_bounds__(256, 2)
void gpu_mining_wmma_kernel<16, 16>(
    const int8_t* __restrict__ d_a_noised,
    const int8_t* __restrict__ d_b_noised_t,
    const int32_t* __restrict__ d_a_rows,
    const int32_t* __restrict__ d_b_cols,
    const MiningJob* __restrict__ d_jobs,
    uint32_t* __restrict__ d_jackpots,
    uint32_t* __restrict__ d_hashes,
    uint8_t*  __restrict__ d_winner_flags,
    uint32_t num_candidates,
    uint32_t P, uint32_t Q,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
    cudaStream_t stream);

template __global__ __launch_bounds__(256, 2)
void gpu_mining_wmma_kernel<32, 32>(
    const int8_t* __restrict__ d_a_noised,
    const int8_t* __restrict__ d_b_noised_t,
    const int32_t* __restrict__ d_a_rows,
    const int32_t* __restrict__ d_b_cols,
    const MiningJob* __restrict__ d_jobs,
    uint32_t* __restrict__ d_jackpots,
    uint32_t* __restrict__ d_hashes,
    uint8_t*  __restrict__ d_winner_flags,
    uint32_t num_candidates,
    uint32_t P, uint32_t Q,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
    cudaStream_t stream);

template __global__ __launch_bounds__(256, 2)
void gpu_mining_wmma_kernel<64, 64>(
    const int8_t* __restrict__ d_a_noised,
    const int8_t* __restrict__ d_b_noised_t,
    const int32_t* __restrict__ d_a_rows,
    const int32_t* __restrict__ d_b_cols,
    const MiningJob* __restrict__ d_jobs,
    uint32_t* __restrict__ d_jackpots,
    uint32_t* __restrict__ d_hashes,
    uint8_t*  __restrict__ d_winner_flags,
    uint32_t num_candidates,
    uint32_t P, uint32_t Q,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
    cudaStream_t stream);

template __global__ __launch_bounds__(256, 2)
void gpu_mining_wmma_kernel<128, 128>(
    const int8_t* __restrict__ d_a_noised,
    const int8_t* __restrict__ d_b_noised_t,
    const int32_t* __restrict__ d_a_rows,
    const int32_t* __restrict__ d_b_cols,
    const MiningJob* __restrict__ d_jobs,
    uint32_t* __restrict__ d_jackpots,
    uint32_t* __restrict__ d_hashes,
    uint8_t*  __restrict__ d_winner_flags,
    uint32_t num_candidates,
    uint32_t P, uint32_t Q,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
    cudaStream_t stream);

void launch_gpu_mining_wmma_16(
    const int8_t*  d_a_noised,
    const int8_t*  d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols,
    const MiningJob* d_jobs,
    uint32_t* d_jackpots,
    uint32_t* d_hashes,
    uint8_t*  d_winner_flags,
    uint32_t num_candidates,
    uint32_t P, uint32_t Q,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
    cudaStream_t stream) {
    constexpr int TM = 16;
    constexpr int TN = 16;
    const uint32_t total_warps = num_candidates * P * Q;
    constexpr int THREADS = 256;
    const int WARPS_PER_BLOCK = THREADS / 32;
    const int grid = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gpu_mining_wmma_kernel<TM, TN><<<grid, THREADS, 0, stream>>>(
        d_a_noised, d_b_noised_t,
        d_a_rows, d_b_cols,
        d_jobs, d_jackpots, d_hashes, d_winner_flags,
        num_candidates, P, Q, m, n, k, rank);
}

void launch_gpu_mining_wmma_32(
    const int8_t*  d_a_noised,
    const int8_t*  d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols,
    const MiningJob* d_jobs,
    uint32_t* d_jackpots,
    uint32_t* d_hashes,
    uint8_t*  d_winner_flags,
    uint32_t num_candidates,
    uint32_t P, uint32_t Q,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
    cudaStream_t stream) {
    constexpr int TM = 32;
    constexpr int TN = 32;
    const uint32_t total_warps = num_candidates * P * Q;
    constexpr int THREADS = 256;
    const int WARPS_PER_BLOCK = THREADS / 32;
    const int grid = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gpu_mining_wmma_kernel<TM, TN><<<grid, THREADS, 0, stream>>>(
        d_a_noised, d_b_noised_t,
        d_a_rows, d_b_cols,
        d_jobs, d_jackpots, d_hashes, d_winner_flags,
        num_candidates, P, Q, m, n, k, rank);
}

void launch_gpu_mining_wmma_64(
    const int8_t*  d_a_noised,
    const int8_t*  d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols,
    const MiningJob* d_jobs,
    uint32_t* d_jackpots,
    uint32_t* d_hashes,
    uint8_t*  d_winner_flags,
    uint32_t num_candidates,
    uint32_t P, uint32_t Q,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
    cudaStream_t stream) {
    constexpr int TM = 64;
    constexpr int TN = 64;
    const uint32_t total_warps = num_candidates * P * Q;
    constexpr int THREADS = 256;
    const int WARPS_PER_BLOCK = THREADS / 32;
    const int grid = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gpu_mining_wmma_kernel<TM, TN><<<grid, THREADS, 0, stream>>>(
        d_a_noised, d_b_noised_t,
        d_a_rows, d_b_cols,
        d_jobs, d_jackpots, d_hashes, d_winner_flags,
        num_candidates, P, Q, m, n, k, rank);
}

void launch_gpu_mining_wmma_128(
    const int8_t*  d_a_noised,
    const int8_t*  d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols,
    const MiningJob* d_jobs,
    uint32_t* d_jackpots,
    uint32_t* d_hashes,
    uint8_t*  d_winner_flags,
    uint32_t num_candidates,
    uint32_t P, uint32_t Q,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
    cudaStream_t stream) {
    constexpr int TM = 128;
    constexpr int TN = 128;
    const uint32_t total_warps = num_candidates * P * Q;
    constexpr int THREADS = 256;
    const int WARPS_PER_BLOCK = THREADS / 32;
    const int grid = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gpu_mining_wmma_kernel<TM, TN><<<grid, THREADS, 0, stream>>>(
        d_a_noised, d_b_noised_t,
        d_a_rows, d_b_cols,
        d_jobs, d_jackpots, d_hashes, d_winner_flags,
        num_candidates, P, Q, m, n, k, rank);
}

} // namespace mining
} // namespace pearl