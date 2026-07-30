#pragma once
// FIX #12/#18/#20: WMMA (Tensor Core) path for the tile dot-product.
// Uses mma.sync.aligned.m16n16k16.row.col.s32.s8.s8.s32 available on SM80+.
// Each warp computes a 16x16 output tile in one instruction group instead of
// the scalar loop, giving ~16x throughput vs. DP4A scalar on Blackwell SM120.
// Templated on TILE_M and TILE_N to support multiple tile sizes.
// Each combo covers a TILE_MxTILE_N region processed as
// (TILE_M/16)x(TILE_N/16) WMMA sub-tiles per k-step.

#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include "gpu_mining_kernel.cuh"

using namespace nvcuda::wmma;

namespace pearl {
namespace mining {

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

template <int TILE_M, int TILE_N>
__global__ __launch_bounds__(256, 2)
void gpu_mining_wmma_kernel(
    const int8_t* __restrict__ d_a_noised,
    const int8_t* __restrict__ d_b_noised_t,
    const int32_t* __restrict__ d_a_rows,
    const int32_t* __restrict__ d_b_cols,
    const MiningJob* __restrict__ d_jobs,
    uint32_t* __restrict__ d_jackpots,
    uint32_t* __restrict__ d_hashes,
    uint8_t* __restrict__ d_winner_flags,
    uint32_t num_candidates,
    uint32_t P, uint32_t Q,
    uint32_t m, uint32_t n, uint32_t k, uint32_t rank) {

    static_assert(TILE_M % WMMA_M == 0, "TILE_M must be a multiple of WMMA_M (16)");
    static_assert(TILE_N % WMMA_N == 0, "TILE_N must be a multiple of WMMA_N (16)");

    constexpr int WMMA_M_TILES = TILE_M / WMMA_M;
    constexpr int WMMA_N_TILES = TILE_N / WMMA_N;

    const uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t num_combos = P * Q;
    const uint32_t total_work = num_candidates * num_combos;
    if (warp_id >= total_work) return;

    const uint32_t cid = warp_id / num_combos;
    const uint32_t combo = warp_id % num_combos;
    const uint32_t p = combo % P;
    const uint32_t q = combo / P;
    const uint32_t lane = threadIdx.x & 31;

    const MiningJob job = d_jobs[cid];
    if (job.status == JOB_STATUS_STOP) return;

    const int8_t* a_base = d_a_noised + (cid * m * k);
    const int8_t* b_base = d_b_noised_t + (cid * n * k);

    const int32_t* a_rows_base = d_a_rows + p * TILE_M;
    const int32_t* b_cols_base = d_b_cols + q * TILE_N;

    __shared__ __align__(128) int8_t s_a[WMMA_M * WMMA_K];
    __shared__ __align__(128) int8_t s_b[WMMA_K * WMMA_N];

    uint32_t jackpot[JACKPOT_SIZE] = {};
    const uint32_t num_steps = k / rank;

    #pragma unroll 1
    for (uint32_t step = 0; step < num_steps; step++) {
        const uint32_t k_off = step * rank;

        for (int wm = 0; wm < WMMA_M_TILES; wm++) {
            for (int wn = 0; wn < WMMA_N_TILES; wn++) {
                if (lane < WMMA_M) {
                    const int a_row = a_rows_base[wm * WMMA_M + lane];
                    const int8_t* src = a_base + a_row * k + k_off;
                    *reinterpret_cast<int4*>(&s_a[lane * WMMA_K]) =
                        *reinterpret_cast<const int4*>(src);
                }
                if (lane < WMMA_N) {
                    const int b_col = b_cols_base[wn * WMMA_N + lane];
                    const int8_t* src = b_base + b_col * k + k_off;
                    *reinterpret_cast<int4*>(&s_b[lane * WMMA_K]) =
                        *reinterpret_cast<const int4*>(src);
                }
                __syncwarp();

                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag;
                fill_fragment(acc_frag, 0);
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, col_major> b_frag;
                load_matrix_sync(a_frag, s_a, WMMA_K);
                load_matrix_sync(b_frag, s_b, WMMA_K);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                __syncwarp();

                uint32_t partial = 0;
                for (int i = 0; i < acc_frag.num_elements; i++) {
                    partial ^= static_cast<uint32_t>(acc_frag.x[i]);
                }
                for (int offset = 16; offset > 0; offset >>= 1)
                    partial ^= __shfl_down_sync(0xFFFFFFFF, partial, offset);

                if (lane == 0) {
                    const int tid = step % JACKPOT_SIZE;
                    jackpot[tid] = (jackpot[tid] << 1) | (jackpot[tid] >> 31);
                    jackpot[tid] ^= partial;
                }
                __syncwarp();
            }
        }
    }

    if (lane == 0) {
        const uint32_t joff = cid * num_combos * JACKPOT_SIZE + combo * JACKPOT_SIZE;
        for (int i = 0; i < JACKPOT_SIZE; i++) d_jackpots[joff + i] = jackpot[i];

        const uint32_t hoff = cid * num_combos * 8 + combo * 8;
        uint32_t* dst_hash = d_hashes + hoff;
        blake3_keyed_hash_inline(jackpot, job.pow_key, dst_hash);

        if (blake3_check_difficulty(dst_hash, job.pow_target)) {
            const uint32_t flag_idx = cid * num_combos + combo;
            atomicOr(reinterpret_cast<uint32_t*>(
                d_winner_flags + (flag_idx / 32) * 4),
                1u << (flag_idx % 32));
        }
    }
}

// Inline launch wrappers — defined here so NVCC can see the kernel launch
// and instantiate the template in the same translation unit that calls them.

inline void launch_gpu_mining_wmma_16(
    const int8_t* d_a_noised,
    const int8_t* d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols,
    const MiningJob* d_jobs,
    uint32_t* d_jackpots,
    uint32_t* d_hashes,
    uint8_t* d_winner_flags,
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

inline void launch_gpu_mining_wmma_32(
    const int8_t* d_a_noised,
    const int8_t* d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols,
    const MiningJob* d_jobs,
    uint32_t* d_jackpots,
    uint32_t* d_hashes,
    uint8_t* d_winner_flags,
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

inline void launch_gpu_mining_wmma_64(
    const int8_t* d_a_noised,
    const int8_t* d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols,
    const MiningJob* d_jobs,
    uint32_t* d_jackpots,
    uint32_t* d_hashes,
    uint8_t* d_winner_flags,
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

inline void launch_gpu_mining_wmma_128(
    const int8_t* d_a_noised,
    const int8_t* d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols,
    const MiningJob* d_jobs,
    uint32_t* d_jackpots,
    uint32_t* d_hashes,
    uint8_t* d_winner_flags,
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