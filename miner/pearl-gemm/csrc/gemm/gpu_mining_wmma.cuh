#pragma once
// FIX #12/#18/#20: WMMA (Tensor Core) path for the tile dot-product.
// Uses mma.sync.aligned.m16n16k16.row.col.s32.s8.s8.s32 available on SM80+.
// Each warp computes a 16x16 output tile in one instruction group instead of
// the scalar loop, giving ~16x throughput vs. DP4A scalar on Blackwell SM120.
// Templated on TILE_M and TILE_N to support 16x16, 32x32, 64x64, and 128x128
// tiles by composing multiple 16x16 WMMA sub-tiles per combo per step.

#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include "gpu_mining_kernel.cuh"

using namespace nvcuda::wmma;

namespace pearl {
namespace mining {

// Base WMMA dimensions (hardware instruction tile).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Templated WMMA mining kernel.
// One warp per (candidate x combo).
// Each combo covers a TILE_M x TILE_N region, processed as
// (TILE_M/16) x (TILE_N/16) WMMA sub-tiles per k-step.
template <int TILE_M, int TILE_N>
__global__ __launch_bounds__(256, 2)
void gpu_mining_wmma_kernel(
    const int8_t* __restrict__ d_a_noised,   // [batch, m, k] int8
    const int8_t* __restrict__ d_b_noised_t, // [batch, n, k] int8 (N-major)
    const int32_t* __restrict__ d_a_rows,     // [P * TILE_M]
    const int32_t* __restrict__ d_b_cols,     // [Q * TILE_N]
    const MiningJob* __restrict__ d_jobs,
    uint32_t* __restrict__ d_jackpots,
    uint32_t* __restrict__ d_hashes,
    uint8_t*  __restrict__ d_winner_flags,
    uint32_t num_candidates,
    uint32_t P,
    uint32_t Q,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t rank) {

    static_assert(TILE_M % WMMA_M == 0, "TILE_M must be a multiple of WMMA_M (16)");
    static_assert(TILE_N % WMMA_N == 0, "TILE_N must be a multiple of WMMA_N (16)");

    constexpr int WMMA_M_TILES = TILE_M / WMMA_M;
    constexpr int WMMA_N_TILES = TILE_N / WMMA_N;

    // One warp per candidate x combo (blockDim.x=256 -> 8 warps per block).
    const uint32_t warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t num_combos = P * Q;
    const uint32_t total_work = num_candidates * num_combos;
    if (warp_id >= total_work) return;

    const uint32_t cid   = warp_id / num_combos;
    const uint32_t combo = warp_id % num_combos;
    const uint32_t p     = combo % P;
    const uint32_t q     = combo / P;
    const uint32_t lane  = threadIdx.x & 31;

    const MiningJob job = d_jobs[cid];
    if (job.status == JOB_STATUS_STOP) return;

    const int8_t* a_base = d_a_noised   + (cid * m * k);
    const int8_t* b_base = d_b_noised_t + (cid * n * k);

    // Base row/col offset for this combo's tile region.
    const int32_t* a_rows_base = d_a_rows + p * TILE_M;
    const int32_t* b_cols_base = d_b_cols + q * TILE_N;

    // Shared memory staging for one 16x16 sub-tile at a time.
    __shared__ __align__(128) int8_t s_a[WMMA_M * WMMA_K]; // 16x16 int8
    __shared__ __align__(128) int8_t s_b[WMMA_K * WMMA_N]; // 16x16 int8

    // Jackpot accumulator (register, warp lane 0 owns it).
    uint32_t jackpot[JACKPOT_SIZE] = {};

    const uint32_t num_steps = k / rank;

    #pragma unroll 1
    for (uint32_t step = 0; step < num_steps; step++) {
        const uint32_t k_off = step * rank;

        // Process each WMMA sub-tile within this combo's tile region.
        for (int wm = 0; wm < WMMA_M_TILES; wm++) {
            for (int wn = 0; wn < WMMA_N_TILES; wn++) {

                // --- Load A sub-tile: gather rows ---
                // 16 threads each load one row of WMMA_K=16 int8 values.
                if (lane < WMMA_M) {
                    const int a_row = a_rows_base[wm * WMMA_M + lane];
                    const int8_t* src = a_base + a_row * k + k_off;
                    *reinterpret_cast<int4*>(&s_a[lane * WMMA_K]) =
                        *reinterpret_cast<const int4*>(src);
                }
                // --- Load B sub-tile: gather cols ---
                if (lane < WMMA_N) {
                    const int b_col = b_cols_base[wn * WMMA_N + lane];
                    const int8_t* src = b_base + b_col * k + k_off;
                    *reinterpret_cast<int4*>(&s_b[lane * WMMA_K]) =
                        *reinterpret_cast<const int4*>(src);
                }
                __syncwarp();

                // --- WMMA compute ---
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag;
                fill_fragment(acc_frag, 0);
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, col_major> b_frag;
                load_matrix_sync(a_frag, s_a, WMMA_K);
                load_matrix_sync(b_frag, s_b, WMMA_K);
                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                __syncwarp();

                // --- XOR-reduce accumulator into jackpot ---
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

    // --- Lane 0: store jackpot, compute BLAKE3, check difficulty ---
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

// Explicit instantiation declarations (defined in a separate .cu or here).
// Forward declarations for launch functions.
extern void launch_gpu_mining_wmma_16(
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
    cudaStream_t stream);

extern void launch_gpu_mining_wmma_32(
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
    cudaStream_t stream);

extern void launch_gpu_mining_wmma_64(
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
    cudaStream_t stream);

extern void launch_gpu_mining_wmma_128(
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
    cudaStream_t stream);

} // namespace mining
} // namespace pearl