#pragma once
// FIX #12/#18/#20: WMMA (Tensor Core) path for the tile dot-product.
// Uses mma.sync.aligned.m16n16k16.row.col.s32.s8.s8.s32 available on SM80+.
// Each warp computes a 16×16 output tile in one instruction group instead of
// the scalar loop, giving ~16× throughput vs. DP4A scalar on Blackwell SM120.

#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include "gpu_mining_kernel.cuh"

using namespace nvcuda::wmma;

namespace pearl {
namespace mining {

// Tile dimensions matched to the m16n16k16 INT8 WMMA shape.
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// FIX #18: fused kernel — computes jackpot accumulation AND BLAKE3 hash in
// the same kernel launch, eliminating the separate gpu_jackpot_hash_kernel.
// FIX #20: uses nvcuda::wmma fragments for warp-level mma.sync instruction.
__global__ __launch_bounds__(256, 2)
void gpu_mining_wmma_kernel(
    const int8_t*  __restrict__ d_a_noised,   // [batch, m, k] int8
    const int8_t*  __restrict__ d_b_noised_t, // [batch, n, k] int8 (N-major)
    const int32_t* __restrict__ d_a_rows,     // [P * WMMA_M]
    const int32_t* __restrict__ d_b_cols,     // [Q * WMMA_N]
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
    uint32_t rank
) {
    // One warp per candidate×combo (blockDim.x=256 → 8 warps per block).
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

    // Row/col indices for this tile (WMMA_M rows, WMMA_N cols).
    const int32_t* a_rows = d_a_rows + p * WMMA_M;
    const int32_t* b_cols = d_b_cols + q * WMMA_N;

    // Accumulator fragment — cleared once per tile.
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag;
    fill_fragment(acc_frag, 0);

    // Shared memory staging for A and B tiles (one k-block at a time).
    __shared__ __align__(128) int8_t s_a[WMMA_M * WMMA_K]; // 16×16 int8
    __shared__ __align__(128) int8_t s_b[WMMA_K * WMMA_N]; // 16×16 int8

    // Jackpot accumulator (register, warp lane 0 owns it).
    uint32_t jackpot[JACKPOT_SIZE] = {};

    const uint32_t num_steps = k / rank;

    #pragma unroll 1
    for (uint32_t step = 0; step < num_steps; step++) {
        const uint32_t k_off = step * rank;

        // --- Load A tile: gather rows ---
        // 16 threads each load one row of WMMA_K=16 int8 values.
        if (lane < WMMA_M) {
            const int a_row = a_rows[lane];
            const int8_t* src = a_base + a_row * k + k_off;
            // 16 bytes per row — 128-bit store.
            *reinterpret_cast<int4*>(&s_a[lane * WMMA_K]) =
                *reinterpret_cast<const int4*>(src);
        }
        // --- Load B tile: gather cols ---
        if (lane < WMMA_N) {
            const int b_col = b_cols[lane];
            const int8_t* src = b_base + b_col * k + k_off;
            *reinterpret_cast<int4*>(&s_b[lane * WMMA_K]) =
                *reinterpret_cast<const int4*>(src);
        }
        __syncwarp();

        // --- WMMA compute ---
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, col_major> b_frag;
        load_matrix_sync(a_frag, s_a, WMMA_K);
        load_matrix_sync(b_frag, s_b, WMMA_K);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();

        // --- XOR-reduce accumulator into jackpot ---
        // Each lane XORs its acc_frag elements; lane 0 gathers via shfl.
        uint32_t partial = 0;
        for (int i = 0; i < acc_frag.num_elements; i++) {
            partial ^= static_cast<uint32_t>(acc_frag.x[i]);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            partial ^= __shfl_down_sync(0xFFFFFFFF, partial, offset);

        if (lane == 0) {
            const int tid = step % JACKPOT_SIZE;
            jackpot[tid] = (jackpot[tid] << 1) | (jackpot[tid] >> 31); // rotl32 by 1
            jackpot[tid] ^= partial;
        }
        __syncwarp();
    }

    // --- Lane 0: store jackpot, compute BLAKE3, check difficulty ---
    if (lane == 0) {
        const uint32_t joff = cid * num_combos * JACKPOT_SIZE + combo * JACKPOT_SIZE;
        for (int i = 0; i < JACKPOT_SIZE; i++) d_jackpots[joff + i] = jackpot[i];

        const uint32_t hoff = cid * num_combos * 8 + combo * 8;
        uint32_t* dst_hash = d_hashes + hoff;
        // Inline BLAKE3 keyed hash (already defined in gpu_mining_kernel.cu
        // as blake3_keyed_hash_inline — declared extern here via the .cuh).
        blake3_keyed_hash_inline(jackpot, job.pow_key, dst_hash);

        if (blake3_check_difficulty(dst_hash, job.pow_target)) {
            const uint32_t flag_idx = cid * num_combos + combo;
            atomicOr(reinterpret_cast<uint32_t*>(
                d_winner_flags + (flag_idx / 32) * 4),
                1u << (flag_idx % 32));
        }
    }
}

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
    cudaStream_t stream);

} // namespace mining
} // namespace pearl
