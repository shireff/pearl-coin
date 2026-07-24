#include "gpu_mining_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace pearl {
namespace mining {

__device__ __forceinline__ uint32_t rotl32(uint32_t v, int r) {
    return (v << r) | (v >> (32 - r));
}

__global__ void gpu_mining_kernel(
    const int32_t* __restrict__ d_a_noised,
    const int32_t* __restrict__ d_b_noised_t,
    const int32_t* __restrict__ d_a_rows_data,
    const int32_t* __restrict__ d_b_cols_data,
    const MiningJob* __restrict__ d_jobs,
    uint32_t* __restrict__ d_jackpots,
    uint32_t num_candidates,
    uint32_t P,
    uint32_t Q,
    uint32_t tile_h,
    uint32_t tile_w,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t rank
) {
    const uint32_t cid = blockIdx.x;
    if (cid >= num_candidates) return;

    const MiningJob job = d_jobs[cid];
    if (job.status == JOB_STATUS_STOP) return;

    const int32_t* a_noised = d_a_noised + cid * m * k;
    const int32_t* b_noised_t = d_b_noised_t + cid * n * k;

    const uint32_t num_combos = P * Q;
    const uint32_t num_rank_steps = (k - rank) / rank + 1;
    const int tile_size = tile_h * tile_w;

    extern __shared__ int32_t s_tile[];
    uint32_t* s_jackpot = reinterpret_cast<uint32_t*>(s_tile + tile_size);
    __shared__ uint32_t s_warp_xor[32];

    for (uint32_t combo = 0; combo < num_combos; combo++) {
        const uint32_t p = combo % P;
        const uint32_t q = combo / P;

        const int32_t* a_rows = d_a_rows_data + p * tile_h;
        const int32_t* b_cols = d_b_cols_data + q * tile_w;

        for (uint32_t step = 0; step < num_rank_steps; step++) {
            const int ll = rank + step * rank;

            for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
                const int u = idx % tile_h;
                const int v = idx / tile_h;
                const int a_idx = a_rows[u];
                const int b_idx = b_cols[v];
                int32_t sum = 0;
                for (int l = ll - rank; l < ll; l++) {
                    sum += a_noised[a_idx * k + l] * b_noised_t[b_idx * k + l];
                }
                s_tile[idx] = sum;
            }
            __syncthreads();

            uint32_t partial = 0;
            for (int idx = threadIdx.x; idx < tile_size; idx += blockDim.x) {
                partial ^= static_cast<uint32_t>(s_tile[idx]);
            }

            uint32_t warp_xor = partial;
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_xor ^= __shfl_down_sync(0xFFFFFFFF, warp_xor, offset);
            }

            if (threadIdx.x % 32 == 0) {
                s_warp_xor[threadIdx.x / 32] = warp_xor;
            }
            __syncthreads();

            uint32_t xored = 0;
            if (threadIdx.x == 0) {
                const int num_warps = (blockDim.x + 31) / 32;
                for (int i = 0; i < num_warps; i++) {
                    xored ^= s_warp_xor[i];
                }
                const int tid = step % JACKPOT_SIZE;
                s_jackpot[tid] = rotl32(s_jackpot[tid], 1) ^ xored;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            uint32_t* dst = d_jackpots + cid * num_combos * JACKPOT_SIZE + combo * JACKPOT_SIZE;
            for (int i = 0; i < JACKPOT_SIZE; i++) {
                dst[i] = s_jackpot[i];
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < JACKPOT_SIZE; i += blockDim.x) {
            s_jackpot[i] = 0;
        }
        __syncthreads();
    }
}

} // namespace mining
} // namespace pearl
