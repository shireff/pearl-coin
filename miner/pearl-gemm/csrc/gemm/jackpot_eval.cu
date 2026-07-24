#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace pearl {
namespace jackpot {

constexpr int JACKPOT_SIZE = 16;
constexpr int LROT_PER_TILE = 1;

__device__ __forceinline__ uint32_t rotl32(uint32_t v, int r) {
    return (v << r) | (v >> (32 - r));
}

__global__ void evaluate_tile_kernel(
    const int32_t* __restrict__ a_noised,  // m x k
    const int32_t* __restrict__ b_noised_t, // n x k
    const int32_t* __restrict__ a_rows,     // tile_h
    const int32_t* __restrict__ b_cols,     // tile_w
    int m, int n, int k, int rank,
    int tile_h, int tile_w,
    uint32_t* __restrict__ jackpot_tile,    // tile_h x tile_w
    uint32_t* __restrict__ jackpot          // 16
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;

    if (u >= tile_h || v >= tile_w) return;

    int a_idx = a_rows[u];
    int b_idx = b_cols[v];

    int32_t sum = 0;
    for (int l = 0; l < rank; l++) {
        sum += a_noised[a_idx * k + l] * b_noised_t[b_idx * k + l];
    }
    jackpot_tile[u * tile_w + v] = static_cast<uint32_t>(sum);
}

__global__ void accumulate_jackpot_kernel(
    const uint32_t* __restrict__ jackpot_tile, // tile_h x tile_w
    int tile_h, int tile_w,
    uint32_t* __restrict__ jackpot,            // 16
    int rank_idx, int num_ranks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= JACKPOT_SIZE) return;

    uint32_t xored = 0;
    for (int u = 0; u < tile_h; u++) {
        for (int v = 0; v < tile_w; v++) {
            xored ^= jackpot_tile[u * tile_w + v];
        }
    }

    uint32_t current = jackpot[tid];
    current = rotl32(current, LROT_PER_TILE) ^ xored;
    jackpot[tid] = current;
}

__global__ void fused_evaluate_accumulate_kernel(
    const int32_t* __restrict__ a_noised,  // m x k
    const int32_t* __restrict__ b_noised_t, // n x k
    const int32_t* __restrict__ a_rows,     // tile_h
    const int32_t* __restrict__ b_cols,     // tile_w
    int m, int n, int k, int rank,
    int tile_h, int tile_w,
    uint32_t* __restrict__ jackpot          // 16
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= JACKPOT_SIZE) return;

    int u = tid / tile_w;
    int v = tid % tile_w;

    if (u >= tile_h || v >= tile_w) return;

    int a_idx = a_rows[u];
    int b_idx = b_cols[v];

    int32_t sum = 0;
    for (int l = 0; l < rank; l++) {
        sum += a_noised[a_idx * k + l] * b_noised_t[b_idx * k + l];
    }

    uint32_t xored = 0;
    for (int u2 = 0; u2 < tile_h; u2++) {
        for (int v2 = 0; v2 < tile_w; v2++) {
            int a_idx2 = a_rows[u2];
            int b_idx2 = b_cols[v2];
            int32_t sum2 = 0;
            for (int l = 0; l < rank; l++) {
                sum2 += a_noised[a_idx2 * k + l] * b_noised_t[b_idx2 * k + l];
            }
            xored ^= static_cast<uint32_t>(sum2);
        }
    }

    uint32_t current = jackpot[tid];
    current = rotl32(current, LROT_PER_TILE) ^ xored;
    jackpot[tid] = current;
}

} // namespace jackpot
} // namespace pearl
