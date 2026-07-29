#pragma once

#include <cstdint>
#include <cstddef>

namespace pearl {
namespace jackpot {

constexpr int JACKPOT_SIZE = 16;
constexpr int LROT_PER_TILE = 1;

__global__ void evaluate_tile_kernel(
    const int32_t* __restrict__ a_noised,
    const int32_t* __restrict__ b_noised_t,
    const int32_t* __restrict__ a_rows,
    const int32_t* __restrict__ b_cols,
    int m, int n, int k, int rank,
    int tile_h, int tile_w,
    uint32_t* __restrict__ jackpot_tile,
    uint32_t* __restrict__ jackpot);

__global__ void accumulate_jackpot_kernel(
    const uint32_t* __restrict__ jackpot_tile,
    int tile_h, int tile_w,
    uint32_t* __restrict__ jackpot,
    int rank_idx, int num_ranks);

__global__ void fused_evaluate_accumulate_kernel(
    const int32_t* __restrict__ a_noised,
    const int32_t* __restrict__ b_noised_t,
    const int32_t* __restrict__ a_rows,
    const int32_t* __restrict__ b_cols,
    int m, int n, int k, int rank,
    int tile_h, int tile_w,
    uint32_t* __restrict__ jackpot);

void launch_evaluate_tile(
    const int32_t* a_noised,
    const int32_t* b_noised_t,
    const int32_t* a_rows,
    const int32_t* b_cols,
    int m, int n, int k, int rank,
    int tile_h, int tile_w,
    uint32_t* jackpot_tile,
    uint32_t* jackpot,
    cudaStream_t stream);

void launch_accumulate_jackpot(
    const uint32_t* jackpot_tile,
    int tile_h, int tile_w,
    uint32_t* jackpot,
    int rank_idx,
    cudaStream_t stream);

void launch_fused_evaluate_accumulate(
    const int32_t* a_noised,
    const int32_t* b_noised_t,
    const int32_t* a_rows,
    const int32_t* b_cols,
    int m, int n, int k, int rank,
    int tile_h, int tile_w,
    uint32_t* jackpot,
    cudaStream_t stream);

} // namespace jackpot
} // namespace pearl
