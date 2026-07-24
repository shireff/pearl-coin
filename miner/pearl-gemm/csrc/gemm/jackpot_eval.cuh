#pragma once

#include <cstdint>
#include <cstddef>

namespace pearl {
namespace jackpot {

constexpr int JACKPOT_SIZE = 16;
constexpr int LROT_PER_TILE = 1;

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
