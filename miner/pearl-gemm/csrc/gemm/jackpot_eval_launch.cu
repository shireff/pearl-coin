#include "jackpot_eval.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace pearl {
namespace jackpot {

void launch_evaluate_tile(
    const int32_t* a_noised,
    const int32_t* b_noised_t,
    const int32_t* a_rows,
    const int32_t* b_cols,
    int m, int n, int k, int rank,
    int tile_h, int tile_w,
    uint32_t* jackpot_tile,
    uint32_t* jackpot,
    cudaStream_t stream) {

    dim3 block(16, 16);
    dim3 grid((tile_w + block.x - 1) / block.x,
              (tile_h + block.y - 1) / block.y);

    evaluate_tile_kernel<<<grid, block, 0, stream>>>(
        a_noised, b_noised_t, a_rows, b_cols,
        m, n, k, rank, tile_h, tile_w,
        jackpot_tile, jackpot
    );
}

void launch_accumulate_jackpot(
    const uint32_t* jackpot_tile,
    int tile_h, int tile_w,
    uint32_t* jackpot,
    int rank_idx,
    cudaStream_t stream) {

    int num_elements = tile_h * tile_w;
    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);

    accumulate_jackpot_kernel<<<grid, block, 0, stream>>>(
        jackpot_tile, tile_h, tile_w, jackpot, rank_idx, 1
    );
}

void launch_fused_evaluate_accumulate(
    const int32_t* a_noised,
    const int32_t* b_noised_t,
    const int32_t* a_rows,
    const int32_t* b_cols,
    int m, int n, int k, int rank,
    int tile_h, int tile_w,
    uint32_t* jackpot,
    cudaStream_t stream) {

    dim3 block(JACKPOT_SIZE);
    dim3 grid(1);

    fused_evaluate_accumulate_kernel<<<grid, block, 0, stream>>>(
        a_noised, b_noised_t, a_rows, b_cols,
        m, n, k, rank, tile_h, tile_w, jackpot
    );
}

} // namespace jackpot
} // namespace pearl
