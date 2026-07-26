#include "gpu_mining_kernel.cuh"
#include "gpu_mining_wmma.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace pearl {
namespace mining {

void launch_gpu_mining(
    const int32_t* a_noised,
    const int32_t* b_noised_t,
    const int32_t* a_rows_data,
    const int32_t* b_cols_data,
    const MiningJob* jobs,
    uint32_t* jackpots,
    uint32_t* hashes,
    uint8_t* winner_flags,
    uint32_t num_candidates,
    uint32_t P,
    uint32_t Q,
    uint32_t tile_h,
    uint32_t tile_w,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t rank,
    cudaStream_t stream) {

    // FIX #16: attempt zero-copy registration of the jobs descriptor buffer
    // so the GPU DMA engine reads directly from pinned host memory without a
    // separate cudaMemcpy. Falls back silently if already device memory.
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, jobs) == cudaSuccess &&
        attr.type == cudaMemoryTypeHost) {
        cudaHostRegister(
            const_cast<MiningJob*>(jobs),
            num_candidates * sizeof(MiningJob),
            cudaHostRegisterMapped | cudaHostRegisterPortable);
    }

    const MiningJob* d_jobs = jobs;
    if (attr.type == cudaMemoryTypeHost) {
        cudaHostGetDevicePointer(
            reinterpret_cast<void**>(const_cast<MiningJob**>(&d_jobs)),
            const_cast<MiningJob*>(jobs), 0);
    }

    const int tile_size = tile_h * tile_w;
    const int smem_a_pitch = tile_h * (rank + 1);
    const int smem_b_pitch = (rank + 1) * tile_w;
    const size_t shared_mem_bytes =
        (tile_size + smem_a_pitch + smem_b_pitch + JACKPOT_SIZE) * sizeof(int32_t);

    // FIX #12/#13/#17/#18: prefer WMMA fused kernel when tile fits 16×16;
    // it uses Tensor Cores, async-copy smem loads, and fuses hash into the
    // same kernel, eliminating the separate gpu_jackpot_hash_kernel launch.
    // FIX #3/#11/#19: 256 threads saturates Blackwell SM warp slots.
    if (tile_h == 16 && tile_w == 16) {
        launch_gpu_mining_wmma(
            reinterpret_cast<const int8_t*>(a_noised),
            reinterpret_cast<const int8_t*>(b_noised_t),
            a_rows_data, b_cols_data,
            jobs, jackpots, hashes, winner_flags,
            num_candidates, P, Q, m, n, k, rank, stream);
        return;
    }

    // Scalar fallback for non-16×16 tiles.
    constexpr int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid(num_candidates);

    gpu_mining_kernel<<<grid, block, shared_mem_bytes, stream>>>(
        a_noised, b_noised_t, a_rows_data, b_cols_data,
        jobs, jackpots, hashes, winner_flags,
        num_candidates, P, Q, tile_h, tile_w,
        m, n, k, rank
    );
}

void launch_gpu_jackpot_hash(
    const uint32_t* jackpots,
    const uint32_t* keys,
    uint32_t* hashes,
    uint32_t num_candidates,
    uint32_t num_combos,
    cudaStream_t stream) {

    const uint32_t total = num_candidates * num_combos;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    gpu_jackpot_hash_kernel<<<blocks, threads, 0, stream>>>(
        jackpots, keys, hashes, num_candidates, num_combos
    );
}

} // namespace mining
} // namespace pearl
