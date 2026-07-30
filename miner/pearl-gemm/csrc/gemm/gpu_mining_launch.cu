    #include "gpu_mining_kernel.cuh"
    #include "gpu_mining_wmma.cuh"
    #include <cuda_runtime.h>
    #include <cassert>
    #include <cstdio>

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
        auto align4 = [](int x) { return (x + 3) & ~3; };
        const int smem_a_pitch = tile_h * align4(rank + 1);
        const int smem_b_pitch = align4(rank + 1) * tile_w;
        const size_t shared_mem_bytes =
            (tile_size + smem_a_pitch + smem_b_pitch + JACKPOT_SIZE) * sizeof(int32_t);

        // Non-invasive runtime logging: print the final runtime values and decision
        printf("[pearl_gemm] launch_gpu_mining: num_candidates=%u P=%u Q=%u tile_h=%u tile_w=%u m=%u n=%u k=%u rank=%u shared_mem_bytes=%zu\n",
            num_candidates, P, Q, tile_h, tile_w, m, n, k, rank, shared_mem_bytes);

        // FIX #12/#13/#17/#18: prefer WMMA fused kernel when tile fits 16×16;
        // it uses Tensor Cores, async-copy smem loads, and fuses hash into the
        // same kernel, eliminating the separate gpu_jackpot_hash_kernel launch.
        // FIX #3/#11/#19: 256 threads saturates Blackwell SM warp slots.
        if (tile_h == 16 && tile_w == 16) {
            printf("[pearl_gemm] launch decision: FUSED WMMA path selected (launch_gpu_mining_wmma)\n");
            launch_gpu_mining_wmma(
                reinterpret_cast<const int8_t*>(a_noised),
                reinterpret_cast<const int8_t*>(b_noised_t),
                a_rows_data, b_cols_data,
                jobs, jackpots, hashes, winner_flags,
                num_candidates, P, Q, m, n, k, rank, stream);
            return;
        }
        // Scalar fallback for non-16×16 tiles.
        printf("[pearl_gemm] launch decision: SCALAR fallback selected (gpu_mining_kernel) - tile mismatch or unsupported fused specialization\n");
        constexpr int BLOCK = 256;
        dim3 block(BLOCK);
        dim3 grid(num_candidates);

        // Opt-in to extended dynamic shared memory (required for > 48 KB).
        // Without this, CUDA silently rejects any shared_mem_bytes > 48 KB
        // with cudaErrorInvalidValue.
        if (shared_mem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(
                gpu_mining_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared_mem_bytes)
            );
        }

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
        // Log the hash kernel launch parameters
        printf("[pearl_gemm] launch_gpu_jackpot_hash: total_hashes=%u num_combos=%u threads=%d blocks=%d\n",
            total, num_combos, threads, blocks);
gpu_jackpot_hash_kernel<<<blocks, threads, 0, stream>>>(
        jackpots, keys, hashes, num_candidates, num_combos
    );
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
    cudaStream_t stream) {

    const uint32_t total_warps = num_candidates * P * Q;
    constexpr int THREADS = 256;
    const int WARPS_PER_BLOCK = THREADS / 32;
    const int grid = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    gpu_mining_wmma_kernel<<<grid, THREADS, 0, stream>>>(
        d_a_noised, d_b_noised_t,
        d_a_rows, d_b_cols,
        d_jobs, d_jackpots, d_hashes, d_winner_flags,
        num_candidates, P, Q, m, n, k, rank
    );
}

} // namespace mining
} // namespace pearl
