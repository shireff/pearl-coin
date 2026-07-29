#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace pearl {
namespace mining {

// CUDA Graph handle for batch mining
struct MiningGraphState {
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    bool captured;
    bool instantiated;
    uint32_t last_num_candidates;
    uint32_t last_num_combos;
    uint32_t last_tile_h;
    uint32_t last_tile_w;
};

// Create and capture CUDA Graph for batch mining
cudaError_t create_mining_graph(MiningGraphState* state, uint32_t num_candidates, uint32_t num_combos,
                                uint32_t tile_h, uint32_t tile_w, uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
                                size_t shared_mem_bytes);

// Update existing graph execution with new parameters
cudaError_t update_mining_graph(MiningGraphState* state, uint32_t num_candidates, uint32_t num_combos,
                                uint32_t tile_h, uint32_t tile_w, uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
                                size_t shared_mem_bytes);

// Launch pre-captured graph
cudaError_t launch_mining_graph(MiningGraphState* state, const void* a_noised, const void* b_noised_t,
                                const void* a_rows, const void* b_cols, const void* jobs, void* jackpots,
                                const void* keys, void* hashes);

// Synchronize graph completion
cudaError_t synchronize_mining_graph(MiningGraphState* state);

// Destroy graph and free resources
cudaError_t destroy_mining_graph(MiningGraphState* state);

} // namespace mining
} // namespace pearl
