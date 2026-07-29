#include "gpu_mining_graph.cuh"
#include "gpu_mining_kernel.cuh"
#include "blake3/blake3.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace pearl {
namespace mining {

cudaError_t create_mining_graph(MiningGraphState* state, uint32_t num_candidates, uint32_t num_combos,
                                uint32_t tile_h, uint32_t tile_w, uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
                                size_t shared_mem_bytes) {
    if (state == nullptr) return cudaErrorInvalidValue;

    memset(state, 0, sizeof(MiningGraphState));

    cudaError_t err = cudaStreamCreate(&state->stream);
    if (err != cudaSuccess) return err;

    err = cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream);
        return err;
    }

    dim3 block(128);
    dim3 grid(num_candidates);
    const int hash_threads = 256;
    const int hash_blocks = (num_candidates * num_combos + hash_threads - 1) / hash_threads;

    MiningJob dummy_jobs[1] = {};
    MiningResult dummy_results[1] = {};
    uint32_t dummy_keys[8] = {};
    uint32_t dummy_hashes[1] = {};

    launch_gpu_mining(
        nullptr, nullptr, nullptr, nullptr,
        dummy_jobs, dummy_results,
        num_candidates, tile_h, tile_w,
        state->stream
    );

    launch_gpu_jackpot_hash(
        reinterpret_cast<const uint32_t*>(dummy_results),
        dummy_keys, dummy_hashes,
        num_candidates, 1,
        state->stream
    );

    err = cudaStreamEndCapture(state->stream, &state->graph);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream);
        return err;
    }

    err = cudaGraphInstantiate(&state->graph_exec, state->graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(state->graph);
        cudaStreamDestroy(state->stream);
        return err;
    }

    state->captured = true;
    state->instantiated = true;
    state->last_num_candidates = num_candidates;
    state->last_num_combos = num_combos;
    state->last_tile_h = tile_h;
    state->last_tile_w = tile_w;

    return cudaSuccess;
}

cudaError_t update_mining_graph(MiningGraphState* state, uint32_t num_candidates, uint32_t num_combos,
                                uint32_t tile_h, uint32_t tile_w, uint32_t m, uint32_t n, uint32_t k, uint32_t rank,
                                size_t shared_mem_bytes) {
    if (state == nullptr || !state->instantiated) return cudaErrorInvalidValue;

    if (state->last_num_candidates == num_candidates &&
        state->last_num_combos == num_combos &&
        state->last_tile_h == tile_h &&
        state->last_tile_w == tile_w) {
        return cudaSuccess;
    }

    cudaGraph_t new_graph;
    cudaError_t err = cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) return err;

    dim3 block(128);
    dim3 grid(num_candidates);
    const int hash_threads = 256;
    const int hash_blocks = (num_candidates * num_combos + hash_threads - 1) / hash_threads;

    MiningJob dummy_jobs[1] = {};
    MiningResult dummy_results[1] = {};
    uint32_t dummy_keys[8] = {};
    uint32_t dummy_hashes[1] = {};

    launch_gpu_mining(
        nullptr, nullptr, nullptr, nullptr,
        dummy_jobs, dummy_results,
        num_candidates, tile_h, tile_w,
        state->stream
    );

    launch_gpu_jackpot_hash(
        reinterpret_cast<const uint32_t*>(dummy_results),
        dummy_keys, dummy_hashes,
        num_candidates, 1,
        state->stream
    );

    err = cudaStreamEndCapture(state->stream, &new_graph);
    if (err != cudaSuccess) return err;

    cudaGraphExec_t new_exec;
    err = cudaGraphInstantiate(&new_exec, new_graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(new_graph);
        return err;
    }

    cudaGraphExecUpdateResult update_result;
    cudaGraphNode_t error_node = nullptr;
    err = cudaGraphExecUpdate(state->graph_exec, new_graph, &error_node, &update_result);
    if (err == cudaSuccess && update_result == cudaGraphExecUpdateSuccess) {
        cudaGraphExecDestroy(new_exec);
        cudaGraphDestroy(new_graph);
    } else {
        cudaGraphExecDestroy(state->graph_exec);
        cudaGraphDestroy(state->graph);
        state->graph_exec = new_exec;
        state->graph = new_graph;
    }

    state->last_num_candidates = num_candidates;
    state->last_num_combos = num_combos;
    state->last_tile_h = tile_h;
    state->last_tile_w = tile_w;

    return cudaSuccess;
}

cudaError_t launch_mining_graph(MiningGraphState* state, const void* a_noised, const void* b_noised_t,
                                const void* a_rows, const void* b_cols, const void* jobs, void* jackpots,
                                const void* keys, void* hashes) {
    if (state == nullptr || !state->instantiated) return cudaErrorInvalidValue;

    cudaError_t err = cudaGraphLaunch(state->graph_exec, state->stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

cudaError_t synchronize_mining_graph(MiningGraphState* state) {
    if (state == nullptr) return cudaErrorInvalidValue;
    return cudaStreamSynchronize(state->stream);
}

cudaError_t destroy_mining_graph(MiningGraphState* state) {
    if (state == nullptr) return cudaErrorInvalidValue;

    if (state->instantiated) {
        cudaGraphExecDestroy(state->graph_exec);
    }
    if (state->captured) {
        cudaGraphDestroy(state->graph);
    }
    if (state->stream) {
        cudaStreamDestroy(state->stream);
    }

    memset(state, 0, sizeof(MiningGraphState));
    return cudaSuccess;
}

} // namespace mining
} // namespace pearl
