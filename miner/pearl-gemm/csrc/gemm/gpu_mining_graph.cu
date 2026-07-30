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

    const uint32_t P = num_combos;
    const uint32_t Q = 1;

    // ── Allocate device buffers BEFORE stream capture begins ─────────────
    // cudaMalloc inside a captured stream is undefined behaviour.
    MiningJob* d_jobs = nullptr;
    int32_t*   d_a_rows = nullptr;
    int32_t*   d_b_cols = nullptr;
    uint32_t*  d_jackpots = nullptr;
    uint32_t*  d_hashes = nullptr;
    uint8_t*   d_winner_flags = nullptr;

    err = cudaMalloc(&d_jobs,       num_candidates * sizeof(MiningJob));
    if (err != cudaSuccess) { cudaStreamDestroy(state->stream); return err; }
    err = cudaMalloc(&d_a_rows,     P * tile_h * sizeof(int32_t));
    if (err != cudaSuccess) { cudaFree(d_jobs); cudaStreamDestroy(state->stream); return err; }
    err = cudaMalloc(&d_b_cols,     Q * tile_w * sizeof(int32_t));
    if (err != cudaSuccess) { cudaFree(d_a_rows); cudaFree(d_jobs); cudaStreamDestroy(state->stream); return err; }
    err = cudaMalloc(&d_jackpots,   num_candidates * num_combos * JACKPOT_SIZE * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_b_cols); cudaFree(d_a_rows); cudaFree(d_jobs); cudaStreamDestroy(state->stream); return err; }
    err = cudaMalloc(&d_hashes,     num_candidates * num_combos * 8 * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_jackpots); cudaFree(d_b_cols); cudaFree(d_a_rows); cudaFree(d_jobs); cudaStreamDestroy(state->stream); return err; }
    err = cudaMalloc(&d_winner_flags, (num_candidates * num_combos + 7) / 8 * sizeof(uint8_t));
    if (err != cudaSuccess) { cudaFree(d_hashes); cudaFree(d_jackpots); cudaFree(d_b_cols); cudaFree(d_a_rows); cudaFree(d_jobs); cudaStreamDestroy(state->stream); return err; }

    // Initialize jobs to STOP status using cudaMemset-compatible pattern
    // (do NOT dereference device pointers from CPU — that causes segfault)
    err = cudaMemset(d_jobs, 0xFF, num_candidates * sizeof(MiningJob));  // 0xFF fills all fields
    if (err != cudaSuccess) {
        cudaFree(d_winner_flags); cudaFree(d_hashes); cudaFree(d_jackpots);
        cudaFree(d_b_cols); cudaFree(d_a_rows); cudaFree(d_jobs);
        cudaStreamDestroy(state->stream);
        return err;
    }

    // ── Begin stream capture ──────────────────────────────────────────────
    err = cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        cudaFree(d_winner_flags); cudaFree(d_hashes); cudaFree(d_jackpots);
        cudaFree(d_b_cols); cudaFree(d_a_rows); cudaFree(d_jobs);
        cudaStreamDestroy(state->stream);
        return err;
    }

    launch_gpu_mining(
        nullptr, nullptr, d_a_rows, d_b_cols,
        d_jobs, d_jackpots, d_hashes, d_winner_flags,
        num_candidates, P, Q, tile_h, tile_w, m, n, k, rank,
        state->stream
    );

    uint32_t dummy_keys[8] = {};
    launch_gpu_jackpot_hash(
        d_jackpots, dummy_keys, d_hashes,
        num_candidates, num_combos,
        state->stream
    );

    // ── End capture ───────────────────────────────────────────────────────
    cudaGraph_t graph_tmp;
    err = cudaStreamEndCapture(state->stream, &graph_tmp);

    // Free temporary capture buffers — graph no longer needs them
    cudaFree(d_winner_flags);
    cudaFree(d_hashes);
    cudaFree(d_jackpots);
    cudaFree(d_b_cols);
    cudaFree(d_a_rows);
    cudaFree(d_jobs);

    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream);
        return err;
    }

    err = cudaGraphInstantiate(&state->graph_exec, graph_tmp, nullptr, nullptr, 0);
    cudaGraphDestroy(graph_tmp);  // graph object no longer needed after instantiation
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream);
        return err;
    }

    state->graph      = graph_tmp;  // kept for reference (already destroyed, just for state tracking)
    state->captured   = true;
    state->instantiated = true;
    state->last_num_candidates = num_candidates;
    state->last_num_combos     = num_combos;
    state->last_tile_h         = tile_h;
    state->last_tile_w         = tile_w;

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

    cudaError_t err = cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) return err;

    const uint32_t P = num_combos;
    const uint32_t Q = 1;

    int32_t* d_a_rows = nullptr;
    int32_t* d_b_cols = nullptr;
    uint32_t* d_jackpots = nullptr;
    uint32_t* d_hashes = nullptr;
    uint8_t* d_winner_flags = nullptr;

    MiningJob* d_jobs = nullptr;
    cudaMalloc(&d_jobs, num_candidates * sizeof(MiningJob));
    for (uint32_t i = 0; i < num_candidates; i++) {
        d_jobs[i].status = JOB_STATUS_STOP;
    }
    cudaMalloc(&d_a_rows, P * tile_h * sizeof(int32_t));
    cudaMalloc(&d_b_cols, Q * tile_w * sizeof(int32_t));
    cudaMalloc(&d_jackpots, num_candidates * num_combos * JACKPOT_SIZE * sizeof(uint32_t));
    cudaMalloc(&d_hashes, num_candidates * num_combos * 8 * sizeof(uint32_t));
    cudaMalloc(&d_winner_flags, (num_candidates * num_combos + 7) / 8 * sizeof(uint8_t));

    launch_gpu_mining(
        nullptr, nullptr, d_a_rows, d_b_cols,
        d_jobs, d_jackpots, d_hashes, d_winner_flags,
        num_candidates, P, Q, tile_h, tile_w, m, n, k, rank,
        state->stream
    );

    uint32_t dummy_keys[8] = {};
    launch_gpu_jackpot_hash(
        d_jackpots, dummy_keys, d_hashes,
        num_candidates, num_combos,
        state->stream
    );

    cudaFree(d_jobs);
    cudaFree(d_a_rows);
    cudaFree(d_b_cols);
    cudaFree(d_jackpots);
    cudaFree(d_hashes);
    cudaFree(d_winner_flags);

    cudaGraph_t new_graph;
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

    if (state->instantiated && state->graph_exec) {
        cudaGraphExecDestroy(state->graph_exec);
    }
    // Note: graph is destroyed right after instantiation in create_mining_graph,
    // so we do NOT call cudaGraphDestroy here.
    if (state->stream) {
        cudaStreamDestroy(state->stream);
    }

    memset(state, 0, sizeof(MiningGraphState));
    return cudaSuccess;
}

} // namespace mining
} // namespace pearl
