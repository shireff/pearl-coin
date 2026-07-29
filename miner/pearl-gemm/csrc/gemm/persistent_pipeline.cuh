#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime_api.h>
#include "persistent_mining.cuh"

namespace pearl {
namespace persistent {

constexpr uint32_t MAX_QUEUE_JOBS = 256;
constexpr uint32_t MAX_BATCH_SIZE = 256;

struct alignas(128) PersistentJobQueue {
    volatile uint32_t head;
    volatile uint32_t tail;
    uint32_t capacity;
    uint32_t num_blocks;
    uint32_t running;
    uint32_t count;
    uint32_t batch_id;
    uint32_t completed_count;
    uint32_t padding[8];
    PersistentJob jobs[MAX_QUEUE_JOBS];
    uint32_t results[MAX_QUEUE_JOBS];
};

cudaError_t pipeline_set_buffers(
    PersistentPipelineState* state,
    const int32_t* d_a_noised,
    const int32_t* d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols
);

cudaError_t pipeline_get_jackpots(PersistentPipelineState* state, uint32_t** jackpots, size_t* size);
cudaError_t pipeline_get_hashes(PersistentPipelineState* state, uint32_t** hashes, size_t* size);
void launch_persistent_worker(PersistentPipelineState* state);
cudaError_t stop_persistent_worker(PersistentPipelineState* state);

} // namespace persistent
} // namespace pearl
