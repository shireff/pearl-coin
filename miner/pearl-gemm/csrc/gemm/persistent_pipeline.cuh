#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime_api.h>

namespace pearl {
namespace persistent {

constexpr uint32_t JOB_STATUS_PENDING = 0;
constexpr uint32_t JOB_STATUS_RUNNING = 1;
constexpr uint32_t JOB_STATUS_DONE = 2;
constexpr uint32_t JOB_STATUS_STOP = 0xFFFFFFFF;

constexpr uint32_t MAX_QUEUE_JOBS = 256;
constexpr uint32_t MAX_BATCH_SIZE = 256;

struct alignas(16) PersistentJob {
    uint32_t status;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t rank;
    uint32_t tile_h;
    uint32_t tile_w;
    uint32_t pow_target[8];
    uint32_t pow_key[8];
    uint32_t a_noised_offset;
    uint32_t b_noised_t_offset;
    uint32_t a_rows_offset;
    uint32_t b_cols_offset;
    uint32_t jackpot_offset;
    uint32_t result_offset;
};

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

// Opaque handle for persistent pipeline state
struct PersistentPipelineState;

// Create a persistent pipeline with pre-allocated GPU buffers
PersistentPipelineState* create_persistent_pipeline(uint32_t max_jackpots);

// Destroy persistent pipeline and free all GPU resources
void destroy_persistent_pipeline(PersistentPipelineState* state);

// Set the global data buffers (matrices, indices) for the pipeline
cudaError_t pipeline_set_buffers(
    PersistentPipelineState* state,
    const int32_t* d_a_noised,
    const int32_t* d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols
);

// Enqueue a batch of jobs for the persistent worker
cudaError_t pipeline_enqueue_jobs(PersistentPipelineState* state, const PersistentJob* jobs, uint32_t num_jobs);

// Wait for all queued jobs to complete
cudaError_t pipeline_wait_for_completion(PersistentPipelineState* state);

// Get results from completed jobs
cudaError_t pipeline_get_results(PersistentPipelineState* state, uint32_t* results, uint32_t num_jobs);

// Get raw jackpot buffer pointer (for CPU readback)
cudaError_t pipeline_get_jackpots(PersistentPipelineState* state, uint32_t** jackpots, size_t* size);

// Get raw hash buffer pointer (for CPU readback)
cudaError_t pipeline_get_hashes(PersistentPipelineState* state, uint32_t** hashes, size_t* size);

// Launch the persistent worker kernel
void launch_persistent_worker(PersistentPipelineState* state);

// Stop the persistent worker
cudaError_t stop_persistent_worker(PersistentPipelineState* state);

} // namespace persistent
} // namespace pearl
