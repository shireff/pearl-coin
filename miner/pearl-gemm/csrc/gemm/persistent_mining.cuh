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

void launch_persistent_mining(
    const int32_t* a_noised,
    const int32_t* b_noised_t,
    const int32_t* a_rows,
    const int32_t* b_cols,
    PersistentJob* jobs,
    uint32_t num_jobs,
    cudaStream_t stream);

struct PersistentPipelineState;

PersistentPipelineState* create_persistent_pipeline(uint32_t max_jackpots);
void destroy_persistent_pipeline(PersistentPipelineState* state);
cudaError_t pipeline_enqueue_jobs(PersistentPipelineState* state, const PersistentJob* jobs, uint32_t num_jobs);
cudaError_t pipeline_wait_for_completion(PersistentPipelineState* state);
cudaError_t pipeline_get_results(PersistentPipelineState* state, uint32_t* results, uint32_t num_jobs);

} // namespace persistent
} // namespace pearl
