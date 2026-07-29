#pragma once

#include <cstdint>
#include <cstddef>

namespace pearl {
namespace mining {

constexpr uint32_t JOB_STATUS_PENDING = 0;
constexpr uint32_t JOB_STATUS_RUNNING = 1;
constexpr uint32_t JOB_STATUS_DONE = 2;
constexpr uint32_t JOB_STATUS_STOP = 0xFFFFFFFF;

constexpr int JACKPOT_SIZE = 16;
constexpr int LROT_PER_TILE = 1;

struct alignas(16) MiningJob {
    uint32_t status;
    uint32_t candidate_id;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t rank;
    uint32_t tile_h;
    uint32_t tile_w;
    uint32_t num_tiles;
    uint32_t a_offset;
    uint32_t b_offset;
    uint32_t a_rows_offset;
    uint32_t b_cols_offset;
    uint32_t result_offset;
    uint32_t pow_target[8];
    uint32_t pow_key[8];
    uint32_t a_noise_seed;
    uint32_t b_noise_seed;
};

struct alignas(16) MiningResult {
    uint32_t status;
    uint32_t candidate_id;
    uint32_t tile_idx;
    uint32_t a_rows_offset;
    uint32_t b_cols_offset;
};

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
    cudaStream_t stream);

__device__ __forceinline__ bool blake3_check_difficulty(
    const uint32_t* hash,
    const uint32_t* target);

__device__ __forceinline__ void blake3_keyed_hash_inline(
    const uint32_t* msg,
    const uint32_t* key,
    uint32_t* output);

void launch_gpu_jackpot_hash(
    const uint32_t* jackpots,
    const uint32_t* keys,
    uint32_t* hashes,
    uint32_t num_candidates,
    uint32_t num_combos,
    cudaStream_t stream);

cudaError_t launch_gpu_candidate_generation(
    int8_t* d_a_raw,
    int8_t* d_b_raw_t,
    uint32_t batch_size,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint64_t base_seed,
    cudaStream_t stream);

} // namespace mining
} // namespace pearl
