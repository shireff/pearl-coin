#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <cuda_runtime_api.h>

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

__device__ __forceinline__ uint32_t rotl32(uint32_t v, int r) {
    return (v << r) | (v >> (32 - r));
}

__global__ void persistent_mining_kernel(
    const int32_t* __restrict__ global_a_noised,
    const int32_t* __restrict__ global_b_noised_t,
    const int32_t* __restrict__ global_a_rows,
    const int32_t* __restrict__ global_b_cols,
    PersistentJob* jobs,
    uint32_t num_jobs,
    uint32_t max_tile_h,
    uint32_t max_tile_w
) {
    int job_id = blockIdx.x;
    if (job_id >= num_jobs) return;

    PersistentJob job = jobs[job_id];

    if (job.status == JOB_STATUS_STOP) return;

    jobs[job_id].status = JOB_STATUS_RUNNING;

    int32_t* a_noised = const_cast<int32_t*>(global_a_noised) + job.a_noised_offset;
    int32_t* b_noised_t = const_cast<int32_t*>(global_b_noised_t) + job.b_noised_t_offset;
    int32_t* a_rows = const_cast<int32_t*>(global_a_rows) + job.a_rows_offset;
    int32_t* b_cols = const_cast<int32_t*>(global_b_cols) + job.b_cols_offset;
    uint32_t* jackpot = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(jobs) + job.jackpot_offset);

    int tile_h = job.tile_h;
    int tile_w = job.tile_w;
    int m = job.m;
    int n = job.n;
    int k = job.k;
    int rank = job.rank;

    for (int ll = rank; ll <= k; ll += rank) {
        for (int u = 0; u < tile_h; u++) {
            int a_idx = a_rows[u];
            for (int v = 0; v < tile_w; v++) {
                int b_idx = b_cols[v];
                int32_t sum = 0;
                for (int l = ll - rank; l < ll; l++) {
                    sum += a_noised[a_idx * k + l] * b_noised_t[b_idx * k + l];
                }
                jackpot[u * tile_w + v] = static_cast<uint32_t>(sum);
            }
        }

        uint32_t xored = 0;
        for (int u = 0; u < tile_h; u++) {
            for (int v = 0; v < tile_w; v++) {
                xored ^= jackpot[u * tile_w + v];
            }
        }

        int tid = (ll / rank - 1) % 16;
        jackpot[tid] = rotl32(jackpot[tid], 1) ^ xored;
    }

    uint32_t jackpot_hash[8];
    for (int i = 0; i < 8; i++) {
        jackpot_hash[i] = jackpot[i];
    }

    bool found = true;
    for (int i = 0; i < 8; i++) {
        if (jackpot_hash[i] > job.pow_target[i]) {
            found = false;
            break;
        }
    }

    if (found) {
        jobs[job_id].status = JOB_STATUS_DONE;
    }
}

__global__ void persistent_mining_kernel_v2(
    const int32_t* __restrict__ global_a_noised,
    const int32_t* __restrict__ global_b_noised_t,
    const int32_t* __restrict__ global_a_rows,
    const int32_t* __restrict__ global_b_cols,
    PersistentJob* jobs,
    uint32_t num_jobs
) {
    int job_id = blockIdx.x;
    if (job_id >= num_jobs) return;

    PersistentJob job = jobs[job_id];

    if (job.status == JOB_STATUS_STOP) return;

    jobs[job_id].status = JOB_STATUS_RUNNING;

    int32_t* a_noised = const_cast<int32_t*>(global_a_noised) + job.a_noised_offset;
    int32_t* b_noised_t = const_cast<int32_t*>(global_b_noised_t) + job.b_noised_t_offset;
    int32_t* a_rows = const_cast<int32_t*>(global_a_rows) + job.a_rows_offset;
    int32_t* b_cols = const_cast<int32_t*>(global_b_cols) + job.b_cols_offset;
    uint32_t* jackpot = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(jobs) + job.jackpot_offset);

    int tile_h = job.tile_h;
    int tile_w = job.tile_w;
    int m = job.m;
    int n = job.n;
    int k = job.k;
    int rank = job.rank;

    for (int ll = rank; ll <= k; ll += rank) {
        for (int u = 0; u < tile_h; u++) {
            int a_idx = a_rows[u];
            for (int v = 0; v < tile_w; v++) {
                int b_idx = b_cols[v];
                int32_t sum = 0;
                for (int l = ll - rank; l < ll; l++) {
                    sum += a_noised[a_idx * k + l] * b_noised_t[b_idx * k + l];
                }
                jackpot[u * tile_w + v] = static_cast<uint32_t>(sum);
            }
        }

        uint32_t xored = 0;
        for (int u = 0; u < tile_h; u++) {
            for (int v = 0; v < tile_w; v++) {
                xored ^= jackpot[u * tile_w + v];
            }
        }

        int tid = (ll / rank - 1) % 16;
        jackpot[tid] = rotl32(jackpot[tid], 1) ^ xored;
    }

    uint32_t jackpot_hash[8];
    for (int i = 0; i < 8; i++) {
        jackpot_hash[i] = jackpot[i];
    }

    bool found = true;
    for (int i = 0; i < 8; i++) {
        if (jackpot_hash[i] > job.pow_target[i]) {
            found = false;
            break;
        }
    }

    if (found) {
        jobs[job_id].status = JOB_STATUS_DONE;
    }
}
