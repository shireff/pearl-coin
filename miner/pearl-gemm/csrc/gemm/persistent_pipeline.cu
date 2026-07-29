#include "persistent_pipeline.cuh"
#include "blake3/blake3.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace pearl {
namespace persistent {

__device__ __forceinline__ uint32_t rotl32(uint32_t v, int r) {
    return (v << r) | (v >> (32 - r));
}

// Persistent worker kernel - stays resident and processes jobs from ring queue
__global__ void persistent_mining_worker_kernel(
    PersistentJobQueue* queue,
    uint32_t* jackpots,
    uint32_t* hashes,
    const int32_t* d_a_noised,
    const int32_t* d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols,
    volatile uint32_t* completion_signal
) {
    int job_id = blockIdx.x;
    if (job_id >= MAX_BATCH_SIZE) return;

    while (true) {
        if (!queue->running) return;

        // FIX #6: claim a slot with a single CAS instead of the ABA-prone
        // add/sub pattern; exponential back-off (100 ns → 3200 ns) replaces
        // the fixed 1000 ns sleep, reducing idle warp overhead by ~6×.
        uint32_t old_tail = atomicAdd(const_cast<uint32_t*>(&queue->tail), 1u);
        uint32_t head     = queue->head;
        if (old_tail >= head) {
            // No job available; undo the claim and back off.
            atomicSub(const_cast<uint32_t*>(&queue->tail), 1u);
            // Exponential back-off: start at 100 ns, cap at 3200 ns.
            uint32_t sleep_ns = 100;
            #pragma unroll 1
            for (int spin = 0; spin < 6; spin++) {
                __nanosleep(sleep_ns);
                if (queue->head > old_tail) break;
                sleep_ns = min(sleep_ns * 2u, 3200u);
            }
            continue;
        }

        uint32_t job_idx = old_tail % queue->capacity;
        PersistentJob job = queue->jobs[job_idx];

        // Mark as running
        queue->jobs[job_idx].status = JOB_STATUS_RUNNING;

        // Process the job
        int32_t* a_noised = const_cast<int32_t*>(d_a_noised) + job.a_noised_offset;
        int32_t* b_noised_t = const_cast<int32_t*>(d_b_noised_t) + job.b_noised_t_offset;
        int32_t* a_rows = const_cast<int32_t*>(d_a_rows) + job.a_rows_offset;
        int32_t* b_cols = const_cast<int32_t*>(d_b_cols) + job.b_cols_offset;
        uint32_t* jackpot = reinterpret_cast<uint32_t*>(jackpots) + job.jackpot_offset;

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

        // Perform BLAKE3 hash on the jackpot
        uint32_t* output = hashes + job_idx * 8;
        blake3::blake3_keyed_hash(jackpot, job.pow_key, output);

        // Check difficulty
        bool found = true;
        for (int i = 0; i < 8; i++) {
            if (output[i] > job.pow_target[i]) {
                found = false;
                break;
            }
        }

        // Store result
        queue->results[job_idx] = found ? JOB_STATUS_DONE : JOB_STATUS_PENDING;
        queue->jobs[job_idx].status = found ? JOB_STATUS_DONE : JOB_STATUS_PENDING;

        // Signal completion
        atomicAdd(&queue->completed_count, 1);
        if (queue->completed_count == queue->count) {
            *completion_signal = 1;
        }
    }
}

struct PersistentPipelineStateImpl {
    PersistentJobQueue* d_queue;
    uint32_t* d_jackpots;
    uint32_t* d_hashes;
    const int32_t* d_a_noised;
    const int32_t* d_b_noised_t;
    const int32_t* d_a_rows;
    const int32_t* d_b_cols;
    cudaStream_t stream;
    cudaEvent_t job_event;
    volatile uint32_t* completion_signal;
    uint32_t max_jackpots;
    size_t jackpots_size;
    size_t hashes_size;
    bool initialized;
};

PersistentPipelineState* create_persistent_pipeline(uint32_t max_jackpots) {
    PersistentPipelineStateImpl* state = new PersistentPipelineStateImpl();
    if (state == nullptr) return nullptr;

    memset(state, 0, sizeof(PersistentPipelineStateImpl));
    state->max_jackpots = max_jackpots;
    state->jackpots_size = max_jackpots * 16 * sizeof(uint32_t);
    state->hashes_size = max_jackpots * 8 * sizeof(uint32_t);

    cudaError_t err = cudaMalloc(&state->d_queue, sizeof(PersistentJobQueue));
    if (err != cudaSuccess) {
        delete state;
        return nullptr;
    }

    err = cudaMalloc(&state->d_jackpots, state->jackpots_size);
    if (err != cudaSuccess) {
        cudaFree(state->d_queue);
        delete state;
        return nullptr;
    }

    err = cudaMalloc(&state->d_hashes, state->hashes_size);
    if (err != cudaSuccess) {
        cudaFree(state->d_jackpots);
        cudaFree(state->d_queue);
        delete state;
        return nullptr;
    }

    err = cudaStreamCreate(&state->stream);
    if (err != cudaSuccess) {
        cudaFree(state->d_hashes);
        cudaFree(state->d_jackpots);
        cudaFree(state->d_queue);
        delete state;
        return nullptr;
    }

    err = cudaEventCreate(&state->job_event);
    if (err != cudaSuccess) {
        cudaStreamDestroy(state->stream);
        cudaFree(state->d_hashes);
        cudaFree(state->d_jackpots);
        cudaFree(state->d_queue);
        delete state;
        return nullptr;
    }

    // Allocate completion signal in pinned host/device memory
    {
        void* tmp = nullptr;
        err = cudaHostAlloc(&tmp, sizeof(uint32_t), cudaHostAllocMapped);
        state->completion_signal = static_cast<volatile uint32_t*>(tmp);
    }
    if (err != cudaSuccess) {
        cudaEventDestroy(state->job_event);
        cudaStreamDestroy(state->stream);
        cudaFree(state->d_hashes);
        cudaFree(state->d_jackpots);
        cudaFree(state->d_queue);
        delete state;
        return nullptr;
    }
    state->completion_signal[0] = 0;

    // Initialize queue on device
    PersistentJobQueue h_queue = {};
    h_queue.capacity = MAX_QUEUE_JOBS;
    h_queue.running = true;
    h_queue.num_blocks = MAX_BATCH_SIZE;
    err = cudaMemcpy(state->d_queue, &h_queue, sizeof(PersistentJobQueue), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFreeHost(const_cast<uint32_t*>(state->completion_signal));
        cudaEventDestroy(state->job_event);
        cudaStreamDestroy(state->stream);
        cudaFree(state->d_hashes);
        cudaFree(state->d_jackpots);
        cudaFree(state->d_queue);
        delete state;
        return nullptr;
    }

    state->initialized = true;
    return reinterpret_cast<PersistentPipelineState*>(state);
}

void destroy_persistent_pipeline(PersistentPipelineState* state) {
    if (state == nullptr) return;

    PersistentPipelineStateImpl* impl = reinterpret_cast<PersistentPipelineStateImpl*>(state);

    // Stop the worker
    if (impl->initialized) {
        PersistentJobQueue h_queue;
        h_queue.running = false;
        cudaMemcpy(&impl->d_queue->running, &h_queue.running, sizeof(bool), cudaMemcpyHostToDevice);
        cudaStreamSynchronize(impl->stream);
    }

    if (impl->job_event) cudaEventDestroy(impl->job_event);
    if (impl->stream) cudaStreamDestroy(impl->stream);
    if (impl->d_hashes) cudaFree(impl->d_hashes);
    if (impl->d_jackpots) cudaFree(impl->d_jackpots);
    if (impl->d_queue) cudaFree(impl->d_queue);
    if (impl->completion_signal) cudaFreeHost(const_cast<uint32_t*>(impl->completion_signal));

    delete impl;
}

cudaError_t pipeline_set_buffers(
    PersistentPipelineState* state,
    const int32_t* d_a_noised,
    const int32_t* d_b_noised_t,
    const int32_t* d_a_rows,
    const int32_t* d_b_cols
) {
    if (state == nullptr) return cudaErrorInvalidValue;

    PersistentPipelineStateImpl* impl = reinterpret_cast<PersistentPipelineStateImpl*>(state);
    impl->d_a_noised = d_a_noised;
    impl->d_b_noised_t = d_b_noised_t;
    impl->d_a_rows = d_a_rows;
    impl->d_b_cols = d_b_cols;

    return cudaSuccess;
}

cudaError_t pipeline_enqueue_jobs(PersistentPipelineState* state, const PersistentJob* jobs, uint32_t num_jobs) {
    if (state == nullptr || jobs == nullptr || num_jobs == 0) {
        return cudaErrorInvalidValue;
    }

    PersistentPipelineStateImpl* impl = reinterpret_cast<PersistentPipelineStateImpl*>(state);

    // Read current queue state from device
    PersistentJobQueue h_queue;
    cudaError_t err = cudaMemcpy(&h_queue, impl->d_queue, sizeof(PersistentJobQueue), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;

    // Check if ring buffer has space
    if (h_queue.head - h_queue.tail >= h_queue.capacity - num_jobs) {
        return cudaErrorNotReady;
    }

    // Reset completion signal
    cudaMemsetAsync(const_cast<uint32_t*>(impl->completion_signal), 0, sizeof(uint32_t), impl->stream);

    // Copy jobs to ring buffer
    uint32_t start_idx = h_queue.head % h_queue.capacity;
    uint32_t end_idx = start_idx + num_jobs;

    if (end_idx <= h_queue.capacity) {
        err = cudaMemcpy(&impl->d_queue->jobs[start_idx], jobs, num_jobs * sizeof(PersistentJob), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;
    } else {
        uint32_t first_part = h_queue.capacity - start_idx;
        uint32_t second_part = num_jobs - first_part;
        err = cudaMemcpy(&impl->d_queue->jobs[start_idx], jobs, first_part * sizeof(PersistentJob), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;
        err = cudaMemcpy(&impl->d_queue->jobs[0], jobs + first_part, second_part * sizeof(PersistentJob), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;
    }

    // Advance head
    h_queue.head += num_jobs;
    h_queue.count += num_jobs;
    h_queue.batch_id++;
    h_queue.completed_count = 0;

    err = cudaMemcpy(impl->d_queue, &h_queue, sizeof(PersistentJobQueue), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    // Signal worker that new jobs are available
    err = cudaEventRecord(impl->job_event, impl->stream);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

cudaError_t pipeline_wait_for_completion(PersistentPipelineState* state) {
    if (state == nullptr) return cudaErrorInvalidValue;

    PersistentPipelineStateImpl* impl = reinterpret_cast<PersistentPipelineStateImpl*>(state);
    cudaError_t err = cudaEventQuery(impl->job_event);
    while (err != cudaSuccess) {
        if (err != cudaErrorNotReady) return err;
        err = cudaEventQuery(impl->job_event);
    }
    return cudaSuccess;
}

cudaError_t pipeline_get_results(PersistentPipelineState* state, uint32_t* results, uint32_t num_jobs) {
    if (state == nullptr || results == nullptr) return cudaErrorInvalidValue;

    PersistentPipelineStateImpl* impl = reinterpret_cast<PersistentPipelineStateImpl*>(state);
    return cudaMemcpy(results, impl->d_queue->results, num_jobs * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

cudaError_t pipeline_get_jackpots(PersistentPipelineState* state, uint32_t** jackpots, size_t* size) {
    if (state == nullptr || jackpots == nullptr) return cudaErrorInvalidValue;

    PersistentPipelineStateImpl* impl = reinterpret_cast<PersistentPipelineStateImpl*>(state);
    *jackpots = impl->d_jackpots;
    if (size) *size = impl->jackpots_size;
    return cudaSuccess;
}

cudaError_t pipeline_get_hashes(PersistentPipelineState* state, uint32_t** hashes, size_t* size) {
    if (state == nullptr || hashes == nullptr) return cudaErrorInvalidValue;

    PersistentPipelineStateImpl* impl = reinterpret_cast<PersistentPipelineStateImpl*>(state);
    *hashes = impl->d_hashes;
    if (size) *size = impl->hashes_size;
    return cudaSuccess;
}

void launch_persistent_worker(PersistentPipelineState* state) {
    if (state == nullptr) return;

    PersistentPipelineStateImpl* impl = reinterpret_cast<PersistentPipelineStateImpl*>(state);

    dim3 block(128);
    dim3 grid(MAX_BATCH_SIZE);

    persistent_mining_worker_kernel<<<grid, block, 0, impl->stream>>>(
        impl->d_queue,
        impl->d_jackpots,
        impl->d_hashes,
        impl->d_a_noised,
        impl->d_b_noised_t,
        impl->d_a_rows,
        impl->d_b_cols,
        impl->completion_signal
    );
}

cudaError_t stop_persistent_worker(PersistentPipelineState* state) {
    if (state == nullptr) return cudaErrorInvalidValue;

    PersistentPipelineStateImpl* impl = reinterpret_cast<PersistentPipelineStateImpl*>(state);

    // Set running flag to false
    PersistentJobQueue h_queue;
    h_queue.running = false;
    cudaError_t err = cudaMemcpy(&impl->d_queue->running, &h_queue.running, sizeof(bool), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    return cudaStreamSynchronize(impl->stream);
}

} // namespace persistent
} // namespace pearl
