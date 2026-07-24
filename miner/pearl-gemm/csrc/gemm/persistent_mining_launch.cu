#include "persistent_mining.cuh"
#include <cuda_runtime.h>
#include <cassert>

namespace pearl {
namespace persistent {

void launch_persistent_mining(
    const int32_t* a_noised,
    const int32_t* b_noised_t,
    const int32_t* a_rows,
    const int32_t* b_cols,
    PersistentJob* jobs,
    uint32_t num_jobs,
    cudaStream_t stream) {

    dim3 block(1);
    dim3 grid(num_jobs);

    persistent_mining_kernel<<<grid, block, 0, stream>>>(
        a_noised, b_noised_t, a_rows, b_cols,
        jobs, num_jobs, 256, 256
    );
}

} // namespace persistent
} // namespace pearl
