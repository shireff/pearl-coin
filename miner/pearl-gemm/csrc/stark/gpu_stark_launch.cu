#include "gpu_stark_kernels.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>

namespace pearl {
namespace stark {

void launch_lde_eval(
    const uint32_t* coeffs,
    const uint32_t* eval_points,
    uint32_t* evaluations,
    int num_coeffs,
    int num_points,
    int poly_degree,
    cudaStream_t stream) {

    // RTX 5090 optimized: 512 threads per block for maximum occupancy
    dim3 block(512);
    dim3 grid((num_points + block.x - 1) / block.x);

    lde_eval_kernel<<<grid, block, 0, stream>>>(
        coeffs, eval_points, evaluations,
        num_coeffs, num_points, poly_degree
    );
}

void launch_batch_lde_eval(
    const uint32_t* coeffs,
    const uint32_t* eval_points,
    uint32_t* evaluations,
    int num_polys,
    int num_points,
    int poly_degree,
    cudaStream_t stream) {

    int total = num_polys * num_points;
    dim3 block(512);
    dim3 grid((total + block.x - 1) / block.x);

    batch_lde_eval_kernel<<<grid, block, 0, stream>>>(
        coeffs, eval_points, evaluations,
        num_polys, num_points, poly_degree
    );
}

void launch_fri_fold(
    const uint32_t* layer,
    uint32_t* folded_layer,
    const uint32_t* challenges,
    int layer_size,
    cudaStream_t stream) {

    dim3 block(512);
    dim3 grid((layer_size / 2 + block.x - 1) / block.x);

    fri_fold_kernel<<<grid, block, 0, stream>>>(
        layer, folded_layer, challenges, layer_size
    );
}

void launch_merkle_hash(
    const uint32_t* leaves,
    uint32_t* nodes,
    int num_leaves,
    int level,
    cudaStream_t stream) {

    dim3 block(512);
    dim3 grid((num_leaves / 2 + block.x - 1) / block.x);

    merkle_hash_kernel<<<grid, block, 0, stream>>>(
        leaves, nodes, num_leaves, level
    );
}

void launch_init_roots_of_unity(
    uint32_t* roots,
    uint32_t primitive_root,
    int num_roots,
    cudaStream_t stream) {

    dim3 block(512);
    dim3 grid((num_roots + block.x - 1) / block.x);

    init_roots_of_unity_kernel<<<grid, block, 0, stream>>>(
        roots, primitive_root, num_roots
    );
}

} // namespace stark
} // namespace pearl
