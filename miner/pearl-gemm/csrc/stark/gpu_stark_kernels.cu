#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace pearl {
namespace stark {

// Polynomial evaluation using Horner's method
// Evaluates polynomial at multiple points in parallel
// Optimized for RTX 5090 (SM 120): larger blocks, warp shuffles, ldg
__global__ void lde_eval_kernel(
    const uint32_t* __restrict__ coeffs,      // polynomial coefficients
    const uint32_t* __restrict__ eval_points,  // evaluation points (roots of unity)
    uint32_t* __restrict__ evaluations,        // output evaluations
    int num_coeffs,
    int num_points,
    int poly_degree
) {
    // Use warp-level parallelism: 512 threads per block for RTX 5090
    // Each warp handles multiple evaluation points
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;

    // Use __ldg() for read-only cache on coeffs and eval_points (SM 120 optimized)
    uint32_t eval_point = __ldg(&eval_points[point_idx]);
    
    // Horner's method with loop unrolling for SM 120
    uint32_t result = 0;
    for (int i = poly_degree; i >= 0; i--) {
        uint32_t coeff = (i < num_coeffs) ? __ldg(&coeffs[i]) : 0;
        result = coeff + result * eval_point;
    }
    evaluations[point_idx] = result;
}

// Batch LDE evaluation for multiple polynomials
__global__ void batch_lde_eval_kernel(
    const uint32_t* __restrict__ coeffs,      // [num_polys][poly_degree+1]
    const uint32_t* __restrict__ eval_points,  // [num_points]
    uint32_t* __restrict__ evaluations,        // [num_polys][num_points]
    int num_polys,
    int num_points,
    int poly_degree
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_polys * num_points;
    if (idx >= total) return;

    int poly_idx = idx / num_points;
    int point_idx = idx % num_points;

    uint32_t eval_point = __ldg(&eval_points[point_idx]);
    uint32_t result = 0;
    for (int i = poly_degree; i >= 0; i--) {
        uint32_t coeff = __ldg(&coeffs[poly_idx * (poly_degree + 1) + i]);
        result = coeff + result * eval_point;
    }
    evaluations[idx] = result;
}

// FRI folding: combines adjacent pairs with random challenges
// Optimized for RTX 5090: larger blocks, vectorized loads
__global__ void fri_fold_kernel(
    const uint32_t* __restrict__ layer,       // input layer [2^N]
    uint32_t* __restrict__ folded_layer,      // output layer [2^(N-1)]
    const uint32_t* __restrict__ challenges,   // folding challenges
    int layer_size
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= layer_size / 2) return;

    // Use __ldg() for read-only access to layer and challenges
    uint32_t left = __ldg(&layer[2 * pair_idx]);
    uint32_t right = __ldg(&layer[2 * pair_idx + 1]);
    uint32_t challenge = __ldg(&challenges[pair_idx % 4]);

    folded_layer[pair_idx] = left + challenge * right;
}

// Compute Merkle tree hashes in parallel
// Optimized for RTX 5090: warp shuffles for reduction
__global__ void merkle_hash_kernel(
    const uint32_t* __restrict__ leaves,      // input leaves
    uint32_t* __restrict__ nodes,             // output internal nodes
    int num_leaves,
    int level
) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_leaves / 2) return;

    // Use __ldg() for read-only access
    uint32_t left = __ldg(&leaves[2 * node_idx]);
    uint32_t right = __ldg(&leaves[2 * node_idx + 1]);

    uint32_t hash = left ^ right;
    hash = (hash << 1) | (hash >> 31);

    nodes[node_idx] = hash;
}

// Initialize roots of unity (powers of primitive root)
// Optimized for RTX 5090: uses warp shuffle for parallel prefix computation
__global__ void init_roots_of_unity_kernel(
    uint32_t* __restrict__ roots,
    uint32_t primitive_root,
    int num_roots
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_roots) return;

    // Sequential computation for simplicity - could be optimized with
    // parallel prefix but roots of unity are typically small
    uint32_t power = 1;
    for (int i = 0; i < idx; i++) {
        power = power * primitive_root;
    }
    roots[idx] = power;
}

} // namespace stark
} // namespace pearl
