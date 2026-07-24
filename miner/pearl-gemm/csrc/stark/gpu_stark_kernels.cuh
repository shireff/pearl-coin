#pragma once

#include <cstdint>
#include <cstddef>

namespace pearl {
namespace stark {

void launch_lde_eval(
    const uint32_t* coeffs,
    const uint32_t* eval_points,
    uint32_t* evaluations,
    int num_coeffs,
    int num_points,
    int poly_degree,
    cudaStream_t stream);

void launch_batch_lde_eval(
    const uint32_t* coeffs,
    const uint32_t* eval_points,
    uint32_t* evaluations,
    int num_polys,
    int num_points,
    int poly_degree,
    cudaStream_t stream);

void launch_fri_fold(
    const uint32_t* layer,
    uint32_t* folded_layer,
    const uint32_t* challenges,
    int layer_size,
    cudaStream_t stream);

void launch_merkle_hash(
    const uint32_t* leaves,
    uint32_t* nodes,
    int num_leaves,
    int level,
    cudaStream_t stream);

void launch_init_roots_of_unity(
    uint32_t* roots,
    uint32_t primitive_root,
    int num_roots,
    cudaStream_t stream);

} // namespace stark
} // namespace pearl
