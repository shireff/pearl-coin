#pragma once
// FIX #14: transpose B from row-major [k,n] to column-major [n,k] on GPU,
// eliminating the CPU-side transpose that was a blocking H2D transfer per batch.
#include <cuda_runtime.h>
#include <cstdint>

namespace pearl {
namespace mining {

// Transposes int32 matrix src[k*n] (row-major k×n) → dst[n*k] (row-major n×k).
// Uses 32×32 shared-memory tile to coalesce both reads and writes.
__global__ void transpose_i32_kernel(
    const int32_t* __restrict__ src,  // [k * n]
    int32_t*       __restrict__ dst,  // [n * k]
    int k, int n)
{
    __shared__ int32_t tile[32][33]; // +1 to avoid bank conflicts

    const int bx = blockIdx.x * 32;
    const int by = blockIdx.y * 32;

    int row = by + threadIdx.y;
    int col = bx + threadIdx.x;
    if (row < k && col < n)
        tile[threadIdx.y][threadIdx.x] = src[row * n + col];
    __syncthreads();

    row = bx + threadIdx.y;
    col = by + threadIdx.x;
    if (row < n && col < k)
        dst[row * k + col] = tile[threadIdx.x][threadIdx.y];
}

inline void launch_transpose_i32(
    const int32_t* src, int32_t* dst, int k, int n, cudaStream_t stream)
{
    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (k + 31) / 32);
    transpose_i32_kernel<<<grid, block, 0, stream>>>(src, dst, k, n);
}

} // namespace mining
} // namespace pearl
