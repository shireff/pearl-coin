// CUDA kernel implementations extracted from pearl_gemm_api.cpp.
// This file is compiled by nvcc; pearl_gemm_api.cpp is compiled by g++.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/python.h>

#include "../blake3/blake3.cuh"

#define CHECK_DEVICE(x)     TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace {

__global__ void blake3_keyed_hash_kernel(
    const uint32_t* messages,
    const uint32_t* key,
    uint32_t*       outputs,
    int             num_hashes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_hashes) {
        blake3::blake3_keyed_hash(messages + idx * 16, key, outputs + idx * 8);
    }
}

} // anonymous namespace

at::Tensor blake3_keyed_hash(at::Tensor messages, at::Tensor key) {
    CHECK_DEVICE(messages);
    CHECK_DEVICE(key);
    CHECK_CONTIGUOUS(messages);
    CHECK_CONTIGUOUS(key);

    TORCH_CHECK(messages.dtype() == torch::kUInt32, "messages must be uint32 tensor");
    TORCH_CHECK(key.dtype()      == torch::kUInt32, "key must be uint32 tensor");
    TORCH_CHECK(messages.dim() == 2,                "messages must be 2D tensor (num_hashes, 16)");
    TORCH_CHECK(key.dim()      == 1,                "key must be 1D tensor (8)");
    TORCH_CHECK(messages.size(1) == 16,             "messages must have 16 uint32s per hash");
    TORCH_CHECK(key.size(0)      == 8,              "key must have 8 uint32s");

    const int num_hashes = static_cast<int>(messages.size(0));
    at::cuda::CUDAGuard device_guard{static_cast<char>(messages.get_device())};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto outputs = torch::empty({num_hashes, 8}, messages.options());

    const int threads = 256;
    const int blocks  = (num_hashes + threads - 1) / threads;
    blake3_keyed_hash_kernel<<<blocks, threads, 0, stream>>>(
        messages.data_ptr<uint32_t>(),
        key.data_ptr<uint32_t>(),
        outputs.data_ptr<uint32_t>(),
        num_hashes);

    cudaStreamSynchronize(stream);
    return outputs;
}
