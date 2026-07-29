#include "moe/build_routing_data.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <cub/cub.cuh>

namespace {

constexpr size_t kBitsPerExpertId = 8;
constexpr size_t kMaxNumExperts = 1LL << kBitsPerExpertId;
constexpr size_t kNumScratchBuffers = 2;

inline size_t scratch_buffers_bytes(int64_t numel) {
  return static_cast<size_t>(numel) * sizeof(int32_t) * kNumScratchBuffers;
}

__device__ inline int32_t upper_bound_count(const int32_t* sorted_keys,
                                            int32_t numel, int32_t target) {
  int32_t lo = 0;
  int32_t hi = numel;
  while (lo < hi) {
    const int32_t mid = lo + (hi - lo) / 2;
    if (sorted_keys[mid] <= target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Step 1: fill iota [0, 1, ..., numel-1]
__global__ void iota_kernel(int32_t* buf, int32_t numel) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) buf[idx] = idx;
}

// Step 3: per-expert upper-bound offsets
__global__ void routing_offsets_kernel(int32_t* routing_offsets,
                                       const int32_t* sorted_keys_buf,
                                       int32_t numel, int32_t num_experts) {
  int32_t expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert < num_experts)
    routing_offsets[expert] = upper_bound_count(sorted_keys_buf, numel, expert);
}

// Step 4: convert flat slot indices to token indices
__global__ void slot_to_token_kernel(const int32_t* slot_indices,
                                     int32_t* routing_data, int32_t top_k,
                                     int32_t numel) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) routing_data[idx] = slot_indices[idx] / top_k;
}

static constexpr int kBlock = 256;

void run_build_routing_data(const int32_t* topk_ids, int32_t* routing_data,
                            int32_t* slot_indices, int32_t* routing_offsets,
                            int32_t num_experts, int32_t num_tokens,
                            int32_t top_k, void* scratchpad,
                            size_t scratchpad_bytes, cudaStream_t stream) {
  const int32_t numel = num_tokens * top_k;

  auto* values_buf = static_cast<int32_t*>(scratchpad);
  auto* sorted_keys_buf = values_buf + numel;
  auto* cub_temp = reinterpret_cast<void*>(sorted_keys_buf + numel);
  size_t cub_temp_bytes = scratchpad_bytes - scratch_buffers_bytes(numel);

  // Step 1: fill iota
  iota_kernel<<<(numel + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      values_buf, numel);

  // Step 2: stable radix sort
  cub::DeviceRadixSort::SortPairs(cub_temp, cub_temp_bytes, topk_ids,
                                  sorted_keys_buf, values_buf, slot_indices,
                                  numel, 0, kBitsPerExpertId, stream);

  // Step 3: per-expert offsets
  routing_offsets_kernel<<<(num_experts + kBlock - 1) / kBlock, kBlock, 0,
                            stream>>>(routing_offsets, sorted_keys_buf, numel,
                                      num_experts);

  // Step 4: slot indices -> token indices
  slot_to_token_kernel<<<(numel + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      slot_indices, routing_data, top_k, numel);
}

}  // namespace

int64_t get_build_routing_data_scratchpad_bytes(int64_t numel) {
  size_t cub_temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, cub_temp_bytes, static_cast<const int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr), static_cast<const int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr), static_cast<int32_t>(numel), 0,
      kBitsPerExpertId);
  return static_cast<int64_t>(scratch_buffers_bytes(numel) + cub_temp_bytes);
}

void build_routing_data(const at::Tensor& topk_ids,
                        const at::Tensor& routing_data,
                        const at::Tensor& slot_indices,
                        const at::Tensor& routing_offsets,
                        const at::Tensor& scratchpad, int64_t num_experts) {
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be on CUDA");
  TORCH_CHECK(routing_data.is_cuda(), "routing_data must be on CUDA");
  TORCH_CHECK(slot_indices.is_cuda(), "slot_indices must be on CUDA");
  TORCH_CHECK(routing_offsets.is_cuda(), "routing_offsets must be on CUDA");
  TORCH_CHECK(scratchpad.is_cuda(), "scratchpad must be on CUDA");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(routing_data.is_contiguous(), "routing_data must be contiguous");
  TORCH_CHECK(slot_indices.is_contiguous(), "slot_indices must be contiguous");
  TORCH_CHECK(routing_offsets.is_contiguous(),
              "routing_offsets must be contiguous");
  TORCH_CHECK(scratchpad.is_contiguous(), "scratchpad must be contiguous");
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32, "topk_ids must be int32");
  TORCH_CHECK(routing_data.dtype() == torch::kInt32,
              "routing_data must be int32");
  TORCH_CHECK(slot_indices.dtype() == torch::kInt32,
              "slot_indices must be int32");
  TORCH_CHECK(routing_offsets.dtype() == torch::kInt32,
              "routing_offsets must be int32");
  TORCH_CHECK(scratchpad.dtype() == torch::kUInt8, "scratchpad must be uint8");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2D (m, K)");

  const int32_t num_tokens = static_cast<int32_t>(topk_ids.size(0));
  const int32_t top_k = static_cast<int32_t>(topk_ids.size(1));
  const int32_t numel = num_tokens * top_k;
  TORCH_CHECK(routing_data.numel() == numel,
              "routing_data must have m*K elements");
  TORCH_CHECK(slot_indices.numel() == numel,
              "slot_indices must have m*K elements");
  TORCH_CHECK(num_experts <= static_cast<int64_t>(kMaxNumExperts),
              "build_routing_data sorts on 8-bit expert IDs, got num_experts=",
              num_experts);
  TORCH_CHECK(routing_offsets.numel() == num_experts,
              "routing_offsets must have num_experts elements");

  const int64_t required = get_build_routing_data_scratchpad_bytes(numel);
  TORCH_CHECK(scratchpad.numel() >= required, "scratchpad too small: need ",
              required, " bytes, got ", scratchpad.numel());

  at::cuda::CUDAGuard device_guard{(char)topk_ids.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  run_build_routing_data(
      topk_ids.data_ptr<int32_t>(), routing_data.data_ptr<int32_t>(),
      slot_indices.data_ptr<int32_t>(), routing_offsets.data_ptr<int32_t>(),
      static_cast<int32_t>(num_experts), num_tokens, top_k,
      scratchpad.data_ptr(), static_cast<size_t>(scratchpad.numel()), stream);
}
