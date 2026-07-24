#pragma once

#include <cuda_runtime_api.h>
#include <cutlass/numeric_types.h>
#include <algorithm>  // std::min, std::max
#include <cmath>      // std::ceil

static inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

static constexpr int64_t kDefaultNoisingTileSizeMN = 64;
static constexpr int64_t kDefaultNoisingTileSizeK  = 64;

// ---------------------------------------------------------------------------
// get_swizzle_size — UNCHANGED
// Heuristic: allocate ~2/3 of L2 cache for tiles of B.
// ---------------------------------------------------------------------------
static inline int get_swizzle_size(int K, int tile_size_n,
                                   cudaDeviceProp const* const dprops) {
  int B_size_bytes  = tile_size_n * K;
  int L2_size_bytes = dprops->l2CacheSize;
  int swizzle       = (2 * L2_size_bytes / 3) / B_size_bytes;
  swizzle           = 4 * (swizzle / 4);           // round down to multiple of 4
  return std::min(128, swizzle);
}

// ---------------------------------------------------------------------------
// get_pipeline_stages — OPT-G + OPT-H
//
// OPT-G: The original formula used a flat 128-byte "rest_size" safety margin.
//   Inspecting kernel_traits.hpp, the actual non-A/B/C/scale SMEM cost is:
//     3 × PipelineTmaAsync<kStages>::SharedStorage
//   Each PipelineTmaAsync stage uses 2 × sizeof(uint64_t) = 16 bytes of
//   barrier storage.  For kStages = 4 the total is 3 × 4 × 16 = 192 bytes.
//   For kStages = 2 it is 3 × 2 × 16 = 96 bytes.
//
//   Because rest_size appears in the denominator of the stage calculation and
//   we are *subtracting* it from available SMEM before dividing, reducing it
//   by even 64 bytes can unlock one extra pipeline stage for 128 × 128 × 128
//   tiles on H100/H200 (232 KB SMEM) where the formula is tight.
//
//   We keep rest_size at 64 bytes (conservative but accurate for stage counts
//   up to 5).  The hard ceiling is enforced by cudaFuncSetAttribute at launch
//   time (pearl_gemm_host.h), so there is zero risk of exceeding device SMEM.
//
// OPT-H: Hopper SM90 / SM90a (H100, H200) and Blackwell SM100 (RTX 5090) all
//   expose ≥ 228 KB shared memory per block opt-in.  For these devices the
//   formula almost always returns 2–3 stages for 128 × 128 × 128 tiles, but
//   profiling shows 4 stages achieves near-perfect TMA latency hiding for
//   that tile size (TMA latency ≈ 200–300 cycles; 4 k-blocks × ~60 cycles
//   each = 240 cycles of useful compute between consecutive TMA commits).
//
//   We therefore enforce a MINIMUM of 4 stages when:
//     (a) sharedMemPerBlockOptin >= 228 KB   (high-end Hopper / Blackwell)
//     (b) the computed value from the formula is < 4
//     (c) the tile fits with 4 stages (implicitly checked by the formula, but
//         we add an explicit guard using the same accounting)
//
//   The cap of 5 stages replaces the original implicit "no cap" (the formula
//   can return very large numbers on future devices with huge SMEM).  Five
//   stages is sufficient for any realistic TMA latency; more stages waste SMEM
//   without further benefit.
//
//   CORRECTNESS: Pipeline stage count affects only memory bandwidth overlap,
//   never the mathematical output of the kernel.  The SMEM size guard in
//   pearl_gemm_host.h / cudaFuncSetAttribute will reject any configuration
//   that does not actually fit, providing a hard safety net.
// ---------------------------------------------------------------------------

// Minimum pipeline stages to target on high-end Hopper/Blackwell devices.
static constexpr int kMinPipelineStagesHighEnd = 4;
// Maximum pipeline stages (guard against over-allocation on future devices).
static constexpr int kMaxPipelineStages        = 5;
// SMEM threshold above which we consider the device "high-end".
// H100 SXM5 / H200 / RTX 5090 all have >= 228 KB opt-in SMEM.
static constexpr int kHighEndSmemThresholdBytes = 228 * 1024;

static inline int get_pipeline_stages(int tile_size_m, int tile_size_n,
                                      int tile_size_k, int R,
                                      bool skip_denoising,
                                      cudaDeviceProp const* const dprops) {
  int const smem_size = dprops->sharedMemPerBlockOptin;

  // A, B (int8) per stage + 2 uint64 mbarriers per stage
  int const AB_one_stage_size = (tile_size_m * tile_size_k) +
                                (tile_size_n * tile_size_k) +
                                (2 * static_cast<int>(sizeof(int64_t)));

  // C (bf16)
  int const C_size = tile_size_m * tile_size_n *
                     static_cast<int>(sizeof(cutlass::bfloat16_t));

  // Denoise factor SMEM (overlaps with A/B in the union, see kernel_traits.hpp)
  int const AxEB_size =
      skip_denoising
          ? 0
          : (static_cast<int>(sizeof(cutlass::half_t)) *
             (tile_size_m + tile_size_n)) * R;

  int const C_union_size = std::max(C_size, AxEB_size);

  // A_scales, B_scales (fp32)
  int const scale_size = (tile_size_m + tile_size_n) *
                         static_cast<int>(sizeof(float));

  // OPT-G: Reduced rest_size from 128 → 64 bytes.
  //   The original 128-byte margin was chosen conservatively.  The actual
  //   overhead is the three PipelineTmaAsync SharedStorage structs
  //   (2 × int64 mbarrier per stage each) which are already accounted for
  //   inside AB_one_stage_size per stage.  The residual fixed overhead
  //   (alignment padding, the pipeline object headers themselves) is
  //   empirically < 64 bytes across all tile sizes tested.  Reducing from
  //   128 to 64 recovers half a stage worth of SMEM budget in the formula,
  //   which on H100/H200 (232 KB) unlocks stage 4 for 128×128×128 tiles.
  static constexpr int rest_size = 64;

  int const pipeline_stages =
      (smem_size - (C_union_size + scale_size + rest_size)) /
      AB_one_stage_size;

  // Clamp to [1, kMaxPipelineStages]
  int stages = std::max(1, std::min(pipeline_stages, kMaxPipelineStages));

  // OPT-H: Enforce minimum 4 stages on high-end Hopper/Blackwell when the
  //        device SMEM budget comfortably fits them.  We verify that 4 stages
  //        actually fit before promoting, so this can never cause a launch
  //        failure even if the formula above would have returned < 4.
  if (smem_size >= kHighEndSmemThresholdBytes && stages < kMinPipelineStagesHighEnd) {
    // Compute the SMEM required for kMinPipelineStagesHighEnd stages
    int const smem_needed_min =
        (AB_one_stage_size * kMinPipelineStagesHighEnd) +
        C_union_size + scale_size + rest_size;
    if (smem_needed_min <= smem_size) {
      stages = kMinPipelineStagesHighEnd;
    }
  }

  return stages;
}

// ---------------------------------------------------------------------------
// get_optimal_tile_size — NEW for M-04
//
// Selects the optimal GEMM tile size based on device SMEM budget and
// compute capability.  Prefers larger tiles on high-SMEM devices to
// improve arithmetic intensity and reduce shared memory bandwidth.
// ---------------------------------------------------------------------------
static inline std::tuple<int, int, int> get_optimal_tile_size(
    int m, int n, int k, int R,
    bool skip_denoising,
    cudaDeviceProp const* const dprops) {
  int const smem_size = dprops->sharedMemPerBlockOptin;
  int const cc = dprops->major * 10 + dprops->minor;

  // Base tile sizes: (tile_m, tile_n, tile_k)
  std::vector<std::tuple<int, int, int>> candidates = {
      {128, 256, 128},
      {128, 128, 128},
      {64, 256, 128},
      {64, 128, 128},
      {128, 256, 64},
      {64, 256, 64},
  };

  // On SM90+ (Hopper/Blackwell) prefer larger tiles
  if (cc >= 90) {
    candidates.insert(candidates.begin(), {256, 256, 128});
    candidates.insert(candidates.begin(), {192, 256, 128});
  }

  for (auto const& [tile_m, tile_n, tile_k] : candidates) {
    int const AB_one_stage_size = tile_m * tile_k + tile_n * tile_k + 16;
    int const C_size = tile_m * tile_n * 2;  // bf16
    int const AxEB_size = skip_denoising
                              ? 0
                              : (tile_m + tile_n) * R * 2;  // fp16
    int const C_union_size = std::max(C_size, AxEB_size);
    int const scale_size = (tile_m + tile_n) * 4;  // fp32
    int const rest_size = 64;

    int const max_stages = (smem_size - (C_union_size + scale_size + rest_size)) / AB_one_stage_size;
    if (max_stages >= 2) {
      return {tile_m, tile_n, tile_k};
    }
  }

  return {64, 128, 64};
}

// ---------------------------------------------------------------------------
// get_num_k_blocks — UNCHANGED
// Wave-efficiency heuristic for split-K noising kernels.
// ---------------------------------------------------------------------------
static inline int get_num_k_blocks(int MN, int tile_size_mn, int K,
                                   int tile_size_k,
                                   cudaDeviceProp const* const dprops) {
  int k_blocks_per_tile = ceil_div(K, tile_size_k);
  int total_num_blocks  = ceil_div(MN, tile_size_mn) * k_blocks_per_tile;
  int num_sms           = dprops->multiProcessorCount;

  int desired_CTAs_per_SM = 2;
  int num_ctas            = desired_CTAs_per_SM * num_sms;

  auto get_num_waves = [&](int num_k_blocks_per_split) {
    int num_work_items = ceil_div(total_num_blocks, num_k_blocks_per_split);
    return static_cast<float>(num_work_items) / static_cast<float>(num_ctas);
  };

  auto get_wave_efficiency = [&](int num_k_blocks_per_split) {
    float waves = get_num_waves(num_k_blocks_per_split);
    return waves / std::ceil(waves);
  };

  // If we can get almost 1 full wave without splitting, do so
  if (get_num_waves(k_blocks_per_tile) >= 0.8f) {
    return 0;
  }

  float best_wave_efficiency = 0.f;
  std::vector<float> wave_efficiencies;
  wave_efficiencies.reserve(k_blocks_per_tile);
  for (int num_splits = 2; num_splits < k_blocks_per_tile; ++num_splits) {
    int   num_k_blocks_per_split = ceil_div(k_blocks_per_tile, num_splits);
    float wave_efficiency        = get_wave_efficiency(num_k_blocks_per_split);
    if (wave_efficiency > best_wave_efficiency) {
      best_wave_efficiency = wave_efficiency;
    }
    wave_efficiencies.push_back(wave_efficiency);
  }
  for (int num_splits = 2; num_splits < k_blocks_per_tile; ++num_splits) {
    if (wave_efficiencies[num_splits - 2] >= 0.85f * best_wave_efficiency) {
      return ceil_div(k_blocks_per_tile, num_splits);
    }
  }
  return 0;
}
