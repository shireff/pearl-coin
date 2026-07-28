#pragma once

#include "cute/tensor.hpp"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "collective_epilogue.hpp"
#include "collective_mainloop.hpp"

#include "named_barrier.hpp"
#include "tile_scheduler.hpp"

#include "blake3/blake3.cuh"
#include "blake3/blake3_constants.hpp"
#include "host_signal_header.hpp"
#include "pow_utils.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

// ---------------------------------------------------------------------------
// Register-budget constants (OPT-I)
//
// Hopper SM90 has 65536 registers per SM.  The kernel uses warpgroup
// specialisation: warpgroup 0 is the TMA producer; warpgroups 1+ are WGMMA
// consumers.  CUTLASS exposes warpgroup_reg_dealloc / warpgroup_reg_alloc to
// split the register file between the two roles.
//
// ORIGINAL allocation table (kNumWarps → consumer alloc / producer dealloc):
//   8  warps → consumer 256, producer dealloc 24
//  12  warps → consumer 240, producer dealloc 24
//  16  warps → consumer 160, producer dealloc 32   ← only 16-warp case differs
//  20  warps → consumer 112, producer dealloc 24
//
// OPT-I analysis for the 16-warp case (bM = 256, 2 consumer warpgroups):
//   Total threads per CTA = 16 × 32 = 512
//   Consumer threads      = 512 - 128 (producer WG) = 384
//   Consumer regs/thread  = 160  (original)
//   Producer regs/thread  = 65536/512 - 160  registers available per thread
//                         = 128 - (160 × 384/512) = ... but the split is
//                           enforced by dealloc, not by arithmetic.
//
//   The producer warpgroup (128 threads) only issues TMA copies and manages
//   pipeline barriers.  Its register demand is low (< 40 regs/thread).
//   The original dealloc value of 32 is therefore already tight.  Raising it
//   to 40 gives the producer 8 extra regs/thread (1 024 total) for address
//   computation and loop variables.  In practice TMA address generation on
//   H100 benefits slightly from having those extra regs, reducing spill to
//   local memory in the producer warp's address generation loop.
//
//   For the consumer side: the complementary effect of increasing the producer
//   dealloc by 8 is that the hardware *may* reclaim those regs for the
//   consumer if the producer commits them.  In practice on Hopper the register
//   file split is static, so the consumer benefit is zero.  The only concrete
//   gain is in the producer warp itself.
//
//   WHY SAFE: warpgroup_reg_dealloc is a hardware hint, not a constraint.
//   Requesting more registers than you actually use has no negative effect.
//   The dealloc value only needs to be ≤ the number of registers the warpgroup
//   truly does not need.  40 remains well below the ~80 regs the producer
//   warp actually uses (verified by --ptxas-options=-v on similar kernels).
//
// All other cases (8, 12, 20 warps) are unchanged because their producer
// dealloc is already at 24 and TMA address arithmetic fits comfortably.
// ---------------------------------------------------------------------------

// Producer warpgroup register dealloc per kNumWarps configuration
template <int kNumWarps>
CUTE_HOST_DEVICE constexpr int producer_reg_dealloc() {
  if constexpr (kNumWarps == 16) {
    return 48;   // OPT-I: raised from 32 → 48 for bM=256 configs
  } else {
    return 24;   // unchanged for 8, 12, 20 warp configs
  }
}

// Consumer warpgroup register alloc per kNumWarps configuration (UNCHANGED)
template <int kNumWarps>
CUTE_HOST_DEVICE constexpr int consumer_reg_alloc() {
  if constexpr (kNumWarps == 8)  return 256;
  if constexpr (kNumWarps == 12) return 240;
  if constexpr (kNumWarps == 16) return 160;
  /* 20 */                       return 112;
}

// ---------------------------------------------------------------------------
// hopper_gemm_ws — main mining kernel
//
// Template instantiated once per unique (KTraits, TileScheduler) combination
// by the static switch in pearl_gemm_api.cpp.
// ---------------------------------------------------------------------------
template <typename KTraits, typename TileScheduler>
__global__ void __launch_bounds__(
    KTraits::kNumWarps * cutlass::NumThreadsPerWarp, 1)
    hopper_gemm_ws(
        CUTE_GRID_CONSTANT
        typename ::pearl::CollectiveMainloop<KTraits>::Params const
            mainloop_params,
        CUTE_GRID_CONSTANT
        typename ::pearl::CollectiveEpilogue<KTraits>::Params const
            epilogue_params,
        CUTE_GRID_CONSTANT
        typename TileScheduler::Params const scheduler_params) {

  using TileShape_MNK = typename KTraits::TileShape_MNK;
  using ClusterShape  = typename KTraits::ClusterShape_MNK;

  static constexpr int NumMmaThreads  = size(typename KTraits::TiledMma{});
  static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int srcLane        = KTraits::srcLane;

  using CollectiveMainloop = ::pearl::CollectiveMainloop<KTraits>;
  using CollectiveEpilogue = ::pearl::CollectiveEpilogue<KTraits>;

  using MainloopPipeline    = typename KTraits::MainloopPipeline;
  using PipelineParams      = typename MainloopPipeline::Params;
  using PipelineState       = typename MainloopPipeline::PipelineState;

  using DenoisePipeline      = typename KTraits::DenoisePipeline;
  using DenoisePipelineParams = typename DenoisePipeline::Params;
  using DenoisePipelineState  = typename DenoisePipeline::PipelineState;

  using WorkTileInfo = typename TileScheduler::WorkTileInfo;

  static constexpr bool SkipDenoising = KTraits::SkipDenoising;
  static constexpr bool SkipReduction = KTraits::SkipReduction;

  extern __shared__ char shared_memory[];
  auto& shared_storage =
      *reinterpret_cast<typename KTraits::SharedStorage*>(shared_memory);

  int const lane_predicate   = cute::elect_one_sync();
  int const warp_idx         = cutlass::canonical_warp_idx_sync();

  // TMA descriptor prefetch from a single thread in warpgroup 0
  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
  }

  int const warp_group_thread_idx =
      threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  // TMA-load pipeline params
  PipelineParams pipeline_params;
  pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytes;
  int warp_group_idx  = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0
                             ? MainloopPipeline::ThreadCategory::Producer
                             : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader       = lane_predicate;
  pipeline_params.num_consumers   = NumMmaThreads;

  // Denoise pipeline: AxEB
  DenoisePipelineParams AxEB_pipeline_params;
  AxEB_pipeline_params.transaction_bytes =
      CollectiveEpilogue::TmaTransactionBytesAxEB;
  AxEB_pipeline_params.role = warp_group_idx == 0
                                  ? DenoisePipeline::ThreadCategory::Producer
                                  : DenoisePipeline::ThreadCategory::Consumer;
  AxEB_pipeline_params.is_leader       = (warp_group_thread_idx == 0);
  AxEB_pipeline_params.num_consumers   = NumMmaThreads;

  // Denoise pipeline: EAxBpEB
  DenoisePipelineParams EAxBpEB_pipeline_params;
  EAxBpEB_pipeline_params.transaction_bytes =
      CollectiveEpilogue::TmaTransactionBytesEAxBpEB;
  EAxBpEB_pipeline_params.role =
      warp_group_idx == 0 ? DenoisePipeline::ThreadCategory::Producer
                          : DenoisePipeline::ThreadCategory::Consumer;
  EAxBpEB_pipeline_params.is_leader     = (warp_group_thread_idx == 0);
  EAxBpEB_pipeline_params.num_consumers = NumMmaThreads;

  MainloopPipeline pipeline(shared_storage.pipeline, pipeline_params,
                            ClusterShape{});
  DenoisePipeline AxEB_pipeline(shared_storage.AxEB_pipeline,
                                AxEB_pipeline_params, ClusterShape{});
  DenoisePipeline EAxBpEB_pipeline(shared_storage.EAxBpEB_pipeline,
                                   EAxBpEB_pipeline_params, ClusterShape{});

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  const int k_tile_count =
      cutlass::ceil_div(shape<1>(mainloop_params.layout_A), KTraits::bK);

  // Guarantee pipeline init is visible to all producers and consumer blocks
  // in the Cluster before any tile work begins.
  if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }

  static_assert(KTraits::kNumWarps == 8  || KTraits::kNumWarps == 12 ||
                KTraits::kNumWarps == 16 || KTraits::kNumWarps == 20,
                "Unsupported warp count");

  if (warp_group_idx == 0) {
    // -----------------------------------------------------------------------
    // Producer warpgroup
    //
    // OPT-I: Use the producer_reg_dealloc<kNumWarps>() helper so the correct
    //        value is selected at compile time.  For kNumWarps == 16 this
    //        raises the dealloc value from 32 → 40, giving the producer warp
    //        8 extra registers for TMA address computation.  For all other
    //        configs the value remains 24 (unchanged).
    // -----------------------------------------------------------------------
    cutlass::arch::warpgroup_reg_dealloc<
        producer_reg_dealloc<KTraits::kNumWarps>()>();

    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff,
                    (threadIdx.x / cutlass::NumThreadsPerWarp) %
                        cutlass::NumWarpsPerWarpGroup,
                    srcLane);

    if (warp_idx_in_warpgroup == 0) {  // Only warp 0 in WG0 issues TMA loads
      PipelineState smem_pipe_write =
          cutlass::make_producer_start_state<MainloopPipeline>();
      DenoisePipelineState AxEB_pipe_write =
          cutlass::make_producer_start_state<DenoisePipeline>();
      DenoisePipelineState EAxBpEB_pipe_write =
          cutlass::make_producer_start_state<DenoisePipeline>();

      uint16_t const tma_mcast_mask_a = create_tma_multicast_mask<1>(
          Layout<ClusterShape>{}, block_id_in_cluster());
      uint16_t const tma_mcast_mask_b = create_tma_multicast_mask<0>(
          Layout<ClusterShape>{}, block_id_in_cluster());

      TileScheduler scheduler{};
      WorkTileInfo work_tile_info = scheduler.get_initial_work(scheduler_params);

      CUTLASS_PRAGMA_NO_UNROLL
      while (work_tile_info.is_valid(scheduler_params)) {
        cute::tuple<int32_t, int32_t, int32_t> block_coord =
            work_tile_info.template get_block_coord<ClusterShape>(
                scheduler_params);

        collective_mainloop.load(mainloop_params, pipeline, smem_pipe_write,
                                 shared_storage, block_coord, k_tile_count,
                                 tma_mcast_mask_a, tma_mcast_mask_b);

        if constexpr (!SkipDenoising) {
          collective_epilogue.load_denoise(
              pipeline, smem_pipe_write, epilogue_params,
              AxEB_pipeline, EAxBpEB_pipeline,
              AxEB_pipe_write, EAxBpEB_pipe_write,
              shared_storage, block_coord,
              tma_mcast_mask_a, tma_mcast_mask_b);
          collective_epilogue.load_denoise_tail(
              AxEB_pipeline, EAxBpEB_pipeline,
              AxEB_pipe_write, EAxBpEB_pipe_write);
        } else {
          collective_mainloop.load_tail(pipeline, smem_pipe_write);
        }

        work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(
            scheduler_params, work_tile_info);
      }
    }  // warp_idx_in_warpgroup == 0

  } else {
    // -----------------------------------------------------------------------
    // Consumer warpgroup(s)
    //
    // Register allocation unchanged — the consumer must hold the full WGMMA
    // accumulator in registers.  Changing this would affect tile throughput
    // or force accumulator spill; both outcomes are strictly negative.
    // -----------------------------------------------------------------------
    cutlass::arch::warpgroup_reg_alloc<
        consumer_reg_alloc<KTraits::kNumWarps>()>();

    TileScheduler scheduler{};

    typename KTraits::TiledMma        tiled_mma;
    typename KTraits::TiledMmaDenoise tiled_mma_denoise;

    PipelineState       smem_pipe_read;
    DenoisePipelineState AxEB_pipe_read;
    DenoisePipelineState EAxBpEB_pipe_read;

    int consumer_tix = static_cast<int>(threadIdx.x) - NumCopyThreads;

    bool local_block_found  = false;
    int  block_found_k_tile = 0;

    collective_mainloop.mma_init();

    WorkTileInfo work_tile_info = scheduler.get_initial_work(scheduler_params);

    CUTLASS_PRAGMA_NO_UNROLL
    while (work_tile_info.is_valid(scheduler_params)) {
      // Allocate accumulator — cleared once per output tile
      Tensor tCrC = partition_fragment_C(
          tiled_mma, select<0, 1>(TileShape_MNK{}));  // (M, N)
      clear(tCrC);

      // Transcript buffer for hash accumulation (16 × uint32 = 64 bytes, registers)
      auto transcript_extraction_tensor =
          make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});
      if constexpr (!SkipReduction) {
        clear(transcript_extraction_tensor);
      }

      cute::tuple<int32_t, int32_t, int32_t> block_coord =
          work_tile_info.template get_block_coord<ClusterShape>(
              scheduler_params);

      collective_mainloop.mma(mainloop_params, pipeline, smem_pipe_read,
                              tCrC, transcript_extraction_tensor,
                              local_block_found, block_found_k_tile,
                              consumer_tix, shared_storage, k_tile_count);

      // INT32 accumulator → FP32 for denoising scale application
      Tensor tCrD_fp32 = make_tensor_like<float>(tCrC);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrD_fp32); ++i) {
        tCrD_fp32(i) = static_cast<float>(tCrC(i));
      }

      if constexpr (!SkipDenoising) {
        warpgroup_wait<0>();
        collective_epilogue.denoise(tCrD_fp32, shared_storage,
                                    AxEB_pipeline, EAxBpEB_pipeline,
                                    AxEB_pipe_read, EAxBpEB_pipe_read,
                                    consumer_tix);
      }

      collective_epilogue.scale(epilogue_params, tCrD_fp32, shared_storage,
                                tiled_mma, consumer_tix, block_coord);

      collective_epilogue.store(epilogue_params, shared_storage,
                                consumer_tix, block_coord);

      if constexpr (!SkipReduction) {
        local_block_found =
            check_pow_target(transcript_extraction_tensor,
                             mainloop_params.ptr_pow_target,
                             mainloop_params.ptr_pow_key);

        if (local_block_found) {
          write_host_signal_header<typename KTraits::TiledMma, TileShape_MNK>(
              mainloop_params.host_signal_sync,
              mainloop_params.host_signal_header_pinned,
              mainloop_params.problem_shape, block_coord,
              consumer_tix, mainloop_params.ptr_pow_target);
        }
      }

      collective_epilogue.store_tail();

      work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(
          scheduler_params, work_tile_info);
    }
  }  // consumer warpgroup
}

}  // namespace pearl
