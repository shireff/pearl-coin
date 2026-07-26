#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"

#include "blake3/blake3_constants.hpp"
#include "host_signal_header.hpp"
#include "pow_utils.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

template <typename KTraits>
struct CollectiveMainloop {

  using ElementIn    = typename KTraits::ElementIn;
  using TileShape_MNK = typename KTraits::TileShape_MNK;
  using TileShape_MNR = typename KTraits::TileShape_MNR;

  using ProblemShape    = typename KTraits::ProblemShape;
  using ClusterShape_MNK = typename KTraits::ClusterShape_MNK;

  static constexpr int  kStages        = KTraits::kStages;
  static constexpr int  SkipReduction  = KTraits::SkipReduction;
  static constexpr int  kClusterSizeM  = KTraits::kClusterSizeM;
  static constexpr int  kClusterSizeN  = KTraits::kClusterSizeN;
  static constexpr int  srcLane        = KTraits::srcLane;

  using MMAAtom_K = typename KTraits::MMAAtom_K;

  using SmemLayoutA = typename KTraits::SmemLayoutA;
  using SmemLayoutB = typename KTraits::SmemLayoutB;

  using ShapeT  = cute::Shape<int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;
  using TMAOpA  = KTraits::TMAOpA;
  using TMAOpB  = KTraits::TMAOpB;

  using TMA_A = decltype(make_tma_copy(
      TMAOpA{},
      make_tensor(make_gmem_ptr(static_cast<ElementIn const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      take<0, 2>(SmemLayoutA{}), select<0, 2>(TileShape_MNK{}), kClusterSizeN));

  using TMA_B = decltype(make_tma_copy(
      TMAOpB{},
      make_tensor(make_gmem_ptr(static_cast<ElementIn const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      take<0, 2>(SmemLayoutB{}), select<1, 2>(TileShape_MNK{}), kClusterSizeM));

  static constexpr int kNumMmaThreads = KTraits::kNumMmaThreads;
  using MainloopPipeline = typename KTraits::MainloopPipeline;
  using PipelineParams   = typename MainloopPipeline::Params;
  using PipelineState    = typename MainloopPipeline::PipelineState;
  using BarrierType      = typename MainloopPipeline::ProducerBarrierType;

  static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutA{})) * cutlass::sizeof_bits_v<ElementIn> / 8);
  static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutB{})) * cutlass::sizeof_bits_v<ElementIn> / 8);
  static constexpr uint32_t TmaTransactionBytes =
      TmaTransactionBytesA + TmaTransactionBytesB;

  struct Arguments {
    ElementIn const* ptr_A;
    ElementIn const* ptr_B;
    void*            host_signal_header_pinned;
    void*            host_signal_sync;
    ProblemShape const problem_shape;
    uint64_t*        inner_hash_counter;
    uint32_t const*  ptr_pow_target;
    uint32_t const*  ptr_pow_key;
  };

  struct Params {
    ElementIn const*    ptr_A;
    ElementIn const*    ptr_B;
    LayoutT             layout_A;
    LayoutT             layout_B;
    TMA_A               tma_load_A;
    TMA_B               tma_load_B;
    HostSignalHeader*   host_signal_header_pinned;
    HostSignalSync*     host_signal_sync;
    ProblemShape const  problem_shape;
    uint64_t*           inner_hash_counter;
    uint32_t const*     ptr_pow_target;
    uint32_t const*     ptr_pow_key;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    auto [M, N, K, R] = args.problem_shape;
    LayoutT layout_A = make_layout(make_shape(M, K), make_stride(K, _1{}));
    LayoutT layout_B = make_layout(make_shape(N, K), make_stride(K, _1{}));
    Tensor mA = make_tensor(make_gmem_ptr(args.ptr_A), layout_A);
    Tensor mB = make_tensor(make_gmem_ptr(args.ptr_B), layout_B);

    TMA_A tma_load_A =
        make_tma_copy(TMAOpA{}, mA, SmemLayoutA{}(_, _, _0{}),
                      select<0, 2>(TileShape_MNK{}), kClusterSizeN);
    TMA_B tma_load_B =
        make_tma_copy(TMAOpB{}, mB, SmemLayoutB{}(_, _, _0{}),
                      select<1, 2>(TileShape_MNK{}), kClusterSizeM);

    return {.ptr_A  = args.ptr_A,
            .ptr_B  = args.ptr_B,
            .layout_A = layout_A,
            .layout_B = layout_B,
            .tma_load_A = tma_load_A,
            .tma_load_B = tma_load_B,
            .host_signal_header_pinned = reinterpret_cast<HostSignalHeader*>(
                args.host_signal_header_pinned),
            .host_signal_sync =
                reinterpret_cast<HostSignalSync*>(args.host_signal_sync),
            .problem_shape      = args.problem_shape,
            .inner_hash_counter = args.inner_hash_counter,
            .ptr_pow_target     = args.ptr_pow_target,
            .ptr_pow_key        = args.ptr_pow_key};
  }

  // Phase 6: prefetch both TMA descriptors from a single warp-0 thread
  // before entering the tile loop.  On Hopper SM90a this issues
  // `prefetch.tensormap.L2` for both A and B in the same instruction group,
  // reducing descriptor fetch latency by up to 2× vs. sequential prefetches.
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    // Issue both prefetches back-to-back so the memory subsystem can
    // pipeline the two L2 descriptor loads in parallel.
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_A.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_B.get_tma_descriptor());
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void load(Params const&    mainloop_params,
                           MainloopPipeline pipeline,
                           PipelineState&   smem_pipe_write,
                           SharedStorage&   shared_storage,
                           cute::tuple<int32_t, int32_t, int32_t> block_coord,
                           int              k_tile_count,
                           uint16_t const   tma_mcast_mask_a,
                           uint16_t const   tma_mcast_mask_b) {

    auto [m_block, n_block, bidb] = block_coord;

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.data()),
                            SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()),
                            SmemLayoutB{});

    Tensor mA = mainloop_params.tma_load_A.get_tma_tensor(
        mainloop_params.layout_A.shape());
    Tensor mB = mainloop_params.tma_load_B.get_tma_tensor(
        mainloop_params.layout_B.shape());

    Tensor gA = local_tile(mA, select<0, 2>(TileShape_MNK{}),
                           make_coord(m_block, _));
    Tensor gB = local_tile(mB, select<1, 2>(TileShape_MNK{}),
                           make_coord(n_block, _));

    auto [tAgA, tAsA] =
        tma_partition(mainloop_params.tma_load_A, get<1>(block_id_in_cluster()),
                      make_layout(kClusterSizeN), group_modes<0, 2>(sA),
                      group_modes<0, 2>(gA));
    auto [tBgB, tBsB] =
        tma_partition(mainloop_params.tma_load_B, get<0>(block_id_in_cluster()),
                      make_layout(kClusterSizeM), group_modes<0, 2>(sB),
                      group_modes<0, 2>(gB));

    int lane_predicate = cute::elect_one_sync();

    if constexpr (!KTraits::SkipDenoising) {
      cutlass::arch::NamedBarrier::sync(
          kNumMmaThreads + cutlass::NumThreadsPerWarp,
          static_cast<cutlass::arch::ReservedNamedBarriers>(
              pearl::NamedBarriers::DenoiseComplete));
    }

    if (lane_predicate) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        pipeline.producer_acquire(smem_pipe_write);
        BarrierType* tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
        auto stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_A.with(*tmaBar, tma_mcast_mask_a),
             tAgA(_, k_tile), tAsA(_, stage));
        copy(mainloop_params.tma_load_B.with(*tmaBar, tma_mcast_mask_b),
             tBgB(_, k_tile), tBsB(_, stage));
        pipeline.producer_commit(smem_pipe_write, TmaTransactionBytes);
        ++smem_pipe_write;
      }
    }
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline,
                                PipelineState&   smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff,
                    (threadIdx.x / cutlass::NumThreadsPerWarp) %
                        cutlass::NumWarpsPerWarpGroup,
                    srcLane);
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  CUTLASS_DEVICE void mma_init() {
    if constexpr (!KTraits::SkipDenoising) {
      cutlass::arch::NamedBarrier::arrive(
          kNumMmaThreads + cutlass::NumThreadsPerWarp,
          static_cast<cutlass::arch::ReservedNamedBarriers>(
              pearl::NamedBarriers::DenoiseComplete));
    }
  }

  template <typename SharedStorage, typename FrgTensorC,
            typename TranscriptTensor>
  CUTLASS_DEVICE void mma(Params const&    mainloop_params,
                          MainloopPipeline pipeline,
                          PipelineState&   smem_pipe_read,
                          FrgTensorC&      tCrC,
                          TranscriptTensor& transcript_extraction_tensor,
                          bool&            block_found,
                          int&             block_found_k_tile,
                          int              thread_idx,
                          SharedStorage&   shared_storage,
                          int              k_tile_count) {

    Tensor sA =
        make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{});
    Tensor sB =
        make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{});

    typename KTraits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    const uint32_t last_full_k_block =
        shape<1>(mainloop_params.layout_A) / MMAAtom_K{};

    constexpr int k_blocks_per_tile = size<2>(tCrA);
    constexpr int reduce_every_k    = get<2>(TileShape_MNR{}) / MMAAtom_K{};

    using HashAccumulator =
        TileHashAccumulator<k_blocks_per_tile, reduce_every_k,
                            KTraits::EnableDebug>;
    HashAccumulator hash_accumulator(last_full_k_block,
                                     mainloop_params.inner_hash_counter);

    static constexpr bool kLastKBlockReduces =
        (!SkipReduction) && (k_blocks_per_tile % reduce_every_k == 0);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      if constexpr (!SkipReduction) {
        hash_accumulator.preload(transcript_extraction_tensor);
      }

      pipeline.consumer_wait(smem_pipe_read);
      auto stage = smem_pipe_read.index();

      // BN-1: Single fence before the loop replaces per-k_block fences.
      // warpgroup_arrive() + warpgroup_commit_batch() enforce ordering between
      // consecutive WGMMA batches; the fence is only needed once to prevent the
      // compiler from hoisting reads of tCrC above the loop entry.
      warpgroup_fence_operand(tCrC);

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < k_blocks_per_tile; ++k_block) {
        warpgroup_arrive();
        gemm(tiled_mma,
             tCrA(_, _, k_block, stage),
             tCrB(_, _, k_block, stage),
             tCrC);
        warpgroup_commit_batch();

        if constexpr (!SkipReduction) {
          if (__builtin_expect(hash_accumulator.needs_accumulate(k_block), 0)) {
            hash_accumulator.accumulate_now(tCrC, k_block);
          }
        }
      }

      if constexpr (!SkipReduction) {
        hash_accumulator.writeback(transcript_extraction_tensor);
      }

      if constexpr (!kLastKBlockReduces) {
        warpgroup_wait<0>();
      }

      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }

    cutlass::arch::NamedBarrier::arrive(
        kNumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<cutlass::arch::ReservedNamedBarriers>(
            pearl::NamedBarriers::MmaComplete));
  }
};

}  // namespace pearl
