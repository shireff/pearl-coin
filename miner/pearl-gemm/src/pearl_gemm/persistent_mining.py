"""
Persistent GPU Mining Pipeline

This module provides a persistent GPU mining pipeline that keeps the GPU worker
resident across multiple mining batches, eliminating kernel launch overhead.
"""
import torch
from typing import Optional, Tuple

from pearl_gemm.test_components import (
    gpu_mine_batch,
    gpu_jackpot_hash,
)


class PersistentGpuMiner:
    """Persistent GPU mining pipeline.

    Keeps GPU worker resident across multiple mining batches,
    eliminating kernel launch overhead and reducing CPU-GPU synchronization.

    Usage:
        miner = PersistentGpuMiner(max_jackpots=65536)
        miner.set_buffers(a_noised, b_noised_t, a_rows, b_cols)
        miner.launch_worker()

        for batch in batches:
            miner.enqueue_jobs(jobs, num_jobs)
            miner.wait_for_completion()
            jackpots = miner.get_jackpots()
            hashes = miner.get_hashes()
            # Check results on CPU...

        miner.stop_worker()
        miner.destroy()
    """

    def __init__(self, max_jackpots: int = 65536):
        """Initialize persistent GPU miner.

        Args:
            max_jackpots: Maximum number of jackpot arrays to store.
        """
        self.state = torch.ops.pearl_gemm.create_persistent_pipeline(max_jackpots)
        if self.state == 0:
            raise RuntimeError("Failed to create persistent pipeline")
        self.max_jackpots = max_jackpots
        self._jackpots_ptr = None
        self._hashes_ptr = None
        self._jackpots_size = 0
        self._hashes_size = 0

    def set_buffers(
        self,
        a_noised: torch.Tensor,
        b_noised_t: torch.Tensor,
        a_rows: torch.Tensor,
        b_cols: torch.Tensor
    ) -> None:
        """Set global data buffers for the pipeline.

        Args:
            a_noised: Noised A matrices (num_jobs x m x k, int32)
            b_noised_t: Noised B^T matrices (num_jobs x n x k, int32)
            a_rows: Row indices (P x tile_h, int32)
            b_cols: Column indices (Q x tile_w, int32)
        """
        torch.ops.pearl_gemm.pipeline_set_buffers(
            self.state, a_noised, b_noised_t, a_rows, b_cols
        )

    def launch_worker(self) -> None:
        """Launch the persistent GPU worker kernel."""
        torch.ops.pearl_gemm.launch_persistent_worker(self.state)

    def stop_worker(self) -> None:
        """Stop the persistent GPU worker kernel."""
        torch.ops.pearl_gemm.stop_persistent_worker(self.state)

    def enqueue_jobs(self, jobs: torch.Tensor, num_jobs: int) -> None:
        """Enqueue jobs for GPU processing.

        Args:
            jobs: Job descriptors tensor (num_jobs x PersistentJob)
            num_jobs: Number of jobs to enqueue
        """
        torch.ops.pearl_gemm.pipeline_enqueue_jobs(self.state, jobs, num_jobs)

    def wait_for_completion(self) -> None:
        """Wait for all queued jobs to complete."""
        torch.ops.pearl_gemm.pipeline_wait_for_completion(self.state)

    def get_jackpots(self) -> torch.Tensor:
        """Get jackpot buffer for CPU readback.

        Returns:
            Jackpot tensor (num_jobs x num_combos x 16, uint32)
        """
        if self._jackpots_ptr is None:
            size = torch.ops.pearl_gemm.pipeline_get_jackpots_size(self.state)
            self._jackpots_size = size
            # Create a tensor that wraps the GPU memory
            # Note: This requires careful memory management
            raise NotImplementedError("Direct GPU memory access not yet implemented")

    def get_hashes(self) -> torch.Tensor:
        """Get hash buffer for CPU readback.

        Returns:
            Hash tensor (num_jobs x num_combos x 8, uint32)
        """
        if self._hashes_ptr is None:
            size = torch.ops.pearl_gemm.pipeline_get_hashes_size(self.state)
            self._hashes_size = size
            raise NotImplementedError("Direct GPU memory access not yet implemented")

    def destroy(self) -> None:
        """Destroy the persistent pipeline and free all resources."""
        if self.state != 0:
            torch.ops.pearl_gemm.destroy_persistent_pipeline(self.state)
            self.state = 0


# Convenience function for single-batch usage (backward compatible)
def gpu_mine_batch_persistent(
    a_noised: torch.Tensor,
    b_noised_t: torch.Tensor,
    a_rows_data: torch.Tensor,
    b_cols_data: torch.Tensor,
    jobs: torch.Tensor,
    num_jobs: int,
    P: int,
    Q: int,
    tile_h: int,
    tile_w: int,
    m: int,
    n: int,
    k: int,
    rank: int,
) -> torch.Tensor:
    """GPU-native batch mining with persistent pipeline.

    This function uses a persistent GPU pipeline to process mining jobs,
    reducing kernel launch overhead.

    Args:
        a_noised: Noised A matrices (num_jobs x m x k, int32)
        b_noised_t: Noised B^T matrices (num_jobs x n x k, int32)
        a_rows_data: Row indices (P x tile_h, int32)
        b_cols_data: Column indices (Q x tile_w, int32)
        jobs: Mining job descriptors (num_jobs x MiningJob)
        num_jobs: Number of mining jobs
        P: Number of a_rows partitions
        Q: Number of b_cols partitions
        tile_h: Tile height
        tile_w: Tile width
        m: Matrix A rows
        n: Matrix B cols
        k: Matrix inner dim
        rank: Noise rank

    Returns:
        (num_jobs x P*Q x 16) uint32 tensor with jackpot arrays
    """
    # For now, fall back to the original implementation
    # Once persistent pipeline is fully integrated, this will use it
    return gpu_mine_batch(
        a_noised, b_noised_t, a_rows_data, b_cols_data,
        jobs, num_jobs, P, Q, tile_h, tile_w, m, n, k, rank
    )
