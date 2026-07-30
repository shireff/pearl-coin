#!/usr/bin/env python3
"""Standalone GPU mining node without vLLM dependency.

Replaces the vLLM inference path with direct pearl_gemm CUDA kernel
calls: noise_gen() -> gpu_mine_batch() -> gpu_jackpot_hash().

All job acquisition, gateway communication, and proof submission use
the existing miner-base and pearl-gateway infrastructure unchanged.
"""

import torch
from miner_base.async_loop_manager import AsyncLoopManager
from miner_base.gateway_client import MiningClient
from miner_base.settings import MinerSettings
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.submission_service import SubmissionService

from pearl_gemm import gpu_jackpot_hash, gpu_mine_batch, noise_gen


class GpuMiningBackend:
    """Replaces vLLM inference with direct GPU mining kernels."""

    @staticmethod
    def mine(
        A_noised,
        B_noised_t,
        a_rows,
        b_cols,
        pow_target,
        pow_key,
        num_jobs,
        m,
        n,
        k,
        rank,
    ):
        """Run noise_gen, gpu_mine_batch, gpu_jackpot_hash pipeline."""
        noise_gen(
            R=rank,
            num_threads=64,
            key_A=pow_key,
            key_B=pow_key,
        )

        jackpots = gpu_mine_batch(
            a_noised=A_noised,
            b_noised_t=B_noised_t,
            a_rows_data=a_rows,
            b_cols_data=b_cols,
            jobs=torch.empty((num_jobs,), dtype=torch.int32, device=A_noised.device),
            num_jobs=num_jobs,
            P=a_rows.shape[0],
            Q=b_cols.shape[0],
            tile_h=16,
            tile_w=16,
            m=m,
            n=n,
            k=k,
            rank=rank,
        )

        keys = pow_key.unsqueeze(0).expand(num_jobs, 8).to(torch.uint32, device=A_noised.device)
        num_combos = a_rows.shape[0] * b_cols.shape[0]
        hashes = torch.empty(
            (num_jobs, num_combos, 8),
            dtype=torch.uint32,
            device=A_noised.device,
        )

        gpu_jackpot_hash(
            jackpots=jackpots,
            keys=keys,
            hashes=hashes,
            num_candidates=num_jobs,
            num_combos=num_combos,
        )

        return jackpots, hashes


class PearlGpuMiner(AsyncLoopManager):
    """Drop-in replacement for the vLLM-based miner that uses
    AsyncLoopManager from miner_base with GPU mining kernels.
    """

    def __init__(self, miner_rpc_config, miner_settings=None):
        super().__init__(miner_rpc_config, miner_settings)
        self._submission_service: SubmissionService | None = None

    def handle_mining_job(self, mining_job: MiningJob):
        """Process a single mining job using pearl_gemm CUDA kernels."""
        from pearl_gateway.comm.dataclasses import OpenedBlockInfo
        from pearl_mining import PlainProof
        from pearl_gateway.proof_generator import ProofGenerator

        A_noised = mining_job.A_noised
        B_noised_t = mining_job.B_noised_t
        a_rows = mining_job.a_rows
        b_cols = mining_job.b_cols
        pow_target = mining_job.pow_target
        pow_key = mining_job.pow_key

        jackpots, hashes = GpuMiningBackend.mine(
            A_noised=A_noised,
            B_noised_t=B_noised_t,
            a_rows=a_rows,
            b_cols=b_cols,
            pow_target=pow_target,
            pow_key=pow_key,
            num_jobs=mining_job.num_jobs,
            m=mining_job.m,
            n=mining_job.n,
            k=mining_job.k,
            rank=mining_job.noise_rank,
        )

        winner_indices = torch.nonzero(
            hashes <= pow_target.unsqueeze(0).unsqueeze(0), as_tuple=False
        )

        if len(winner_indices) == 0:
            return None

        open_block_info = OpenedBlockInfo(
            A=A_noised,
            B_t=B_noised_t,
            A_row_indices=a_rows,
            B_column_indices=b_cols,
            noise_rank=mining_job.noise_rank,
        )

        plain_proof = ProofGenerator.generate_proof(
            jackpots=jackpots,
            hashes=hashes,
            open_block_info=open_block_info,
        )

        return plain_proof

    def run(self):
        """Start the mining loop."""
        self.start()
        try:
            while True:
                job = self.get_mining_job()
                if job is None:
                    continue
                proof = self.handle_mining_job(job)
                if proof is not None:
                    self.handle_submit_block(proof, job)
        except KeyboardInterrupt:
            self.stop()


def main():
    from miner_utils import get_logger
    from pearl_gateway.config import MinerRpcConfig

    _LOGGER = get_logger("miner_gpu")

    miner_rpc_config = MinerRpcConfig()
    miner_settings = MinerSettings()

    miner = PearlGpuMiner(miner_rpc_config, miner_settings)
    _LOGGER.info("Starting GPU miner (no vLLM)")
    miner.run()


if __name__ == "__main__":
    main()