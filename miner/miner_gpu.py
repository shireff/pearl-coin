#!/usr/bin/env python3
"""Standalone GPU mining node without vLLM dependency.

Uses pearl_gemm CUDA kernels directly for noise generation,
batch mining, and jackpot hashing on the GPU. Communicates
with the Pearl node gateway to receive mining jobs and submit
winning proofs.

Flow:
  1. Connect to gateway node
  2. Receive MiningJob (block template) from node
  3. Prepare A_noised, B_noised_t, a_rows, b_cols tensors
  4. noise_gen()  ->  generate noise matrices E{A,B}{L,R}
  5. gpu_mine_batch()  ->  evaluate tiles on GPU
  6. gpu_jackpot_hash()  ->  compute BLAKE3 hashes on GPU
  7. Extract winner_flags  ->  find winning tiles
  8. submit_solution()  ->  send proof back to node
  9. Loop back to step 2
"""

import argparse
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch

from pearl_gemm import (
    gpu_jackpot_hash,
    gpu_mine_batch,
    noise_gen,
    BLAKE3_DIGEST_SIZE_U32,
)


@dataclass
class MinerConfig:
    """Configuration for the standalone GPU miner."""

    gateway_url: str = "http://localhost:8545"
    noise_rank: int = 256
    tile_h: int = 16
    tile_w: int = 16
    pipeline_stages: int = 2
    batch_size: int = 1
    gpu_device: str = "cuda"
    poll_interval_s: float = 1.0
    max_concurrent_jobs: int = 4


@dataclass
class MinerJob:
    """A single mining job received from the gateway node."""

    job_id: int
    m: int
    n: int
    k: int
    A: torch.Tensor
    B_T: torch.Tensor
    a_row_indices: torch.Tensor
    b_col_indices: torch.Tensor
    pow_target: torch.Tensor
    pow_key: torch.Tensor
    num_tiles_h: int
    num_tiles_w: int


@dataclass
class MinerResult:
    """Result of mining a single job."""

    job_id: int
    winners: list[int]
    jackpots: torch.Tensor
    hashes: torch.Tensor


class GatewayClient:
    """Minimal gateway client for job retrieval and proof submission."""

    def __init__(self, url: str):
        self._url = url

    def get_mining_job(self) -> Optional[MinerJob]:
        """Fetch a mining job from the gateway node.

        Returns None if no job is currently available.
        """
        try:
            import json

            import urllib.request

            with urllib.request.urlopen(
                self._url + "/get_mining_job", timeout=10
            ) as resp:
                data = json.loads(resp.read().decode())

            if data is None:
                return None

            A = torch.tensor(data["A"], dtype=torch.int32, device=self._device)
            B_T = torch.tensor(data["B_T"], dtype=torch.int32, device=self._device)
            a_rows = torch.tensor(
                data["a_row_indices"], dtype=torch.int32, device=self._device
            )
            b_cols = torch.tensor(
                data["b_col_indices"], dtype=torch.int32, device=self._device
            )
            pow_target = torch.tensor(
                data["pow_target"], dtype=torch.uint32, device="cpu"
            )
            pow_key = torch.tensor(
                data["pow_key"], dtype=torch.uint32, device="cpu"
            )

            return MinerJob(
                job_id=data.get("job_id", 0),
                m=data["m"],
                n=data["n"],
                k=data["k"],
                A=A,
                B_T=B_T,
                a_row_indices=a_rows,
                b_col_indices=b_cols,
                pow_target=pow_target,
                pow_key=pow_key,
                num_tiles_h=data.get("num_tiles_h", 1),
                num_tiles_w=data.get("num_tiles_w", 1),
            )
        except Exception as e:
            print(f"[miner_gpu] Failed to fetch job: {e}", file=sys.stderr)
            return None

    def submit_solution(
        self,
        job_id: int,
        winner_indices: list[int],
        jackpots: torch.Tensor,
        hashes: torch.Tensor,
    ) -> bool:
        """Submit a winning solution back to the gateway node."""
        try:
            import json

            import urllib.request

            payload = {
                "job_id": job_id,
                "winner_indices": winner_indices,
                "jackpots": jackpots.cpu().tolist(),
                "hashes": hashes.cpu().tolist(),
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url + "/submit_solution",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
            return result.get("accepted", False)
        except Exception as e:
            print(f"[miner_gpu] Failed to submit solution: {e}", file=sys.stderr)
            return False

    @property
    def _device(self):
        return "cpu"


def prepare_noised_tensors(
    A: torch.Tensor,
    B_T: torch.Tensor,
    noise_rank: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate noise and produce noised A and B^T matrices.

    Args:
        A: Original A matrix (m x k), int32
        B_T: Original B^T matrix (n x k), int32
        noise_rank: Rank of the noise matrices
        device: Target CUDA device

    Returns:
        A_noised, B_noised_t, a_rows, b_cols
    """
    m, k = A.shape
    n, k2 = B_T.shape
    assert k == k2, f"Dimension mismatch: k={k} != k2={k2}"

    tile_h = noise_rank
    tile_w = noise_rank
    P = math.ceil(m / tile_h)
    Q = math.ceil(n / tile_w)

    EAL = torch.empty((m, noise_rank), dtype=torch.int8, device=device)
    EAL_fp16 = torch.empty((m, noise_rank), dtype=torch.float16, device=device)
    EAR_R_major = torch.empty((k, noise_rank), dtype=torch.int8, device=device)
    EAR_K_major = torch.empty(
        (noise_rank, k), dtype=torch.int8, device=device
    )
    EBL_R_major = torch.empty((k, noise_rank), dtype=torch.int8, device=device)
    EBL_K_major = torch.empty(
        (noise_rank, k), dtype=torch.int8, device=device
    )
    EBR = torch.empty((n, noise_rank), dtype=torch.int8, device=device)
    EBR_fp16 = torch.empty((n, noise_rank), dtype=torch.float16, device=device)

    noise_gen(
        R=noise_rank,
        num_threads=64,
        EAL=EAL,
        EAL_fp16=EAL_fp16,
        EAR_R_major=EAR_R_major,
        EAR_K_major=EAR_K_major,
        EBL_R_major=EBL_R_major,
        EBL_K_major=EBL_K_major,
        EBR=EBR,
        EBR_fp16=EBR_fp16,
    )

    from pearl_gemm import noise_A, noise_B

    a_rows = torch.empty((P, tile_h), dtype=torch.int32, device=device)
    b_cols = torch.empty((Q, tile_w), dtype=torch.int32, device=device)
    ApEA = torch.empty((m, k), dtype=torch.int32, device=device)
    BpEB = torch.empty((n, k), dtype=torch.int32, device=device)
    AxEBL = torch.empty((m, noise_rank), dtype=torch.int32, device=device)
    EARxBpEB = torch.empty((n, noise_rank), dtype=torch.int32, device=device)

    noise_A(
        A=A,
        EAL=EAL,
        AxEBL=AxEBL,
        ApEA=ApEA,
        EAR=EAR_R_major,
        EBL=EBL_R_major,
        tile_size_m=tile_h,
        tile_size_k=noise_rank,
        pipeline_stages=2,
    )

    noise_B(
        B=B_T,
        EBR=EBR,
        EARxBpEB=EARxBpEB,
        BpEB=BpEB,
        EAR=EAR_K_major,
        EBL=EBL_K_major,
        tile_size_n=tile_w,
        tile_size_k=noise_rank,
        pipeline_stages=2,
    )

    a_rows = ApEA
    b_cols = BpEB

    A_noised = ApEA
    B_noised_t = BpEB

    return A_noised, B_noised_t, a_rows, b_cols


def mine_single_job(
    job: MinerJob,
    config: MinerConfig,
) -> MinerResult:
    """Mine a single job on the GPU.

    Runs the full pipeline:
      noise_gen -> gpu_mine_batch -> gpu_jackpot_hash -> winner detection
    """
    device = config.gpu_device
    m, n, k = job.m, job.n, job.k
    rank = config.noise_rank
    P = job.num_tiles_h
    Q = job.num_tiles_w

    A_noised = job.A.to(device, dtype=torch.int32)
    B_noised_t = job.B_T.to(device, dtype=torch.int32)
    a_rows = job.a_row_indices.to(device, dtype=torch.int32)
    b_cols = job.b_col_indices.to(device, dtype=torch.int32)

    jackpots = gpu_mine_batch(
        a_noised=A_noised,
        b_noised_t=B_noised_t,
        a_rows_data=a_rows,
        b_cols_data=b_cols,
        jobs=torch.empty((1,), dtype=torch.int32, device=device),
        num_jobs=1,
        P=P,
        Q=Q,
        tile_h=config.tile_h,
        tile_w=config.tile_w,
        m=m,
        n=n,
        k=k,
        rank=rank,
    )

    keys = job.pow_key.unsqueeze(0).expand(1, 8).to(device, dtype=torch.uint32)
    hashes = torch.empty(
        (1, P * Q, BLAKE3_DIGEST_SIZE_U32),
        dtype=torch.uint32,
        device=device,
    )

    gpu_jackpot_hash(
        jackpots=jackpots,
        keys=keys,
        hashes=hashes,
        num_candidates=1,
        num_combos=P * Q,
    )

    pow_target = job.pow_target.to(device, dtype=torch.uint32)
    winner_mask = hashes <= pow_target.unsqueeze(0).unsqueeze(0)
    winner_positions = torch.nonzero(winner_mask, as_tuple=False)

    winners = (
        winner_positions[:, 1].tolist() if len(winner_positions) > 0 else []
    )

    return MinerResult(
        job_id=job.job_id,
        winners=winners,
        jackpots=jackpots,
        hashes=hashes,
    )


def mining_loop(
    config: MinerConfig,
    gateway: GatewayClient,
    executor: ThreadPoolExecutor,
):
    """Main mining loop: fetch jobs, mine them, submit winners."""
    print(f"[miner_gpu] Starting GPU mining loop on {config.gpu_device}")
    print(f"[miner_gpu] Gateway: {config.gateway_url}")
    print(f"[miner_gpu] Noise rank: {config.noise_rank}, Tile: {config.tile_h}x{config.tile_w}")

    while True:
        try:
            job = gateway.get_mining_job()
            if job is None:
                time.sleep(config.poll_interval_s)
                continue

            print(f"[miner_gpu] Received job {job.job_id}: {job.m}x{job.n}x{job.k}")

            result = executor.submit(mine_single_job, job, config).result()

            if result.winners:
                print(
                    f"[miner_gpu] Job {result.job_id}: "
                    f"{len(result.winners)} winner(s) found"
                )
                accepted = gateway.submit_solution(
                    job_id=result.job_id,
                    winner_indices=result.winners,
                    jackpots=result.jackpots,
                    hashes=result.hashes,
                )
                if accepted:
                    print(f"[miner_gpu] Solution for job {result.job_id} accepted")
                else:
                    print(f"[miner_gpu] Solution for job {result.job_id} rejected")
            else:
                print(f"[miner_gpu] Job {result.job_id}: no winners this round")

        except KeyboardInterrupt:
            print("\n[miner_gpu] Shutting down...")
            break
        except Exception as e:
            print(f"[miner_gpu] Error in mining loop: {e}", file=sys.stderr)
            time.sleep(config.poll_interval_s)


def main():
    """Entry point for the standalone GPU miner."""
    parser = argparse.ArgumentParser(
        description="Standalone GPU miner for Pearl GEMM (no vLLM)"
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default="http://localhost:8545",
        help="URL of the Pearl gateway node",
    )
    parser.add_argument(
        "--noise-rank",
        type=int,
        default=256,
        help="Rank of noise matrices (default: 256)",
    )
    parser.add_argument(
        "--tile-h", type=int, default=16, help="Tile height (default: 16)"
    )
    parser.add_argument(
        "--tile-w", type=int, default=16, help="Tile width (default: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device (default: cuda)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between job polls (default: 1.0)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent GPU jobs (default: 4)",
    )
    args = parser.parse_args()

    config = MinerConfig(
        gateway_url=args.gateway_url,
        noise_rank=args.noise_rank,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        gpu_device=args.device,
        poll_interval_s=args.poll_interval,
        max_concurrent_jobs=args.max_concurrent,
    )

    gateway = GatewayClient(config.gateway_url)
    executor = ThreadPoolExecutor(max_workers=config.max_concurrent_jobs)

    try:
        mining_loop(config, gateway, executor)
    finally:
        executor.shutdown(wait=False)
        gateway.close()


if __name__ == "__main__":
    main()