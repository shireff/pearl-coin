#!/usr/bin/env python3
"""Standalone GPU mining node without vLLM dependency.

Directly uses pearl_gemm CUDA kernels for noise generation,
batch mining, and jackpot hashing on the GPU.

Flow:
  noise_gen() -> gpu_mine_batch() -> gpu_jackpot_hash() -> submit winner
"""

import torch
import numpy as np
from pearl_gemm import (
    noise_gen,
    gpu_mine_batch,
    gpu_jackpot_hash,
    BLAKE3_DIGEST_SIZE_U32,
    make_pow_target_tensor,
)

JACKPOT_SIZE = 16


def run_mining_job(
    m,
    n,
    k,
    rank,
    P,
    Q,
    tile_h,
    tile_w,
    num_jobs,
    pow_target_int,
    pow_key_bytes,
    device="cuda",
):
    """Run a single batch mining job on the GPU.

    Args:
        m, n, k: Matrix dimensions
        rank: Noise rank
        P, Q: Partition counts for row/column tiling
        tile_h, tile_w: Tile dimensions
        num_jobs: Number of mining jobs to process
        pow_target_int: Proof-of-work target as uint256 integer
        pow_key_bytes: 32-byte BLAKE3 key for PoW hash

    Returns:
        dict with winners, jackpots, hashes
    """
    # ---- 1. Generate noise matrices ----
    num_threads = 64
    EAL = torch.empty((m, rank), dtype=torch.int8, device=device)
    EAL_fp16 = torch.empty((m, rank), dtype=torch.float16, device=device)
    EAR_R_major = torch.empty((k, rank), dtype=torch.int8, device=device)
    EAR_K_major = torch.empty((rank, k), dtype=torch.int8, device=device)
    EBL_R_major = torch.empty((k, rank), dtype=torch.int8, device=device)
    EBL_K_major = torch.empty((rank, k), dtype=torch.int8, device=device)
    EBR = torch.empty((n, rank), dtype=torch.int8, device=device)
    EBR_fp16 = torch.empty((n, rank), dtype=torch.float16, device=device)

    noise_gen(
        R=rank,
        num_threads=num_threads,
        EAL=EAL,
        EAL_fp16=EAL_fp16,
        EAR_R_major=EAR_R_major,
        EAR_K_major=EAR_K_major,
        EBL_R_major=EBL_R_major,
        EBL_K_major=EBL_K_major,
        EBR=EBR,
        EBR_fp16=EBR_fp16,
        key_A=pow_key_bytes,
        key_B=pow_key_bytes,
    )

    # ---- 2. Build job descriptors ----
    MiningJob = torch.classes.pearl_gemm.MiningJob if hasattr(
        torch.classes, "pearl_gemm"
    ) else None

    a_noised = torch.empty((num_jobs, m, k), dtype=torch.int32, device=device)
    b_noised_t = torch.empty((num_jobs, n, k), dtype=torch.int32, device=device)
    a_rows_data = torch.empty((P, tile_h), dtype=torch.int32, device=device)
    b_cols_data = torch.empty((Q, tile_w), dtype=torch.int32, device=device)

    jobs = torch.empty(
        (num_jobs,),
        dtype=torch.int32,
        device=device,
    )

    # ---- 3. GPU batch mining ----
    jackpots = gpu_mine_batch(
        a_noised=a_noised,
        b_noised_t=b_noised_t,
        a_rows_data=a_rows_data,
        b_cols_data=b_cols_data,
        jobs=jobs,
        num_jobs=num_jobs,
        P=P,
        Q=Q,
        tile_h=tile_h,
        tile_w=tile_w,
        m=m,
        n=n,
        k=k,
        rank=rank,
    )

    # ---- 4. GPU jackpot hash ----
    keys = torch.empty((num_jobs, 8), dtype=torch.uint32, device=device)
    hashes = torch.empty(
        (num_jobs, P * Q, BLAKE3_DIGEST_SIZE_U32),
        dtype=torch.uint32,
        device=device,
    )

    gpu_jackpot_hash(
        jackpots=jackpots,
        keys=keys,
        hashes=hashes,
        num_candidates=num_jobs,
        num_combos=P * Q,
    )

    # ---- 5. Difficulty check on CPU ----
    pow_target = make_pow_target_tensor(pow_target_int, device="cpu")
    winning_mask = hashes <= pow_target.unsqueeze(0).unsqueeze(0)
    winning_indices = torch.nonzero(winning_mask, as_tuple=True)

    return {
        "winning_indices": winning_indices,
        "jackpots": jackpots,
        "hashes": hashes,
    }


def main():
    """Entry point for standalone GPU mining node."""
    import argparse

    parser = argparse.ArgumentParser(description="Standalone GPU miner for Pearl GEMM")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--P", type=int, default=4)
    parser.add_argument("--Q", type=int, default=4)
    parser.add_argument("--tile-h", type=int, default=16)
    parser.add_argument("--tile-w", type=int, default=16)
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--pow-target", type=int, default=0, help="PoW target as uint256")
    parser.add_argument("--pow-key", type=str, default="", help="32-byte hex PoW key")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pow_key_bytes = (
        bytes.fromhex(args.pow_key) if args.pow_key else b"\x00" * 32
    )

    result = run_mining_job(
        m=args.m,
        n=args.n,
        k=args.k,
        rank=args.rank,
        P=args.P,
        Q=args.Q,
        tile_h=args.tile_h,
        tile_w=args.tile_w,
        num_jobs=args.num_jobs,
        pow_target_int=args.pow_target,
        pow_key_bytes=pow_key_bytes,
        device=args.device,
    )

    print(f"Mining result: {len(result['winning_indices'][0])} winners found")
    return result


if __name__ == "__main__":
    main()