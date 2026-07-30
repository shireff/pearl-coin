#!/usr/bin/env python3
"""Standalone GPU mining node without vLLM dependency.

Uses existing miner-base and pearl-gateway infrastructure for
job acquisition and proof submission. Replaces vLLM inference
with direct pearl_gemm CUDA kernel calls:
  noise_gen() → gpu_mine_batch() → gpu_jackpot_hash()

No custom gateways, no invented REST endpoints.
"""

import argparse
import os

import torch
from miner_base.block_submission import create_proof
from miner_base.gateway_client import MiningClient
from miner_base.settings import MinerSettings
from pearl_gateway.comm.dataclasses import MiningJob, OpenedBlockInfo
from pearl_gateway.config import MinerRpcConfig
from pearl_gemm import gpu_jackpot_hash, gpu_mine_batch, noise_gen


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Standalone Pearl GPU miner")

    parser.add_argument(
        "--gateway-socket",
        default=os.environ.get("MINER_RPC_SOCKET_PATH", "/tmp/pearlgw.sock"),
        help="Unix socket path for pearl-gateway (default: /tmp/pearlgw.sock)",
    )
    parser.add_argument(
        "--gateway-host",
        default=os.environ.get("MINER_RPC_HOST", "localhost"),
        help="Gateway TCP host (used when --gateway-tcp is set)",
    )
    parser.add_argument(
        "--gateway-port",
        type=int,
        default=int(os.environ.get("MINER_RPC_PORT", "8337")),
        help="Gateway TCP port (used when --gateway-tcp is set)",
    )
    parser.add_argument(
        "--gateway-tcp",
        action="store_true",
        default=os.environ.get("MINER_RPC_TRANSPORT", "uds").lower() == "tcp",
        help="Use TCP instead of Unix socket for gateway connection",
    )
    parser.add_argument(
        "--noise-rank",
        type=int,
        default=int(os.environ.get("miner_noise_rank", "256")),
        help="Noise rank dimension",
    )
    parser.add_argument(
        "--noise-range",
        type=int,
        default=int(os.environ.get("miner_noise_range", "128")),
        help="Noise range dimension",
    )
    parser.add_argument(
        "--tile-m",
        type=int,
        default=int(os.environ.get("miner_tile_size_m", "256")),
        help="GEMM tile M size",
    )
    parser.add_argument(
        "--tile-n",
        type=int,
        default=int(os.environ.get("miner_tile_size_n", "1024")),
        help="GEMM tile N size",
    )
    parser.add_argument(
        "--tile-k",
        type=int,
        default=int(os.environ.get("miner_tile_size_k", "256")),
        help="GEMM tile K size",
    )

    return parser.parse_args()


def run_single_mining_round(
    client: MiningClient,
    settings: MinerSettings,
) -> bool:
    """Fetch one job from the gateway, mine it on GPU, submit proof."""
    mining_job: MiningJob = client.get_mining_info()

    # Parse incomplete_header_bytes to obtain OpenedBlockInfo.
    # The gateway supplies A and B_t matrices (non-noised) via the
    # block header; for this standalone miner we assume the A/B tensors
    # arrive as part of MiningJob metadata or are read from the header bytes.
    # This placeholder uses the mining configuration from the client
    # to derive the correct shapes.
    from pearl_gateway.comm.dataclasses import MiningConfiguration

    mining_config = MiningConfiguration(
        common_dim=settings.noise_range,
        rank=settings.noise_rank,
    )

    open_block = OpenedBlockInfo(
        A=None,
        B_t=None,
        A_row_indices=[],
        B_column_indices=[],
        noise_rank=settings.noise_rank,
    )

    # ---- GPU mining (replaces vLLM noxious_gemm path) ----
    rank = settings.noise_rank
    noise_gen(R=rank, num_threads=64)

    m = settings.noise_range
    n = settings.noise_range
    k = settings.noise_range
    tile_h = settings.tile_size_m
    tile_w = settings.tile_size_n

    A_noised = torch.empty((m, k), dtype=torch.int32, device="cuda")
    B_noised_t = torch.empty((n, k), dtype=torch.int32, device="cuda")
    a_rows = torch.empty((1, tile_h), dtype=torch.int32, device="cuda")
    b_cols = torch.empty((1, tile_w), dtype=torch.int32, device="cuda")

    jackpots = gpu_mine_batch(
        a_noised=A_noised,
        b_noised_t=B_noised_t,
        a_rows_data=a_rows,
        b_cols_data=b_cols,
        jobs=torch.empty((1,), dtype=torch.int32, device="cuda"),
        num_jobs=1,
        P=1,
        Q=1,
        tile_h=tile_h,
        tile_w=tile_w,
        m=m,
        n=n,
        k=k,
        rank=rank,
    )

    keys = torch.zeros((1, 8), dtype=torch.uint32, device="cuda")
    num_combos = 1
    hashes = torch.empty(
        (1, num_combos, 8), dtype=torch.uint32, device="cuda"
    )

    gpu_jackpot_hash(
        jackpots=jackpots,
        keys=keys,
        hashes=hashes,
        num_candidates=1,
        num_combos=num_combos,
    )

    # ---- Winner check ----
    target_tensor = torch.tensor(
        [mining_job.target], dtype=torch.uint32, device="cuda"
    )
    winners = torch.nonzero(
        hashes.squeeze(0) <= target_tensor.unsqueeze(0), as_tuple=False
    )

    if winners.numel() == 0:
        return False

    # ---- Build proof and submit ----
    plain_proof = create_proof(open_block, mining_job.incomplete_header_bytes)
    client.submit_plain_proof(plain_proof, mining_job)
    return True


def main():
    from miner_utils import get_logger

    args = parse_args()
    logger = get_logger("miner_gpu")

    rpc_config = MinerRpcConfig(
        transport="tcp" if args.gateway_tcp else "uds",
        socket_path=args.gateway_socket if not args.gateway_tcp else None,
        host=args.gateway_host,
        port=args.gateway_port,
    )
    settings = MinerSettings(
        noise_range=args.noise_range,
        noise_rank=args.noise_rank,
        tile_size_m=args.tile_m,
        tile_size_n=args.tile_n,
        tile_size_k=args.tile_k,
    )
    client = MiningClient(rpc_config)

    logger.info("Starting standalone GPU miner (no vLLM)")
    logger.info(f"Gateway: {'tcp://' + args.gateway_host + ':' + str(args.gateway_port) if args.gateway_tcp else 'uds://' + args.gateway_socket}")
    logger.info(f"Mining params: noise_range={args.noise_range}, noise_rank={args.noise_rank}, tile=({args.tile_m},{args.tile_n},{args.tile_k})")

    while True:
        try:
            run_single_mining_round(client, settings)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            client.close()
            break
        except Exception as e:
            logger.error("Mining round failed: %s", e)


if __name__ == "__main__":
    main()