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
import subprocess
import sys
import threading
import time

import torch
from miner_base.gateway_client import MiningClient
from miner_base.settings import MinerSettings
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.config import MinerRpcConfig
from pearl_gemm import gpu_jackpot_hash, gpu_mine_batch, noise_gen


DEFAULT_GATEWAY_SOCKET = "/tmp/pearlgw.sock"
DEFAULT_GATEWAY_TCP_PORT = 8337
SUPPORTED_NOISE_RANKS = (64, 128)
SUPPORTED_NOISE_GEN_THREAD_COUNTS = (32, 64, 128)
JOB_STRUCT_BYTES = 128
DEFAULT_TILE_SIZE_M = 64
DEFAULT_TILE_SIZE_N = 64
DEFAULT_TILE_SIZE_K = 128
DEFAULT_MAX_SHARED_MEMORY_BYTES = 100_000


def allocate_aligned_byte_tensor(num_bytes: int, align: int = 16, device: str = "cuda") -> torch.Tensor:
    buffer = torch.empty(num_bytes + align - 1, dtype=torch.uint8, device=device)
    ptr = buffer.data_ptr()
    offset = (-ptr) % align
    if offset == 0:
        return buffer[:num_bytes]
    return buffer[offset:offset + num_bytes]


def estimate_shared_mem_bytes(tile_h: int, tile_w: int, rank: int) -> int:
    return (
        tile_h * tile_w
        + tile_h * (rank + 1)
        + (rank + 1) * tile_w
        + 16
    ) * 4


def get_cuda_shared_memory_limit() -> int:
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            return int(
                getattr(props, "max_shared_memory_per_block", None)
                or getattr(props, "shared_memory_per_block", None)
                or DEFAULT_MAX_SHARED_MEMORY_BYTES
            )
    except Exception:
        pass
    return DEFAULT_MAX_SHARED_MEMORY_BYTES


def get_gpu_name() -> str | None:
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            return props.name
    except Exception:
        pass
    return None


def choose_safe_tile_size(tile_h: int, tile_w: int, rank: int, shared_mem_limit: int) -> tuple[int, int]:
    if estimate_shared_mem_bytes(tile_h, tile_w, rank) <= shared_mem_limit:
        return tile_h, tile_w

    for candidate in ((128, 128), (64, 64), (32, 32), (16, 16)):
        if estimate_shared_mem_bytes(candidate[0], candidate[1], rank) <= shared_mem_limit:
            return candidate

    return DEFAULT_TILE_SIZE_M, DEFAULT_TILE_SIZE_N


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone Pearl GPU miner",
    )

    pearl_node = parser.add_argument_group("Pearl node connection")
    pearl_node.add_argument("--rpc-url", default=None, help="Pearl node JSON-RPC URL, e.g. http://127.0.0.1:44107")
    pearl_node.add_argument("--rpc-user", default=None, help="RPC username for pearl node")
    pearl_node.add_argument("--rpc-password", default=None, help="RPC password for pearl node")
    pearl_node.add_argument("--mining-address", default=None, help="Taproot mining address")

    gateway = parser.add_argument_group("Gateway connection")
    gateway.add_argument(
        "--gateway-socket",
        default=os.environ.get("MINER_RPC_SOCKET_PATH", DEFAULT_GATEWAY_SOCKET),
        help="Unix socket path for pearl-gateway (default: /tmp/pearlgw.sock)",
    )
    gateway.add_argument(
        "--gateway-tcp",
        action="store_true",
        default=os.environ.get("MINER_RPC_TRANSPORT", "uds").lower() == "tcp",
        help="Use TCP instead of Unix socket for gateway connection",
    )
    gateway.add_argument(
        "--gateway-host",
        default=os.environ.get("MINER_RPC_HOST", "localhost"),
        help="Gateway TCP host (used when --gateway-tcp is set)",
    )
    gateway.add_argument(
        "--gateway-port",
        type=int,
        default=int(os.environ.get("MINER_RPC_PORT", str(DEFAULT_GATEWAY_TCP_PORT))),
        help="Gateway TCP port (used when --gateway-tcp is set)",
    )

    mining = parser.add_argument_group("Mining parameters")
    mining.add_argument("--noise-rank", type=int, default=int(os.environ.get("miner_noise_rank", "128")))
    mining.add_argument("--noise-range", type=int, default=int(os.environ.get("miner_noise_range", "128")))
    mining.add_argument("--tile-m", type=int, default=int(os.environ.get("miner_tile_size_m", "64")))
    mining.add_argument("--tile-n", type=int, default=int(os.environ.get("miner_tile_size_n", "64")))
    mining.add_argument("--tile-k", type=int, default=int(os.environ.get("miner_tile_size_k", "128")))

    return parser.parse_args()


def start_gateway_process(rpc_url, rpc_user, rpc_password, mining_address):
    """Start pearl-gateway as a subprocess configured with the given pearl node settings."""
    env = os.environ.copy()
    if rpc_url is not None:
        env["PEARLD_RPC_URL"] = rpc_url
    if rpc_user is not None:
        env["PEARLD_RPC_USER"] = rpc_user
    if rpc_password is not None:
        env["PEARLD_RPC_PASSWORD"] = rpc_password
    if mining_address is not None:
        env["PEARLD_MINING_ADDRESS"] = mining_address
    env.setdefault("PEARL_LOG_LEVEL", "INFO")

    cmd = [sys.executable, "-m", "pearl_gateway", "start"]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_socket(socket_path, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(socket_path):
            return True
        time.sleep(0.5)
    return False


def stream_logs(name, proc):
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"[{name}] {line}", end="", flush=True)


def wait_for_gateway_socket(proc, socket_path, timeout=30, logger=None):
    """Wait for gateway socket while streaming logs and checking process health."""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(socket_path):
            return True
        if proc.poll() is not None:
            if logger is not None:
                logger.error("Managed pearl-gateway exited early with code %s", proc.returncode)
            else:
                print(f"Managed pearl-gateway exited early with code {proc.returncode}")
            if proc.stdout:
                remaining = proc.stdout.read()
                if remaining:
                    print(remaining, end="", flush=True)
            return False
        time.sleep(0.5)
    return False


def run_single_mining_round(
    client: MiningClient,
    settings: MinerSettings,
) -> bool:
    """Fetch one job from the gateway, mine it on GPU, submit proof."""
    mining_job: MiningJob = client.get_mining_info()

    rank  = settings.noise_rank
    # m/n/k are the matrix inner dimensions — must equal noise_range.
    # tile_h/tile_w are the output tile dimensions used by gpu_mine_batch.
    m     = settings.noise_range
    n     = settings.noise_range
    k     = settings.noise_range
    tile_h = settings.tile_size_m
    tile_w = settings.tile_size_n

    if rank not in SUPPORTED_NOISE_RANKS:
        nearest_rank = min(SUPPORTED_NOISE_RANKS, key=lambda supported: abs(supported - rank))
        print(
            f"[WARN] Unsupported noise rank {rank}; switching to supported rank {nearest_rank}. "
            f"Supported values are {SUPPORTED_NOISE_RANKS}."
        )
        rank = nearest_rank

    # ── Allocate noise matrices ──────────────────────────────────────────────
    EAL          = torch.empty((m, rank), dtype=torch.int8,    device="cuda")
    EAL_fp16     = torch.empty((m, rank), dtype=torch.float16, device="cuda")
    EBR          = torch.empty((n, rank), dtype=torch.int8,    device="cuda")
    EBR_fp16     = torch.empty((n, rank), dtype=torch.float16, device="cuda")
    EAR_R_major  = torch.empty((k, rank), dtype=torch.int8,    device="cuda")
    EAR_K_major  = torch.empty((rank, k), dtype=torch.int8,    device="cuda")
    EBL_R_major  = torch.empty((k, rank), dtype=torch.int8,    device="cuda")
    EBL_K_major  = torch.empty((rank, k), dtype=torch.int8,    device="cuda")
    key_A        = torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda")
    key_B        = torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda")

    # ── Generate noise on GPU ────────────────────────────────────────────────
    kernel_start = time.time()
    noise_gen(
        R=rank,
        num_threads=64,
        EAL=EAL,
        EAL_fp16=EAL_fp16,
        EAR_R_major=EAR_R_major,
        EAR_K_major=EAR_K_major,
        EBL_R_major=EBL_R_major,
        EBL_K_major=EBL_K_major,
        EBR=EBR,
        EBR_fp16=EBR_fp16,
        key_A=key_A,
        key_B=key_B,
    )

    # ── Allocate noised A / B matrices ───────────────────────────────────────
    A_noised   = torch.zeros((1, m, k), dtype=torch.int32, device="cuda")
    B_noised_t = torch.zeros((1, n, k), dtype=torch.int32, device="cuda")

    # ── Mine: row/col indices for the tile ───────────────────────────────────
    a_rows = torch.arange(tile_h, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, tile_h)
    b_cols = torch.arange(tile_w, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, tile_w)

    jobs_buffer = allocate_aligned_byte_tensor(JOB_STRUCT_BYTES, align=16, device="cuda")
    jackpots = gpu_mine_batch(
        a_noised=A_noised,
        b_noised_t=B_noised_t,
        a_rows_data=a_rows,
        b_cols_data=b_cols,
        jobs=jobs_buffer,
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
    gpu_mine_batch_time = time.time() - kernel_start

    # ── Hash jackpots ────────────────────────────────────────────────────────
    keys = torch.zeros((1, 8), dtype=torch.uint32, device="cuda")
    hashes = gpu_jackpot_hash(jackpots=jackpots, keys=keys)

    elapsed = time.time() - kernel_start
    tmok    = (tile_h * tile_w) / rank / 1000.0

    logger = None
    try:
        from miner_utils import get_logger
        logger = get_logger("miner_gpu")
    except Exception:
        pass

    msg = f"[stats] GPU round time: {elapsed:.3f}s, tmok/s={tmok / elapsed if elapsed > 0 else 0.0:.2f}"
    if logger is not None:
        logger.info("GPU round time: %.3fs, tmok/s=%.2f", elapsed, tmok / elapsed if elapsed > 0 else 0.0)
    else:
        print(msg)

    # ── Check for winner ─────────────────────────────────────────────────────
    target_tensor = torch.tensor(
        [mining_job.target], dtype=torch.uint32, device="cuda"
    )
    winners = torch.nonzero(
        hashes.squeeze(0) <= target_tensor.unsqueeze(0), as_tuple=False
    )

    if winners.numel() == 0:
        return False

    # ── Build open_block with actual matrix data for proof ───────────────────
    from pearl_gateway.comm.dataclasses import OpenedBlockInfo
    open_block = OpenedBlockInfo(
        A=A_noised.cpu().numpy(),
        B_t=B_noised_t.cpu().numpy(),
        A_row_indices=a_rows.squeeze(0).cpu().tolist(),
        B_column_indices=b_cols.squeeze(0).cpu().tolist(),
        commitment_hash=None,
        noise_rank=rank,
    )
    from miner_base.block_submission import create_proof
    plain_proof = create_proof(open_block, mining_job.incomplete_header_bytes)
    client.submit_plain_proof(plain_proof, mining_job)
    return True


def main():
    from miner_utils import get_logger

    args = parse_args()
    logger = get_logger("miner_gpu")

    client = None
    gateway_proc = None
    managed_gateway = False
    log_thread = None

    try:
        if args.rpc_url is not None:
            managed_gateway = True
            socket_path = args.gateway_socket if not args.gateway_tcp else None

            if not args.gateway_tcp and os.path.exists(socket_path):
                os.remove(socket_path)

            logger.info("Starting managed pearl-gateway process...")
            gateway_proc = start_gateway_process(
                args.rpc_url,
                args.rpc_user,
                args.rpc_password,
                args.mining_address,
            )

            if not args.gateway_tcp:
                log_thread = threading.Thread(target=stream_logs, args=("gateway", gateway_proc), daemon=True)
                log_thread.start()

                if not wait_for_gateway_socket(gateway_proc, socket_path, timeout=30, logger=logger):
                    logger.error("Gateway socket %s never appeared", socket_path)
                    return 1
                logger.info("Gateway socket ready: %s", socket_path)
            else:
                log_thread = threading.Thread(target=stream_logs, args=("gateway", gateway_proc), daemon=True)
                log_thread.start()
                time.sleep(2)
                logger.info("Waiting for gateway TCP on %s:%s", args.gateway_host, args.gateway_port)

        rpc_config = MinerRpcConfig(
            transport="tcp" if args.gateway_tcp else "uds",
            socket_path=args.gateway_socket if not args.gateway_tcp else None,
            host=args.gateway_host,
            port=args.gateway_port,
        )

        shared_mem_limit = get_cuda_shared_memory_limit()
        gpu_name = get_gpu_name()
        tile_m, tile_n = choose_safe_tile_size(args.tile_m, args.tile_n, args.noise_rank, shared_mem_limit)
        if (tile_m, tile_n) != (args.tile_m, args.tile_n):
            logger.warning(
                "Requested tile size (%s,%s) is unsafe for this GPU; using safe tile size (%s,%s).",
                args.tile_m, args.tile_n, tile_m, tile_n,
            )

        if gpu_name is not None:
            logger.info("Detected GPU: %s, shared_mem_limit=%s", gpu_name, shared_mem_limit)

        settings = MinerSettings(
            noise_range=args.noise_range,
            noise_rank=args.noise_rank,
            tile_size_m=tile_m,
            tile_size_n=tile_n,
            tile_size_k=args.tile_k,
        )
        client = MiningClient(rpc_config)

        logger.info("Starting standalone GPU miner (no vLLM)")
        logger.info(
            "Gateway: %s",
            "tcp://" + args.gateway_host + ":" + str(args.gateway_port)
            if args.gateway_tcp
            else "uds://" + args.gateway_socket,
        )
        logger.info(
            "Mining params: noise_range=%s, noise_rank=%s, tile=(%s,%s,%s)",
            args.noise_range,
            args.noise_rank,
            args.tile_m,
            args.tile_n,
            args.tile_k,
        )

        while True:
            try:
                run_single_mining_round(client, settings)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                import traceback as _tb
                tb = _tb.extract_tb(exc.__traceback__)
                last = tb[-1] if tb else None
                location = f"{last.filename}:{last.lineno} in {last.name}" if last else "unknown"
                logger.error(
                    f"Mining round failed — {type(exc).__name__}: {exc}  [at {location}]"
                )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if client is not None:
            client.close()
        if managed_gateway and gateway_proc is not None:
            logger.info("Stopping managed gateway process...")
            gateway_proc.terminate()
            try:
                gateway_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gateway_proc.kill()


if __name__ == "__main__":
    import sys as _sys
    import traceback as _tb

    try:
        main()
    except SystemExit:
        raise
    except Exception as _exc:
        frames = _tb.extract_tb(_exc.__traceback__)
        print("\n=== FATAL ERROR ===", file=_sys.stderr)
        print(f"Type   : {type(_exc).__name__}", file=_sys.stderr)
        print(f"Message: {_exc}", file=_sys.stderr)
        if frames:
            last = frames[-1]
            print(f"File   : {last.filename}", file=_sys.stderr)
            print(f"Line   : {last.lineno}  in  {last.name}", file=_sys.stderr)
            print(f"Code   : {last.line}", file=_sys.stderr)
        print("===================\n", file=_sys.stderr)
        _sys.exit(1)
