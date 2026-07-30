#!/usr/bin/env python3
"""Standalone GPU mining node without vLLM dependency.

Uses CUDA Graph capture to eliminate per-round kernel launch overhead:
  noise_gen() → [CUDA Graph: mine_batch + jackpot_hash] (captured once, replayed every round)
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

_local_root = Path(__file__).resolve().parent
_local_pearl_gemm_src = str(_local_root.parent / "pearl-gemm" / "src")
if _local_pearl_gemm_src not in sys.path:
    sys.path.insert(0, _local_pearl_gemm_src)

import torch
from miner_base.gateway_client import MiningClient
from miner_base.settings import MinerSettings
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.config import MinerRpcConfig
from pearl_gemm import (
    noise_gen,
    create_mining_graph,
    launch_mining_graph,
    synchronize_mining_graph,
    destroy_mining_graph,
)

DEFAULT_GATEWAY_SOCKET = "/tmp/pearlgw.sock"
DEFAULT_GATEWAY_TCP_PORT = 8337
SUPPORTED_NOISE_RANKS = (64, 128)
JOB_STRUCT_BYTES = 128
DEFAULT_MAX_SHARED_MEMORY_BYTES = 100_000


def allocate_aligned_byte_tensor(num_bytes: int, align: int = 16, device: str = "cuda") -> torch.Tensor:
    buffer = torch.empty(num_bytes + align - 1, dtype=torch.uint8, device=device)
    ptr = buffer.data_ptr()
    offset = (-ptr) % align
    if offset == 0:
        return buffer[:num_bytes]
    return buffer[offset:offset + num_bytes]


def estimate_shared_mem_bytes(tile_h: int, tile_w: int, rank: int) -> int:
    """Match gpu_mining_launch.cu formula: (tile_h*tile_w + tile_h*(rank+1) + (rank+1)*tile_w + 16) * 4"""
    smem_a_pitch = tile_h * (rank + 1)
    smem_b_pitch = (rank + 1) * tile_w
    return (tile_h * tile_w + smem_a_pitch + smem_b_pitch + 16) * 4


def get_cuda_shared_memory_limit() -> int:
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return DEFAULT_MAX_SHARED_MEMORY_BYTES
        device_idx = torch.cuda.current_device()
        import ctypes
        try:
            _libcuda = ctypes.CDLL("libcuda.so.1")
            _val = ctypes.c_int(0)
            # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
            if _libcuda.cuDeviceGetAttribute(ctypes.byref(_val), ctypes.c_int(97), ctypes.c_int(device_idx)) == 0 and _val.value > 0:
                return _val.value
        except Exception:
            pass
        try:
            _libcudart = ctypes.CDLL("libcudart.so")
            _val = ctypes.c_int(0)
            # cudaDevAttrMaxSharedMemoryPerBlockOptin = 78
            if _libcudart.cudaDeviceGetAttribute(ctypes.byref(_val), ctypes.c_int(78), ctypes.c_int(device_idx)) == 0 and _val.value > 0:
                return _val.value
        except Exception:
            pass
        props = torch.cuda.get_device_properties(device_idx)
        sm = props.major * 10 + props.minor
        if sm >= 90:
            return 232 * 1024
        elif sm >= 80:
            return 163 * 1024
        elif sm >= 70:
            return 96 * 1024
        else:
            return 100 * 1024
    except Exception:
        pass
    return DEFAULT_MAX_SHARED_MEMORY_BYTES


def get_gpu_name() -> str | None:
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.cuda.get_device_properties(0).name
    except Exception:
        pass
    return None


def choose_safe_tile_size(tile_h: int, tile_w: int, rank: int, shared_mem_limit: int) -> tuple[int, int]:
    """Return the largest tile that fits in shared memory. (16,16) triggers WMMA fused path."""
    if estimate_shared_mem_bytes(tile_h, tile_w, rank) <= shared_mem_limit:
        return tile_h, tile_w
    for candidate in ((128, 128), (64, 64), (32, 32), (16, 16)):
        if estimate_shared_mem_bytes(candidate[0], candidate[1], rank) <= shared_mem_limit:
            return candidate
    return 16, 16


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Pearl GPU miner")
    g = parser.add_argument_group("Pearl node connection")
    g.add_argument("--rpc-url", default=None)
    g.add_argument("--rpc-user", default=None)
    g.add_argument("--rpc-password", default=None)
    g.add_argument("--mining-address", default=None)
    g = parser.add_argument_group("Gateway connection")
    g.add_argument("--gateway-socket", default=os.environ.get("MINER_RPC_SOCKET_PATH", DEFAULT_GATEWAY_SOCKET))
    g.add_argument("--gateway-tcp", action="store_true", default=os.environ.get("MINER_RPC_TRANSPORT", "uds").lower() == "tcp")
    g.add_argument("--gateway-host", default=os.environ.get("MINER_RPC_HOST", "localhost"))
    g.add_argument("--gateway-port", type=int, default=int(os.environ.get("MINER_RPC_PORT", str(DEFAULT_GATEWAY_TCP_PORT))))
    g = parser.add_argument_group("Mining parameters")
    g.add_argument("--noise-rank", type=int, default=int(os.environ.get("miner_noise_rank", "128")))
    g.add_argument("--noise-range", type=int, default=int(os.environ.get("miner_noise_range", "128")))
    g.add_argument("--tile-m", type=int, default=int(os.environ.get("miner_tile_size_m", "64")))
    g.add_argument("--tile-n", type=int, default=int(os.environ.get("miner_tile_size_n", "64")))
    g.add_argument("--tile-k", type=int, default=int(os.environ.get("miner_tile_size_k", "128")))
    return parser.parse_args()


def start_gateway_process(rpc_url, rpc_user, rpc_password, mining_address):
    env = os.environ.copy()
    if rpc_url:
        env["PEARLD_RPC_URL"] = rpc_url
    if rpc_user:
        env["PEARLD_RPC_USER"] = rpc_user
    if rpc_password:
        env["PEARLD_RPC_PASSWORD"] = rpc_password
    if mining_address:
        env["PEARLD_MINING_ADDRESS"] = mining_address
    env.setdefault("PEARL_LOG_LEVEL", "INFO")
    return subprocess.Popen(
        [sys.executable, "-m", "pearl_gateway", "start"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def stream_logs(name, proc):
    if proc.stdout is None:
        return
    for line in proc.stdout:
        print(f"[{name}] {line}", end="", flush=True)


def wait_for_gateway_socket(proc, socket_path, timeout=30, logger=None):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(socket_path):
            return True
        if proc.poll() is not None:
            msg = f"Managed pearl-gateway exited early with code {proc.returncode}"
            if logger:
                logger.error(msg)
            else:
                print(msg)
            if proc.stdout:
                remaining = proc.stdout.read()
                if remaining:
                    print(remaining, end="", flush=True)
            return False
        time.sleep(0.5)
    return False


class MiningGraphSession:
    """Allocates all GPU buffers once, captures the CUDA Graph once,
    then replays it every round — no realloc, no per-kernel sync."""

    def __init__(self, settings: MinerSettings, num_jobs: int = 1, P: int = 1, Q: int = 1):
        self.num_jobs = num_jobs
        self.num_combos = P * Q
        rank = settings.noise_rank
        m = n = k = settings.noise_range
        tile_h = settings.tile_size_m
        tile_w = settings.tile_size_n
        self.rank = rank
        self.m = self.n = self.k = m
        self.tile_h = tile_h
        self.tile_w = tile_w

        self.EAL         = torch.empty((m, rank), dtype=torch.int8,    device="cuda")
        self.EAL_fp16    = torch.empty((m, rank), dtype=torch.float16, device="cuda")
        self.EBR         = torch.empty((n, rank), dtype=torch.int8,    device="cuda")
        self.EBR_fp16    = torch.empty((n, rank), dtype=torch.float16, device="cuda")
        self.EAR_R_major = torch.empty((k, rank), dtype=torch.int8,    device="cuda")
        self.EAR_K_major = torch.empty((rank, k), dtype=torch.int8,    device="cuda")
        self.EBL_R_major = torch.empty((k, rank), dtype=torch.int8,    device="cuda")
        self.EBL_K_major = torch.empty((rank, k), dtype=torch.int8,    device="cuda")
        self.key_A       = allocate_aligned_byte_tensor(32, align=16, device="cuda")
        self.key_B       = allocate_aligned_byte_tensor(32, align=16, device="cuda")

        self.A_noised   = torch.zeros((num_jobs, m, k), dtype=torch.int32, device="cuda")
        self.B_noised_t = torch.zeros((num_jobs, n, k), dtype=torch.int32, device="cuda")
        self.a_rows     = torch.arange(tile_h, dtype=torch.int32, device="cuda").unsqueeze(0)
        self.b_cols     = torch.arange(tile_w, dtype=torch.int32, device="cuda").unsqueeze(0)
        self.jobs       = allocate_aligned_byte_tensor(JOB_STRUCT_BYTES, align=16, device="cuda")
        self.jackpots   = torch.empty((num_jobs, self.num_combos, 16), dtype=torch.uint32, device="cuda")
        self.keys       = torch.zeros((num_jobs, 8), dtype=torch.uint32, device="cuda")
        self.hashes     = torch.empty((num_jobs, self.num_combos, 8), dtype=torch.uint32, device="cuda")

        smem = estimate_shared_mem_bytes(tile_h, tile_w, rank)
        self.graph_state = create_mining_graph(num_jobs, self.num_combos, tile_h, tile_w, m, n, k, rank, smem)
        if self.graph_state is None:
            raise RuntimeError(
                f"create_mining_graph failed for tile=({tile_h},{tile_w}) smem={smem} bytes — "
                f"try smaller tile size"
            )

    def run_round(self, mining_job: MiningJob) -> bool:
        self.key_A.copy_(torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda"))
        self.key_B.copy_(torch.randint(0, 256, (32,), dtype=torch.uint8, device="cuda"))
        noise_gen(
            R=self.rank, num_threads=64,
            EAL=self.EAL, EAL_fp16=self.EAL_fp16,
            EAR_R_major=self.EAR_R_major, EAR_K_major=self.EAR_K_major,
            EBL_R_major=self.EBL_R_major, EBL_K_major=self.EBL_K_major,
            EBR=self.EBR, EBR_fp16=self.EBR_fp16,
            key_A=self.key_A, key_B=self.key_B,
        )
        err = launch_mining_graph(
            self.graph_state,
            self.A_noised, self.B_noised_t,
            self.a_rows, self.b_cols,
            self.jobs, self.jackpots, self.keys, self.hashes,
        )
        if err != 0:
            raise RuntimeError(f"launch_mining_graph failed: cudaError {err}")
        synchronize_mining_graph(self.graph_state)

        _target_int = int(mining_job.target)
        target_words = [(_target_int >> (32 * i)) & 0xFFFFFFFF for i in range(8)]
        target_tensor = torch.tensor(target_words, dtype=torch.int64, device="cuda")
        hashes_i64 = self.hashes.squeeze(0).to(torch.int64)
        winners = torch.nonzero(
            (hashes_i64 <= target_tensor.unsqueeze(0)).all(dim=-1), as_tuple=False
        )
        return winners.numel() > 0

    def build_proof(self, mining_job: MiningJob):
        from pearl_gateway.comm.dataclasses import OpenedBlockInfo
        from miner_base.block_submission import create_proof
        open_block = OpenedBlockInfo(
            A=self.A_noised.cpu().numpy(),
            B_t=self.B_noised_t.cpu().numpy(),
            A_row_indices=self.a_rows.squeeze(0).cpu().tolist(),
            B_column_indices=self.b_cols.squeeze(0).cpu().tolist(),
            commitment_hash=None,
            noise_rank=self.rank,
        )
        return create_proof(open_block, mining_job.incomplete_header_bytes)

    def close(self):
        if self.graph_state is not None:
            destroy_mining_graph(self.graph_state)
            self.graph_state = None


def main():
    from miner_utils import get_logger

    args = parse_args()
    logger = get_logger("miner_gpu")

    client = None
    gateway_proc = None
    managed_gateway = False
    session = None

    try:
        if args.rpc_url is not None:
            managed_gateway = True
            socket_path = args.gateway_socket if not args.gateway_tcp else None
            if not args.gateway_tcp and os.path.exists(socket_path):
                os.remove(socket_path)
            logger.info("Starting managed pearl-gateway process...")
            gateway_proc = start_gateway_process(
                args.rpc_url, args.rpc_user, args.rpc_password, args.mining_address,
            )
            if not args.gateway_tcp:
                threading.Thread(target=stream_logs, args=("gateway", gateway_proc), daemon=True).start()
                if not wait_for_gateway_socket(gateway_proc, socket_path, timeout=30, logger=logger):
                    logger.error("Gateway socket %s never appeared", socket_path)
                    return 1
                logger.info("Gateway socket ready: %s", socket_path)
            else:
                threading.Thread(target=stream_logs, args=("gateway", gateway_proc), daemon=True).start()
                time.sleep(2)

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
                "Tile (%s,%s) exceeds GPU shared memory (%d KB) — using (%s,%s)",
                args.tile_m, args.tile_n, shared_mem_limit // 1024, tile_m, tile_n,
            )

        logger.info("GPU: %s  shared_mem_limit=%d KB", gpu_name or "unknown", shared_mem_limit // 1024)
        logger.info("tile=(%d,%d)  rank=%d  noise_range=%d", tile_m, tile_n, args.noise_rank, args.noise_range)

        settings = MinerSettings(
            noise_range=args.noise_range,
            noise_rank=args.noise_rank,
            tile_size_m=tile_m,
            tile_size_n=tile_n,
            tile_size_k=args.tile_k,
        )
        client = MiningClient(rpc_config)

        logger.info("Initialising CUDA Graph session...")
        session = MiningGraphSession(settings, num_jobs=1, P=1, Q=1)
        smem_kb = estimate_shared_mem_bytes(session.tile_h, session.tile_w, session.rank) // 1024
        logger.info("CUDA Graph ready — tile=(%d,%d)  smem=%d KB", session.tile_h, session.tile_w, smem_kb)

        while True:
            try:
                mining_job: MiningJob = client.get_mining_info()
                t0 = time.time()
                won = session.run_round(mining_job)
                elapsed = time.time() - t0
                tmok = (session.tile_h * session.tile_w) / session.rank / 1000.0
                tmok_per_s = tmok / elapsed if elapsed > 0 else 0.0
                logger.info(f"GPU round time: {elapsed:.3f}s, tmok/s={tmok_per_s:.2f}")
                if won:
                    plain_proof = session.build_proof(mining_job)
                    client.submit_plain_proof(plain_proof, mining_job)
                    logger.info("Block found and submitted!")
            except KeyboardInterrupt:
                raise
            except BrokenPipeError:
                logger.warning("Gateway connection lost — reconnecting in 3s...")
                time.sleep(3)
                try:
                    client.close()
                except Exception:
                    pass
                try:
                    client = MiningClient(rpc_config)
                    logger.info("Reconnected to gateway")
                except Exception as e:
                    logger.error(f"Reconnect failed: {e}")
            except Exception as exc:
                import traceback as _tb
                tb = _tb.extract_tb(exc.__traceback__)
                last = tb[-1] if tb else None
                location = f"{last.filename}:{last.lineno} in {last.name}" if last else "unknown"
                logger.error(f"Mining round failed — {type(exc).__name__}: {exc}  [at {location}]")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if session is not None:
            session.close()
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
