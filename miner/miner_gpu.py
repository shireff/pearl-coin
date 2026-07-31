#!/usr/bin/env python3
"""Standalone GPU mining node without vLLM dependency.

Uses direct gpu_mine_batch kernel calls each round (no CUDA Graph):
  noise_gen() → gpu_mine_batch() → gpu_jackpot_hash()
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_local_root = Path(__file__).resolve().parent
_local_pearl_gemm_src = str(_local_root / "pearl-gemm" / "src")
if _local_pearl_gemm_src not in sys.path:
    sys.path.insert(0, _local_pearl_gemm_src)

import torch
from miner_base.gateway_client import MiningClient
from miner_base.settings import MinerSettings
from pearl_gateway.comm.dataclasses import MiningJob
from pearl_gateway.config import MinerRpcConfig
from pearl_gemm import (
    noise_gen,
    gpu_mine_batch,
    gpu_jackpot_hash,
)

from pool_mining_client import PoolMiningClient, PoolMiningJob


DEFAULT_GATEWAY_SOCKET = "/tmp/pearlgw.sock"
DEFAULT_GATEWAY_TCP_PORT = 8337
SUPPORTED_NOISE_RANKS = (64, 128)
JOB_STRUCT_BYTES = 128
DEFAULT_MAX_SHARED_MEMORY_BYTES = 100_000


@dataclass
class MiningResult:
    won: bool
    nonce_hex: str = ""
    result_hex: str = ""
    job_id: str = ""
    combo_index: int = -1


def allocate_aligned_byte_tensor(num_bytes: int, align: int = 16, device: str = "cuda") -> torch.Tensor:
    buffer = torch.empty(num_bytes + align - 1, dtype=torch.uint8, device=device)
    ptr = buffer.data_ptr()
    offset = (-ptr) % align
    if offset == 0:
        return buffer[:num_bytes]
    return buffer[offset:offset + num_bytes]


def estimate_shared_mem_bytes(tile_h: int, tile_w: int, rank: int) -> int:
    """Match gpu_mining_launch.cu formula exactly, including align4 padding.

    align4(x) = (x + 3) & ~3
    smem_a_pitch = tile_h * align4(rank + 1)
    smem_b_pitch = align4(rank + 1) * tile_w
    total = (tile_h*tile_w + smem_a_pitch + smem_b_pitch + JACKPOT_SIZE) * 4
    """
    aligned_rank_plus_1 = (rank + 1 + 3) & ~3
    smem_a_pitch = tile_h * aligned_rank_plus_1
    smem_b_pitch = aligned_rank_plus_1 * tile_w
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
            if _libcuda.cuDeviceGetAttribute(ctypes.byref(_val), ctypes.c_int(97), ctypes.c_int(device_idx)) == 0 and _val.value > 0:
                return _val.value
        except Exception:
            pass
        try:
            _libcudart = ctypes.CDLL("libcudart.so")
            _val = ctypes.c_int(0)
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


def get_gpu_sm_version() -> tuple[int, int]:
    """Return (major, minor) CUDA compute capability for device 0."""
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.cuda.get_device_capability(0)
    except Exception:
        pass
    return (0, 0)


def get_optimal_tile_size(rank: int, shared_mem_limit: int) -> tuple[int, int]:
    """Auto-select the best tile size for the current GPU.

    All returned tile sizes are multiples of 16, enabling the fused WMMA
    (Tensor Core) kernel path for maximum throughput.

    Selection logic:
      - Blackwell  (SM >= 120, e.g. RTX 5090):  128x128 if SMEM allows, else 64x64
      - Ada        (SM 8.x, e.g. RTX 4060/4090): 64x64 if SMEM allows, else 32x32
      - Older GPUs: 32x32 or 16x16

    The tile is always clamped to fit within the GPU's shared memory budget.
    """
    major, minor = get_gpu_sm_version()
    sm = major * 10 + minor

    if sm >= 120:
        candidates = ((128, 128), (64, 64), (32, 32), (16, 16))
    elif sm >= 80:
        candidates = ((64, 64), (32, 32), (16, 16))
    else:
        candidates = ((32, 32), (16, 16))

    for tile in candidates:
        if estimate_shared_mem_bytes(tile[0], tile[1], rank) <= shared_mem_limit:
            return tile
    return 16, 16


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Pearl GPU miner")
    g = parser.add_argument_group("Pearl node connection")
    g.add_argument("--rpc-url", default=None)
    g.add_argument("--rpc-user", default=None)
    g.add_argument("--rpc-password", default=None)
    g.add_argument("--mining-address", default=None)
    g = parser.add_argument_group("Pool mining (Stratum)")
    g.add_argument("--pool-url", default=None,
                   help="Stratum pool URL (e.g. stratum+tcp://pool.example.com:3333)")
    g.add_argument("--pool-username", default=None,
                   help="Pool username (wallet.worker format)")
    g.add_argument("--pool-password", default="x",
                   help="Pool password (default: x)")
    g.add_argument("--pool-worker", default="",
                   help="Worker name (default: auto from username)")
    g = parser.add_argument_group("Gateway connection")
    g.add_argument("--gateway-socket", default=os.environ.get("MINER_RPC_SOCKET_PATH", DEFAULT_GATEWAY_SOCKET))
    g.add_argument("--gateway-tcp", action="store_true", default=os.environ.get("MINER_RPC_TRANSPORT", "uds").lower() == "tcp")
    g.add_argument("--gateway-host", default=os.environ.get("MINER_RPC_HOST", "localhost"))
    g.add_argument("--gateway-port", type=int, default=int(os.environ.get("MINER_RPC_PORT", str(DEFAULT_GATEWAY_TCP_PORT))))
    g = parser.add_argument_group("Mining parameters")
    g.add_argument("--noise-rank", type=int, default=int(os.environ.get("miner_noise_rank", "128")))
    g.add_argument("--noise-range", type=int, default=int(os.environ.get("miner_noise_range", "256")))
    g.add_argument("--tile-m", type=int, default=int(os.environ.get("miner_tile_size_m", "64")))
    g.add_argument("--tile-n", type=int, default=int(os.environ.get("miner_tile_size_n", "64")))
    g.add_argument("--tile-k", type=int, default=int(os.environ.get("miner_tile_size_k", "128")))
    g.add_argument("--P", type=int, default=int(os.environ.get("miner_P", "8")),
                   help="Number of A-row partitions (more = more nonces per round)")
    g.add_argument("--Q", type=int, default=int(os.environ.get("miner_Q", "8")),
                   help="Number of B-col partitions (more = more nonces per round)")
    g.add_argument("--num-jobs", type=int, default=int(os.environ.get("miner_num_jobs", "1")),
                   help="Number of parallel mining jobs per round")
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
    then replays it every round — no realloc, no per-kernel sync.

    Optimizations:
    - key_A/key_B refreshed in-place with .random_() — no temp tensor allocation
    - target_tensor pre-allocated as class attribute and updated in-place
    - Single graph launch + single sync per round
    - hashes buffer reused across rounds
    """

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

        # Allow P and Q larger than m//tile_h by using strided offsets.
        # Each partition samples tile_h rows starting at a different stride,
        # so even overlapping partitions cover different combinations.
        # No hard clamping — just ensure offsets wrap modulo m.
        self.clamped_P = P
        self.clamped_Q = Q

        # ── Noise matrices (allocated once) ──────────────────────────────
        self.EAL         = torch.empty((m, rank), dtype=torch.int8,    device="cuda")
        self.EAL_fp16    = torch.empty((m, rank), dtype=torch.float16, device="cuda")
        self.EBR         = torch.empty((n, rank), dtype=torch.int8,    device="cuda")
        self.EBR_fp16    = torch.empty((n, rank), dtype=torch.float16, device="cuda")
        self.EAR_R_major = torch.empty((k, rank), dtype=torch.int8,    device="cuda")
        self.EAR_K_major = torch.empty((rank, k), dtype=torch.int8,    device="cuda")
        self.EBL_R_major = torch.empty((k, rank), dtype=torch.int8,    device="cuda")
        self.EBL_K_major = torch.empty((rank, k), dtype=torch.int8,    device="cuda")
        # Keys as contiguous aligned buffers (refreshed in-place each round)
        self.key_A       = allocate_aligned_byte_tensor(32, align=16, device="cuda")
        self.key_B       = allocate_aligned_byte_tensor(32, align=16, device="cuda")

        # ── Mining I/O buffers (allocated once, reused every round) ──────
        self.A_noised   = torch.zeros((num_jobs, m, k), dtype=torch.int32, device="cuda").contiguous()
        self.B_noised_t = torch.zeros((num_jobs, n, k), dtype=torch.int32, device="cuda").contiguous()
        self.a_rows     = torch.arange(tile_h, dtype=torch.int32, device="cuda").unsqueeze(0)
        self.b_cols     = torch.arange(tile_w, dtype=torch.int32, device="cuda").unsqueeze(0)

        # Generate P diverse row partitions from [0, m) — each partition samples
        # tile_h rows without replacement from the full m rows.
        # This ensures each (p,q) combination covers different parts of the matrix.
        a_rows_list = []
        for p in range(P):
            offset = (p * tile_h) % m
            rows = torch.arange(offset, offset + tile_h, dtype=torch.int32) % m
            a_rows_list.append(rows)
        self.a_rows_data = torch.stack(a_rows_list).to(device="cuda").contiguous()  # P x tile_h, contiguous

        b_cols_list = []
        for q in range(Q):
            offset = (q * tile_w) % n
            cols = torch.arange(offset, offset + tile_w, dtype=torch.int32) % n
            b_cols_list.append(cols)
        self.b_cols_data = torch.stack(b_cols_list).to(device="cuda").contiguous()  # Q x tile_w, contiguous
        
        self.P = self.clamped_P
        self.Q = self.clamped_Q
        self.jobs       = allocate_aligned_byte_tensor(JOB_STRUCT_BYTES, align=16, device="cuda")
        self.jackpots   = torch.empty((num_jobs, self.num_combos, 16), dtype=torch.uint32, device="cuda")
        self.keys       = torch.zeros((num_jobs, 8), dtype=torch.uint32, device="cuda")
        self.hashes     = torch.empty((num_jobs, self.num_combos, 8), dtype=torch.uint32, device="cuda")

        self._target_buf = torch.zeros(8, dtype=torch.int64, device="cuda")
        self._last_hashes: Optional[torch.Tensor] = None
        self._last_keys: Optional[torch.Tensor] = None

        # ── Pre-allocated target buffer (updated in-place each round) ────
        self._target_buf = torch.zeros(8, dtype=torch.int64, device="cuda")

        # ── CUDA Graph for noise_gen (stable kernel, captured once) ──────
        # noise_gen has fixed input shapes every round — safe to capture.
        # Warmup runs first (required by torch.cuda.graph).
        self._noise_stream = torch.cuda.Stream()
        self._noise_graph = None
        with torch.cuda.stream(self._noise_stream):
            # Warmup: 3 runs to ensure CUDA caches are warm
            for _ in range(3):
                noise_gen(
                    R=self.rank, num_threads=64,
                    EAL=self.EAL, EAL_fp16=self.EAL_fp16,
                    EAR_R_major=self.EAR_R_major, EAR_K_major=self.EAR_K_major,
                    EBL_R_major=self.EBL_R_major, EBL_K_major=self.EBL_K_major,
                    EBR=self.EBR, EBR_fp16=self.EBR_fp16,
                    key_A=self.key_A, key_B=self.key_B,
                )
            torch.cuda.synchronize()
            # Capture
            self._noise_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._noise_graph, stream=self._noise_stream):
                noise_gen(
                    R=self.rank, num_threads=64,
                    EAL=self.EAL, EAL_fp16=self.EAL_fp16,
                    EAR_R_major=self.EAR_R_major, EAR_K_major=self.EAR_K_major,
                    EBL_R_major=self.EBL_R_major, EBL_K_major=self.EBL_K_major,
                    EBR=self.EBR, EBR_fp16=self.EBR_fp16,
                    key_A=self.key_A, key_B=self.key_B,
                )

        # ── CUDA Events for accurate GPU timing ──────────────────────────
        self._t_start = torch.cuda.Event(enable_timing=True)
        self._t_end   = torch.cuda.Event(enable_timing=True)

        # ── Async copy stream for non-blocking D2H operations ────────────
        self._copy_stream = torch.cuda.Stream()

        # ── Mining stream for async kernel dispatch ───────────────────────
        self._mining_stream = torch.cuda.Stream()

    def run_round(self, mining_job: MiningJob) -> bool:
        self._t_start.record()

        self.key_A.random_(0, 256)
        self.key_B.random_(0, 256)
        if self._noise_graph is not None:
            self._noise_graph.replay()
        else:
            noise_gen(
                R=self.rank, num_threads=64,
                EAL=self.EAL, EAL_fp16=self.EAL_fp16,
                EAR_R_major=self.EAR_R_major, EAR_K_major=self.EAR_K_major,
                EBL_R_major=self.EBL_R_major, EBL_K_major=self.EBL_K_major,
                EBR=self.EBR, EBR_fp16=self.EBR_fp16,
                key_A=self.key_A, key_B=self.key_B,
            )

        with torch.cuda.stream(self._mining_stream):
            jackpots = gpu_mine_batch(
                self.A_noised,
                self.B_noised_t,
                self.a_rows_data,
                self.b_cols_data,
                self.jobs,
                self.num_jobs,
                self.P, self.Q,
                self.tile_h, self.tile_w,
                self.m, self.n, self.k, self.rank,
            )
        self._mining_stream.synchronize()

        hashes = gpu_jackpot_hash(jackpots, self.keys)
        _target_int = int(mining_job.target)
        _shifts = torch.arange(8, dtype=torch.int64, device="cuda") * 32
        self._target_buf.copy_(
            torch.tensor(
                [(_target_int >> (32 * i)) & 0xFFFFFFFF for i in range(8)],
                dtype=torch.int64, device="cpu"
            ).cuda(non_blocking=True)
        )
        hashes_i64 = hashes.view(-1, 8).to(dtype=torch.int64)
        any_winner = (hashes_i64 <= self._target_buf.unsqueeze(0)).all(dim=-1).any()
        self._last_hashes = hashes_i64.cpu()
        self._last_keys = self.keys.cpu()
        self._t_end.record()
        return bool(any_winner)

    def get_last_result(self) -> tuple[bool, Optional[str], Optional[str]]:
        if not hasattr(self, "_last_hashes") or self._last_hashes is None:
            return False, None, None
        flat = self._last_hashes.view(-1, 8)
        target_cpu = self._target_buf.cpu()
        winners = (flat <= target_cpu.unsqueeze(0)).all(dim=-1)
        if not winners.any():
            return False, None, None
        idx = int(winners.argmax().item())
        winner_hash = flat[idx]
        nonce_bytes = self._last_keys[idx % self.num_jobs].numpy().tobytes()
        result_hex = bytes(winner_hash.numpy().astype("<I").tobytes()).hex()
        nonce_hex = nonce_bytes.hex()
        return True, nonce_hex, result_hex

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
        if self._noise_graph is not None:
            self._noise_graph.reset()
            self._noise_graph = None


def main():
    from miner_utils import get_logger

    args = parse_args()
    logger = get_logger("miner_gpu")

    client = None
    gateway_proc = None
    managed_gateway = False
    pool_mode = args.pool_url is not None
    session = None

    try:
        if pool_mode:
            logger.info(f"Starting in POOL mode: {args.pool_url}")
            client = PoolMiningClient(
                pool_url=args.pool_url,
                username=args.pool_username,
                password=args.pool_password,
                worker_name=args.pool_worker,
            )
            if not client.connect():
                logger.error("Failed to connect to pool")
                return 1
            logger.info("Connected to pool successfully")
        else:
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
                        logger.error(f"Gateway socket {socket_path} never appeared")
                        return 1
                    logger.info(f"Gateway socket ready: {socket_path}")
                else:
                    threading.Thread(target=stream_logs, args=("gateway", gateway_proc), daemon=True).start()
                    time.sleep(2)

            rpc_config = MinerRpcConfig(
                transport="tcp" if args.gateway_tcp else "uds",
                socket_path=args.gateway_socket if not args.gateway_tcp else None,
                host=args.gateway_host,
                port=args.gateway_port,
            )
            client = MiningClient(rpc_config)

        shared_mem_limit = get_cuda_shared_memory_limit()
        gpu_name = get_gpu_name()
        tile_m, tile_n = get_optimal_tile_size(args.noise_rank, shared_mem_limit)
        if (tile_m, tile_n) != (args.tile_m, args.tile_n):
            logger.warning(
                f"Auto-selected tile ({tile_m},{tile_n}) differs from "
                f"user setting ({args.tile_m},{args.tile_n}) — "
                f"shared_mem_limit={shared_mem_limit // 1024} KB"
            )
        else:
            major, minor = get_gpu_sm_version()
            logger.info(
                f"GPU SM {major}.{minor}: selected tile ({tile_m},{tile_n}) "
                f"for WMMA path (shared_mem_limit={shared_mem_limit // 1024} KB)"
            )

        logger.info(f"GPU: {gpu_name or 'unknown'}  shared_mem_limit={shared_mem_limit // 1024} KB")
        logger.info(f"tile=({tile_m},{tile_n})  rank={args.noise_rank}  noise_range={args.noise_range}")
        if pool_mode:
            logger.info(f"Pool: {args.pool_url}  user={args.pool_username}")

        settings = MinerSettings(
            noise_range=args.noise_range,
            noise_rank=args.noise_rank,
            tile_size_m=tile_m,
            tile_size_n=tile_n,
            tile_size_k=args.tile_k,
        )

        P = args.P
        Q = args.Q
        num_jobs = args.num_jobs
        logger.info(f"Initialising mining session num_jobs={num_jobs} P={P} Q={Q} num_combos={P*Q}...")
        session = MiningGraphSession(settings, num_jobs=num_jobs, P=P, Q=Q)
        smem_kb = estimate_shared_mem_bytes(session.tile_h, session.tile_w, session.rank) // 1024
        logger.info(f"Session ready — tile=({session.tile_h},{session.tile_w})  smem={smem_kb} KB  num_jobs={num_jobs}  P={session.P}  Q={session.Q}  combos={session.P*session.Q}")

        import queue as _queue
        _job_queue: _queue.Queue = _queue.Queue(maxsize=2)
        _stop_prefetch = threading.Event()

        _client_ref = [client]

        def _prefetch_worker():
            import queue as _q
            while not _stop_prefetch.is_set():
                try:
                    job = _client_ref[0].get_mining_info()
                    try:
                        _job_queue.put_nowait(job)
                    except _q.Full:
                        try:
                            _job_queue.get_nowait()
                        except _q.Empty:
                            pass
                        try:
                            _job_queue.put_nowait(job)
                        except _q.Full:
                            pass
                            time.sleep(0.02)
                except Exception as e:
                    if pool_mode and isinstance(e, RuntimeError):
                        logger.warning(f"Prefetch: pool job unavailable — {e}")
                        time.sleep(1)
                        continue
                    logger.debug(f"Prefetch error: {type(e).__name__}: {e}")
                    time.sleep(0.05)

        prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
        prefetch_thread.start()
        logger.info("Job prefetch thread started")

        round_count = 0
        total_elapsed = 0.0

        _current_job = None

        while True:
            try:
                # Non-blocking: get new job if available, otherwise reuse last job
                try:
                    _current_job = _job_queue.get_nowait()
                except _queue.Empty:
                    if _current_job is None:
                        # No job yet — block until first job arrives
                        _current_job = _job_queue.get(timeout=5.0)

                mining_job = _current_job
                won = session.run_round(mining_job)

                session._t_end.synchronize()
                elapsed = session._t_start.elapsed_time(session._t_end) / 1000.0
                tmok = (session.tile_h * session.tile_w * session.P * session.Q * session.num_jobs) / session.rank / 1000.0
                tmok_per_s = tmok / elapsed if elapsed > 0 else 0.0

                logger.info(f"GPU round time: {elapsed*1000:.1f}ms, tmok/s={tmok_per_s:.2f}  jobs={session.num_jobs}  combos={session.P*session.Q}")

                # In pool mode, immediately run the next round without waiting
                # for a new job — keeps GPU at P0 and maximizes tmok/s.
                # Only check for a new job between rounds (non-blocking).

                if won:
                    if pool_mode:
                        ok, nonce_hex, result_hex = session.get_last_result()
                        if ok and nonce_hex and result_hex:
                            job_id = getattr(mining_job, "job_id", "")
                            _client_ref[0].submit_plain_proof(nonce_hex, result_hex, job_id)
                            logger.info("Pool share submitted!")
                        else:
                            logger.warning("Won but could not extract nonce/result for pool")
                    else:
                        plain_proof = session.build_proof(mining_job)
                        _client_ref[0].submit_plain_proof(plain_proof, mining_job)
                        logger.info("Block found and submitted!")

            except _queue.Empty:
                logger.warning("Job queue empty — waiting for jobs...")
            except KeyboardInterrupt:
                _stop_prefetch.set()
                raise
            except Exception as exc:
                import traceback as _tb
                tb = _tb.extract_tb(exc.__traceback__)
                last = tb[-1] if tb else None
                location = f"{last.filename}:{last.lineno} in {last.func}" if last else "unknown"
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
