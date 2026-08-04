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

import blake3

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
    alph_mine_batch,
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



def alephium_double_blake3(header_blob: bytes, nonce24: bytes) -> bytes:
    """Compute the Alephium double-BLAKE3 hash.

    Reference: gpu-miner/src/blake3/inlined-blake.hpp — DOUBLE_HASH macro
      input = buf[0..326) where buf[0:24] = nonce24, buf[24:326] = headerBlob
      hash  = blake3(blake3(nonce24 + headerBlob))

    Args:
        header_blob: raw headerBlob bytes (302 bytes from pool)
        nonce24:     full 24-byte nonce (buf[0:24] from the kernel buffer)
    """
    inner = blake3.blake3(nonce24 + header_blob).digest()
    return blake3.blake3(inner).digest()


def alephium_chain_index(hash_bytes: bytes) -> int:
    """Derive the Alephium chain index from the hash.

    Reference: gpu-miner/src/blake3/inlined-blake.hpp — CHECK_INDEX macro:
      uint32_t big_index = (H7 & 0x0F000000) >> 24;
      H7 is the last uint32 of the hash stored little-endian on GPU,
      but when read back as bytes: big_index = hash[30] (in big-endian byte order)

    Reference: gpu-miner/src/pow.h (commented-out check_index):
      uint8_t big_index = hash[31] % chain_nums;   ← older version used hash[31]

    The pool's mining-pool/lib/util.js blockChainIndex uses:
      bigIndex = hash[30] << 8 | hash[31]
      index    = bigIndex % 16
    """
    big_index = (int(hash_bytes[30]) << 8) | int(hash_bytes[31])
    return big_index % 16


def alephium_group_valid(hash_bytes: bytes, from_group: int | None, to_group: int | None) -> bool:
    """Validate the final Alephium chain-index gate if applicable."""
    if from_group is None or to_group is None:
        return True
    big_index = alephium_chain_index(hash_bytes)
    return (big_index // 4) == int(from_group) and (big_index % 4) == int(to_group)


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
    g.add_argument("--pool-algorithm", default=None,
                   choices=["pearl", "alephium"],
                   help="Force pool mining algorithm (default: auto-detect from URL)")
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
    g.add_argument("--alph-nonces-per-round", type=int,
                   default=int(os.environ.get("miner_alph_nonces_per_round", str(1 << 24))),
                   help="Alephium nonces per GPU round (default: 16M = 1<<24, higher = more MH/s)")
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


class AlephiumMiningSession:
    """GPU mining session for Alephium (Blake3 standard, non-keyed).

    Reference: gpu-miner/src/blake3/inlined-blake.hpp + worker.h

    Buffer layout (matches BLAKE3_BUF_LEN = 326):
        buf[0:24]   = nonce24  (24 random bytes, kernel mutates buf[0:4] only)
        buf[24:326] = headerBlob (302 bytes from pool)

    Hash: blake3(blake3(buf[0:326])) = blake3(blake3(nonce24 + headerBlob))

    Stratum submit:
        full_nonce_24       = nonce24  (24 bytes)
        nonceSansExtraNonce = nonce24[2:]  (22 bytes, pool prepends extraNonce[2])
        pool reconstructs:  extraNonce(2) + nonceSans(22) = nonce24  ✓

    Kernel behaviour (inlined-blake.hpp — blake3_hasher_mine):
        short_nonce = &input[0x00]  ← points to buf[0:4]
        per thread: *short_nonce = (*short_nonce) / stride * stride + tid
        per step:   *short_nonce += stride   ← only buf[0:4] changes
        buf[4:24] stays as set by reset_worker() (random bytes)

    Our kernel (alph_mine_batch) receives base_nonce (uint64) and writes it
    into buf[0:8] as little-endian uint64, then increments by 1 per thread.
    buf[8:24] stays zero (our allocation is zeroed).

    So our nonce24 layout:
        buf[0:8]  = base_nonce + thread_id  (little-endian uint64)
        buf[8:24] = zeros

    nonceSansExtraNonce = nonce24[2:] = nonce24 bytes [2:24] = 22 bytes
    """

    BLOB_PADDED_LEN = 320  # 302 bytes + 18 zero bytes padding to 5×64

    def __init__(self, nonces_per_round: int = 1 << 23):
        self.nonces_per_round = nonces_per_round
        self._base_nonce: int = 0
        from miner_utils import get_logger as _get_logger
        self._logger = _get_logger("alephium_session")
        # Pre-allocate zero-padded blob (320 bytes) — last 18 bytes stay zero
        self._blob_gpu = torch.zeros(self.BLOB_PADDED_LEN, dtype=torch.uint8, device="cuda")
        self._target_gpu = torch.empty(32, dtype=torch.uint8, device="cuda")
        self._t_start = torch.cuda.Event(enable_timing=True)
        self._t_end = torch.cuda.Event(enable_timing=True)
        self._last_found: bool = False
        self._last_nonce24: bytes = b"\x00" * 24   # full 24-byte nonce from kernel
        self._last_hash_hex: str = ""
        self._last_chain_ok: bool = False
        self._last_blob_bytes: bytes = b"\x00" * 302
        self._last_from_group: int | None = None
        self._last_to_group: int | None = None
        self._last_block_target: int = 0

    def run_round(self, job: PoolMiningJob) -> tuple[bool, bytes]:
        """Run one GPU mining round.

        Reference kernel (inlined-blake.hpp):
          - buf[0:24]   = nonce24 (our kernel writes uint64 little-endian into buf[0:8])
          - buf[24:326] = headerBlob
          - hash = blake3(blake3(buf[0:326]))
          - winner when hash <= target AND chain_index matches

        Returns:
            (found, nonce24_bytes) where nonce24_bytes is the 24-byte nonce
            that produced the winning hash, or b'\\x00'*24 if not found.
        """
        self._last_job_id = getattr(job, "job_id", "")
        blob_bytes = bytes(job.blob)
        self._last_blob_bytes = blob_bytes

        # Load headerBlob into GPU buffer (302 bytes, last 18 stay zero)
        blob_tensor = torch.frombuffer(blob_bytes, dtype=torch.uint8)
        self._blob_gpu[:302].copy_(blob_tensor)

        # Target as big-endian uint8[32]
        target_bytes = job.target.to_bytes(32, "big")
        self._target_gpu.copy_(torch.frombuffer(target_bytes, dtype=torch.uint8))

        self._t_start.record()
        # alph_mine_batch writes base_nonce as little-endian uint64 into buf[0:8],
        # increments by 1 per thread. buf[8:24] stays zero.
        # Returned nonce is the winning uint64 value.
        _base_signed = (self._base_nonce if self._base_nonce < (1 << 63)
                        else self._base_nonce - (1 << 64))
        found, raw_nonce = alph_mine_batch(
            self._blob_gpu,
            _base_signed,
            self.nonces_per_round,
            self._target_gpu,
        )
        self._t_end.record()
        self._base_nonce += self.nonces_per_round
        self._last_found = bool(found)

        if found:
            # Reconstruct the exact 24-byte nonce the kernel used:
            # kernel writes uint64 little-endian into buf[0:8], buf[8:24] = zeros.
            # Reference: inlined-blake.hpp buf layout.
            nonce_uint64 = int(raw_nonce) & 0xFFFFFFFFFFFFFFFF
            if nonce_uint64 == 0xFFFFFFFFFFFFFFFF:
                # Sentinel — kernel found nothing despite found=True
                self._last_found = False
                self._last_nonce24 = b"\x00" * 24
                self._last_hash_hex = ""
                self._last_chain_ok = False
                return False, b"\x00" * 24

            # nonce24 = uint64 little-endian (8 bytes) + zeros (16 bytes)
            nonce24 = nonce_uint64.to_bytes(8, "little") + b"\x00" * 16

            # Verify hash (ground truth — same as pool will compute)
            hash_bytes = alephium_double_blake3(blob_bytes, nonce24)
            self._last_hash_hex = hash_bytes.hex()
            self._last_nonce24 = nonce24

            self._last_from_group = getattr(job, "from_group", None)
            self._last_to_group = getattr(job, "to_group", None)
            self._last_block_target = getattr(job, "block_target", 0)
            self._last_chain_ok = alephium_group_valid(
                hash_bytes, self._last_from_group, self._last_to_group
            )

            if not self._last_chain_ok:
                self._logger.debug(
                    f"[chain-mismatch] hash chain≠job chain, skipping: "
                    f"job=({self._last_from_group},{self._last_to_group}) "
                    f"hash={self._last_hash_hex[:16]}"
                )

            # Diagnostic: log first winner for pool verification
            if not getattr(self, "_diag_logged", False):
                self._diag_logged = True
                self._logger.info(
                    f"[DIAG] job_id={job.job_id}  "
                    f"blob_len={len(blob_bytes)}  "
                    f"nonce24={nonce24.hex()}  "
                    f"nonceSans={nonce24[2:].hex()}  "
                    f"hash={self._last_hash_hex}  "
                    f"target={job.target:#x}  "
                    f"hash_le_target={int(self._last_hash_hex, 16) <= job.target}  "
                    f"chain_ok={self._last_chain_ok}"
                )
        else:
            self._last_nonce24 = b"\x00" * 24
            self._last_hash_hex = ""
            self._last_chain_ok = False

        # Periodic diagnostic
        if not hasattr(self, "_round_count"):
            self._round_count = 0
        self._round_count += 1
        if self._round_count % 1000 == 1:
            self._logger.info(
                f"[diag] round={self._round_count}  base_nonce={self._base_nonce:#018x}  "
                f"found={found}  target={job.target:#x}"
            )
        if found:
            self._logger.info(
                f"[WINNER] job_id={getattr(job, 'job_id', '?')}  "
                f"nonce24={self._last_nonce24.hex()}  chain_ok={self._last_chain_ok}"
            )

        return self._last_found, self._last_nonce24

    def get_last_result(self) -> tuple[bool, str, str]:
        """Return (ok, nonceSansExtraNonce_hex, hash_hex) for pool submission.

        Reference: alephium stratum spec + gpu-miner/src/worker.h write_new_block()

        Pool submission protocol:
            mining.submit params: [jobId, nonceSansExtraNonce, workerId?]
            full_nonce_24       = nonce24  (24 bytes the kernel found)
            nonceSansExtraNonce = nonce24[2:]  (22 bytes)
            pool reconstructs:  extraNonce(2) + nonceSans(22) = nonce24  ✓
            pool verifies:      blake3(blake3(nonce24 + headerBlob)) <= target

        nonce24 layout from our kernel (alph_mine_batch writes little-endian uint64):
            nonce24[0:8]  = winning uint64 (little-endian)
            nonce24[8:24] = zeros
        """
        if not self._last_found:
            return False, "", ""

        if self._last_nonce24 == b"\x00" * 24:
            self._logger.error("[get_last_result] nonce24 is all zeros — kernel found nothing")
            return False, "", ""

        if not self._last_chain_ok:
            nonce24 = self._last_nonce24
            bi = (nonce24[30] << 8 | nonce24[31]) % 16  # not the hash — use hash bytes
            hash_b = bytes.fromhex(self._last_hash_hex) if self._last_hash_hex else b"\x00" * 32
            bi = (hash_b[30] << 8 | hash_b[31]) % 16
            self._logger.debug(
                f"[chain-skip] hash chain=({bi//4},{bi%4}) "
                f"job=({self._last_from_group},{self._last_to_group}) — discarding"
            )
            return False, "", ""

        # nonceSansExtraNonce = nonce24[2:] — exactly 22 bytes
        # Reference: stratum spec "full_nonce = extraNonce(2) + nonceSansExtraNonce(22)"
        nonce_sans_hex = self._last_nonce24[2:].hex()  # 22 bytes = 44 hex chars

        self._logger.debug(
            f"[submit] nonce24={self._last_nonce24.hex()}  "
            f"nonceSans={nonce_sans_hex}  ({len(nonce_sans_hex)//2}B)"
        )
        return True, nonce_sans_hex, self._last_hash_hex

    def close(self) -> None:
        pass


def detect_pool_algorithm(pool_url: str) -> str:
    """Detect mining algorithm from pool URL."""
    url_lower = pool_url.lower()
    if "alph" in url_lower:
        return "alephium"
    return "pearl"


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
        pool_algorithm = "pearl"

        if pool_mode:
            logger.info(f"Starting in POOL mode: {args.pool_url}")
            # Detect algorithm BEFORE connect so StratumClient is built with correct algorithm
            if args.pool_algorithm is not None:
                pool_algorithm = args.pool_algorithm
                logger.info(f"Pool algorithm forced by --pool-algorithm: {pool_algorithm}")
            else:
                pool_algorithm = detect_pool_algorithm(args.pool_url)
                logger.info(f"Detected pool algorithm: {pool_algorithm}")
            client = PoolMiningClient(
                pool_url=args.pool_url,
                username=args.pool_username,
                password=args.pool_password,
                worker_name=args.pool_worker,
                algorithm=pool_algorithm,
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
        if pool_mode and pool_algorithm == "alephium":
            session = AlephiumMiningSession(nonces_per_round=args.alph_nonces_per_round)
            logger.info(f"Alephium session ready — nonces_per_round={session.nonces_per_round}  ({session.nonces_per_round/1e6:.1f}M)")
        else:
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
        _job_cycle_idx = 0  # round-robin index into job_map for Alephium

        while True:
            try:
                # For Alephium: cycle through all available chain jobs in round-robin.
                # The pool sends one job per chain (up to 16 chains). By rotating
                # through them, each round mines a different chain so the winner's
                # chain will naturally match the current job.
                if pool_mode and pool_algorithm == "alephium" and isinstance(_client_ref[0], PoolMiningClient):
                    with _client_ref[0]._job_lock:
                        all_jobs = list(_client_ref[0]._job_map.values())
                    if all_jobs:
                        _job_cycle_idx = _job_cycle_idx % len(all_jobs)
                        _current_job = all_jobs[_job_cycle_idx]
                        _job_cycle_idx += 1
                    else:
                        # No jobs in map yet — fall back to queue
                        try:
                            _current_job = _job_queue.get_nowait()
                        except _queue.Empty:
                            if _current_job is None:
                                _current_job = _job_queue.get(timeout=5.0)
                else:
                    # Non-blocking: drain queue and take the latest job only.
                    try:
                        latest = _job_queue.get_nowait()
                        while True:
                            try:
                                latest = _job_queue.get_nowait()
                            except _queue.Empty:
                                break
                        _current_job = latest
                    except _queue.Empty:
                        if _current_job is None:
                            _current_job = _job_queue.get(timeout=5.0)

                mining_job = _current_job
                if isinstance(session, AlephiumMiningSession):
                    won, _winning_nonce24 = session.run_round(mining_job)
                else:
                    won = session.run_round(mining_job)

                session._t_end.synchronize()
                elapsed = session._t_start.elapsed_time(session._t_end) / 1000.0
                if isinstance(session, AlephiumMiningSession):
                    tmok_per_s = session.nonces_per_round / elapsed / 1e6 if elapsed > 0 else 0.0
                    logger.info(
                        f"[ROUND] alephium  time={elapsed*1000:.1f}ms  "
                        f"nonces={session.nonces_per_round}  "
                        f"MH/s={tmok_per_s:.2f}  "
                        f"base_nonce={session._base_nonce:#018x}"
                    )
                else:
                    tmok = (session.tile_h * session.tile_w * session.P * session.Q * session.num_jobs) / session.rank / 1000.0
                    tmok_per_s = tmok / elapsed if elapsed > 0 else 0.0
                    logger.info(f"GPU round time: {elapsed*1000:.1f}ms, tmok/s={tmok_per_s:.2f}  jobs={session.num_jobs}  combos={session.P*session.Q}")

                # In pool mode, immediately run the next round without waiting
                # for a new job — keeps GPU at P0 and maximizes tmok/s.
                # Only check for a new job between rounds (non-blocking).

                if won:
                    if pool_mode:
                        ok, nonce_hex, result_hex = session.get_last_result()
                        if ok and nonce_hex:
                            job_id = getattr(mining_job, "job_id", "")
                            success = _client_ref[0].submit_plain_proof(nonce_hex, result_hex, job_id)
                            if success:
                                logger.info(
                                    f"[SUBMIT OK] Pool received share  job_id={job_id}  "
                                    f"nonce={nonce_hex}"
                                )
                            else:
                                logger.debug(f"[SUBMIT SKIP] Rate-limited  job_id={job_id}")
                        else:
                            # chain_ok=False — the hash chain doesn't match this job.
                            # The nonce is valid (hash<=target) but for a different chain.
                            # We cannot submit it against a different job's blob —
                            # the pool recomputes hash(nonce + THIS_JOB_blob) and it won't match.
                            # The only correct action is to keep mining and wait for a
                            # winner whose chain matches the current job.
                            logger.debug(
                                f"[chain-skip] winner on wrong chain for job "
                                f"({session._last_from_group},{session._last_to_group})"
                            )
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
