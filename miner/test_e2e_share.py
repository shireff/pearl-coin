#!/usr/bin/env python3
"""
End-to-end share test: connect to pool → get real job → mine GPU → submit share → verify accepted.

Usage:
    cd ~/pearl-cion
    python miner/test_e2e_share.py \
        --pool-url stratum+tcp://alph.kryptex.network:7010 \
        --pool-username 3cUrDWnDxgFVy1V8dEWjWR4npf5EcJKESoDNjL2Nmn4mssdW7n2Ui.worker \
        --pool-password x

Nonce protocol (reference: docs.alephium.org/mining/alephium-stratum + gpu-miner/src/worker.h):
    full_nonce_24       = nonce24  (24 bytes produced by kernel)
    nonceSansExtraNonce = nonce24[2:]  (22 bytes sent to pool)
    pool reconstructs:  extraNonce(2) + nonceSans(22) = nonce24
    pool verifies:      blake3(blake3(nonce24 + headerBlob)) <= target

Exit codes:
    0 = share found AND accepted by pool  (test PASSED)
    1 = share found but rejected by pool  (submit pipeline broken)
    2 = no share found in time limit      (kernel / target problem)
    3 = connection / setup failure        (network / auth problem)
"""
import argparse
import os
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent
_gemm_src = str(_root / "pearl-gemm" / "src")
if _gemm_src not in sys.path:
    sys.path.insert(0, _gemm_src)

import torch
import blake3 as _blake3
from pearl_gemm import alph_mine_batch

# ─── path fix so stratum_client / pool_mining_client are importable ───────────
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from stratum_client import StratumClient, StratumJob


PASS  = "\033[92m[PASS]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
INFO  = "\033[94m[INFO]\033[0m"
WARN  = "\033[93m[WARN]\033[0m"


# ─── helpers ─────────────────────────────────────────────────────────────────

def py_double_blake3(nonce24: bytes, blob: bytes) -> bytes:
    """Reference: inlined-blake.hpp DOUBLE_HASH — blake3(blake3(nonce24 + headerBlob)).

    Args:
        nonce24: exact 24-byte nonce (buf[0:24] from kernel buffer)
        blob:    headerBlob (302 bytes from pool)
    """
    assert len(nonce24) == 24, f"nonce24 must be 24 bytes, got {len(nonce24)}"
    inner = _blake3.blake3(nonce24 + blob).digest()
    return _blake3.blake3(inner).digest()


def nonce_sans_from_nonce24(nonce24: bytes) -> bytes:
    """Compute nonceSansExtraNonce from the full 24-byte nonce.

    Reference: Alephium Stratum spec (docs.alephium.org/mining/alephium-stratum):
        mining.submit params: [jobId, nonceSansExtraNonce]
        full_nonce_24 = extraNonce(2) + nonceSansExtraNonce(22)
        → nonceSansExtraNonce = full_nonce_24[2:]

    Args:
        nonce24: the 24-byte nonce buf[0:24] from the kernel

    Returns:
        22-byte nonceSansExtraNonce (= nonce24[2:])
    """
    assert len(nonce24) == 24
    return nonce24[2:]  # exactly 22 bytes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2E share submission test")
    p.add_argument("--pool-url",      required=True)
    p.add_argument("--pool-username", required=True)
    p.add_argument("--pool-password", default="x")
    p.add_argument("--timeout",       type=int, default=120,
                   help="Max seconds to wait for a share (default: 120)")
    p.add_argument("--nonces-per-round", type=int, default=1 << 24,
                   help="Nonces per GPU round (default: 16M)")
    return p.parse_args()


# ─── Step 1: connect & collect first real job ─────────────────────────────────

def wait_for_job(pool_url: str, username: str, password: str,
                 timeout: int = 30) -> tuple[StratumClient, StratumJob]:
    """Connect to pool and block until a mining job arrives."""
    job_holder: list[StratumJob] = []

    import threading
    job_event = threading.Event()

    def on_job(j: StratumJob) -> None:
        if not job_holder:
            job_holder.append(j)
        else:
            job_holder[0] = j          # always keep latest
        job_event.set()

    def on_error(msg: str) -> None:
        print(f"{FAIL} Pool error: {msg}")

    client = StratumClient(
        pool_url=pool_url,
        username=username,
        password=password,
        algorithm="alephium",
        on_job=on_job,
        on_error=on_error,
    )

    print(f"{INFO} Connecting to {pool_url} ...")
    if not client.connect():
        print(f"{FAIL} Connection/auth failed")
        sys.exit(3)

    print(f"{INFO} Connected. worker_id={client.get_worker_id()!r}  "
          f"extranonce={client._extranonce1!r}")

    # Pool may push a job immediately after authorize
    existing = client.get_current_job()
    if existing:
        print(f"{INFO} Job already available: id={existing.job_id}")
        return client, existing

    print(f"{INFO} Waiting up to {timeout}s for first job ...")
    if not job_event.wait(timeout):
        print(f"{FAIL} No job received within {timeout}s")
        client.disconnect()
        sys.exit(3)

    return client, job_holder[0]


# ─── Step 2: mine until a share is found ─────────────────────────────────────

def mine_until_share(client: StratumClient, initial_job: StratumJob,
                     nonces_per_round: int, timeout: int) -> tuple[str, str, str, bytes, bytes]:
    """Mine until a valid share is found.

    Returns:
        (job_id, nonce_sans_hex, hash_hex, blob_bytes, nonce24)

    Nonce layout (reference: inlined-blake.hpp + stratum spec):
        kernel writes uint64 little-endian into buf[0:8], buf[8:24] = zeros

        Pool protocol:
            full_nonce_24 = extraNonce(2) + nonceSans(22)
            → nonce24[0:2] MUST equal extraNonce

        Our kernel increments the full uint64 by 1 per step, so bytes [0:2]
        change with every nonce. We cannot guarantee nonce24[0:2]==extranonce
        for arbitrary nonces — we simply accept the winner if it happens to
        satisfy nonce24[0:2]==extranonce, and search more rounds if not.

        To maximize valid hits: we set base_nonce so that the START of the
        range has nonce24[0:2]==extranonce, and use nonces_per_round=0x10000
        so the range wraps exactly through one period of the low 16 bits,
        guaranteeing exactly one valid nonce24[0:2]==extranonce match per 64K.
        For large nonces_per_round, we scan multiple 64K windows per round.
    """
    blob_gpu   = torch.zeros(320, dtype=torch.uint8, device="cuda")
    target_gpu = torch.empty(32,  dtype=torch.uint8, device="cuda")

    # Parse extranonce — 2 bytes from pool
    extranonce_hex = client._extranonce1 or ""
    extranonce_bytes = b"\x00\x00"
    extranonce_le16  = 0
    if extranonce_hex and len(extranonce_hex) >= 4:
        try:
            eb = bytes.fromhex(extranonce_hex)[:2]
            extranonce_bytes = eb
            # In little-endian uint64: byte[0]=eb[0], byte[1]=eb[1]
            # → uint64 & 0xFFFF = eb[0] | (eb[1]<<8)
            extranonce_le16 = eb[0] | (eb[1] << 8)
        except Exception:
            pass
    print(f"{INFO} extranonce={extranonce_bytes.hex()}  "
          f"le16={extranonce_le16:#06x}  (nonce24[0:2] must equal this)")

    # Use 64K window: guarantees exactly one extranonce-matching nonce per window
    WINDOW = 0x10000
    # How many windows fit in the requested nonces_per_round
    windows_per_round = max(1, nonces_per_round // WINDOW)
    print(f"{INFO} Using window=64K  windows_per_round={windows_per_round}  "
          f"effective_nonces={windows_per_round * WINDOW:,}")

    window_counter = 0  # increments each window, shifts bytes [2:8] of nonce
    deadline = time.time() + timeout
    rounds   = 0
    job      = initial_job

    while time.time() < deadline:
        rounds += 1

        latest = client.get_current_job()
        if latest and latest.job_id != job.job_id:
            job = latest
            print(f"{INFO} New job: id={job.job_id}  target={job.target:#x}  "
                  f"height={job.height}")

        blob_bytes = bytes(job.blob)
        if len(blob_bytes) != 302:
            print(f"{WARN} Unexpected blob length {len(blob_bytes)}, skipping")
            time.sleep(0.05)
            continue

        blob_gpu[:302].copy_(torch.frombuffer(blob_bytes, dtype=torch.uint8))
        blob_gpu[302:].zero_()
        target_bytes = job.target.to_bytes(32, "big")
        target_gpu.copy_(torch.frombuffer(target_bytes, dtype=torch.uint8))

        found_any = False
        for w in range(windows_per_round):
            # base_nonce: low 16 bits = extranonce_le16, upper 48 bits = window_counter
            # → nonce_uint64.to_bytes(8,'little')[0:2] = extranonce_bytes for ALL nonces in window
            base_nonce = extranonce_le16 | ((window_counter + w) << 16)
            bn_signed  = base_nonce if base_nonce < (1 << 63) else base_nonce - (1 << 64)

            t0 = time.perf_counter()
            found, raw_nonce = alph_mine_batch(
                blob_gpu, bn_signed, WINDOW, target_gpu
            )
            elapsed = time.perf_counter() - t0

            if not found:
                continue

            nonce_uint64 = int(raw_nonce) & 0xFFFFFFFFFFFFFFFF
            if nonce_uint64 == 0xFFFFFFFFFFFFFFFF:
                continue

            nonce24 = nonce_uint64.to_bytes(8, "little") + b"\x00" * 16

            # Sanity: nonce24[0:2] must be extranonce
            if nonce24[:2] != extranonce_bytes:
                # Shouldn't happen with correct base_nonce, but log if it does
                print(f"{WARN} unexpected: nonce24[0:2]={nonce24[:2].hex()} "
                      f"!= extranonce={extranonce_bytes.hex()} — base_nonce={base_nonce:#018x}")
                continue

            hash_bytes = py_double_blake3(nonce24, blob_bytes)
            hash_int   = int.from_bytes(hash_bytes, "big")

            bi = (hash_bytes[30] << 8 | hash_bytes[31]) % 16
            hash_fg = bi // 4
            hash_tg = bi % 4
            job_fg  = getattr(job, "_from_group", None)
            job_tg  = getattr(job, "_to_group",   None)

            mhs = WINDOW / elapsed / 1e6
            print(f"\n{INFO} Candidate found!")
            print(f"  round={rounds}  window={w}  MH/s={mhs:.1f}")
            print(f"  nonce_uint64      = {nonce_uint64:#018x}")
            print(f"  nonce24           = {nonce24.hex()}")
            print(f"  nonce24[0:2]      = {nonce24[:2].hex()}  == extranonce ✓")
            print(f"  nonceSans         = {nonce24[2:].hex()}  (22B)")
            print(f"  pool_reconstructs = {extranonce_bytes.hex()}{nonce24[2:].hex()}")
            print(f"  hash              = {hash_bytes.hex()}")
            print(f"  target            = {job.target:#x}")
            print(f"  hash <= target    = {hash_int <= job.target}")
            print(f"  chain: hash=({hash_fg},{hash_tg})  job=({job_fg},{job_tg})")

            if hash_int > job.target:
                print(f"{WARN} hash > target — skipping")
                continue

            if job_fg is not None and job_tg is not None:
                if hash_fg != job_fg or hash_tg != job_tg:
                    print(f"{WARN} chain mismatch — skipping")
                    continue

            nonce_sans_hex = nonce24[2:].hex()
            window_counter += windows_per_round
            return job.job_id, nonce_sans_hex, hash_bytes.hex(), blob_bytes, nonce24

        window_counter += windows_per_round

        if rounds % 100 == 0:
            remaining = int(deadline - time.time())
            print(f"{INFO} round={rounds}  window_counter={window_counter}  "
                  f"timeout_in={remaining}s")

    print(f"{FAIL} Timeout: no share found in {timeout}s after {rounds} rounds")
    sys.exit(2)


# ─── Step 3: submit the share and verify pool response ───────────────────────

def submit_and_verify(client: StratumClient, job_id: str,
                      nonce_sans_hex: str, hash_hex: str,
                      blob_bytes: bytes, nonce24: bytes) -> None:
    """Run pre-submit checkpoints then submit nonceSansExtraNonce to pool.

    Args:
        nonce_sans_hex: nonce24[2:].hex() — 22 bytes / 44 hex chars
        hash_hex:       blake3(blake3(nonce24 + blob)).hex()
        blob_bytes:     raw 302-byte headerBlob from pool
        nonce24:        full 24-byte nonce as bytes
    """
    print(f"\n{INFO} Submitting share ...")
    print(f"  job_id         = {job_id}")
    print(f"  nonce24        = {nonce24.hex()}  (24B)")
    print(f"  nonceSans      = {nonce_sans_hex}  ({len(nonce_sans_hex)//2}B, sent to pool)")
    print(f"  hash           = {hash_hex}")
    print(f"  worker_id      = {client.get_worker_id()!r}")
    print(f"  extranonce     = {client._extranonce1!r}")

    # ── CHECKPOINT 1: nonceSans must be exactly 22 bytes (44 hex chars) ──────
    # Reference: stratum spec — nonceSansExtraNonce is 22 bytes
    if len(nonce_sans_hex) != 44:
        print(f"{FAIL} CHECKPOINT 1: nonce_sans_hex len={len(nonce_sans_hex)} != 44")
        sys.exit(1)
    print(f"  CHECKPOINT 1: nonce_sans length=22B ✓")

    # ── CHECKPOINT 2: nonce24 must be exactly 24 bytes ────────────────────────
    if len(nonce24) != 24:
        print(f"{FAIL} CHECKPOINT 2: nonce24 len={len(nonce24)} != 24")
        sys.exit(1)
    print(f"  CHECKPOINT 2: nonce24 length=24B ✓")

    # ── CHECKPOINT 3: nonceSans == nonce24[2:] ────────────────────────────────
    # Reference: stratum spec — nonceSansExtraNonce = full_nonce[2:]
    expected_sans = nonce24[2:].hex()
    if nonce_sans_hex != expected_sans:
        print(f"{FAIL} CHECKPOINT 3: nonce_sans_hex != nonce24[2:].hex()")
        print(f"    got:      {nonce_sans_hex}")
        print(f"    expected: {expected_sans}")
        sys.exit(1)
    print(f"  CHECKPOINT 3: nonce_sans == nonce24[2:] ✓")

    # ── CHECKPOINT 4: Python blake3 must match hash_hex ──────────────────────
    # Reference: inlined-blake.hpp DOUBLE_HASH = blake3(blake3(nonce24 + headerBlob))
    py_hash = py_double_blake3(nonce24, blob_bytes)
    if py_hash.hex() != hash_hex:
        print(f"{FAIL} CHECKPOINT 4: hash mismatch")
        print(f"    expected: {py_hash.hex()}")
        print(f"    got:      {hash_hex}")
        sys.exit(1)
    print(f"  CHECKPOINT 4: blake3(blake3(nonce24+blob)) matches ✓")

    # ── CHECKPOINT 5: pool reconstruction round-trip ──────────────────────────
    # Pool receives nonceSans(22), prepends extraNonce(2):
    #   reconstructed_nonce24 = extraNonce(2) + nonceSans(22)
    # This must equal our nonce24 for the pool to get the same hash.
    extranonce_hex = client._extranonce1 or ""
    if extranonce_hex and len(extranonce_hex) >= 4:
        try:
            extranonce_bytes = bytes.fromhex(extranonce_hex)[:2]
            nonce_sans_bytes = bytes.fromhex(nonce_sans_hex)
            reconstructed = extranonce_bytes + nonce_sans_bytes  # 2 + 22 = 24
            pool_hash = py_double_blake3(reconstructed, blob_bytes)
            if pool_hash != py_hash:
                print(f"{FAIL} CHECKPOINT 5: pool reconstruction gives DIFFERENT hash")
                print(f"    extranonce          = {extranonce_bytes.hex()}")
                print(f"    reconstructed nonce = {reconstructed.hex()}")
                print(f"    pool_hash           = {pool_hash.hex()}")
                print(f"    our_hash            = {py_hash.hex()}")
                print(f"    → pool will reject. nonce24[0:2]={nonce24[0:2].hex()} must equal extranonce")
                sys.exit(1)
            print(f"  CHECKPOINT 5: pool reconstruction hash matches ✓")
            print(f"    extranonce={extranonce_bytes.hex()}  reconstructed={reconstructed.hex()}")
        except Exception as e:
            print(f"{WARN} CHECKPOINT 5: skipped (extranonce parse error: {e})")
    else:
        print(f"{WARN} CHECKPOINT 5: skipped (no extranonce from pool yet)")

    # ── CHECKPOINT 6: hash <= target ──────────────────────────────────────────
    current_job = client.get_current_job()
    if current_job:
        hash_int = int.from_bytes(py_hash, "big")
        if hash_int > current_job.target:
            print(f"{FAIL} CHECKPOINT 6: hash > target — share would be rejected")
            sys.exit(1)
        print(f"  CHECKPOINT 6: hash <= target ✓")

    # ── CHECKPOINT 7: chain index matches job ─────────────────────────────────
    # Reference: inlined-blake.hpp CHECK_INDEX + pool blockChainIndex
    bi = (py_hash[30] << 8 | py_hash[31]) % 16
    hash_fg, hash_tg = bi // 4, bi % 4
    job_fg = getattr(current_job, "_from_group", None) if current_job else None
    job_tg = getattr(current_job, "_to_group",   None) if current_job else None
    if job_fg is not None and job_tg is not None:
        if hash_fg != job_fg or hash_tg != job_tg:
            print(f"{FAIL} CHECKPOINT 7: chain mismatch "
                  f"hash=({hash_fg},{hash_tg}) job=({job_fg},{job_tg})")
            sys.exit(1)
    print(f"  CHECKPOINT 7: chain index ({hash_fg},{hash_tg}) matches job ✓")

    # ── CHECKPOINT 8: still connected ────────────────────────────────────────
    if not client.is_connected():
        print(f"{FAIL} CHECKPOINT 8: not connected")
        sys.exit(1)
    print(f"  CHECKPOINT 8: connected ✓")

    # ── CHECKPOINT 9: job freshness warning ──────────────────────────────────
    if current_job and current_job.job_id != job_id:
        print(f"{WARN} CHECKPOINT 9: stale job_id={job_id} "
              f"current={current_job.job_id} — submitting anyway")
    else:
        print(f"  CHECKPOINT 9: job_id fresh ✓")

    # ── SUBMIT ────────────────────────────────────────────────────────────────
    t0 = time.time()
    success = client.submit_share(job_id, nonce_sans_hex)
    latency = (time.time() - t0) * 1000
    print(f"\n  submit latency: {latency:.0f}ms")

    if success:
        print(f"\n{PASS} Share ACCEPTED by pool ✓")
        print("       Dashboard should show +1 valid share")
        sys.exit(0)
    else:
        print(f"\n{FAIL} Share REJECTED by pool")
        print("  Diagnose:")
        print("  • Check checkpoint 5 — does extranonce match nonce24[0:2]?")
        print("  • Is the job stale?")
        print("  • Does pool log show an error?")
        sys.exit(1)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Alephium E2E Share Submission Test")
    print("=" * 60)
    print(f"  pool:    {args.pool_url}")
    print(f"  user:    {args.pool_username}")
    print(f"  timeout: {args.timeout}s")
    print(f"  nonces:  {args.nonces_per_round:,} per round")
    print("=" * 60)

    # Step 1: connect and get first job
    client, job = wait_for_job(
        args.pool_url, args.pool_username, args.pool_password,
        timeout=30,
    )
    print(f"{INFO} First job: id={job.job_id}  target={job.target:#x}  "
          f"blob_len={len(job.blob)}  height={job.height}")

    # Step 2: mine until a valid share is found
    job_id, nonce_sans_hex, hash_hex, blob_bytes, nonce24 = mine_until_share(
        client, job,
        nonces_per_round=args.nonces_per_round,
        timeout=args.timeout,
    )

    # Step 3: verify and submit
    submit_and_verify(client, job_id, nonce_sans_hex, hash_hex, blob_bytes, nonce24)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(3)
