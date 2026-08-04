#!/usr/bin/env python3
"""
End-to-end share test: connect to pool → get real job → mine GPU → submit share → verify accepted.

Usage:
    cd ~/pearl-cion
    python miner/test_e2e_share.py \
        --pool-url stratum+tcp://alph.kryptex.network:7010 \
        --pool-username 3cUrDWnDxgFVy1V8dEWjWR4npf5EcJKESoDNjL2Nmn4mssdW7n2Ui.worker \
        --pool-password x

Nonce submission mode (mirrors get_last_result PEARL_NONCE_SUBMIT_MODE):
    ref22   (default) — pool protocol nonceSansExtraNonce: zeros(14) + counter(8) = 22 bytes
                        Pool prepends its extranonce(2) → 24-byte nonce for verification.
    full24            — raw 24-byte big-endian nonce sent as-is.

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

def py_double_blake3(nonce: int, blob: bytes) -> bytes:
    nonce24 = nonce.to_bytes(24, "big")
    inner = _blake3.blake3(nonce24 + blob).digest()
    return _blake3.blake3(inner).digest()


def build_nonce_sans_ref22(nonce_int: int) -> bytes:
    """Mirror get_last_result(mode='ref22') exactly.

    Kernel layout:
        nonce_int = extranonce(2B) << 48 | counter(6B)
        nonce_full_24 big-endian = zeros(16) + extranonce(2) + counter(6)

    Pool protocol (nonceSansExtraNonce):
        The pool prepends its own extranonce(2) to the 22-byte nonce_sans it
        receives, reconstructing:
            extranonce(2) + nonce_sans(22) = 24-byte full nonce

        For that reconstruction to equal our nonce_full_24, nonce_sans must be:
            zeros(14) + counter_8(8)
        where counter_8 = b'\\x00\\x00' + counter_6  (zero-pad 6→8 bytes, big-endian)

    Returns:
        22-byte nonceSansExtraNonce bytes (not hex).
    """
    nonce_full_24: bytes = nonce_int.to_bytes(24, "big")
    counter_6: bytes = nonce_full_24[18:24]          # low 6 bytes of nonce_int
    counter_8: bytes = b"\x00\x00" + counter_6       # padded to 8 bytes
    nonce_sans: bytes = b"\x00" * 14 + counter_8     # 14 + 8 = 22 bytes
    return nonce_sans


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2E share submission test")
    p.add_argument("--pool-url",      required=True)
    p.add_argument("--pool-username", required=True)
    p.add_argument("--pool-password", default="x")
    p.add_argument("--timeout",       type=int, default=120,
                   help="Max seconds to wait for a share (default: 120)")
    p.add_argument("--nonces-per-round", type=int, default=1 << 24,
                   help="Nonces per GPU round (default: 16M)")
    p.add_argument("--nonce-mode",    default="ref22",
                   choices=["ref22", "full24"],
                   help="Nonce submit mode matching PEARL_NONCE_SUBMIT_MODE "
                        "(default: ref22 — nonceSansExtraNonce 22 bytes)")
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
                     nonces_per_round: int, timeout: int,
                     nonce_mode: str = "ref22") -> tuple[str, str, str, bytes, int, str]:
    """
    Returns (job_id, nonce_submit_hex, hash_hex, blob_bytes, nonce_int, nonce_full24_hex).

    nonce_submit_hex is the value actually sent to the pool:
      - ref22  → 22-byte nonceSansExtraNonce (zeros(14)+counter(8))
      - full24 → 24-byte raw big-endian nonce

    Exits with code 2 if timeout expires.
    """
    blob_gpu   = torch.zeros(320, dtype=torch.uint8, device="cuda")
    target_gpu = torch.empty(32,  dtype=torch.uint8, device="cuda")

    # Initialise base_nonce from extranonce
    extranonce_hex = client._extranonce1 or ""
    base_nonce: int = 0
    if extranonce_hex and len(extranonce_hex) >= 2:
        try:
            en_bytes = bytes.fromhex(extranonce_hex)[:2]
            base_nonce = int.from_bytes(en_bytes, "big") << 48
            print(f"{INFO} extranonce={en_bytes.hex()}  base_nonce={base_nonce:#018x}")
        except Exception:
            pass

    deadline = time.time() + timeout
    rounds   = 0
    job      = initial_job

    print(f"{INFO} Starting GPU mining loop  nonces_per_round={nonces_per_round:,}  "
          f"nonce_mode={nonce_mode}")

    while time.time() < deadline:
        rounds += 1

        # Refresh job from pool (non-blocking)
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

        # Load blob into GPU (302 bytes, last 18 stay zero)
        blob_gpu[:302].copy_(torch.frombuffer(blob_bytes, dtype=torch.uint8))
        blob_gpu[302:].zero_()

        # Load target
        target_bytes = job.target.to_bytes(32, "big")
        target_gpu.copy_(torch.frombuffer(target_bytes, dtype=torch.uint8))

        # Sign-extend if needed
        bn_signed = (base_nonce if base_nonce < (1 << 63)
                     else base_nonce - (1 << 64))

        t0 = time.perf_counter()
        found, raw_nonce = alph_mine_batch(blob_gpu, bn_signed,
                                           nonces_per_round, target_gpu)
        elapsed = time.perf_counter() - t0
        mhs = nonces_per_round / elapsed / 1e6

        base_nonce = (base_nonce + nonces_per_round) & 0xFFFFFFFFFFFFFFFF

        if rounds % 20 == 0:
            remaining = int(deadline - time.time())
            print(f"{INFO} round={rounds}  MH/s={mhs:.1f}  "
                  f"base_nonce={base_nonce:#018x}  timeout_in={remaining}s")

        if not found:
            continue

        nonce_int = int(raw_nonce) & 0xFFFFFFFFFFFFFFFF
        if nonce_int == 0xFFFFFFFFFFFFFFFF:
            print(f"{WARN} Kernel returned sentinel nonce — skipping")
            continue

        # ── Strict check: nonce must be within the range we just searched ──────
        range_start = (base_nonce - nonces_per_round) & 0xFFFFFFFFFFFFFFFF
        range_end   = (base_nonce - 1) & 0xFFFFFFFFFFFFFFFF
        # Handle wraparound: if range_start > range_end the range wrapped
        in_range = (
            range_start <= nonce_int <= range_end
            if range_start <= range_end
            else (nonce_int >= range_start or nonce_int <= range_end)
        )
        if not in_range:
            print(f"{FAIL} Nonce {nonce_int:#018x} is OUTSIDE search range "
                  f"[{range_start:#018x}, {range_end:#018x}] — kernel bug")
            sys.exit(2)

        # ── Verify hash with Python blake3 (ground truth) ─────────────────────
        hash_bytes = py_double_blake3(nonce_int, blob_bytes)
        hash_int   = int.from_bytes(hash_bytes, "big")

        # ── Chain index validation against pool source (util.js blockChainIndex) ──
        bi = (hash_bytes[30] << 8 | hash_bytes[31]) % 16
        hash_from_group = bi // 4
        hash_to_group   = bi % 4
        job_from_group  = getattr(job, "_from_group", None)
        job_to_group    = getattr(job, "_to_group", None)

        print(f"\n{INFO} Candidate found!")
        print(f"  round={rounds}  MH/s={mhs:.1f}")
        print(f"  nonce_int={nonce_int:#018x}")
        print(f"  nonce_hex={nonce_int.to_bytes(24,'big').hex()}  (24B)")
        print(f"  hash={hash_bytes.hex()}")
        print(f"  target={job.target:#x}")
        print(f"  hash <= target: {hash_int <= job.target}")
        print(f"  chain: hash=({hash_from_group},{hash_to_group})  job=({job_from_group},{job_to_group})")

        if hash_int > job.target:
            print(f"{WARN} Python verification FAILED: hash > target — skipping")
            continue

        # Chain index must match job — pool rejects with InvalidBlockChainIndex otherwise
        if job_from_group is not None and job_to_group is not None:
            if hash_from_group != job_from_group or hash_to_group != job_to_group:
                print(f"{WARN} Chain index mismatch: hash=({hash_from_group},{hash_to_group}) "
                      f"job=({job_from_group},{job_to_group}) — skipping, pool would reject")
                continue

        # ── Build both nonce formats (24-byte full and 22-byte sans) ──────────
        nonce_full_24_bytes = nonce_int.to_bytes(24, "big")
        nonce_full_24_hex   = nonce_full_24_bytes.hex()

        if nonce_mode == "ref22":
            nonce_sans_bytes = build_nonce_sans_ref22(nonce_int)
            nonce_submit_hex = nonce_sans_bytes.hex()
            print(f"  nonce_submit (ref22): {nonce_submit_hex}  ({len(nonce_submit_hex)//2}B)")
        elif nonce_mode == "full24":
            nonce_submit_hex = nonce_full_24_hex
            print(f"  nonce_submit (full24): {nonce_submit_hex}  ({len(nonce_submit_hex)//2}B)")
        else:
            print(f"{FAIL} Unknown nonce_mode: {nonce_mode}")
            sys.exit(1)

        return job.job_id, nonce_submit_hex, hash_bytes.hex(), blob_bytes, nonce_int, nonce_full_24_hex

    print(f"{FAIL} Timeout: no share found in {timeout}s after {rounds} rounds")
    print(f"       Check: target difficulty, kernel correctness, nonce range")
    sys.exit(2)


# ─── Step 3: submit the share and verify pool response ───────────────────────

def submit_and_verify(client: StratumClient, job_id: str,
                      nonce_submit_hex: str, result_hex: str,
                      blob_bytes: bytes, nonce_int: int,
                      nonce_full_24_hex: str, nonce_mode: str) -> None:
    print(f"\n{INFO} Submitting share ...")
    print(f"  job_id={job_id}")
    print(f"  nonce_mode={nonce_mode}")
    print(f"  nonce_submit_hex={nonce_submit_hex}  ({len(nonce_submit_hex)//2}B)")
    print(f"  result_hex={result_hex}")
    print(f"  worker_id={client.get_worker_id()!r}")

    # ── checkpoint 1: nonce_submit_hex length matches mode ────────────────────
    expected_len = 44 if nonce_mode == "ref22" else 48
    if len(nonce_submit_hex) != expected_len:
        print(f"{FAIL} CHECKPOINT 1: nonce_submit_hex len={len(nonce_submit_hex)} != {expected_len} "
              f"for mode={nonce_mode}")
        sys.exit(1)
    print(f"  CHECKPOINT 1: nonce_submit_hex length={expected_len} for mode={nonce_mode} ✓")

    # ── checkpoint 2: full24 nonce must encode correct nonce_int ─────────────
    nonce_from_hex = int(nonce_full_24_hex, 16)
    # nonce_int is uint64, nonce_full_24_hex is 24-byte big-endian — high 16 bytes are zeros
    nonce_int_from_hex = nonce_from_hex & 0xFFFFFFFFFFFFFFFF
    if nonce_int_from_hex != nonce_int:
        print(f"{FAIL} CHECKPOINT 2: nonce_full_24_hex encodes {nonce_int_from_hex:#018x} "
              f"but kernel found {nonce_int:#018x}")
        sys.exit(1)
    print(f"  CHECKPOINT 2: nonce_full_24_hex encodes correct nonce {nonce_int:#018x} ✓")

    # ── checkpoint 3: Python blake3 hash must match result_hex ───────────────
    py_hash = py_double_blake3(nonce_int, blob_bytes)
    if py_hash.hex() != result_hex:
        print(f"{FAIL} CHECKPOINT 3: hash mismatch")
        print(f"    expected: {py_hash.hex()}")
        print(f"    got:      {result_hex}")
        sys.exit(1)
    print(f"  CHECKPOINT 3: Python blake3 hash matches ✓")

    # ── checkpoint 4: chain index must match job ──────────────────────────────
    bi = (py_hash[30] << 8 | py_hash[31]) % 16
    hash_fg, hash_tg = bi // 4, bi % 4
    current_job = client.get_current_job()
    job_fg = getattr(current_job, "_from_group", None) if current_job else None
    job_tg = getattr(current_job, "_to_group",   None) if current_job else None
    if job_fg is not None and job_tg is not None:
        if hash_fg != job_fg or hash_tg != job_tg:
            print(f"{FAIL} CHECKPOINT 4: chain index mismatch "
                  f"hash=({hash_fg},{hash_tg}) job=({job_fg},{job_tg}) — pool would reject")
            sys.exit(1)
    print(f"  CHECKPOINT 4: chain index ({hash_fg},{hash_tg}) matches job ✓")

    # ── checkpoint 5: connected? ──────────────────────────────────────────────
    if not client.is_connected():
        print(f"{FAIL} CHECKPOINT 5: client.is_connected()=False before submit")
        sys.exit(1)
    print(f"  CHECKPOINT 5: connected ✓")

    # ── checkpoint 6: job still fresh? ───────────────────────────────────────
    current = client.get_current_job()
    if current and current.job_id != job_id:
        print(f"{WARN} CHECKPOINT 6: stale job_id={job_id} current={current.job_id} — submitting anyway")
    else:
        print(f"  CHECKPOINT 6: job_id fresh ✓")

    # ── checkpoint 7: ref22 nonce_sans internal layout ───────────────────────
    # Pool protocol (Kryptex Alephium):
    #   full_nonce_submitted = extranonce(2) + nonce_sans(22) = 24 bytes
    # Our kernel: nonce_int = extranonce<<48 | counter_6
    #   nonce_full_24 = zeros(16) + extranonce(2) + counter(6)   [big-endian]
    # So extranonce bytes = nonce_full_24[16:18]
    # nonce_sans must be zeros(14) + zero_pad(counter_6→8)
    # Pool reconstruction: extranonce(2) + nonce_sans(22) = nonce_full_24[16:18] + nonce_sans
    #   must equal nonce_full_24[16:]  →  nonce_sans == nonce_full_24[18:] zero-padded to 22B
    if nonce_mode == "ref22":
        nonce_sans_bytes = build_nonce_sans_ref22(nonce_int)

        # 7a: must be exactly 22 bytes
        if len(nonce_sans_bytes) != 22:
            print(f"{FAIL} CHECKPOINT 7a: nonce_sans length={len(nonce_sans_bytes)} != 22")
            sys.exit(1)
        print(f"  CHECKPOINT 7a: nonce_sans length=22 ✓")

        # 7b: high 14 bytes must all be zero
        if nonce_sans_bytes[:14] != b"\x00" * 14:
            print(f"{FAIL} CHECKPOINT 7b: nonce_sans high-14 bytes are not zero: "
                  f"{nonce_sans_bytes[:14].hex()}")
            sys.exit(1)
        print(f"  CHECKPOINT 7b: nonce_sans[:14] = zeros ✓")

        # 7c: bytes [14:16] must be zero (the extra zero padding of counter_6 → counter_8)
        if nonce_sans_bytes[14:16] != b"\x00\x00":
            print(f"{FAIL} CHECKPOINT 7c: nonce_sans[14:16]={nonce_sans_bytes[14:16].hex()} != 0000 "
                  f"(counter_8 padding bytes must be zero)")
            sys.exit(1)
        print(f"  CHECKPOINT 7c: nonce_sans[14:16] = 0x0000 (counter padding) ✓")

        # 7d: bytes [16:22] must match counter_6 from nonce_full_24
        nonce_full_24_bytes = bytes.fromhex(nonce_full_24_hex)
        counter_6 = nonce_full_24_bytes[18:24]
        if nonce_sans_bytes[16:22] != counter_6:
            print(f"{FAIL} CHECKPOINT 7d: nonce_sans[16:22]={nonce_sans_bytes[16:22].hex()} "
                  f"!= counter_6={counter_6.hex()}")
            sys.exit(1)
        print(f"  CHECKPOINT 7d: nonce_sans[16:22] = counter_6 ({counter_6.hex()}) ✓")

        # ── checkpoint 8: pool-reconstructed nonce must round-trip via blake3 ─
        # Pool does: blake3(blake3(extranonce(2) + nonce_sans(22) + headerBlob))
        # extranonce bytes come from nonce_full_24[16:18]
        extranonce_2bytes = nonce_full_24_bytes[16:18]
        reconstructed_nonce24 = extranonce_2bytes + nonce_sans_bytes  # 2 + 22 = 24
        inner_rc = _blake3.blake3(reconstructed_nonce24 + blob_bytes).digest()
        hash_reconstructed = _blake3.blake3(inner_rc).digest()
        hash_original       = py_double_blake3(nonce_int, blob_bytes)
        if hash_reconstructed != hash_original:
            print(f"{FAIL} CHECKPOINT 8: pool-reconstructed nonce produces different hash")
            print(f"    reconstructed_nonce24 = {reconstructed_nonce24.hex()}")
            print(f"    hash_reconstructed    = {hash_reconstructed.hex()}")
            print(f"    hash_original         = {hash_original.hex()}")
            print(f"    This means the nonce_sans layout is WRONG — pool will reject")
            sys.exit(1)
        print(f"  CHECKPOINT 8: pool reconstruction blake3(blake3(extranonce+nonce_sans+blob)) "
              f"= original hash ✓")
        print(f"    extranonce={extranonce_2bytes.hex()}  reconstructed_nonce24={reconstructed_nonce24.hex()}")

        # ── checkpoint 9: submit nonce_hex must equal nonce_sans_hex ─────────
        if nonce_submit_hex != nonce_sans_bytes.hex():
            print(f"{FAIL} CHECKPOINT 9: nonce_submit_hex != nonce_sans_hex — encoding mismatch")
            sys.exit(1)
        print(f"  CHECKPOINT 9: nonce_submit_hex == nonce_sans_hex ✓")

    elif nonce_mode == "full24":
        # For full24 mode just confirm it matches the raw nonce_int
        if nonce_submit_hex != nonce_full_24_hex:
            print(f"{FAIL} CHECKPOINT 7: full24 nonce_submit_hex != nonce_full_24_hex")
            sys.exit(1)
        print(f"  CHECKPOINT 7: full24 nonce_submit_hex matches nonce_full_24_hex ✓")

    # ── submit ────────────────────────────────────────────────────────────────
    t_submit = time.time()
    success = client.submit_share(job_id, nonce_submit_hex)
    latency = (time.time() - t_submit) * 1000

    print(f"\n  submit latency: {latency:.0f}ms")

    if success:
        print(f"\n{PASS} Share ACCEPTED by pool  ✓")
        print("       Dashboard should now show +1 share")
        sys.exit(0)
    else:
        print(f"\n{FAIL} Share REJECTED by pool")
        print("  Possible causes:")
        print("  • Stale job")
        print("  • Wrong nonce format (check PEARL_NONCE_SUBMIT_MODE env var)")
        print("  • Hash does not match pool's verification")
        print("  • Worker not authorized")
        sys.exit(1)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Alephium E2E Share Submission Test")
    print("=" * 60)
    print(f"  pool:       {args.pool_url}")
    print(f"  user:       {args.pool_username}")
    print(f"  timeout:    {args.timeout}s")
    print(f"  nonces:     {args.nonces_per_round:,} per round")
    print(f"  nonce_mode: {args.nonce_mode}")
    print("=" * 60)

    # Step 1
    client, job = wait_for_job(
        args.pool_url, args.pool_username, args.pool_password,
        timeout=30,
    )
    print(f"{INFO} First job: id={job.job_id}  target={job.target:#x}  "
          f"blob_len={len(job.blob)}  height={job.height}")

    # Step 2
    (job_id, nonce_submit_hex, hash_hex,
     blob_bytes, nonce_int, nonce_full_24_hex) = mine_until_share(
        client, job,
        nonces_per_round=args.nonces_per_round,
        timeout=args.timeout,
        nonce_mode=args.nonce_mode,
    )

    # Step 3
    submit_and_verify(
        client, job_id,
        nonce_submit_hex, hash_hex,
        blob_bytes, nonce_int,
        nonce_full_24_hex, args.nonce_mode,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(3)
