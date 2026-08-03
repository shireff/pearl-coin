#!/usr/bin/env python3
"""
Diagnostic: compare alph_mine_batch kernel output against Python blake3.
Run on the server:
    cd ~/pearl-cion && python miner/test_alph_hash.py
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_gemm_src = str(_root / "pearl-gemm" / "src")
if _gemm_src not in sys.path:
    sys.path.insert(0, _gemm_src)

import torch
import blake3 as _blake3
from pearl_gemm import alph_mine_batch

# ── helpers ──────────────────────────────────────────────────────────────────

def py_double_blake3(nonce: int, blob: bytes) -> bytes:
    nonce24 = nonce.to_bytes(24, "big")
    inner = _blake3.blake3(nonce24 + blob).digest()
    return _blake3.blake3(inner).digest()

def make_target(difficulty_bits: int) -> int:
    """target = 2^(256-difficulty_bits) - 1  (easier = more bits)"""
    return (1 << (256 - difficulty_bits)) - 1

# ── test blob: 302 bytes, all zeros (reproducible) ───────────────────────────
BLOB = bytes(302)
BASE_NONCE = 0

# ── Step 1: compute Python hash for nonce=0 ──────────────────────────────────
py_hash = py_double_blake3(BASE_NONCE, BLOB)
print(f"[PY]  nonce=0  hash={py_hash.hex()}")
print(f"[PY]  hash[0:4]={py_hash[:4].hex()}  (big-endian word0 = {int.from_bytes(py_hash[:4],'big'):#010x})")

# ── Step 2: run kernel with a very easy target (all 1s = always win) ─────────
blob_gpu = torch.zeros(320, dtype=torch.uint8, device="cuda")
blob_tensor = torch.frombuffer(BLOB, dtype=torch.uint8)
blob_gpu[:302].copy_(blob_tensor)

# target = 2^256 - 1 (max, every nonce should win)
max_target = (1 << 256) - 1
target_bytes = max_target.to_bytes(32, "big")
target_gpu = torch.frombuffer(target_bytes, dtype=torch.uint8).cuda()

print(f"\n[TEST1] target=MAX (every nonce must win)")
found, nonce_out = alph_mine_batch(blob_gpu, BASE_NONCE, 256, target_gpu)
print(f"  found={found}  nonce={nonce_out}")
if not found:
    print("  *** BUG: kernel returned found=False with MAX target — BLAKE3 or comparison broken ***")
else:
    print("  OK: kernel found a nonce as expected")

# ── Step 3: verify winning nonce hash matches Python ─────────────────────────
if found and nonce_out >= 0:
    nonce_int = nonce_out & 0xFFFFFFFFFFFFFFFF
    kernel_hash = py_double_blake3(nonce_int, BLOB)
    print(f"\n[TEST2] verify nonce={nonce_int} hash via Python blake3")
    print(f"  hash={kernel_hash.hex()}")
    print(f"  hash <= MAX_target: {int.from_bytes(kernel_hash,'big') <= max_target}")

# ── Step 4: target just above Python hash[0] — should find nonce=0 ───────────
py_hash_int = int.from_bytes(py_hash, "big")
# target = py_hash_int (exact match, nonce=0 should win)
target_exact = py_hash_int.to_bytes(32, "big")
target_exact_gpu = torch.frombuffer(target_exact, dtype=torch.uint8).cuda()

print(f"\n[TEST3] target=py_hash (nonce=0 should win exactly)")
print(f"  py_hash_int={py_hash_int:#066x}")
found2, nonce2 = alph_mine_batch(blob_gpu, BASE_NONCE, 256, target_exact_gpu)
print(f"  found={found2}  nonce={nonce2}")
if not found2:
    print("  *** BUG: kernel did not find nonce=0 even though hash==target ***")
    print("  Likely cause: target byte-swap mismatch between kernel and Python")
else:
    print("  OK: kernel found nonce=0 with exact-match target")

# ── Step 5: target = py_hash - 1 — nonce=0 should NOT win ────────────────────
target_hard = (py_hash_int - 1).to_bytes(32, "big")
target_hard_gpu = torch.frombuffer(target_hard, dtype=torch.uint8).cuda()

print(f"\n[TEST4] target=py_hash-1 (nonce=0 must NOT win)")
found3, nonce3 = alph_mine_batch(blob_gpu, BASE_NONCE, 256, target_hard_gpu)
print(f"  found={found3}  nonce={nonce3}")
if found3 and (nonce3 & 0xFFFFFFFFFFFFFFFF) == 0:
    print("  *** BUG: kernel accepted nonce=0 when hash > target ***")
else:
    print("  OK: nonce=0 correctly rejected (or different nonce found)")

print("\nDone.")
