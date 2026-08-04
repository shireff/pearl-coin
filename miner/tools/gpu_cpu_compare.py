#!/usr/bin/env python3
"""Compare GPU kernel hash outputs with a CPU lexicographic comparator.

Usage examples:
  - Load dumped arrays: python miner/tools/gpu_cpu_compare.py --npz /tmp/snapshot.npz
  - Attach to a running session: python -c "import miner.miner_gpu as m; from miner.tools.gpu_cpu_compare import compare_session; compare_session(m.session)"

The script exposes `compare_session(session, samples=1000, seed=None)` which inspects
`session._last_hashes` and `session._target_buf` and verifies the CPU comparator
matches the lexicographic semantics used on GPU.
"""
from __future__ import annotations

import argparse
import importlib
import random
from typing import Optional

import numpy as np
import torch


def words_to_int_be(words) -> int:
    """Convert a sequence of 8 32-bit words (MSW first) to a single big int."""
    v = 0
    for w in words:
        v = (v << 32) | (int(w) & 0xFFFFFFFF)
    return v


def cpu_bigint_compare(hashes: np.ndarray, target_words: np.ndarray) -> np.ndarray:
    """Return boolean array where each entry is True iff hash < target (big-int compare).

    Args:
      hashes: (N,8) array of uint32/ints MSW first.
      target_words: (8,) array of uint32/ints MSW first.
    """
    target_int = words_to_int_be(target_words)
    out = np.empty(hashes.shape[0], dtype=bool)
    for i in range(hashes.shape[0]):
        out[i] = words_to_int_be(hashes[i]) < target_int
    return out


def cpu_lexicographic_numpy(hashes: np.ndarray, target_words: np.ndarray) -> np.ndarray:
    """Reproduce the GPU lexicographic mask logic using numpy boolean ops.

    This implements the same semantics as the torch code in `MiningGraphSession.run_round()`:
      - smaller = hashes < target_row
      - equal = hashes == target_row
      - eq_cumulative = cumprod(equal, axis=1)
      - eq_prefix = [1] + eq_cumulative[:, :-1]
      - win = any(smaller & eq_prefix, axis=1)
    """
    hashes = hashes.astype(np.int64)
    target = target_words.astype(np.int64)
    smaller = hashes < target
    equal = hashes == target
    eq_cumulative = np.cumprod(equal.astype(np.uint8), axis=1).astype(bool)
    ones_col = np.ones((hashes.shape[0], 1), dtype=bool)
    eq_prefix = np.concatenate([ones_col, eq_cumulative[:, :-1]], axis=1)
    win = np.any(smaller & eq_prefix, axis=1)
    return win


def get_arrays_from_session(session) -> tuple[np.ndarray, np.ndarray]:
    """Extract last_hashes (N,8) and target_words (8,) as numpy arrays from a session object."""
    if getattr(session, "_last_hashes", None) is None:
        raise RuntimeError("session._last_hashes is None — run a round first")
    # _last_hashes stored as torch tensor on CPU in MiningGraphSession
    last_hashes = session._last_hashes
    if isinstance(last_hashes, torch.Tensor):
        hashes_np = last_hashes.numpy().astype(np.uint32)
    else:
        hashes_np = np.array(last_hashes, dtype=np.uint32)

    # target buffer may live on CUDA — copy to CPU
    t = getattr(session, "_target_buf", None)
    if t is None:
        raise RuntimeError("session._target_buf not found")
    if isinstance(t, torch.Tensor):
        target_np = t.cpu().numpy().astype(np.uint32)
    else:
        target_np = np.array(t, dtype=np.uint32)

    if target_np.shape[0] != 8 or hashes_np.shape[1] != 8:
        raise RuntimeError(f"Unexpected shapes: hashes={hashes_np.shape} target={target_np.shape}")
    return hashes_np, target_np


def compare_session(session, samples: int = 1000, seed: Optional[int] = None) -> dict:
    """Compare CPU vs alternative implementations on random samples from session.

    Returns a dict with counts and mismatch examples.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    hashes_np, target_np = get_arrays_from_session(session)
    N = hashes_np.shape[0]
    take = min(samples, N)
    if take == 0:
        raise RuntimeError("No hashes available to sample")

    idx = np.random.choice(N, size=take, replace=False)
    sample_hashes = hashes_np[idx]

    bigint_results = cpu_bigint_compare(sample_hashes, target_np)
    lex_results = cpu_lexicographic_numpy(sample_hashes, target_np)

    mism = np.where(bigint_results != lex_results)[0]

    result = {
        "total_sampled": int(take),
        "mismatches": int(mism.size),
        "mismatch_examples": [],
    }
    if mism.size > 0:
        for j in mism[:10]:
            i = idx[j]
            result["mismatch_examples"].append({
                "global_index": int(i),
                "hash_words": sample_hashes[j].tolist(),
                "target_words": target_np.tolist(),
                "bigint": bool(bigint_results[j]),
                "lex": bool(lex_results[j]),
            })
    return result


def main():
    p = argparse.ArgumentParser(description="Compare GPU kernel hashes vs CPU comparator")
    p.add_argument("--npz", help="Load arrays from .npz file created by torch/np.savez (keys: last_hashes, target)")
    p.add_argument("--attach", help="Import target session as MODULE:VAR (e.g. miner.miner_gpu:session)")
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.npz:
        d = np.load(args.npz)
        if "last_hashes" not in d or "target" not in d:
            p.error("npz must contain keys 'last_hashes' and 'target'")
        hashes = d["last_hashes"].astype(np.uint32)
        target = d["target"].astype(np.uint32)
        class DummySession:
            pass
        s = DummySession()
        s._last_hashes = torch.from_numpy(hashes.astype(np.int64))
        s._target_buf = torch.from_numpy(target.astype(np.int64))
        res = compare_session(s, samples=args.samples, seed=args.seed)
        print(res)
        return

    if args.attach:
        try:
            module_path, var = args.attach.split(":", 1)
        except Exception:
            p.error("--attach must be MODULE:VAR")
        m = importlib.import_module(module_path)
        session = getattr(m, var)
        res = compare_session(session, samples=args.samples, seed=args.seed)
        print(res)
        return

    p.print_help()


if __name__ == "__main__":
    main()
