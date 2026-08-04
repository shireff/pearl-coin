#!/usr/bin/env python3
"""
Estimate probability and expected time to find the first share.
Usage examples:
  python miner/tools/share_probability.py --rate 700000 --target 00000000000007ffffffffffffffffffffffffffffffffffffffffffffffffff
  python miner/tools/share_probability.py --rate 700000 --difficulty 1e24
"""
import argparse
import math
from typing import Optional

TWO256 = 2 ** 256


def parse_target(target_hex: str) -> int:
    # Strip optional 0x and whitespace
    s = target_hex.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) == 0:
        raise ValueError("empty target")
    return int(s, 16)


def format_seconds(s: float) -> str:
    if s == float("inf"):
        return "infinite"
    s = float(s)
    if s < 1:
        return f"{s:.3f} s"
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    parts = []
    if d >= 1:
        parts.append(f"{int(d)}d")
    if h >= 1:
        parts.append(f"{int(h)}h")
    if m >= 1:
        parts.append(f"{int(m)}m")
    parts.append(f"{sec:.1f}s")
    return " ".join(parts)


def main():
    p = argparse.ArgumentParser(description="Share-first-arrival probability estimator")
    p.add_argument("--rate", type=float, required=True, help="attempts per second (combos/sec)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--target", type=str, help="hex target (256-bit hex) e.g. 0000... or ffff...)")
    grp.add_argument("--difficulty", type=float, help="difficulty where difficulty = 2^256 / target")
    p.add_argument("--show", type=float, default=0.5, help="probability to compute time for (default 0.5 = median)")
    args = p.parse_args()

    rate = float(args.rate)
    if rate <= 0:
        raise SystemExit("--rate must be positive")

    if args.target:
        target_int = parse_target(args.target)
        prob_per_attempt = target_int / TWO256
    else:
        # difficulty -> probability per attempt = 1 / difficulty (since difficulty = 2^256 / target)
        prob_per_attempt = 1.0 / float(args.difficulty)

    if prob_per_attempt <= 0:
        raise SystemExit("invalid target/difficulty yielding zero probability")

    lambda_rate = rate * prob_per_attempt

    expected_time_s = float("inf") if lambda_rate == 0 else 1.0 / lambda_rate
    median_time_s = float("inf") if lambda_rate == 0 else math.log(2) / lambda_rate

    def prob_in_seconds(t: float) -> float:
        if lambda_rate == 0:
            return 0.0
        return 1.0 - math.exp(-lambda_rate * t)

    def time_for_prob(q: float) -> float:
        if q <= 0:
            return 0.0
        if q >= 1:
            return float("inf")
        return -math.log(1 - q) / lambda_rate

    print(f"Rate (attempts/sec): {rate:,.0f}")
    print(f"Probability per attempt: {prob_per_attempt:.3e}")
    print(f"Lambda (rate * p): {lambda_rate:.3e}  (expected shares/sec)")
    print(f"Expected time to first share (mean): {format_seconds(expected_time_s)}")
    print(f"Median time to first share: {format_seconds(median_time_s)}")
    print("")

    # common intervals
    for label, secs in (
        ("1 minute", 60),
        ("1 hour", 3600),
        ("1 day", 86400),
        ("7 days", 86400 * 7),
        ("30 days", 86400 * 30),
        ("1 year", 86400 * 365),
    ):
        pr = prob_in_seconds(secs)
        print(f"P(>=1 share in {label}): {pr*100:.6f}%")

    print("")
    q = float(args.show)
    t_for_q = time_for_prob(q)
    print(f"Time to reach probability {q*100:.2f}%: {format_seconds(t_for_q)}")


if __name__ == "__main__":
    main()
