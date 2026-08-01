#!/usr/bin/env python3
"""Standalone Alephium (ALPH) GPU miner for Kryptex pool."""

import queue
import sys
import threading
import time
from pathlib import Path

_local_root = Path(__file__).resolve().parent.parent / "miner"
if str(_local_root) not in sys.path:
    sys.path.insert(0, str(_local_root))

from miner_utils import get_logger
from pool_mining_client import PoolMiningClient, PoolMiningJob
from pearl_gemm import alph_mine_batch
import torch

POOL_URL = "stratum+tcp://alph.kryptex.network:7010"
USERNAME = "3cUrDWnDxgFVy1V8dEWjWR4npf5EcJKESoDNjL2Nmn4mssdW7n2Ui.worker"
PASSWORD = "x"
WORKER_NAME = "x"
ALGORITHM = "alephium"

NONCES_PER_ROUND = 1 << 22  # 4M nonces per kernel launch


def main():
    logger = get_logger("mine_alph")

    client = PoolMiningClient(
        pool_url=POOL_URL,
        username=USERNAME,
        password=PASSWORD,
        worker_name=WORKER_NAME,
        algorithm=ALGORITHM,
    )

    if not client.connect():
        logger.error("Failed to connect to pool")
        return 1

    logger.info("Connected to pool successfully")

    job_queue: queue.Queue = queue.Queue(maxsize=2)
    stop_prefetch = threading.Event()
    client_ref = [client]

    def prefetch_worker():
        while not stop_prefetch.is_set():
            try:
                job = client_ref[0].get_mining_info()
                try:
                    job_queue.put_nowait(job)
                except queue.Full:
                    try:
                        job_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        job_queue.put_nowait(job)
                    except queue.Full:
                        pass
                    time.sleep(0.02)
            except Exception as e:
                logger.warning(f"Prefetch error: {e}")
                time.sleep(1)

    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()
    logger.info("Job prefetch thread started")

    blob_gpu = torch.zeros(320, dtype=torch.uint8, device="cuda")
    target_gpu = torch.empty(32, dtype=torch.uint8, device="cuda")
    base_nonce = 0
    current_job = None
    round_count = 0

    logger.info("Starting mining loop...")

    try:
        while True:
            try:
                try:
                    latest = job_queue.get_nowait()
                    while True:
                        try:
                            latest = job_queue.get_nowait()
                        except queue.Empty:
                            break
                    current_job = latest
                except queue.Empty:
                    if current_job is None:
                        current_job = job_queue.get(timeout=5.0)

                blob_bytes = bytes(current_job.blob)
                blob_tensor = torch.frombuffer(blob_bytes, dtype=torch.uint8)
                blob_gpu[:302].copy_(blob_tensor)

                target_bytes = current_job.target.to_bytes(32, "big")
                target_gpu.copy_(torch.frombuffer(target_bytes, dtype=torch.uint8))

                t_start = time.perf_counter()
                found, nonce = alph_mine_batch(
                    blob_gpu, base_nonce, NONCES_PER_ROUND, target_gpu
                )
                elapsed = time.perf_counter() - t_start
                base_nonce += NONCES_PER_ROUND
                round_count += 1

                tmok_per_s = NONCES_PER_ROUND / elapsed / 1e6 if elapsed > 0 else 0.0
                logger.info(
                    f"round={round_count}  tmok/s={tmok_per_s:.0f}  "
                    f"found={found} nonce={nonce}  job={current_job.job_id}"
                )

                if found and nonce >= 0:
                    nonce_hex = nonce.to_bytes(8, "big").hex()
                    logger.info(f"SHARE FOUND: nonce={nonce_hex}")
                    success = client.submit_plain_proof(nonce_hex, "", current_job.job_id)
                    if success:
                        logger.info("Share ACCEPTED!")
                    else:
                        logger.warning("Share REJECTED")

            except queue.Empty:
                logger.warning("Job queue empty — waiting...")
                time.sleep(1)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.error(f"Mining round error: {type(exc).__name__}: {exc}")
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        stop_prefetch.set()
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
