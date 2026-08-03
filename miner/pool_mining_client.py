import hashlib
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from miner_utils import get_logger
from stratum_client import StratumClient, StratumJob


@dataclass
class PoolMiningJob:
    """Mining job from a Stratum pool, compatible with MiningJob interface."""

    incomplete_header_bytes: bytes
    target: int
    job_id: str = ""
    blob: bytes = field(default=b"")
    difficulty: float = 1.0
    height: int = 0
    extranonce: str = ""  # pool extranonce from mining.set_extranonce
    from_group: Optional[int] = None
    to_group: Optional[int] = None
    block_target: int = 0  # targetBlob from pool — pool validates hash <= this

    def to_dict(self) -> dict[str, Any]:
        return {
            "incomplete_header_bytes": self.incomplete_header_bytes.hex(),
            "target": self.target,
            "job_id": self.job_id,
        }


class PoolMiningClient:
    """Mining client that connects to a Stratum pool instead of local gateway."""

    def __init__(
        self,
        pool_url: str,
        username: str,
        password: str = "x",
        worker_name: str = "",
        device_name: str = "",
        algorithm: str = "pearl",
    ):
        self.pool_url = pool_url
        self.username = username
        self.password = password
        self.worker_name = worker_name
        self.device_name = device_name
        self.algorithm = algorithm
        self._stratum: Optional[StratumClient] = None
        self._current_job: Optional[PoolMiningJob] = None
        self._job_map: dict[tuple[int, int], PoolMiningJob] = {}  # (from_group, to_group) → job
        self._job_lock = threading.Lock()
        self._logger = get_logger(__name__)
        self._connected = False
        self._worker_id = ""

    def connect(self) -> bool:
        if self.worker_name and "." not in self.username:
            worker = f"{self.username}.{self.worker_name}"
        else:
            worker = self.username
        self._stratum = StratumClient(
            pool_url=self.pool_url,
            username=worker,
            password=self.password,
            worker_name=self.worker_name or self.username.split(".")[0],
            algorithm=self.algorithm,
            on_job=self._on_pool_job,
            on_error=self._on_pool_error,
        )

        if not self._stratum.connect():
            self._logger.error("Failed to connect to Stratum pool")
            return False

        self._connected = True
        self._logger.info(f"Connected to pool: {self.pool_url}")

        job = self._stratum.get_current_job()
        if job:
            self._convert_job(job)

        return True

    def disconnect(self) -> None:
        self._connected = False
        if self._stratum:
            self._stratum.disconnect()
            self._stratum = None
        with self._job_lock:
            self._current_job = None
            self._job_map.clear()

    def is_connected(self) -> bool:
        return self._connected and self._stratum is not None and self._stratum.is_connected()

    def get_mining_info(self) -> PoolMiningJob:
        with self._job_lock:
            if self._current_job is None:
                raise RuntimeError("No mining job available from pool")
            return self._current_job

    def get_job_for_chain(self, from_group: int, to_group: int) -> Optional[PoolMiningJob]:
        """Return the most recent job for a specific chain combination, or None."""
        with self._job_lock:
            return self._job_map.get((from_group, to_group))

    def submit_plain_proof(self, nonce: str, result_hex: str, job_id: str = "") -> bool:
        if not self.is_connected() or not self._stratum:
            self._logger.error(
                "[SUBMIT FAIL] Cannot submit: not connected to pool  "
                f"connected={self._connected}  stratum={self._stratum is not None}"
            )
            return False

        target_job_id = job_id or (self._current_job.job_id if self._current_job else "")
        if not target_job_id:
            self._logger.error(
                f"[SUBMIT FAIL] Cannot submit: no job ID  "
                f"job_id_arg={job_id!r}  current_job={self._current_job is not None}"
            )
            return False

        worker_id = self._stratum.get_worker_id() if self._stratum else "<no stratum>"
        nonce_preview = nonce[:32] if nonce else "<empty>"

        if not nonce:
            self._logger.error(
                f"[SUBMIT FAIL] nonce is empty  job_id={target_job_id}  worker_id={worker_id}"
            )
            return False

        self._logger.info(
            f"[SUBMIT] algorithm={self.algorithm}  job_id={target_job_id}  "
            f"nonce={nonce_preview}  nonce_full={nonce}  nonce_len={len(nonce)//2}B  "
            f"result_hex={result_hex[:16] if result_hex else '<empty>'}  worker_id={worker_id}"
        )

        if self.algorithm == "alephium":
            success = self._stratum.submit_share(target_job_id, nonce)
        else:
            success = self._stratum.submit_share(target_job_id, nonce, result_hex)

        if success:
            self._logger.info(
                f"[SUBMIT OK] Share sent to pool  job_id={target_job_id}  nonce={nonce_preview}"
            )
        else:
            self._logger.warning(
                f"[SUBMIT SKIP/FAIL] job_id={target_job_id}  nonce={nonce_preview}  "
                f"(rate-limited or send error)"
            )

        return success

    def get_worker_id(self) -> str:
        return self._worker_id

    def _on_pool_job(self, job: StratumJob) -> None:
        self._logger.info(
            f"[JOB] New job from pool: id={job.job_id}  target={job.target:#x}  "
            f"block_target={job.block_target:#x}  height={job.height}  "
            f"blob_len={len(job.blob)}  difficulty={job.difficulty}"
        )
        self._convert_job(job)

    def _on_pool_error(self, error: str) -> None:
        self._logger.error(f"Pool error: {error}")
        self._connected = False
        # Auto-reconnect in background thread
        import threading as _threading
        _threading.Thread(target=self._reconnect_loop, daemon=True).start()

    def _reconnect_loop(self) -> None:
        import time as _time
        attempt = 0
        while not self._connected:
            attempt += 1
            wait = min(5 * attempt, 30)
            self._logger.info(f"Reconnecting in {wait}s (attempt {attempt})...")
            _time.sleep(wait)
            try:
                if self._stratum:
                    self._stratum.disconnect()
                if self.connect():
                    self._logger.info("Reconnected to pool successfully")
                    return
            except Exception as e:
                self._logger.error(f"Reconnect attempt {attempt} failed: {e}")

    def close(self) -> None:
        """Close the pool connection cleanly."""
        self.disconnect()

    def _convert_job(self, stratum_job: StratumJob) -> None:
        # Get current extranonce from stratum client
        extranonce = getattr(self._stratum, "_extranonce1", "") or ""
        # Use from_group/to_group injected directly by _handle_mining_notify
        from_group: Optional[int] = getattr(stratum_job, "_from_group", None)
        to_group: Optional[int] = getattr(stratum_job, "_to_group", None)
        blob = bytes(stratum_job.blob)
        self._logger.info(
            f"[CONVERT JOB] job_id={stratum_job.job_id}  "
            f"target={stratum_job.target:#x}  blob_len={len(blob)}  "
            f"from_group={from_group}  to_group={to_group}  extranonce={extranonce!r}"
        )
        with self._job_lock:
            self._current_job = PoolMiningJob(
                incomplete_header_bytes=blob,
                target=stratum_job.target,
                job_id=stratum_job.job_id,
                blob=blob,
                difficulty=stratum_job.difficulty,
                height=getattr(stratum_job, "height", 0) or 0,
                extranonce=extranonce,
                from_group=from_group,
                to_group=to_group,
                block_target=stratum_job.block_target,
            )
            # Keep per-chain job map so we can find the right job for any winning hash
            if from_group is not None and to_group is not None:
                self._job_map[(from_group, to_group)] = self._current_job
