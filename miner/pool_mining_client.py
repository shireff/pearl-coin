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

    def is_connected(self) -> bool:
        return self._connected and self._stratum is not None and self._stratum.is_connected()

    def get_mining_info(self) -> PoolMiningJob:
        with self._job_lock:
            if self._current_job is None:
                raise RuntimeError("No mining job available from pool")
            return self._current_job

    def submit_plain_proof(self, nonce: str, result_hex: str, job_id: str = "") -> bool:
        if not self.is_connected() or not self._stratum:
            self._logger.error("Cannot submit: not connected to pool")
            return False

        target_job_id = job_id or (self._current_job.job_id if self._current_job else "")
        if not target_job_id:
            self._logger.error("Cannot submit: no job ID")
            return False

        if self.algorithm == "alephium":
            success = self._stratum.submit_share(target_job_id, nonce)
        else:
            success = self._stratum.submit_share(target_job_id, nonce, result_hex)
        if success:
            self._logger.info(f"Share accepted: nonce={nonce[:16]}...")
        else:
            self._logger.warning(f"Share rejected: nonce={nonce[:16]}...")

        return success

    def get_worker_id(self) -> str:
        return self._worker_id

    def _on_pool_job(self, job: StratumJob) -> None:
        self._logger.debug(f"Received new job from pool: {job.job_id}")
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
        from_group: Optional[int] = None
        to_group: Optional[int] = None
        blob = bytes(stratum_job.blob)
        if len(blob) >= 280:
            # Best-effort extraction from the Alephium pool header prefix. The
            # group ids are not included in the Python-side share contract, but
            # the final kernel will use them to validate the group-index gate.
            prefix = blob[278:280]
            if len(prefix) >= 2:
                from_group = int(prefix[0]) & 0x0F
                to_group = int(prefix[1]) & 0x0F
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
            )
