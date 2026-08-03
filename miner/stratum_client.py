import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from miner_utils import get_logger

@dataclass
class StratumJob:
    job_id: str
    blob: bytes
    target: int
    algorithm: str = "prl"
    coin: str = "prl"
    height: int = 0
    clean_jobs: bool = False
    extranonce1: str = ""
    extranonce2_size: int = 4
    difficulty: float = 1.0
    block_target: int = 0  # original block target from targetBlob

class StratumClient:
    def __init__(
        self,
        pool_url: str,
        username: str,
        password: str = "x",
        worker_name: str = "",
        algorithm: str = "pearl",
        on_job: Optional[Callable[[StratumJob], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.pool_url = pool_url
        self.username = username
        self.password = password
        self.worker_name = worker_name
        self.algorithm = algorithm
        self.on_job = on_job
        self.on_error = on_error
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._subscribed = False
        self._authorized = False
        self._current_job: Optional[StratumJob] = None
        self._worker_id = ""
        self._extranonce1 = ""
        # Pool share target — set only via set_target or set_difficulty.
        # When None, targetBlob from mining.notify is used directly as share target.
        # Kryptex Alephium sends the actual pool share target in targetBlob.
        # Default difficulty=512 (Kryptex Alephium default) — updated by set_difficulty
        self._share_target: Optional[int] = None
        self._difficulty_received = threading.Event()
        self._job_id_counter = 0
        self._pending_requests: dict[int, tuple[threading.Event, list]] = {}
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._logger = get_logger(__name__)

    def connect(self) -> bool:
        try:
            if self.pool_url.startswith("stratum+tcp://"):
                pool_url = self.pool_url[len("stratum+tcp://"):]
            else:
                pool_url = self.pool_url

            if ":" in pool_url:
                host, port_str = pool_url.split(":")
                port = int(port_str)
            else:
                host = pool_url
                port = 3333

            self._sock = socket.create_connection((host, port), timeout=10)
            self._sock.settimeout(0.1)
            self._connected = True
            self._stop_event.clear()

            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()

            return self._subscribe_and_authorize()
        except Exception as e:
            self._logger.error(f"Failed to connect to pool: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        self._stop_event.set()
        self._connected = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)

    def is_connected(self) -> bool:
        return self._connected and self._subscribed and self._authorized

    def get_current_job(self) -> Optional[StratumJob]:
        with self._lock:
            return self._current_job

    def submit_share(self, job_id: str, nonce: str, result: str = "") -> bool:
        if not self.is_connected():
            return False

        if self.algorithm == "alephium":
            # Alephium stratum: [jobId, nonceSansExtraNonce]
            # No worker_id in params — pool identifies worker from connection.
            params = [job_id, nonce]
            sent = self._send_fire_and_forget("mining.submit", params)
            if sent:
                self._logger.info(f"Pool submit sent (fire-and-forget): job={job_id} nonce={nonce}...")
            return sent
        else:
            worker = self.worker_name if self.worker_name else self.username.split(".")[0]
            params = [job_id, nonce, result, worker]

        response = self._send_request("mining.submit", params, timeout=30.0)
        if response is None:
            return False

        result_val = response.get("result", False)
        error = response.get("error")
        if error:
            self._logger.warning(f"Pool submit error: {error}")
        else:
            self._logger.info(f"Pool submit response: result={result_val}")
        return result_val is True

    def _subscribe_and_authorize(self) -> bool:
        subscribe_response = self._send_request("mining.subscribe", [])
        if subscribe_response is None or not subscribe_response.get("result"):
            self._logger.error("Failed to subscribe to pool")
            return False

        self._subscribed = True
        result = subscribe_response["result"]
        if isinstance(result, list) and len(result) > 1:
            extranonce = result[1]
            if isinstance(extranonce, list) and len(extranonce) > 0:
                self._extranonce1 = str(extranonce[0])
            else:
                self._extranonce1 = str(extranonce)
        elif isinstance(result, str) and result:
            # Kryptex Alephium: subscribe result is the session/worker id (e.g. "00000000")
            # Store it so submit_share sends [job_id, nonce, worker_id] not [job_id, nonce]
            self._worker_id = str(result).strip()
            self._logger.info(f"Pool subscribe returned worker_id: {self._worker_id}")

        authorize_response = self._send_request(
            "mining.authorize", [self.username, self.password]
        )
        if authorize_response is None or not authorize_response.get("result"):
            self._logger.error(f"Failed to authorize with pool: {self.username}")
            return False

        auth_result = authorize_response.get("result")
        if isinstance(auth_result, str) and auth_result:
            # Pool returned a string worker_id from authorize — use it
            self._worker_id = str(auth_result).strip()
        elif isinstance(auth_result, bool):
            # Some Kryptex Alephium pools respond with a bare true here.
            # Preserve any worker/session id already discovered from subscribe.
            self._worker_id = self._worker_id or ""
        elif not self._worker_id:
            # No worker_id from subscribe or authorize — leave empty
            self._worker_id = ""
        # else: keep existing _worker_id set from subscribe result
        self._authorized = True
        return True

    def _send_fire_and_forget(self, method: str, params: list) -> bool:
        if not self._sock:
            return False
        self._job_id_counter += 1
        request_id = self._job_id_counter
        request = {"id": request_id, "method": method, "params": params}
        try:
            payload = json.dumps(request) + "\n"
            self._logger.debug(f"[STRATUM SEND] {payload.strip()}")
            self._sock.sendall(payload.encode("utf-8"))
            return True
        except Exception as e:
            self._logger.error(f"Failed to send {method}: {e}")
            return False

    def _send_request(self, method: str, params: list, timeout: float = 10.0) -> Optional[dict]:
        if not self._sock:
            return None

        self._job_id_counter += 1
        request_id = self._job_id_counter

        request = {"id": request_id, "method": method, "params": params}
        try:
            payload = json.dumps(request) + "\n"
            self._logger.debug(f"[STRATUM SEND] {payload.strip()}")
            self._sock.sendall(payload.encode("utf-8"))
        except Exception as e:
            self._logger.error(f"Failed to send request {method}: {e}")
            return None

        response_event = threading.Event()
        response_data: list = [None]

        with self._lock:
            self._pending_requests[request_id] = (response_event, response_data)

        if not response_event.wait(timeout):
            with self._lock:
                self._pending_requests.pop(request_id, None)
            self._logger.error(f"Timeout waiting for response to {method}")
            return None

        return response_data[0]

    def _read_loop(self) -> None:
        import socket as _socket
        buffer = ""
        while not self._stop_event.is_set() and self._connected:
            try:
                data = self._sock.recv(4096)
                if not data:
                    self._logger.warning("Pool connection closed by server")
                    self._connected = False
                    break

                buffer += data.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._handle_message(line)
            except _socket.timeout:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    self._logger.error(f"Error reading from pool: {e}")
                    self._connected = False
                break

        if self._connected and self.on_error:
            self.on_error("Connection lost")

    def _handle_message(self, raw: str) -> None:
        self._logger.debug(f"[STRATUM RECV] {raw}")
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            self._logger.warning(f"Invalid JSON from pool: {raw[:100]}")
            return

        if "method" in message:
            self._handle_notification(message)
        elif "id" in message:
            self._handle_response(message)
        else:
            self._logger.debug(f"Unknown message from pool: {message}")

    def _handle_notification(self, message: dict) -> None:
        method = message.get("method", "")
        params = message.get("params", [])

        if method == "mining.notify":
            self._handle_mining_notify(params)
        elif method == "mining.set_difficulty":
            self._handle_set_difficulty(params)
        elif method == "mining.set_extranonce":
            self._handle_set_extranonce(params)
        elif method == "mining.set_target":
            self._handle_set_target(params)
        else:
            self._logger.debug(f"Unhandled notification: {method}")

    def _handle_mining_notify(self, params: list) -> None:
        try:
            if len(params) == 0:
                self._logger.warning("mining.notify with empty params")
                return

            param0 = params[0]

            if isinstance(param0, dict):
                job_id = str(param0.get("jobId", ""))
                header_blob_hex = str(param0.get("headerBlob", ""))
                target_blob_hex = str(param0.get("targetBlob", ""))
                height = param0.get("height", 0)
            elif isinstance(param0, str) and len(params) >= 3:
                job_id = str(param0)
                header_blob_hex = str(params[1])
                target_blob_hex = str(params[2])
                height = 0
            else:
                self._logger.warning(f"Unexpected mining.notify format: {type(param0)}")
                return

            if not header_blob_hex or not target_blob_hex:
                self._logger.warning("Empty header/target from pool")
                return

            blob = bytes.fromhex(header_blob_hex)
            block_target = int(target_blob_hex, 16)

            # Use _share_target if set via set_target/set_difficulty,
            # otherwise use targetBlob directly — Kryptex embeds the pool
            # share target in targetBlob, so this is the correct default.
            with self._lock:
                share_target = self._share_target
            target = share_target if share_target is not None else block_target

            job = StratumJob(
                job_id=job_id,
                blob=blob,
                target=target,
                height=height,
                clean_jobs=True,
                block_target=block_target,
            )

            with self._lock:
                self._current_job = job

            self._logger.info(
                f"New job: id={job_id} target={target:#x} blob_len={len(blob)} height={height}"
            )

            if self.on_job:
                _t = threading.Thread(target=self.on_job, args=(job,), daemon=True)
                _t.start()
        except Exception as e:
            self._logger.error(f"Failed to parse mining.notify: {e}")

    def _handle_set_difficulty(self, params: list) -> None:
        try:
            if params and len(params) > 0:
                difficulty = float(params[0])
                if difficulty > 0:
                    # Verified: share_target = (2^256-1) / difficulty
                    # difficulty=512 → 007fffff... → 197 valid per 100k ✓
                    share_target = int((2**256 - 1) // difficulty)
                else:
                    share_target = 2**256 - 1
                with self._lock:
                    self._share_target = share_target
                    if self._current_job:
                        self._current_job.target = share_target
                        self._current_job.difficulty = difficulty
                self._logger.info(
                    f"Pool set difficulty: {difficulty} → share_target={share_target:#x}"
                )
                self._difficulty_received.set()
        except Exception as e:
            self._logger.error(f"Failed to parse set_difficulty: {e}")

    def _handle_set_extranonce(self, params) -> None:
        try:
            if isinstance(params, list) and len(params) > 0:
                extranonce = str(params[0])
            else:
                extranonce = str(params)
            self._extranonce1 = extranonce
            self._logger.info(f"Pool set extranonce: {extranonce}")
        except Exception as e:
            self._logger.error(f"Failed to parse set_extranonce: {e}")

    def _handle_set_target(self, params) -> None:
        try:
            target_hex = str(params[0]) if isinstance(params, list) and len(params) > 0 else str(params)
            target_val = int(target_hex, 16)
            with self._lock:
                self._share_target = target_val
                if self._current_job:
                    self._current_job.target = target_val
            self._logger.info(f"Pool set target: {target_hex} ({target_val:#x})")
        except Exception as e:
            self._logger.error(f"Failed to parse set_target: {e}")

    def _handle_response(self, message: dict) -> None:
        request_id = message.get("id")
        if request_id is None:
            return

        with self._lock:
            pending = self._pending_requests.pop(request_id, None)

        if pending:
            event, data_list = pending
            data_list[0] = message
            event.set()
        else:
            self._logger.info(f"[Pool response] id={request_id} {message}")
