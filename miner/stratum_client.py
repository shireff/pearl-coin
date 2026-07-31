import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

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


class StratumClient:
    def __init__(
        self,
        pool_url: str,
        username: str,
        password: str = "x",
        worker_name: str = "",
        on_job: Optional[Callable[[StratumJob], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.pool_url = pool_url
        self.username = username
        self.password = password
        self.worker_name = worker_name
        self.on_job = on_job
        self.on_error = on_error
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._subscribed = False
        self._authorized = False
        self._current_job: Optional[StratumJob] = None
        self._job_id_counter = 0
        self._pending_requests: dict[int, list[threading.Event, Any]] = {}
        self._lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._logger = get_logger(__name__)

    def connect(self) -> bool:
        try:
            if self.pool_url.startswith("stratum+tcp://"):
                pool_url = self.pool_url[len("stratum+tcp://") :]
            else:
                pool_url = self.pool_url

            if ":" in pool_url:
                host, port_str = pool_url.split(":")
                port = int(port_str)
            else:
                host = pool_url
                port = 3333

            self._sock = socket.create_connection((host, port), timeout=10)
            self._sock.settimeout(None)
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

    def submit_share(self, job_id: str, nonce: str, result: str) -> bool:
        if not self.is_connected():
            return False

        worker = self.worker_name if self.worker_name else self.username.split(".")[0]
        params = [job_id, nonce, result, worker]

        response = self._send_request("mining.submit", params)
        if response is None:
            return False

        return response.get("result", False) is True

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

        authorize_response = self._send_request(
            "mining.authorize", [self.username, self.password]
        )
        if authorize_response is None or not authorize_response.get("result"):
            self._logger.error(f"Failed to authorize with pool: {self.username}")
            return False

        self._authorized = True
        return True

    def _send_request(
        self, method: str, params: list, timeout: float = 10.0
    ) -> Optional[dict]:
        if not self._sock:
            return None

        self._job_id_counter += 1
        request_id = self._job_id_counter

        request = {"id": request_id, "method": method, "params": params}
        try:
            payload = json.dumps(request) + "\n"
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
            except Exception as e:
                if not self._stop_event.is_set():
                    self._logger.error(f"Error reading from pool: {e}")
                    self._connected = False
                break

        if self._connected and self.on_error:
            self.on_error("Connection lost")

    def _handle_message(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            self._logger.warning(f"Invalid JSON from pool: {raw[:100]}")
            return

        if "method" in message:
            self._logger.debug(f"Notification from pool: {message.get('method')} params={message.get('params')}")
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
        else:
            self._logger.debug(f"Unhandled notification: {method}")

    def _handle_mining_notify(self, params: list) -> None:
        try:
            self._logger.debug(f"mining.notify params: {params}")
            job_id = str(params[0]) if len(params) > 0 else ""
            blob_hex = str(params[1]) if len(params) > 1 else ""
            target_hex = str(params[2]) if len(params) > 2 else ""

            self._logger.debug(f"job_id={job_id} blob_len={len(blob_hex)} target_len={len(target_hex)}")

            if not blob_hex or not target_hex:
                self._logger.warning(f"Empty blob or target from pool: blob_hex='{blob_hex}' target_hex='{target_hex}'")
                return

            blob = bytes.fromhex(blob_hex)
            target = int(target_hex, 16)

            job = StratumJob(
                job_id=job_id,
                blob=blob,
                target=target,
                clean_jobs=True,
            )

            with self._lock:
                self._current_job = job

            self._logger.info(
                f"New job from pool: id={job_id} target={target:#x} blob_len={len(blob)}"
            )

            if self.on_job:
                self.on_job(job)
        except Exception as e:
            self._logger.error(f"Failed to parse mining.notify: {e}")

    def _handle_set_difficulty(self, params: list) -> None:
        try:
            if params and len(params) > 0:
                difficulty = float(params[0])
                with self._lock:
                    if self._current_job:
                        self._current_job.difficulty = difficulty
                self._logger.info(f"Pool set difficulty: {difficulty}")
        except Exception as e:
            self._logger.error(f"Failed to parse set_difficulty: {e}")

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
