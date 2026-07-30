import os
import subprocess
import sys
import threading
import time
import traceback

import pearl_mining

CLONE_DIR = "/root/pearl-cion"
MODEL_NAME = "pearl-ai/Llama-3.2-1B-Instruct-pearl"

TILE_SIZE_M = 256
TILE_SIZE_N = 512
TILE_SIZE_K = 256
NOISE_RANK = 256
IDXS_PER_COL = 4
COLS_PATTERN_SIZE = 64
PINNED_POOL_SIZE = 256
REFRESH_RATE = 0.5

PEARL_GEMM_SO_SRC = "/root/pearl-cion/miner/pearl-gemm/src"

GATEWAY_SOCKET = "/tmp/pearlgw.sock"


def _install_pearl_gemm():
    import glob
    import shutil

    venv_sp = f"{CLONE_DIR}/.venv/lib/python3.12/site-packages"
    pattern = os.path.join(PEARL_GEMM_SO_SRC, "pearl_gemm_cuda*.so")
    matches = glob.glob(pattern)

    if not matches:
        print(f"[WARN] pearl_gemm_cuda.so not found at {PEARL_GEMM_SO_SRC}")
        print("[WARN] Run: cd /root/pearl-cion && bash miner/pearl-gemm/build.sh --no-pull")
        return

    for src in matches:
        dst = os.path.join(venv_sp, os.path.basename(src))
        if os.path.exists(dst):
            print(f"[OK] pearl_gemm_cuda already in venv: {os.path.basename(src)}")
            continue
        try:
            shutil.copy2(src, dst)
            print(f"[OK] Installed pearl_gemm_cuda: {os.path.basename(src)}")
        except Exception as e:
            print(f"[WARN] Could not install {os.path.basename(src)}: {e}")


def _stream_logs(name: str, proc):
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"[{name}] {line}", end="", flush=True)


def _print_startup_banner():
    print("=" * 70)
    print("PEARL MINER — Production Profile (Lightning AI)")
    print("=" * 70)
    print(f"  tile_size_m  = {TILE_SIZE_M}")
    print(f"  tile_size_n  = {TILE_SIZE_N}")
    print(f"  tile_size_k  = {TILE_SIZE_K}")
    print(f"  noise_rank   = {NOISE_RANK}")
    print(f"  idxs_per_col = {IDXS_PER_COL}")
    print(f"  cols_pattern = {COLS_PATTERN_SIZE} entries")
    print(f"  pinned_pool  = {PINNED_POOL_SIZE}")
    print(f"  refresh_rate = {REFRESH_RATE} s")
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
        else:
            gpu_name = "unknown"
    except ImportError:
        gpu_name = "unknown"
    print(f"  gpu          = {gpu_name}")
    print(f"  cuda         = 12.6")
    print("=" * 70)


def _print_gpu_snapshot():
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    mem = torch.cuda.memory_stats(device)
    print(f"GPU Snapshot: {props.name} ({props.total_memory // (1024 * 1024)} MB)")
    clock_mhz = getattr(props, 'clock_rate', 0) // 1000 or getattr(props, 'max_clock_rate', 0) // 1000
    print(f"  SMs: {props.multi_processor_count}  |  Clock: {clock_mhz} MHz")
    print(f"  Allocated: {mem.get('allocated_bytes.all.current', 0) / (1024 * 1024):.1f} MB  |  "
          f"Reserved: {mem.get('reserved_bytes.all.current', 0) / (1024 * 1024):.1f} MB")
    print(f"  Active Allocations: {mem.get('allocation.all.current', 'N/A')}  |  "
          f"Active Reservations: {mem.get('reservation.all.current', 'N/A')}")
    print("")


def _mining_loop():
    import json
    import socket

    from pearl_gateway.comm.dataclasses import MiningJob

    rows_pattern = [0, 8, 64, 72]
    cols_pattern = [0, 1, 8, 9, 32, 33, 40, 41]

    mining_config = pearl_mining.MiningConfiguration(
        common_dim=TILE_SIZE_K,
        rank=NOISE_RANK,
        mma_type=pearl_mining.MMAType.Int7xInt7ToInt32,
        rows_pattern=pearl_mining.PeriodicPattern.from_list(rows_pattern),
        cols_pattern=pearl_mining.PeriodicPattern.from_list(cols_pattern),
    )

    socket_path = GATEWAY_SOCKET
    max_retries = 300
    retry_count = 0

    while retry_count < max_retries:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(socket_path)
            break
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            retry_count += 1
            time.sleep(1)
    else:
        print("[WARN] Could not connect to gateway miner RPC, skipping mining loop")
        return

    reader = sock.makefile("r", encoding="utf-8")
    writer = sock.makefile("w", encoding="utf-8")
    request_id = 0

    def rpc_call(method, params=None):
        nonlocal request_id
        request_id += 1
        request = {"jsonrpc": "2.0", "method": method, "id": request_id}
        request["params"] = params if params is not None else {}
        writer.write(json.dumps(request) + "\n")
        writer.flush()
        line = reader.readline()
        if not line:
            raise ConnectionError("Connection closed by gateway")
        response = json.loads(line.strip())
        if "error" in response:
            raise Exception(f"JSON-RPC error: {response['error']}")
        return response.get("result")

    print("[MINING] Direct mining loop started (no vLLM)")
    _start_time = time.time()
    _iterations = 0

    try:
        while True:
            try:
                job_dict = rpc_call("getMiningInfo")
                mining_job = MiningJob(
                    incomplete_header_bytes=bytes.fromhex(job_dict["incomplete_header_bytes"])
                    if job_dict.get("incomplete_header_bytes", "").startswith("0x")
                    else bytes.fromhex(job_dict["incomplete_header_bytes"]),
                    target=job_dict["target"],
                    cert_version=pearl_mining.CertificateVersion(job_dict.get("cert_version", 2)),
                )
            except Exception as e:
                print(f"[WARN] Failed to get mining job: {e}")
                time.sleep(1)
                continue

            header = pearl_mining.IncompleteBlockHeader.from_bytes(
                mining_job.incomplete_header_bytes
            )

            proof = pearl_mining.mine(
                TILE_SIZE_M,
                TILE_SIZE_N,
                TILE_SIZE_K,
                header,
                mining_config,
            )

            proof_b64 = proof.to_base64()

            try:
                rpc_call("submitPlainProof", {
                    "plain_proof": proof_b64,
                    "mining_job": {
                        "incomplete_header_bytes": job_dict["incomplete_header_bytes"],
                        "target": job_dict["target"],
                    },
                })
                _iterations += 1
                elapsed = time.time() - _start_time
                if _iterations % 10 == 0:
                    print(f"[MINING] iter={_iterations} elapsed={elapsed:.0f}s proofs_submitted={_iterations}")
            except Exception as e:
                print(f"[WARN] Failed to submit proof: {e}")

    except KeyboardInterrupt:
        print("\n[MINING] Mining loop stopped by user")
    except Exception as e:
        print(f"[ERROR] Mining loop error: {e}")
    finally:
        sock.close()


def run_mining():
    os.chdir(CLONE_DIR)
    _install_pearl_gemm()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        print(f"[OK] HuggingFace token set ({len(hf_token)} chars)")
    else:
        print("[INFO] No HF_TOKEN set — running in direct mining mode (no vLLM)")

    os.environ["PEARLD_RPC_URL"] = "http://localhost:44107"
    os.environ["PEARLD_RPC_USER"] = "rpcuser"
    os.environ["PEARLD_RPC_PASSWORD"] = "rpcpass"
    os.environ["PEARLD_MINING_ADDRESS"] = "prl1qtestaddress1234567890abcdefghijklmnopqrstuvwxyz"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["PEARL_LOG_LEVEL"] = "INFO"

    pearld_proc = gateway_proc = None

    try:
        _print_startup_banner()
        _print_gpu_snapshot()

        print("\n[1/3] Starting pearld...")
        pearld_proc = subprocess.Popen(
            [f"{CLONE_DIR}/bin/pearld",
             "--rpcuser=rpcuser", "--rpcpass=rpcpass",
             "--rpclisten=0.0.0.0:44107",
             "--txindex",
             "--datadir=/root/.pearl"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        threading.Thread(target=_stream_logs, args=("pearld", pearld_proc), daemon=True).start()
        time.sleep(3)

        print("[2/3] Starting pearl-gateway...")
        gateway_proc = subprocess.Popen(
            [sys.executable, "-m", "pearl_gateway"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        threading.Thread(target=_stream_logs, args=("gateway", gateway_proc), daemon=True).start()

        socket_path = "/tmp/pearlgw.sock"
        for _ in range(60):
            if os.path.exists(socket_path):
                print(f"[OK] Gateway socket ready at {socket_path}")
                break
            if gateway_proc is not None and gateway_proc.poll() is not None:
                print("[WARN] Gateway process died, continuing without it")
                gateway_proc = None
                break
            time.sleep(1)
        else:
            print("[WARN] Gateway socket never appeared, continuing without it")
            if gateway_proc is not None:
                gateway_proc.terminate()
                gateway_proc = None

        print("[3/3] Starting direct mining loop (no vLLM)...")
        _mining_loop()

        print("[DONE] Mining loop finished")

    except Exception:
        print("\n[FATAL] run_mining crashed:")
        traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nStopping all processes...")
    finally:
        for proc, name in [(pearld_proc, "pearld"), (gateway_proc, "gateway")]:
            if proc is None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("\nAll processes stopped.")


if __name__ == "__main__":
    run_mining()