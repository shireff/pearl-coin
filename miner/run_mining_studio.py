import os
import subprocess
import sys
import threading
import time
import traceback

CLONE_DIR = os.path.expanduser("~/pearl-cion")
DATA_DIR = os.path.expanduser("~/.pearl-cion")
MODEL_NAME = "pearl-ai/Llama-3.2-1B-Instruct-pearl"

TILE_SIZE_M = 256
TILE_SIZE_N = 512
TILE_SIZE_K = 256
NOISE_RANK = 256
IDXS_PER_COL = 4
COLS_PATTERN_SIZE = 64
PINNED_POOL_SIZE = 256
REFRESH_RATE = 0.5


def _stream_logs(name: str, proc):
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(f"[{name}] {line}", end="", flush=True)


def _print_startup_banner():
    print("=" * 70)
    print("PEARL MINER — Production Profile (Lightning AI Studio)")
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
    print(f"  SMs: {props.multi_processor_count}  |  Clock: {props.clock_rate // 1000} MHz")
    print(f"  Allocated: {mem.get('allocated_bytes.all.current', 0) / (1024 * 1024):.1f} MB  |  "
          f"Reserved: {mem.get('reserved_bytes.all.current', 0) / (1024 * 1024):.1f} MB")
    print(f"  Active Allocations: {mem.get('allocation.all.current', 'N/A')}  |  "
          f"Active Reservations: {mem.get('reservation.all.current', 'N/A')}")
    print("")


def run_mining():
    os.chdir(CLONE_DIR)
    os.environ["PEARLD_RPC_URL"] = "http://localhost:44107"
    os.environ["PEARLD_RPC_USER"] = "rpcuser"
    os.environ["PEARLD_RPC_PASSWORD"] = "rpcpass"
    os.environ["PEARLD_MINING_ADDRESS"] = "prl1qtestaddress1234567890abcdefghijklmnopqrstuvwxyz"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["PEARL_LOG_LEVEL"] = "INFO"

    pearld_proc = gateway_proc = vllm_proc = None

    try:
        _print_startup_banner()
        _print_gpu_snapshot()

        # Start pearld
        print("\n[1/4] Starting pearld...")
        pearld_proc = subprocess.Popen(
            [f"{CLONE_DIR}/bin/pearld",
             "--rpcuser=rpcuser", "--rpcpass=rpcpass",
             "--rpclisten=0.0.0.0:44107",
             "--txindex",
             f"--datadir={DATA_DIR}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        threading.Thread(target=_stream_logs, args=("pearld", pearld_proc), daemon=True).start()
        time.sleep(3)

        # Start pearl-gateway using the current Python interpreter
        print("[2/4] Starting pearl-gateway...")
        gateway_proc = subprocess.Popen(
            [sys.executable, "-m", "pearl_gateway"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        threading.Thread(target=_stream_logs, args=("gateway", gateway_proc), daemon=True).start()

        # Wait for gateway socket with polling
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

        # Start vllm serve using the current Python interpreter
        print("[3/4] Starting vllm serve with Pearl mining plugin...")
        vllm_proc = subprocess.Popen(
            [sys.executable, "-m", "vllm", "serve", MODEL_NAME,
             "--host", "0.0.0.0", "--port", "8000",
             "--max-model-len", "2048",
             "--gpu-memory-utilization", "0.9",
             "--enforce-eager"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        threading.Thread(target=_stream_logs, args=("vllm", vllm_proc), daemon=True).start()

        print("[4/4] Monitoring logs (300s)...")
        print("=" * 70)

        _iteration_count = 0
        _start_time = time.time()

        for _ in range(300):
            time.sleep(1)
            _iteration_count += 1

            if pearld_proc.poll() is not None:
                print("[ERROR] pearld stopped!")
                break
            if gateway_proc is not None and gateway_proc.poll() is not None:
                print("[WARN] gateway stopped, continuing without it")
                gateway_proc = None
            if vllm_proc is not None and vllm_proc.poll() is not None:
                print("[WARN] vllm stopped, continuing without it")
                vllm_proc = None

            if _iteration_count == 1:
                _print_gpu_snapshot()

            elapsed = time.time() - _start_time
            if _iteration_count % 10 == 0 and elapsed > 0:
                nonces_per_iter = (TILE_SIZE_M * TILE_SIZE_N) // NOISE_RANK
                total_nonces = nonces_per_iter * _iteration_count
                tmok_per_s = total_nonces / elapsed / 1000.0
                print(f"[stats] iter={_iteration_count} elapsed={elapsed:.0f}s "
                      f"tmok/s={tmok_per_s:.2f}")

    except Exception:
        print("\n[FATAL] run_mining crashed:")
        traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nStopping all processes...")
    finally:
        for proc, name in [(pearld_proc, "pearld"), (gateway_proc, "gateway"), (vllm_proc, "vllm")]:
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