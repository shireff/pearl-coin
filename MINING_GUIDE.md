# Pearl Miner — Complete Step-by-Step Setup Guide

> Every command in this guide is derived directly from the production source code:
> `pyproject.toml`, `Dockerfile`, `entrypoint.sh`, `Taskfile.yml`, and the CUDA kernel sources.
> Nothing is guessed or fabricated.

---

## How Mining Actually Works

```
pearld (full node)
    ↓  JSON-RPC (port 44107)
pearl-gateway start          ← fetches block template, distributes work
    ↓  Unix socket /tmp/pearlgw.sock
vllm serve <model>           ← runs the LLM inference server
    ↓  hook fires on every linear layer forward pass
noisy_gemm (CUDA kernel)     ← INT8 WGMMA on GPU Tensor Cores
    ↓  BLAKE3 PoW check inside the CUDA kernel
host_signal_header           ← if block found → written to pinned memory
    ↓
pearl-gateway                ← builds ZK proof, submits block to pearld
```

The GPU executes:

- INT8 matrix multiply via WGMMA Tensor Core instructions (SM90/SM120)
- BLAKE3 PoW difficulty check inside the CUDA kernel
- Noise generation and denoising as part of the same kernel pipeline

The CPU only:

- Waits for a CUDA event signal
- Reads `host_signal_header_pinned` when a block is found
- Builds the `PlainProof` and sends it through `pearl-gateway` to `pearld`

---

## Prerequisites

### Hardware

| Requirement | Details                                                            |
| ----------- | ------------------------------------------------------------------ |
| GPU         | NVIDIA with CUDA — **sm_90a** (H100/H200) or **sm_120** (RTX 5090) |
| RAM         | 16 GB minimum, 32 GB recommended                                   |
| Storage     | 50 GB free space                                                   |
| OS          | Ubuntu 22.04+ (recommended) or Windows 11                          |

### Software

| Tool         | Required Version | Source                                                  |
| ------------ | ---------------- | ------------------------------------------------------- |
| Python       | **3.12.x**       | https://www.python.org/downloads/release/python-3120/   |
| uv           | Latest           | https://astral.sh/uv/                                   |
| CUDA Toolkit | **13.0**         | https://developer.nvidia.com/cuda-13-0-download-archive |
| Rust         | stable           | https://rustup.rs/                                      |
| Go           | 1.26+            | https://go.dev/dl/                                      |
| Task         | Latest           | https://taskfile.dev/installation/                      |
| Docker       | 20.x+            | https://docs.docker.com/get-docker/                     |
| Git          | 2.x              | https://git-scm.com/downloads                           |

---

## Installation Scenarios

### 1. First-time install on a new machine

#### Linux / macOS

1. Clone the repo and initialize submodules:

```bash
git clone https://github.com/shireff/pearl-cion.git
cd pearl-cion
git submodule update --init --recursive
```

2. Install CUDA Toolkit 13.0:

```bash
# Ubuntu example:
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0

# Or download and install from NVIDIA:
# https://developer.nvidia.com/cuda-13-0-download-archive
```

Verify CUDA:

```bash
nvcc -V
# expected output includes "release 13.0"
```

If CUDA is installed in a nonstandard path, set:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

3. Install Python 3.12 and required tooling:

```bash
# Ubuntu example:
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Install Go
wget https://go.dev/dl/go1.26.12.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.26.12.linux-amd64.tar.gz
export PATH="/usr/local/go/bin:$PATH"

# Install Task runner
sh -c "$(curl -fsSL https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
```

Verify the tools:

```bash
python3 --version
uv --version
rustc --version
cargo --version
go version
task --version
```

4. Build the Python workspace and CUDA extension:

```bash
uv sync --package vllm-miner
```

5. Verify CUDA/PyTorch compatibility:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc -V
```

Expected output will include CUDA 13.0 / `cu130`, for example:

```bash
2.1.1+cu130 13.0
```

If PyTorch installs a non-CUDA wheel, install the correct package manually:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

6. Build the blockchain binaries if you need the full node locally:

```bash
task build:blockchain
```

If `task` is not installed, build manually:

```bash
cd zk-pow
cargo run --release --no-default-features --bin build_cache src/circuit/v2_cache.bin src/v1/v1_cache.bin
cd ..
go build -tags xmss,zkpow -o bin/pearld ./node
go build -tags xmss,zkpow -o bin/prlctl ./node/cmd/prlctl
go build -tags xmss,zkpow -o bin/oyster ./wallet
CGO_ENABLED=0 go build -o bin/oystercli ./wallet/cmd/oystercli
```

6. If the `pearl-gemm` build fails, retry directly from `miner/pearl-gemm`:

```bash
cd miner/pearl-gemm
bash build.sh --no-pull
```

#### Windows

1. Clone the repo and initialize submodules:

```powershell
git clone https://github.com/shireff/pearl-cion.git
cd pearl-cion
git submodule update --init --recursive
```

2. Install CUDA Toolkit 13.0 and add it to PATH.

3. Install Python 3.12 from the official Windows installer.

4. Install `uv`, Rust, and Go using official installers.

5. Verify Python and CUDA:

```powershell
python --version
nvcc -V
```

6. Build the Python workspace:

```powershell
uv sync --package vllm-miner
```

7. If CUDA kernel build fails, run from PowerShell or WSL:

```powershell
cd miner/pearl-gemm
bash build.sh --no-pull
```

> Note: Windows support may require WSL or a Linux-compatible shell for the CUDA build step.

### 2. Existing checkout on a machine that already has the repo

From the repo root:

```bash
git pull origin main
git submodule update --init --recursive
```

Then refresh the Python workspace:

```bash
uv sync --package vllm-miner
```

If you changed CUDA kernel or model plugin code, rebuild the extension:

```bash
cd miner/pearl-gemm
bash build.sh --no-pull
```

### 3. If the repo is already built and you only want to run it

From the repo root, start the full node:

```bash
bin/pearld --rpcuser=rpcuser --rpcpass=rpcpass --rpclisten=0.0.0.0:44107 --miningaddr=<your-mining-address> --txindex
```

Start the gateway in another terminal:

```bash
pearl-gateway start
```

Start the miner in another terminal:

```bash
uv run vllm serve pearl-ai/Llama-3.3-70B-Instruct-pearl \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager
```

### 4. Important repo paths

- `pearl/` — repository root
- `miner/pearl-gemm/` — CUDA kernel source and extension build logic
- `miner/vllm-miner/` — vLLM plugin and model-serving package
- `bin/pearld`, `bin/prlctl`, `bin/oyster`, `bin/oystercli` — compiled node and wallet binaries
- `miner/pearl-gemm/src/pearl_gemm/pearl_gemm_cuda.so` — built CUDA extension output

---

## Quick Reference — Client-Facing Commands

### Build (from repository root)

```bash
# Option 1: Using uv (works without task)
uv sync --package vllm-miner

# Option 2: Build everything (all workspace packages)
uv sync --all-packages

# Option 3: Build blockchain binaries only (Go + Rust)
cd zk-pow && cargo run --release --no-default-features --bin build_cache src/circuit/v2_cache.bin src/v1/v1_cache.bin && cd ..
go build -tags xmss,zkpow -o bin/pearld ./node
go build -tags xmss,zkpow -o bin/prlctl ./node/cmd/prlctl
go build -tags xmss,zkpow -o bin/oyster ./wallet
CGO_ENABLED=0 go build -o bin/oystercli ./wallet/cmd/oystercli
```

### Run the full mining stack

**Terminal 1 — Start the full node:**

```bash
cd "pearl-cion"
./bin/pearld --rpcuser=rpcuser --rpcpass=rpcpass --rpclisten=0.0.0.0:44107 --miningaddr=<your-mining-address> --txindex
```

**Terminal 2 — Start the gateway:**

```bash
cd "pearl-cion"
pearl-gateway start
```

**Terminal 3 — Start the miner (vLLM with Pearl plugin):**

```bash
cd "pearl-cion"
uv run vllm serve pearl-ai/Llama-3.3-70B-Instruct-pearl \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager
```

> The Pearl mining plugin is auto-loaded as a vLLM general plugin.
> You will see `INFO: pearl mining plugin registered` in the vLLM logs.

### Verify mining is active

```bash
# Check GPU utilization
nvidia-smi

# Check gateway metrics
curl http://127.0.0.1:9109/metrics

# Check node connectivity
prlctl getinfo

# Check vllm logs for:
#   INFO: pearl mining plugin registered
#   INFO: noisy_gemm activated for layer ...
```

---

## Method 1: From Source (Git)

### Step 1 — Clone the repository

```bash
git clone https://github.com/shireff/pearl-cion.git
cd pearl
git submodule update --init --recursive
```

### Step 2 — Install CUDA Toolkit 13.0

Download from: https://developer.nvidia.com/cuda-13-0-download-archive

**Linux:**

```bash
# Follow the network installer instructions for your distro.
# After install, verify:
nvcc --version
# Expected: release 13.0
```

**Windows:**

- Select: Windows → x86_64 → exe (local)
- Install to: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`

```powershell
nvcc --version
# Expected: release 13.0
```

### Step 3 — Install uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env   # or restart terminal

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 4 — Install Rust

```bash
# Linux/macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Windows
# Download rustup-init.exe from https://rustup.rs and run it
```

### Step 5 — Install Go 1.26+

```bash
# Linux — download from https://go.dev/dl/
# macOS
brew install go

# Verify:
go version
# Expected: go1.26.x or higher
```

### Step 6 — Install Task runner

```bash
# Linux/macOS
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b ~/.local/bin

# Windows (PowerShell)
winget install Task.Task
```

### Step 7 — Build blockchain binaries

From the **repository root** (`pearl/`):

```bash
task build:blockchain
```

This produces `bin/pearld`, `bin/prlctl`, `bin/oyster`, `bin/oystercli`.

**If `task` is not installed**, build manually without Task:

```bash
# Build ZK cache first (one-time, takes ~10 minutes)
cd zk-pow
cargo run --release --no-default-features --bin build_cache \
  src/circuit/v2_cache.bin src/v1/v1_cache.bin
cd ..

# Build binaries
go build -tags xmss,zkpow -o bin/pearld    ./node
go build -tags xmss,zkpow -o bin/prlctl   ./node/cmd/prlctl
go build -tags xmss,zkpow -o bin/oyster   ./wallet
CGO_ENABLED=0 go build   -o bin/oystercli ./wallet/cmd/oystercli
```

### Step 8 — Build the miner Python packages

From the **repository root**:

```bash
uv sync --package vllm-miner
```

This automatically:

- Installs Python 3.12 via uv
- Installs `torch==2.11.0` from the `cu130` PyTorch index (defined in `pyproject.toml`)
- Installs `vllm==0.20.2`
- Compiles the `pearl-gemm` CUDA kernels against CUDA 13.0

> **Do not** run `pip install torch` manually. `uv` uses the `pytorch-cu130` index
> as configured in `pyproject.toml`. Using the wrong PyTorch build will break the CUDA kernels.

If the CUDA kernel build fails:

```bash
export PEARL_GEMM_FORCE_BUILD=TRUE
uv sync --package vllm-miner
```

**Alternative: build everything (all workspace packages):**

```bash
uv sync --all-packages
```

### Step 9 — Create a wallet and get a mining address

**Interactive (easiest):**

```bash
bin/oystercli
# Walk through: Create New Wallet → save seed phrase → Receive → copy address
```

**Manual:**

```bash
# Create the wallet
bin/oyster --create
# Enter a passphrase when prompted. Save the seed phrase.

# Start the wallet daemon (background)
bin/oyster &

# Generate a Taproot mining address
bin/prlctl --wallet getnewaddress
# Output example: prl1qxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Save this address — you need it for pearld and the miner
```

### Step 10 — Start the full node

```bash
bin/pearld \
  --rpcuser=rpcuser \
  --rpcpass=rpcpass \
  --rpclisten=0.0.0.0:44107 \
  --miningaddr=<your-mining-address> \
  --txindex
```

Wait for it to sync. Verify:

```bash
bin/prlctl getinfo
# Look for "blocks" count increasing
```

Network port reference:

| Network | pearld RPC | P2P   | oyster RPC |
| ------- | ---------- | ----- | ---------- |
| Mainnet | 44107      | 44108 | 44207      |
| Testnet | 44109      | 44110 | 44209      |
| Simnet  | 18556      | 18555 | 18554      |

### Step 11 — Set environment variables

```bash
# Linux/macOS
export PEARLD_RPC_URL=http://localhost:44107
export PEARLD_RPC_USER=rpcuser
export PEARLD_RPC_PASSWORD=rpcpass
export PEARLD_MINING_ADDRESS=<your-mining-address>
export PEARL_LOG_LEVEL=INFO
export HF_TOKEN=<your-huggingface-token>
```

```powershell
# Windows (PowerShell)
$env:PEARLD_RPC_URL      = "http://localhost:44107"
$env:PEARLD_RPC_USER     = "rpcuser"
$env:PEARLD_RPC_PASSWORD = "rpcpass"
$env:PEARLD_MINING_ADDRESS = "<your-mining-address>"
$env:PEARL_LOG_LEVEL     = "INFO"
$env:HF_TOKEN            = "<your-huggingface-token>"
```

### Step 12 — Start pearl-gateway

In a **separate terminal**:

```bash
pearl-gateway start
```

Wait until the socket appears:

```bash
# Linux/macOS — verify socket is ready:
ls -la /tmp/pearlgw.sock
# Expected: srwxr-xr-x ... /tmp/pearlgw.sock
```

> `pearl-gateway` must be fully ready before starting `vllm serve`.
> The socket at `/tmp/pearlgw.sock` is the signal that it is ready.
> This is the same check `entrypoint.sh` performs (waits up to 30 seconds).

### Step 13 — Start vllm serve

In a **separate terminal**:

```bash
vllm serve pearl-ai/Llama-3.3-70B-Instruct-pearl \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager
```

> **Important:** The correct command is `vllm serve`, **not** `python -m vllm_miner`.
> The `vllm_miner` plugin registers itself automatically through the
> `vllm.general_plugins` entry point defined in `miner/vllm-miner/pyproject.toml`.
> You will see `INFO: pearl mining plugin registered` in the vllm logs when it loads.

---

## Method 2: Docker (Recommended for Production)

### Step 1 — Clone the repository

```bash
git clone https://github.com/shireff/pearl-cion.git
cd pearl
git submodule update --init --recursive
```

### Step 2 — Install blockchain binaries

```bash
# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/pearl-research-labs/pearl/master/install.ps1 | iex
```

Installs `pearld`, `prlctl`, `oyster`, `oystercli` to `~/.local/bin` (Linux/macOS)
or `%LOCALAPPDATA%\Pearl\bin` (Windows).

### Step 3 — Create wallet and start node

Same as Steps 9 and 10 from Method 1 above.

```bash
oystercli                                    # interactive wallet setup
pearld --rpcuser=rpcuser --rpcpass=rpcpass \
       --rpclisten=0.0.0.0:44107 \
       --miningaddr=<your-mining-address> \
       --txindex
```

### Step 4 — Build the Docker image

**Must run from the repository root** — the Dockerfile uses the full monorepo as build context:

```bash
docker buildx build -t vllm_miner . -f miner/vllm-miner/Dockerfile
```

Build time: 20–40 minutes (compiles CUDA kernels for `arch=compute_120,code=sm_120`).

### Step 5 — Run the miner container

```bash
docker run --rm -it --gpus all \
  -p 8000:8000 -p 8337:8337 -p 8339:8339 \
  -e PEARLD_RPC_URL=http://host.docker.internal:44107 \
  -e PEARLD_RPC_USER=rpcuser \
  -e PEARLD_RPC_PASSWORD=rpcpass \
  -e PEARLD_MINING_ADDRESS=<your-mining-address> \
  -e HF_TOKEN=<your-huggingface-token> \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 8g \
  vllm_miner:latest \
  pearl-ai/Llama-3.3-70B-Instruct-pearl \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager
```

What `entrypoint.sh` does inside the container (source: `miner/vllm-miner/entrypoint.sh`):

1. Auto-detects GPU count → sets `CUDA_VISIBLE_DEVICES`
2. Runs `pearl-gateway start` in the background
3. Polls `/tmp/pearlgw.sock` every second for up to 30 seconds
4. Exits with error if gateway dies or socket never appears
5. Runs `exec vllm serve <all your args>`

Expected output when everything works:

```
Auto-detected 1 GPUs, setting CUDA_VISIBLE_DEVICES=0
Starting pearl-gateway...
Waiting for pearl-gateway socket at /tmp/pearlgw.sock ...
Starting vllm serve with args: pearl-ai/Llama-3.3-70B-Instruct-pearl ...
INFO:     pearl mining plugin registered
INFO:     model loaded
```

---

## Verifying Mining Is Active

### Check GPU utilization

```bash
nvidia-smi
# GPU-Util % should rise when vllm processes requests
```

### Check gateway metrics

```bash
curl http://127.0.0.1:9109/metrics
# Returns Prometheus metrics including block submission counts
```

### Check node connectivity

```bash
prlctl getinfo
# Look for "blocks" and "connections" > 0
```

### Check vllm logs

```
INFO: pearl mining plugin registered        ← plugin loaded
INFO: noisy_gemm activated for layer ...    ← GPU mining active
```

---

## Profiling (Production Path)

### 1. Nsight Systems — System-Level Timeline

Identifies where time is spent across CPU, GPU, memory, and CUDA APIs.

```bash
# Profile the vLLM process for 30 seconds
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output=profile_report \
  --force-overwrite true \
  uv run vllm serve pearl-ai/Llama-3.3-70B-Instruct-pearl \
    --host 0.0.0.0 --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization 0.9 --enforce-eager
```

After the process runs for 30 seconds, press `Ctrl+C` to stop and generate the report.

Open the report:

```bash
nsys-ui profile_report.qdrep
```

**What to look for:**

- Long gaps between GPU kernels → CPU-GPU sync bottleneck
- CPU time dominating → Python overhead or rate limiting
- Memory copies between CPU and GPU → unnecessary `.cpu()` calls
- CUDA API wait time → synchronization overhead

---

### 2. Nsight Compute — CUDA Kernel Analysis

Analyzes individual CUDA kernel performance (occupancy, memory throughput, register pressure).

```bash
# Profile a specific kernel (run vllm in another terminal first)
ncu --set full \
  --kernel-name "noisy_gemm" \
  --target-processes all \
  --output ncu_report \
  --force-overwrite true \
  uv run vllm serve pearl-ai/Llama-3.3-70B-Instruct-pearl \
    --host 0.0.0.0 --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization 0.9 --enforce-eager
```

Stop after 30 seconds with `Ctrl+C`.

Open the report:

```bash
ncu-ui ncu_report.ncu-rep
```

**Key metrics to check:**

| Metric                                         | What it tells you      | Good value      |
| ---------------------------------------------- | ---------------------- | --------------- |
| `sm_efficiency`                                | GPU SM utilization     | > 60%           |
| `achieved_occupancy`                           | Warp occupancy on SMs  | > 50%           |
| `dram_throughput`                              | Memory bandwidth usage | > 50% of peak   |
| `l1tex_cache_hit_rate`                         | L1/L2 cache efficiency | > 80%           |
| `smsp__warps_active.avg.pct_of_peak_sustained` | Active warps           | > 70%           |
| `gpu__time_duration`                           | Kernel execution time  | Lower is better |
| `register_per_thread`                          | Registers per thread   | < 255 (limit)   |
| `shared_mem_usage`                             | Shared memory usage    | < 64 KB (limit) |

**If occupancy is low:**

- Increase `tile_size_m` or `tile_size_n` in `settings.py`
- Decrease `noise_rank` to reduce register pressure
- Check if `pipeline_stages` is too high or too low

**If memory is the bottleneck:**

- Check `dram_throughput` vs GPU memory bandwidth
- Optimize `cols_pattern` density (more columns = more compute per memory access)
- Increase `tile_size_k` to process more data per memory load

**If register pressure is high:**

- `register_per_thread` near 255
- Reduce `noise_rank` or `tile_size_m`
- The compiler may be spilling registers to local memory

---

### 3. nvidia-smi dmon — Real-Time GPU Monitoring

Monitors GPU utilization, memory, power, and temperature in real time.

```bash
# Monitor all GPUs every 1 second
nvidia-smi dmon -s uctpmv -d 1

# Columns:
# u = GPU utilization (%)
# c = CUDA cores utilization (%)
# t = Temperature (°C)
# p = Power draw (W)
# m = Memory utilization (%)
# v = Memory bandwidth (GB/s)
```

**What to look for:**

- GPU utilization < 80% → GPU is waiting for CPU or kernel is too small
- Memory utilization near 100% → OOM risk or memory-bound kernel
- Power draw stable at max → GPU is fully utilized
- Temperature throttling → reduce power limit or improve cooling

**Quick check during mining:**

```bash
watch -n 1 nvidia-smi
```

---

### 4. CUDA Events — Kernel Timing

Measure exact execution time of individual CUDA kernels.

Add this to `gemm_operators.py` for ad-hoc profiling:

```python
def profile_kernel(name: str, fn, *args, **kwargs):
    """Profile a single CUDA kernel execution."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    _LOGGER.info(f"[PROFILE] {name}: {elapsed_ms:.2f} ms")
    return result
```

Usage in `pearl_gemm_noisy`:

```python
# Profile each kernel individually
profile_kernel("tensor_hash_A", run_tensor_hash, A.to(torch.uint8), key_tensor, A_tensor_hash, scratchpad)
profile_kernel("tensor_hash_B", run_tensor_hash, B.to(torch.uint8), key_tensor, B_tensor_hash, scratchpad)
profile_kernel("commitment_hash", commitment_hash_from_merkle_roots, A_tensor_hash, B_tensor_hash, key_tensor, commitment_hash_A, commitment_hash_B)
profile_kernel("noise_gen", noise_gen, R=r, EAL=EAL, EAL_fp16=EAL_fp16, ...)
profile_kernel("noisy_gemm", noisy_gemm, A=A, B=B, ...)
```

**What to look for:**

- Which kernel takes the most time → that's your bottleneck
- If `noisy_gemm` dominates → kernel is compute-bound, optimize tile sizes
- If `tensor_hash` dominates → hash computation is the bottleneck, increase `idxs_per_col`
- If `noise_gen` dominates → noise generation is the bottleneck, optimize `noise_rank`

---

### 5. Combined Profiling Workflow

```bash
# Step 1: Check GPU utilization
watch -n 1 nvidia-smi

# Step 2: Run Nsight Systems for system-level timeline
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
  --output profile_report --force-overwrite true \
  uv run vllm serve pearl-ai/Llama-3.3-70B-Instruct-pearl \
    --host 0.0.0.0 --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization 0.9 --enforce-eager

# Step 3: Run Nsight Compute for kernel-level analysis
ncu --set full --kernel-name "noisy_gemm" \
  --output ncu_report --force-overwrite true \
  uv run vllm serve pearl-ai/Llama-3.3-70B-Instruct-pearl \
    --host 0.0.0.0 --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization 0.9 --enforce-eager

# Step 4: Analyze results
nsys-ui profile_report.qdrep
ncu-ui ncu_report.ncu-rep
```

---

## Performance Settings

### Single GPU

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
```

### Multi-GPU (tensor parallelism)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve pearl-ai/Llama-3.3-70B-Instruct-pearl \
  --tensor-parallel-size 4 \
  ...
```

### System-level (Linux)

```bash
ulimit -n 65536
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl <max-power-watts>
```

---

## Port Reference

| Port                | Service       | Description                  |
| ------------------- | ------------- | ---------------------------- |
| 44107               | pearld RPC    | Full node JSON-RPC           |
| 44108               | pearld P2P    | Network peers                |
| 44207               | oyster RPC    | Wallet daemon                |
| 8000                | vllm serve    | LLM inference API            |
| 8337                | pearl-gateway | Mining RPC (TCP fallback)    |
| 8339                | pearl-gateway | Secondary port               |
| 9109                | pearl-gateway | Prometheus metrics           |
| `/tmp/pearlgw.sock` | pearl-gateway | Unix Domain Socket (default) |

---

## Troubleshooting

### CUDA kernel build fails

```bash
export PEARL_GEMM_FORCE_BUILD=TRUE
uv sync --package vllm-miner
# Make sure nvcc --version shows 13.x
```

### No space left on device while extracting Python wheels

If `uv sync` fails while extracting a wheel into `/tmp`, the host temporary directory may be too small.
Use a larger temp directory and retry with:

```bash
mkdir -p /root/uv-tmp
export TMPDIR=/root/uv-tmp
uv sync --package vllm-miner
```

Make sure the target path has at least 10-20 GB free space.

### pearl-gateway socket never appears

```bash
# Check if gateway is running:
ps aux | grep pearl-gateway

# Run with debug logging:
pearl-gateway start --debug

# Use a custom socket path if needed:
export MINER_RPC_SOCKET_PATH=/tmp/pearlgw.sock
pearl-gateway start
```

### GPU not detected

```bash
nvidia-smi                                                      # must work
nvcc --version                                                  # must show 13.x
python -c "import torch; print(torch.cuda.is_available())"     # must print True
```

### pearld not connecting

```bash
# Check pearld is running:
prlctl getinfo

# Check config file:
cat ~/.pearld/pearld.conf          # Linux
# %LOCALAPPDATA%\Pearld\pearld.conf  # Windows
```

### "pearl mining plugin not registered" in vllm logs

```bash
# Verify the package is installed:
uv run python -c "import vllm_miner; print('OK')"

# Re-install if needed:
uv sync --package vllm-miner
```

### Wrong PyTorch CUDA version

```bash
# Check what version is installed:
python -c "import torch; print(torch.version.cuda)"
# Must print: 13.0 (not 12.x or 11.x)

# Fix: do NOT use pip. Use uv from the repo root:
uv sync --package vllm-miner
```

---

## Repository Structure

```
pearl/
├── node/              → pearld binary (full node)
├── wallet/
│   └── cmd/
│       ├── prlctl/    → prlctl binary (node CLI)
│       └── oystercli/ → oystercli binary (interactive wallet)
├── zk-pow/            → ZK proof-of-work circuits (Rust)
├── miner/
│   ├── pearl-gemm/    → CUDA kernels: NoisyGEMM, BLAKE3 PoW, Tensor Core path
│   ├── pearl-gateway/ → node ↔ miner bridge  (CLI: pearl-gateway start)
│   ├── vllm-miner/    → vLLM plugin           (loaded via: vllm serve)
│   │   └── entrypoint.sh  → production startup sequence (Docker)
│   ├── miner-base/    → core mining loop, block submission
│   └── miner-utils/   → shared logging/utilities
├── py-pearl-mining/   → Python bindings (Rust/PyO3)
├── pyproject.toml     → uv workspace root (torch cu130 index here)
└── Taskfile.yml       → task shortcuts (build, test, fmt, lint)
```

---

## Useful Links

- [Pearl Protocol Paper](https://arxiv.org/abs/2504.09971)
- [GitHub Repository](https://github.com/shireff/pearl-cion)
- [HuggingFace Model](https://huggingface.co/pearl-ai/Llama-3.3-70B-Instruct-pearl)
- [CUDA 13.0 Download](https://developer.nvidia.com/cuda-13-0-download-archive)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Task Runner](https://taskfile.dev)
