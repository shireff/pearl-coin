# Pearl Miner - Complete Setup and Operations Guide

## Overview

Pearl is an L1 blockchain based on **Proof-of-Useful-Work** protocol, where mining is performed as a by-product of arbitrary matrix multiplication, as proposed in [this paper](https://arxiv.org/abs/2504.09971).

This guide covers setting up and operating the Miner on NVIDIA GPUs.

---

## Prerequisites

### Hardware

- **GPU**: NVIDIA GPU with **CUDA** support and compute capability **sm_90a** (H100/H200)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 50GB minimum free space

### Software

| Tool             | Required Version | Notes                                      |
| ---------------- | ---------------- | ------------------------------------------ |
| **Python**       | 3.12 or later    | Required for the miner                     |
| **uv**           | Latest           | Python package manager                     |
| **CUDA Toolkit** | 13.0 or later    | Required for GPU kernel execution          |
| **Rust**         | Latest stable    | For ZK proving and Blake3 utilities        |
| **Git**          | 2.x              | For cloning the repository                 |
| **Task**         | Latest           | (Optional) for running predefined commands |
| **Docker**       | 20.x+            | (Optional) for containerized deployment    |

#### CUDA Toolkit Setup

Download and install CUDA Toolkit 13.x from NVIDIA:

- **Download link**: https://developer.nvidia.com/cuda-130-download-archive
- **Windows installer**: x86_64 → exe (local)
- **Install path**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- **Add to PATH**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`

Verify installation:

```bash
nvcc --version
```

#### PyTorch with CUDA Support

Install PyTorch with CUDA support (not CPU-only):

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA is available:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
# Expected: 2.x.x+cu128 and 12.8
```

#### Build CUDA Extension

Build the `pearl-gemm` CUDA extension:

```bash
cd miner/pearl-gemm
python setup.py build_ext --inplace
```

Verify the extension was built:

- **Windows**: `ls miner/pearl-gemm/src/pearl_gemm/*.pyd`
- **Linux**: `ls miner/pearl-gemm/build/lib/*.so`

#### Build Rust zk-pow with GPU Support

From the repository root:

```bash
cd zk-pow
cargo build --features gpu_prove
```

If `pearl_gemm_cuda` is not found automatically, set the library path:

```bash
# Windows (PowerShell)
$env:PEARL_GEMM_CUDA_PATH = "C:\path\to\pearl\miner\pearl-gemm\src\pearl_gemm"
```

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/shireff/pearl-cion.git
cd pearl-cion
```

### Initialize Submodules

```bash
git submodule update --init --recursive
```

---

## Step 2: Install Required Tools

### Linux/macOS

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
# Install Rust
irm https://win.rustup.rs -OutFile rustup-init.exe
.\rustup-init.exe -y

# Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Step 3: Build the Project

### Option 1: Using Task Runner (Recommended)

```bash
# Build everything (blockchain + miner)
task build

# Or build miner only
task build:miner
```

### Option 2: Manual Build

```bash
# Build ZK cache first (required for zk-pow)
cd zk-pow
cargo run --release --no-default-features --bin build_cache src/circuit/v2_cache.bin src/v1/v1_cache.bin
cd ..

# Build Python packages
uv sync --all-packages
```

---

## Step 4: Wallet and Node Setup

### 4.1 Create a Wallet

```bash
cd bin
./oystercli
```

Or manually:

```bash
./bin/oyster -u rpcuser -P rpcpass --create
```

### 4.2 Start the Wallet

```bash
./bin/oyster -u rpcuser -P rpcpass &
```

### 4.3 Get a Mining Address

```bash
./bin/prlctl -u rpcuser -P rpcpass -s https://localhost:44207 getnewaddress
```

### 4.4 Start the Node (pearld)

```bash
./bin/pearld \
  --rpcuser=rpcuser \
  --rpcpass=rpcpass \
  --rpclisten=0.0.0.0:44107 \
  --miningaddr=<your-mining-address> \
  --txindex
```

---

## Step 5: Run the Miner

### Environment Variables

```bash
export PEARLD_RPC_URL="http://localhost:44107"
export PEARLD_RPC_USER="rpcuser"
export PEARLD_RPC_PASSWORD="rpcpass"
export PEARLD_MINING_ADDRESS="<your-mining-address>"
export PEARL_LOG_LEVEL="INFO"
```

### Start Pearl Gateway

```bash
pearl-gateway start
```

### Start vLLM Miner

```bash
uv run python -m vllm_miner \
  --model pearl-ai/Llama-3.3-70B-Instruct-pearl \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

---

## Performance Tips for Maximum GPU Throughput

### 1. CUDA Settings

```bash
# Disable TDM if causing issues
export CUDA_LAUNCH_BLOCKING=0

# Enable shared memory caching
export CUDA_CACHE_DISABLE=0
```

### 2. PyTorch / vLLM Settings

```bash
# Enable cuDNN autotuner
export CUDNN_BENCHMARK=1

# Restrict to single GPU
export CUDA_VISIBLE_DEVICES=0
```

### 3. System Settings

```bash
# Increase file descriptor limit (Linux)
ulimit -n 65536

# Disable GPU idle power management (Linux)
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl <max-power-limit-in-watts>
```

### 4. BIOS/OS Recommendations

- Enable **Resizable BAR** in BIOS
- Use **Windows 11** or **Ubuntu 22.04+** for best compatibility
- Close other GPU-intensive applications

---

## Verification

### Verify GPU is Working

```bash
nvidia-smi
```

### Verify Node is Connected

```bash
./bin/prlctl getinfo
```

### Verify Miner is Running

```bash
# Check gateway logs
pearl-gateway start -v

# Check miner logs
uv run python -m vllm_miner --log-level debug
```

---

## Performance Simulation Benchmark (RTX 5090 Model)

### Overview

The `rtx5090_perf_sim` benchmark is a **deterministic mathematical simulator** that estimates the expected mining throughput on an NVIDIA RTX 5090 **without requiring an actual RTX 5090**.

It analyzes the production mining pipeline, intercepts every CUDA kernel, and estimates execution time using:

- Instruction counts from kernel source code
- Memory traffic analysis
- Roofline model
- Blackwell SM100 architectural limits
- Tensor Core utilization models

**This benchmark does NOT depend on the host machine GPU.** Running it on a CPU-only machine, GTX 1060, RTX 3090, or RTX 4090 produces approximately the same result because the simulation is based on a fixed hardware model.

### RTX 5090 Hardware Model

| Parameter                        | Value                  |
| -------------------------------- | ---------------------- |
| Architecture                     | Blackwell              |
| CUDA Cores                       | 21,760                 |
| Boost Clock                      | 2,407 MHz              |
| Memory                           | 32 GB GDDR7            |
| Memory Bus                       | 512-bit                |
| Memory Bandwidth                 | 1,792 GB/s             |
| PCIe                             | Gen5 x16               |
| Tensor Cores                     | Blackwell Tensor Cores |
| Warp Size                        | 32                     |
| SMs                              | 170                    |
| Max Threads per SM               | 2,048                  |
| Registers per SM                 | 65,536                 |
| Shared Memory per Block (opt-in) | 228 KB                 |
| L2 Cache                         | 64 MB (estimated)      |

### Build the Benchmark

```bash
cd zk-pow
cargo build --bin rtx5090_perf_sim
```

For release mode (recommended for accurate timing):

```bash
cargo build --release --bin rtx5090_perf_sim
```

### Run the Benchmark

```bash
cargo run --release --bin rtx5090_perf_sim -- <batch_size> <m> <n> <k> <rank>
```

#### Mining Problem Shape Parameters

| Parameter    | Default | Description                                                                    |
| ------------ | ------- | ------------------------------------------------------------------------------ |
| `batch_size` | 256     | Number of candidates per batch (Number of candidates in each batch)            |
| `m`          | 1024    | Number of rows in matrix A (Number of rows in a matrix A)                      |
| `n`          | 1024    | Number of columns in matrix B^T (Number of columns in a matrix B^T)            |
| `k`          | 1024    | Common dimension of the matmul (Common dimension of the matrix multiplication) |
| `rank`       | 32      | Rank of the noise matrices (Rank of the noise matrices)                        |
| `tile_h`     | 128     | Tile height for jackpot evaluation (derived from `a_rows_list`)                |
| `tile_w`     | 256     | Tile width for jackpot evaluation (derived from `b_cols_list`)                 |
| `p`          | 8       | Number of tile groups in M dimension (derived)                                 |
| `q`          | 8       | Number of tile groups in N dimension (derived)                                 |

### Example Commands

```bash
# Default parameters (256 candidates, 1024x1024x1024 matrices, rank 32)
cargo run --release --bin rtx5090_perf_sim

# Custom parameters: 128 candidates, 2048x2048x2048 matrices, rank 64
cargo run --release --bin rtx5090_perf_sim -- 128 2048 2048 2048 64

# Small test: 32 candidates, 512x512x512 matrices, rank 16
cargo run --release --bin rtx5090_perf_sim -- 32 512 512 512 16
```

### Output Explanation

The benchmark prints a detailed performance report:

```
================================================================================

                     RTX 5090 PERFORMANCE SIMULATION REPORT

================================================================================

Git Commit: <commit-hash>
Compiler: rustc <version>
Build Mode: release
Architecture: Blackwell
GPU Model: NVIDIA RTX 5090
Simulation Mode: Deterministic Mathematical Model
Batch Size: 256
Kernel Count: 7
Average Batch Time: 0.002 ms
Throughput: 7.0 TMOC/s
Peak Throughput: 12.5 TMOC/s
Min Throughput: 6.1 TMOC/s
Median Throughput: 7.0 TMOC/s
95% Confidence Interval: 6.5 - 7.6 TMOC/s
Probability >=1500 TMOC/s: 50.0%
Overall Confidence: 70.0%

================================================================================
                            FINAL PERFORMANCE REPORT
================================================================================

Simulation Target: NVIDIA RTX 5090
Average Batch Latency: 0.002 ms
Average Throughput: 7.0 TMOC/s
Peak Throughput: 12.5 TMOC/s
Minimum Throughput: 6.1 TMOC/s
Median Throughput: 7.0 TMOC/s
95% Confidence Interval: 6.5 - 7.6 TMOC/s
Probability of reaching >=1500 TMOC/s: 50.0%
Overall Confidence: 70.0%

================================================================================
                             FINAL ESTIMATED TMOC/s
================================================================================

7.0

================================================================================
```

### Key Metrics

| Metric                        | Description                                               |
| ----------------------------- | --------------------------------------------------------- |
| **FINAL ESTIMATED TMOC/s**    | Primary KPI — predicted throughput on RTX 5090            |
| **Average Batch Latency**     | Predicted time per batch in milliseconds                  |
| **Peak Throughput**           | Theoretical maximum if all bottlenecks were eliminated    |
| **95% Confidence Interval**   | Expected performance range under realistic variance       |
| **Probability >=1500 TMOC/s** | Likelihood of reaching the target throughput              |
| **Overall Confidence**        | Model confidence percentage based on hardware assumptions |

### How It Works

The simulator executes the real production mining pipeline stages:

1. **Candidate Generation** — `gpu_generate_raw_matrices_kernel`
2. **Noise Generation** — `run_noise_generation` (A and B)
3. **Denoise Converter** — `run_denoise_converter`
4. **GEMM** — `hopper_gemm_ws` (5-stage TMA + WGMMA)
5. **Jackpot Mining** — `gpu_mining_kernel`
6. **BLAKE3 Hash** — `gpu_jackpot_hash_kernel`
7. **Inner Hash Verification** — `inner_hash_kernel`

For each kernel, the simulator calculates:

- Instruction count from loop bounds
- Arithmetic intensity from tensor dimensions
- Occupancy from shared memory and register usage
- Roofline model crossing point
- Estimated execution time from compute/memory bound models

### Validation

The benchmark is deterministic and self-consistent:

- Same inputs always produce same outputs
- No hardware measurements are used
- No CUDA errors can occur
- Running on any machine produces identical results

### Integration with CI

```bash
# Run benchmark as part of CI
cargo run --release --bin rtx5090_perf_sim -- 256 1024 1024 1024 32
```

---

## Troubleshooting

### Issue: CUDA Kernel Build Failure

```bash
# Force rebuild
export PEARL_GEMM_FORCE_BUILD=TRUE
uv sync
```

### Issue: Node Connection Failure

```bash
# Verify pearld is running
ps aux | grep pearld

# Check RPC configuration
cat ~/.pearld/pearld.conf
```

### Issue: GPU Not Detected

```bash
# Verify driver installation
nvidia-smi

# Check CUDA version
nvcc --version
```

---

## Project Structure

```
pearl-cion/
├── node/                    # Full node implementation (pearld)
├── wallet/                  # Oyster HD wallet daemon
├── zk-pow/                  # ZK proof-of-work circuits (Rust)
├── miner/
│   ├── pearl-gemm/         # CUDA kernels for mining
│   ├── pearl-gateway/      # Bridge between node and miner
│   ├── vllm-miner/         # vLLM plugin for mining
│   └── miner-base/         # Core mining logic
├── py-pearl-mining/        # Python bindings for mining
├── plonky2/                # ZK proof system
└── apps/                   # Frontend applications
```

---

## Support

- **Repository**: https://github.com/shireff/pearl-cion
- **Issues**: https://github.com/shireff/pearl-cion/issues
