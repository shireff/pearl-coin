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
| Tool | Required Version | Notes |
|------|-----------------|-------|
| **Python** | 3.12 or later | Required for the miner |
| **uv** | Latest | Python package manager |
| **CUDA Toolkit** | 13.0 or later | Required for GPU kernel execution |
| **Rust** | Latest stable | For ZK proving and Blake3 utilities |
| **Git** | 2.x | For cloning the repository |
| **Task** | Latest | (Optional) for running predefined commands |
| **Docker** | 20.x+ | (Optional) for containerized deployment |

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
