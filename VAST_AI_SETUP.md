# Pearl Mining on vast.ai — Complete Setup Guide

Verified on: RTX 4060 Ti · CUDA 13.0 · torch 2.11.0 · Python 3.12

---

## 1. Picking an Instance

### Docker image

```
vastai/base-image_cuda-13.0.3-auto/jupyter
```

This image ships with `/workspace/pearl-venv` containing **torch 2.11.0+cu130** pre-installed — no torch download needed.

### Search filters

1. vast.ai → **Search** tab
2. Filters: GPU RAM ≥ 16 GB · CUDA 13.0 · Disk ≥ 30 GB
3. Sort by **DLP/$/hr**
4. Click **Rent**

---

## 2. SSH Connection

1. vast.ai → **Instances** → **Connect**
2. Copy the SSH command — the port changes every instance:

```bash
ssh -p <PORT> root@<IP>
```

> The port is different for every instance. Always copy it fresh from the Connect button.

### Use tmux (required for persistence)

```bash
tmux new-session -s mining
# Detach: Ctrl+B then D
# Reattach: tmux attach -t mining
```

---

## 3. First-Time Setup

Run the following commands once on a new instance.

### 3.1 Clone the repository

```bash
git clone https://github.com/shireff/pearl-cion.git ~/pearl-cion
cd ~/pearl-cion
```

### 3.2 Activate the pre-installed Python environment

The instance has `/workspace/pearl-venv` with torch 2.11.0+cu130 ready. Always use this venv.

```bash
source /workspace/pearl-venv/bin/activate

# Make permanent for future sessions
echo 'source /workspace/pearl-venv/bin/activate' >> ~/.bashrc

# Verify
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
# Expected: 2.11.0+cu130 13.0 NVIDIA GeForce RTX 4060 Ti
```

### 3.3 Install miner packages

Install in this exact order — each depends on the previous:

```bash
cd ~/pearl-cion

pip install loguru --quiet
pip install -e miner/miner-utils --no-deps --quiet
pip install -e py-pearl-mining --no-deps --quiet
pip install -e miner/pearl-gateway --no-deps --quiet
pip install -e miner/miner-base --no-deps --quiet

# Verify
python -c "from miner_base.gateway_client import MiningClient; print('miner_base OK')"
```

### 3.4 Build pearld from source

`pearld` is not available as a pre-built binary — it must be compiled from source.

```bash
cd ~/pearl-cion

# Load Rust and Go (pre-installed on the instance)
source "$HOME/.cargo/env"
export PATH="/usr/local/go/bin:$PATH"

# Initialize required submodules
git submodule update --init zk-pow xmss

# Build zk-pow Go bindings
cd zk-pow/bindings/go
cargo build --release --no-default-features
cd ~/pearl-cion

# Build xmss static library
cd xmss && make && cd ~/pearl-cion

# Build pearld
mkdir -p bin
touch bin/sample-pearld.conf
go build -tags xmss,zkpow -o bin/pearld ./node

# Verify
./bin/pearld --version
# Expected: pearld version 1.1.6
```

### 3.5 Configure and start pearld

```bash
mkdir -p ~/.pearld
cat > ~/.pearld/pearld.conf << 'EOF'
[Application Options]
rpcuser=rpcuser
rpcpass=rpcpass
rpclisten=127.0.0.1:44107
txindex=1
notls=1
EOF

# Start pearld in background
cd ~/pearl-cion
./bin/pearld &
sleep 10

# Verify RPC is working
curl -s --user rpcuser:rpcpass \
  --data-binary '{"jsonrpc":"1.0","method":"getblockcount","params":[]}' \
  -H 'content-type:text/plain;' \
  http://127.0.0.1:44107/
# Expected: {"result":0,...}
```

### 3.6 Build the CUDA extension

Must be built against the same torch in `/workspace/pearl-venv`.

```bash
cd ~/pearl-cion
source /workspace/pearl-venv/bin/activate

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK=1
export MAX_JOBS=4

pip install -e miner/pearl-gemm-build-utils --quiet

cd miner/pearl-gemm
rm -rf build src/pearl_gemm_cuda*.so
python setup.py build_ext --inplace
cd ~/pearl-cion

# Verify
python -c "import pearl_gemm_cuda; print('BUILD OK')"
```

This takes 5–15 minutes.

---

## 4. Running the Miner

```bash
cd ~/pearl-cion
source /workspace/pearl-venv/bin/activate

python miner/miner_gpu.py \
  --rpc-url http://127.0.0.1:44107 \
  --rpc-user rpcuser \
  --rpc-password rpcpass \
  --mining-address <YOUR_WALLET_ADDRESS> \
  --noise-range 256 --P 8 --Q 8
```

### Available flags

| Flag | Default | Description |
|------|---------|-------------|
| `--noise-range` | 256 | Matrix size |
| `--P` | 8 | Row partitions |
| `--Q` | 8 | Column partitions |
| `--num-jobs` | 1 | Parallel jobs per round |

### Run persistently with log

```bash
tmux new-session -s mining

cd ~/pearl-cion
source /workspace/pearl-venv/bin/activate

python miner/miner_gpu.py \
  --rpc-url http://127.0.0.1:44107 \
  --rpc-user rpcuser \
  --rpc-password rpcpass \
  --mining-address <YOUR_WALLET_ADDRESS> \
  --noise-range 256 --P 8 --Q 8 \
  2>&1 | tee /tmp/miner.log
```

### Expected output

```
GPU round time: 0.5ms, tmok/s=3800.xx  jobs=1  combos=64
GPU round time: 0.6ms, tmok/s=3612.xx  jobs=1  combos=64
```

---

## ⚠️ **VERIFY THE MINER IS ACTUALLY RUNNING — NOT JUST APPEARING TO**

> **Do not assume the miner is working just because a command ran without errors.**
> Run these steps to confirm real GPU activity and real log output.

### Step 1 — Kill any leftover processes and start fresh

```bash
# Kill any orphan pearl_gateway processes left from previous sessions
# (PIDs shown are examples — use pgrep -f pearl_gateway to find yours)
kill $(pgrep -f "pearl_gateway") 2>/dev/null

# Activate the correct Python environment
source /workspace/pearl-venv/bin/activate

# Go to project root and pull latest code
cd ~/pearl-cion && git pull
```

### Step 2 — Start the miner in the background and capture logs

```bash
# nohup: keeps the process running even if the terminal closes
# > /tmp/miner.log 2>&1: redirects both stdout and stderr to the log file
# &: sends the process to the background so you keep your terminal
nohup python miner/miner_gpu.py \
  --rpc-url http://127.0.0.1:44107 \
  --rpc-user rpcuser \
  --rpc-password rpcpass \
  --mining-address <YOUR_WALLET_ADDRESS> \
  --noise-range 256 --P 8 --Q 8 > /tmp/miner.log 2>&1 &

# Print the process ID — save this number, you can use it to kill the miner later
echo "PID: $!"
```

### Step 3 — Follow the live log output

```bash
# -f: follows the file in real time — new lines appear as they're written
# Press Ctrl+C to stop following (the miner keeps running in the background)
tail -f /tmp/miner.log
```

**Wait 10–20 seconds. You should see lines like:**

```
GPU round time: 0.5ms, tmok/s=3800.xx  jobs=1  combos=64
GPU round time: 0.6ms, tmok/s=3612.xx  jobs=1  combos=64
```

Press `Ctrl+C` to stop watching.

### Step 4 — Check the last 50 lines after stopping

```bash
# Shows the last 50 lines of the log — useful to catch any errors
# that appeared after you stopped watching with tail -f
tail -50 /tmp/miner.log
```

### Step 5 — Confirm GPU is actually working

```bash
# Shows real-time GPU utilization, temperature, and memory usage
# If the miner is running correctly, GPU% should be 60–95%
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used,power.draw \
  --format=csv -l 2
```

**If GPU% is 0%** — the miner is running on CPU only or has crashed. Check `/tmp/miner.log` for errors.

---

## 5. Session Restart Checklist

Every time you reconnect:

```bash
# 1. Activate venv
source /workspace/pearl-venv/bin/activate

# 2. Kill leftover gateway processes
kill $(pgrep -f "pearl_gateway") 2>/dev/null; sleep 1

# 3. Make sure pearld is running
cd ~/pearl-cion
pgrep -x pearld || ./bin/pearld &
sleep 5

# 4. Pull latest code
git pull

# 5. Run miner
python miner/miner_gpu.py \
  --rpc-url http://127.0.0.1:44107 \
  --rpc-user rpcuser \
  --rpc-password rpcpass \
  --mining-address <YOUR_WALLET_ADDRESS> \
  --noise-range 256 --P 8 --Q 8
```

---

## 6. Troubleshooting

### `ssh: connect to host X port XXXXX: Connection refused`

The SSH port changes with every instance.
Go to vast.ai → Instances → **Connect** and copy the new command.

### `ModuleNotFoundError: No module named 'miner_base'`

Wrong venv or packages not installed:
```bash
source /workspace/pearl-venv/bin/activate
cd ~/pearl-cion
pip install loguru --quiet
pip install -e miner/miner-utils --no-deps --quiet
pip install -e py-pearl-mining --no-deps --quiet
pip install -e miner/pearl-gateway --no-deps --quiet
pip install -e miner/miner-base --no-deps --quiet
```

### `ModuleNotFoundError: No module named 'torch'`

Wrong venv — torch only lives in `/workspace/pearl-venv`:
```bash
source /workspace/pearl-venv/bin/activate
python -c "import torch; print(torch.__version__)"
```

### `undefined symbol` in pearl_gemm_cuda.so

The `.so` was built against a different torch. Rebuild:
```bash
source /workspace/pearl-venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK=1
export MAX_JOBS=4
cd ~/pearl-cion/miner/pearl-gemm
rm -rf build src/pearl_gemm_cuda*.so
python setup.py build_ext --inplace
```

### `No block template available` / `mining_paused`

pearld is not running:
```bash
cd ~/pearl-cion
pgrep -x pearld || ./bin/pearld &
sleep 10
curl -s --user rpcuser:rpcpass \
  --data-binary '{"jsonrpc":"1.0","method":"getblockcount","params":[]}' \
  -H 'content-type:text/plain;' http://127.0.0.1:44107/
```

### `CUDA error: illegal memory access`

Rebuild pearl-gemm:
```bash
source /workspace/pearl-venv/bin/activate
cd ~/pearl-cion/miner/pearl-gemm
rm -rf build src/pearl_gemm_cuda*.so
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK=1
export MAX_JOBS=4
python setup.py build_ext --inplace
```

### Multiple gateway processes running

```bash
kill $(pgrep -f "pearl_gateway") 2>/dev/null
```

---

## 7. Performance Reference

| GPU | tmok/s |
|-----|--------|
| RTX 4060 Ti (16 GB) | 1500–4000 |
| RTX 4090 (24 GB) | ~8000 (estimated) |
| A100 (80 GB) | ~12000 (estimated) |

---

*Verified: RTX 4060 Ti · torch 2.11.0+cu130 · CUDA 13.0 · Python 3.12 · pearld 1.1.6*
