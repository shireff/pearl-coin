# Pearl Mining — Setup Guide

> Every command in this guide is verified against the current repository source.  
> Obsolete instructions (vLLM plugin path, Docker vllm-miner image, `noisy_gemm` vLLM hooks) have been removed because the `vllm-miner` package directory is not present in this checkout.  
> **Single source of truth**: the code in `miner/`, `node/`, `wallet/`, and `miner/pearl-gateway/`.

## How Mining Works

```
pearld (full node)
    ↓ JSON-RPC HTTPS (port 44107)
pearl-gateway start            ← fetches block template, distributes work
    ↓ Unix socket /tmp/pearlgw.sock (or TCP 8337)
miner_gpu.py                   ← standalone GPU miner
    ↓ CUDA kernels (pearl_gemm)
gpu_mine_batch / gpu_jackpot_hash
    ↓
Miners submit proofs back through pearl-gateway → pearld
```

The GPU miner connects directly to `pearl-gateway` over a local socket.  
`pearl-gateway` handles all communication with `pearld`.  
`miner_gpu.py` does **not** connect to `pearld` and does **not** require VLLM.

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
| Git          | 2.x              | https://git-scm.com/downloads                           |

## Wallet Setup

Mining requires a **Taproot mining address** that receives block rewards.  
You must use **your own wallet** — do not hardcode a shared address.

### Option A — Interactive (recommended)

```bash
# From the repo root
cd bin && ./oystercli
```

Walk through **Create New Wallet** → save the seed phrase → **Receive** → copy the Taproot address.

### Option B — Command line

```bash
# 1. Create a new wallet
./bin/oyster --create
# Enter a passphrase. Record the seed phrase.

# 2. Start the wallet daemon (background)
./bin/oyster &

# 3. Generate a Taproot address
./bin/prlctl --wallet getnewaddress
# Output: prl1qxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Save the printed address. You will need it for `pearld --miningaddr` and for the gateway.

> **Note:** `--wallet` connects to the wallet RPC server on port 44207 (mainnet).  
> If your node is on a non-mainnet network, add the matching network flag (`--testnet`, `--simnet`, etc.).

## Build

### Clone and initialize

```bash
git clone https://github.com/shireff/pearl-cion.git
cd pearl
git submodule update --init --recursive
```

### Install Python dependencies

```bash
uv sync
```

This installs the workspace packages (`miner-base`, `pearl-gateway`, `pearl-gemm`, `miner-utils`, `py-pearl-mining`) and their dependencies.

> **Requires user input:** `vllm-miner` is listed in `pyproject.toml` as a workspace member but its directory is not present in this checkout. If `uv sync` fails due to the missing package, install only the required dependencies manually or restore the missing package.

### Build blockchain binaries

```bash
task build:blockchain
```

If `task` is not installed:

```bash
cd zk-pow
cargo run --release --no-default-features --bin build_cache \
  src/circuit/v2_cache.bin src/v1/v1_cache.bin
cd ..

go build -tags xmss,zkpow -o bin/pearld    ./node
go build -tags xmss,zkpow -o bin/prlctl   ./node/cmd/prlctl
go build -tags xmss,zkpow -o bin/oyster   ./wallet
CGO_ENABLED=0 go build   -o bin/oystercli ./wallet/cmd/oystercli
```

### Build CUDA kernels

```bash
cd miner/pearl-gemm
bash build.sh --no-pull
```

## Running the Stack

The stack consists of three components that must be started **in order**:

1. `pearld` — the full node
2. `pearl-gateway` — the bridge between the node and miners
3. `miner_gpu.py` — the GPU miner

Alternatively, `run_mining.py` can launch all three automatically (see Appendix).

### Terminal 1 — Start pearld

The gateway connects to `pearld` over plain HTTP by default.  
You must start `pearld` with TLS disabled or configure the gateway for HTTPS.

```bash
./bin/pearld \
  --rpcuser=rpcuser \
  --rpcpass=rpcpass \
  --rpclisten=0.0.0.0:44107 \
  --txindex \
  --notls \
  --miningaddr=<YOUR_WALLET_ADDRESS>
```

Verify it is running:

```bash
./bin/prlctl getinfo
# "blocks" should increment as the node syncs
```

> **Common failure:** If `pearl-gateway` logs `Failed to connect to Pearl node` or returns HTTP 401 / TLS errors, confirm that `pearld` was started with `--notls` and that the `rpcuser` / `rpcpass` match on both sides.

### Terminal 2 — Start pearl-gateway

Set the environment variables that tell the gateway how to reach `pearld`:

```bash
export PEARLD_RPC_URL="http://localhost:44107"
export PEARLD_RPC_USER="rpcuser"
export PEARLD_RPC_PASSWORD="rpcpass"
export PEARLD_MINING_ADDRESS="<YOUR_WALLET_ADDRESS>"
export PEARL_LOG_LEVEL="INFO"
```

Then start the gateway:

```bash
pearl-gateway start
```

Wait until the Unix socket appears:

```bash
# Linux / macOS
ls -la /tmp/pearlgw.sock

# Windows (if using TCP fallback)
# Set MINER_RPC_TRANSPORT=tcp and check port 8337 is listening
```

### Terminal 3 — Start the GPU miner

Run the standalone GPU miner:

```bash
python miner/miner_gpu.py
```

Or with `uv` (recommended if you use `uv` for the workspace):

```bash
uv run python miner/miner_gpu.py
```

The miner reads its configuration from two sources:

- **Gateway connection** — `MinerRpcConfig` (environment prefix `MINER_RPC_`)
- **Mining parameters** — `MinerSettings` (environment prefix `miner_`)

Defaults are sane: UDS at `/tmp/pearlgw.sock`, `noise_rank=256`, `tile_size_m=256`, `tile_size_n=1024`.

## Environment Variables

### pearl-gateway (Terminal 2)

| Variable | Default | Description |
|----------|---------|-------------|
| `PEARLD_RPC_URL` | `http://0.0.0.0:44107` | Pearl node JSON-RPC endpoint |
| `PEARLD_RPC_USER` | `user` | RPC username |
| `PEARLD_RPC_PASSWORD` | `pass` | RPC password |
| `PEARLD_MINING_ADDRESS` | `""` | Your Taproot mining address |
| `MINER_RPC_TRANSPORT` | `uds` | Transport for miner connections: `uds` or `tcp` |
| `MINER_RPC_SOCKET_PATH` | `/tmp/pearlgw.sock` | Unix socket path |
| `MINER_RPC_PORT` | `8337` | TCP port (used when `MINER_RPC_TRANSPORT=tcp`) |
| `MINER_RPC_HOST` | `localhost` | TCP bind host |
| `PEARL_LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARN`, `ERROR`, `NONE`) |

### GPU miner (Terminal 3)

| Variable | Default | Description |
|----------|---------|-------------|
| `miner_noise_range` | `128` | Noise range dimension |
| `miner_noise_rank` | `256` | Noise rank |
| `miner_tile_size_m` | `256` | GEMM tile M |
| `miner_tile_size_n` | `1024` | GEMM tile N |
| `miner_tile_size_k` | `256` | GEMM tile K |
| `PEARL_LOG_LEVEL` | `INFO` | Log verbosity |

## Verifying the Installation

### 1. Verify pearld RPC responds

```bash
curl --user rpcuser:rpcpass --data-binary '{"jsonrpc":"1.0","method":"getblockcount","params":[],"id":"1"}' -H 'content-type:text/plain;' http://localhost:44107/
```

Expected: a JSON response with the current block height.

### 2. Verify pearl-gateway starts successfully

After running `pearl-gateway start`, look for:

```
PearlGateway started successfully
```

Common failure: `Port 44107 already in use` or `Connection refused` — check that `pearld` is running and `--rpcuser` / `--rpcpass` match the gateway env vars.

### 3. Verify the miner connects successfully

After running `python miner/miner_gpu.py`, look for:

```
INFO:     Starting standalone GPU miner (no VLLM)
```

If you see:

```
ConnectionRefusedError: [Errno 111] Connection refused
```

The gateway socket has not appeared yet. Ensure `pearl-gateway start` is running and `/tmp/pearlgw.sock` exists.

### 4. Verify mining jobs are being received

Look for recurring log messages from the miner:

```
INFO:     Mining round completed
INFO:     Mining round failed: ...
```

Even failed rounds confirm that the miner is receiving jobs, computing hashes, and checking the target. If you see only the startup line and then silence, the gateway may not be delivering block templates — check `pearld` sync status and gateway logs.

## Port Reference

| Port / Socket | Service | Description |
| ------------- | ------- | ----------- |
| 44107 | pearld RPC | Full node JSON-RPC |
| 44108 | pearld P2P | Network peers |
| 44207 | oyster RPC | Wallet daemon |
| 8337 | pearl-gateway | Miner RPC (TCP fallback) |
| `/tmp/pearlgw.sock` | pearl-gateway | Unix Domain Socket (default miner transport) |

## Troubleshooting

### `uv sync` fails because `vllm-miner` is missing

`vllm-miner` is listed in `pyproject.toml` but the directory is absent from this checkout.  
Install only the available workspace packages, or restore the missing package before syncing.

```bash
uv sync --package pearl-gemm
```

If dependency resolution still fails, install the required runtime packages manually.

### CUDA kernel build fails

```bash
export PEARL_GEMM_FORCE_BUILD=TRUE
cd miner/pearl-gemm
bash build.sh --no-pull
```

Verify `nvcc --version` reports CUDA 13.0.

### `pearl-gateway` cannot connect to `pearld` (TLS error)

```bash
# Confirm pearld was started with --notls
ps aux | grep pearld
```

If you prefer TLS on the node, start the gateway with `PEARLD_RPC_URL=https://localhost:44107` and ensure the gateway has the correct CA certificate.

### `pearl-gateway` socket never appears

```bash
# Check gateway process
ps aux | grep pearl-gateway

# Check for port conflicts
lsof /tmp/pearlgw.sock   # Linux/macOS

# Restart with debug logs
PEARL_LOG_LEVEL=DEBUG pearl-gateway start
```

### Miner reports `ConnectionRefusedError`

```bash
# Ensure gateway is running and socket exists
ls -la /tmp/pearlgw.sock

# If using TCP (Windows or socket conflicts)
export MINER_RPC_TRANSPORT=tcp
export MINER_RPC_PORT=8337
python miner/miner_gpu.py
```

### Wallet / `getnewaddress` returns an error

```bash
# Make sure oyster daemon is running
ps aux | grep oyster

# Verify wallet RPC is accessible on port 44207
./bin/prlctl --wallet getinfo
```

## Appendix — `run_mining.py`

`miner/run_mining.py` is a production orchestrator that starts `pearld`, `pearl-gateway`, and the direct mining loop in a single process. It is used for automated deployments (e.g., Lightning AI).

```bash
python miner/run_mining.py
```

It hardcodes `PEARLD_RPC_URL`, `PEARLD_RPC_USER`, and `PEARLD_RPC_PASSWORD`. For custom deployments, prefer the manual three-terminal workflow documented above.

## Repository Structure

```
pearl/
├── node/                  → pearld, prlctl, oyster, oystercli
│   ├── sample-pearld.conf → pearld configuration template
│   └── doc.go             → CLI flag help text
├── wallet/                → oyster wallet daemon + CLI
│   └── cmd/
│       ├── prlctl/        → node CLI
│       └── oystercli/     → interactive wallet setup
├── zk-pow/                → ZK proof-of-work circuits (Rust)
├── miner/
│   ├── miner_gpu.py       → standalone GPU miner (current entry point)
│   ├── run_mining.py      → production orchestrator
│   ├── pearl-gateway/     → pearl-gateway bridge (pearl-gateway start)
│   ├── pearl-gemm/        → CUDA kernels
│   ├── miner-base/        → mining loop, block submission, settings
│   └── miner-utils/       → shared logging / utilities
└── py-pearl-mining/       → Python bindings (Rust/PyO3)
```
