# Pearl Mining — Setup Guide

This guide uses the current repository source as the single source of truth.  
Obsolete paths (VLLM plugin, `vllm-miner` Docker image, `noisy_gemm` vLLM hooks) have been removed because the `vllm-miner` package directory is absent from this checkout.

## How Mining Works

```
pearld (full node)
    ↓ JSON-RPC (port 44107)
pearl-gateway start            ← fetches block template, distributes work
    ↓ Unix socket /tmp/pearlgw.sock (or TCP 8337)
miner_gpu.py                   ← standalone GPU miner
    ↓ CUDA kernels (pearl_gemm)
gpu_mine_batch / gpu_jackpot_hash
    ↓ proofs submitted via pearl-gateway → pearld
```

Three components must run in order:
1. **pearld** — the full node
2. **pearl-gateway** — the bridge between the node and miners
3. **miner_gpu.py** — the standalone GPU miner

`miner_gpu.py` does **not** connect to `pearld` directly. It connects to `pearl-gateway` over a local Unix socket or TCP.

## Prerequisites

### Hardware

| Requirement | Details |
| ----------- | ------- |
| GPU         | NVIDIA with CUDA — **sm_90a** (H100/H200) or **sm_120** (RTX 5090) |
| RAM         | 16 GB minimum, 32 GB recommended |
| Storage     | 50 GB free space |
| OS          | Ubuntu 22.04+ (recommended) or Windows 11 |

### Software

| Tool         | Required Version | Source |
| ------------ | ---------------- | ------ |
| Python       | **3.12.x**       | https://www.python.org/downloads/release/python-3120/ |
| uv           | Latest           | https://astral.sh/uv/ |
| CUDA Toolkit | **13.0**         | https://developer.nvidia.com/cuda-13-0-download-archive |
| Rust         | stable           | https://rustup.rs/ |
| Go           | 1.26+            | https://go.dev/dl/ |
| Git          | 2.x              | https://git-scm.com/downloads |

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

This installs `miner-base`, `pearl-gateway`, `pearl-gemm`, `miner-utils`, and `py-pearl-mining`.

> **Note:** `vllm-miner` is listed in `pyproject.toml` as a workspace member, but its directory is absent from this checkout. If `uv sync` fails, install only the available packages or restore the missing package.

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

## Running the Stack

The stack must be started **in order**.

### 1. Start pearld

Start the full node with TLS disabled (the gateway connects over plain HTTP by default):

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

### 2. Start the miner — this automatically starts pearl-gateway

If `pearld` is already running, `miner_gpu.py` can manage `pearl-gateway` for you. Just pass the pearl node connection details:

```bash
python miner/miner_gpu.py \
  --rpc-url http://127.0.0.1:44107 \
  --rpc-user rpcuser \
  --rpc-password rpcpass \
  --mining-address <YOUR_WALLET_ADDRESS>
```

`miner_gpu.py` starts `pearl-gateway` as a background process, waits for the gateway socket to appear, then begins mining.

If you prefer to run `pearl-gateway` manually in a separate terminal, use its documented command instead:

```bash
pearl-gateway start
```

### Environment Variables

All values below can be passed either as CLI flags to `miner_gpu.py` or as environment variables.

| Variable / Flag | Default | Description |
|----------|---------|-------------|
| `PEARLD_RPC_URL` / `--rpc-url` | `http://0.0.0.0:44107` | Pearl node JSON-RPC endpoint |
| `PEARLD_RPC_USER` / `--rpc-user` | `user` | RPC username |
| `PEARLD_RPC_PASSWORD` / `--rpc-password` | `pass` | RPC password |
| `PEARLD_MINING_ADDRESS` / `--mining-address` | `""` | Your Taproot mining address |
| `PEARL_LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARN`, `ERROR`, `NONE`) |
| `MINER_RPC_SOCKET_PATH` / `--gateway-socket` | `/tmp/pearlgw.sock` | Gateway Unix socket |
| `MINER_RPC_TRANSPORT` / `--gateway-tcp` | `uds` | Gateway transport (`uds` or `tcp` via `--gateway-tcp`) |
| `MINER_RPC_PORT` / `--gateway-port` | `8337` | Gateway TCP port |
| `MINER_RPC_HOST` / `--gateway-host` | `localhost` | Gateway TCP host |
| `miner_noise_rank` / `--noise-rank` | `256` | Noise rank dimension |
| `miner_noise_range` / `--noise-range` | `128` | Noise range dimension |
| `miner_tile_size_m` / `--tile-m` | `256` | GEMM tile M |
| `miner_tile_size_n` / `--tile-n` | `1024` | GEMM tile N |
| `miner_tile_size_k` / `--tile-k` | `256` | GEMM tile K |

## Verifying the Installation

### 1. Verify pearld RPC responds

```bash
curl --user rpcuser:rpcpass \
  --data-binary '{"jsonrpc":"1.0","method":"getblockcount","params":[],"id":"1"}' \
  -H 'content-type:text/plain;' \
  http://localhost:44107/
```

Expected: a JSON response with the current block height.

### 2. Verify pearl-gateway starts successfully

If you started `pearl-gateway` manually, look for:

```
PearlGateway started successfully
```

If `miner_gpu.py` started it for you, look for:

```
INFO: Starting managed pearl-gateway process...
INFO: Gateway socket ready: /tmp/pearlgw.sock
```

Common failure: `Port 44107 already in use` or `Connection refused` — check that `pearld` is running and `--rpcuser` / `--rpcpass` match.

### 3. Verify the miner connects successfully

After running `python miner/miner_gpu.py`, look for:

```
INFO: Starting standalone GPU miner (no vLLM)
INFO: Gateway: uds:///tmp/pearlgw.sock
INFO: Mining params: noise_range=128, noise_rank=256, tile=(256,1024,256)
```

If you see:

```
ConnectionRefusedError: [Errno 111] Connection refused
```

The gateway socket has not appeared yet. Ensure `pearl-gateway` is running and `/tmp/pearlgw.sock` exists.

### 4. Verify mining jobs are being received

Look for recurring log messages from the miner:

```
INFO: Mining round completed
INFO: Mining round failed: ...
```

Even failed rounds confirm that the miner is receiving jobs, computing hashes, and checking the target. If you see only the startup line and then silence, check `pearld` sync status and gateway logs.

## Port Reference

| Port / Socket | Service | Description |
| ------------- | ------- | ----------- |
| 44107 | pearld RPC | Full node JSON-RPC |
| 44108 | pearld P2P | Network peers |
| 44207 | oyster RPC | Wallet daemon |
| 8337 | pearl-gateway | Miner RPC (TCP fallback) |
| `/tmp/pearlgw.sock` | pearl-gateway | Unix Domain Socket (default miner transport) |

## Troubleshooting

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

# Check for socket conflicts
lsof /tmp/pearlgw.sock   # Linux/macOS

# Restart with debug logs
PEARL_LOG_LEVEL=DEBUG pearl-gateway start
```

### Miner reports `ConnectionRefusedError`

```bash
# Ensure pearld is running and the miner can reach it
curl --user rpcuser:rpcpass http://127.0.0.1:44107

# Ensure gateway socket exists
ls -la /tmp/pearlgw.sock
```

If the socket is missing, either start `pearl-gateway` manually or pass `--rpc-url` to `miner_gpu.py` so it starts the gateway for you.

### Wallet / `getnewaddress` returns an error

```bash
# Make sure oyster daemon is running
ps aux | grep oyster

# Verify wallet RPC is accessible on port 44207
./bin/prlctl --wallet getinfo
```

## Appendix — `run_mining.py`

`miner/run_mining.py` is a production orchestrator that starts `pearld`, `pearl-gateway`, and a direct mining loop in a single process. It is used for automated deployments.

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
