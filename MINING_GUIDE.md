# Pearl (PRL) Mining Guide

## Quick Start

### PRL Pool (Kryptex)
```bash
cd ~/pearl-cion
pkill -f miner_gpu.py; sleep 1
nohup env PEARL_LOG_LEVEL=DEBUG python miner/miner_gpu.py \
  --pool-url stratum+tcp://prl.kryptex.network:7048 \
  --pool-username krxYM43GWD.worker \
  --pool-password x \
  --pool-algorithm pearl \
  --noise-range 256 --P 8 --Q 8 \
  > /tmp/miner_live.log 2>&1 & echo "PID: $!"
```

### ALPH Pool (Kryptex)
```bash
cd ~/pearl-cion
pkill -f miner_gpu.py; sleep 1
nohup env PEARL_LOG_LEVEL=DEBUG python miner/miner_gpu.py \
  --pool-url stratum+tcp://alph.kryptex.network:7010 \
  --pool-username 3cUrDWnDxgFVy1V8dEWjWR4npf5EcJKESoDNjL2Nmn4mssdW7n2Ui.worker \
  --pool-password x \
  --pool-algorithm alephium \
  --noise-range 256 --P 8 --Q 8 \
  > /tmp/miner_live.log 2>&1 & echo "PID: $!"
```

## Rebuild CUDA Kernel

```bash
source /workspace/pearl-venv/bin/activate
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK=1 MAX_JOBS=4
cd ~/pearl-cion/miner/pearl-gemm && rm -rf build src/pearl_gemm_cuda*.so
python setup.py build_ext --inplace
```

### Verify rebuild:
```bash
stat -c '%y' ~/pearl-cion/miner/pearl-gemm/src/pearl_gemm_cuda*.so
python3 -c "import sys; sys.path.insert(0,'src'); from pearl_gemm import alph_mine_batch; print('OK')"
```

## Monitor Live

```bash
tail -f /tmp/miner_live.log | strings | grep -E "WINNER|submit|SHARE|Error|closed|difficulty|ACCEPTED|REJECTED"
```

## Key Files

| File | Purpose |
|------|---------|
| `miner/miner_gpu.py` | Main miner entry point |
| `miner/stratum_client.py` | Pool protocol (Stratum) |
| `miner/pool_mining_client.py` | Pool connection manager |
| `miner/pearl-gemm/csrc/gemm/alph_mining_kernel.cu` | CUDA Blake3 kernel |
| `TODO.md` | Performance optimization tasks |

## Protocol Differences

| Feature | PRL Pool | ALPH Pool |
|---------|----------|-----------|
| mining.hello | No | Yes |
| subscribe params | `[]` | `["alephium-gpu-miner/1.0.0", "AlephiumStratum/1.0.0"]` |
| notify params format | dict `{header, job_id, target, height}` | list `[{jobId, headerBlob, targetBlob, ...}]` |
| submit response | yes | fire-and-forget (no response) |
