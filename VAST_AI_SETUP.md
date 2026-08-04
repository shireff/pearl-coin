# Vast.ai Instance Setup for Pearl Mining

## Instance Requirements

- GPU: RTX 4060 or better
- CUDA: 12.x
- RAM: 16GB+
- Disk: 30GB+
- Image: PyTorch with CUDA (pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel)

## First-Time Setup

```bash
# 1. Clone repo
git clone https://github.com/shireff/pearl-coin ~/pearl-cion
cd ~/pearl-cion

# 2. Create venv
python3 -m venv /workspace/pearl-venv
source /workspace/pearl-venv/bin/activate

# 3. Install dependencies
cd miner && pip install -r requirements.txt 2>/dev/null || pip install torch loguru

# 4. Build CUDA kernel
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK=1 MAX_JOBS=4
cd pearl-gemm && rm -rf build src/pearl_gemm_cuda*.so
python setup.py build_ext --inplace

# 5. Verify
python3 -c "import sys; sys.path.insert(0,'src'); from pearl_gemm import alph_mine_batch; print('Kernel OK')"
```

## Start Mining (after setup)

```bash
source /workspace/pearl-venv/bin/activate
cd ~/pearl-cion
# See MINING_GUIDE.md for pool commands
```

## After Instance Restart

```bash
source /workspace/pearl-venv/bin/activate
cd ~/pearl-cion && git pull
# Kernel .so persists — no rebuild needed unless kernel changed
# Start miner directly (see MINING_GUIDE.md)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named pearl_gemm` | Rebuild kernel |
| `Failed to subscribe to pool` | Check `--pool-algorithm` matches pool |
| `GPU round time: 0.3ms` | Wrong algorithm — Pearl session running instead of Alephium |
| Dashboard shows 0 H/s | Wait 10 min after first share |
| Worker offline | Check pool URL and username format |
