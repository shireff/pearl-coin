#!/bin/bash
set -e

echo "=== Pearl GEMM Installer for vast.ai (RTX 4060 Ti) ==="

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "=== Step 0/6: Pulling latest changes ==="
git stash || true
git pull origin main || echo "WARNING: git pull failed, continuing with current code"

echo "=== Step 1/6: Installing system dependencies ==="
apt-get update -qq && apt-get install -y -qq ninja-build curl

echo "=== Step 2/6: Checking GPU ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || {
    echo "ERROR: nvidia-smi not found. Is this a GPU instance?"
    exit 1
}

echo "=== Step 3/6: Installing CUDA toolkit ==="
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA 12.4 toolkit..."
    apt-get install -y -qq cuda-nvcc-12-4 cuda-cudart-12-4 cuda-cudart-dev-12-4 cuda-nvtx-12-4 \
        libcurand-dev-12-4 libcublas-dev-12-4 libcusparse-dev-12-4 libcusolver-dev-12-4 || {
        echo "ERROR: Failed to install CUDA 12.4 toolkit."
        exit 1
    }
    ln -sf /usr/local/cuda-12.4 /usr/local/cuda
fi

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

nvcc --version | grep release

echo "=== Step 4/6: Installing Python dependencies ==="
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy packaging wheel ninja
pip install -e "$REPO_DIR/miner/pearl-gemm-build-utils" --no-build-isolation

echo "=== Step 5/6: Building pearl-gemm CUDA extension ==="
cd "$REPO_DIR/miner/pearl-gemm"
export TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK=1
export MAX_JOBS=2
export PEARL_GEMM_ARCH="arch=compute_89,code=sm_89"
python setup.py build_ext --inplace

echo "=== Step 6/6: Verifying build ==="
python -c "import pearl_gemm_cuda; print('CUDA extension OK')"

echo "=== DONE ==="
