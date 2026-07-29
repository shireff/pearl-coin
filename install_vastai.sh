#!/bin/bash
set -e

echo "=== Pearl GEMM Installer for vast.ai (RTX 4060 Ti) ==="

echo "=== Step 1/5: Installing system dependencies ==="
apt-get update -qq && apt-get install -y -qq ninja-build curl

echo "=== Step 2/5: Checking GPU ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || {
    echo "ERROR: nvidia-smi not found. Is this a GPU instance?"
    exit 1
}

echo "=== Step 3/5: Checking CUDA toolkit (nvcc) ==="
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA toolkit..."
    apt-get install -y -qq cuda-toolkit-12-4 || apt-get install -y -qq cuda-toolkit-13-0 || apt-get install -y -qq cuda-toolkit || {
        echo "ERROR: Failed to install CUDA toolkit."
        exit 1
    }
    export PATH="/usr/local/cuda/bin:$PATH"
fi

nvcc --version | grep release

echo "=== Step 4/5: Installing Python dependencies ==="
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy packaging wheel ninja

echo "=== Step 5/5: Building pearl-gemm CUDA extension ==="
cd "$(dirname "$0")/miner/pearl-gemm"
export MAX_JOBS=2
python setup.py build_ext --inplace

echo "=== Verifying build ==="
python -c "import pearl_gemm_cuda; print('CUDA extension OK')"

echo "=== DONE ==="
