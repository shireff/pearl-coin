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

echo "=== Step 3/5: Installing CUDA toolkit ==="
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

echo "=== Step 4/5: Installing Python dependencies ==="
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy packaging wheel ninja

echo "=== Step 5/5: Building pearl-gemm CUDA extension ==="
cd "$(dirname "$0")/miner/pearl-gemm"
export TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK=1
export MAX_JOBS=2
export PEARL_GEMM_ARCH="arch=compute_89,code=sm_89"
python setup.py build_ext --inplace

echo "=== Verifying build ==="
python -c "import pearl_gemm_cuda; print('CUDA extension OK')"

echo "=== DONE ==="
