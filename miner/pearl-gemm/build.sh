#!/usr/bin/env bash
# pearl-gemm build script
# Usage: bash build.sh [--no-pull] [--jobs N]
set -euo pipefail

PULL=true
JOBS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-pull) PULL=false; shift ;;
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "${CUDA_HOME:-}" ]; then
    TORCH_CUDA="$(python -c 'import torch; print(getattr(torch.version, "cuda", ""))' 2>/dev/null || true)"
    candidates=()
    if [ -n "${TORCH_CUDA:-}" ]; then
        candidates+=( "/usr/local/cuda-${TORCH_CUDA}" "/usr/local/cuda-${TORCH_CUDA%.*}" )
    fi
    candidates+=( /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-13.1 /usr/local/cuda-13.2 /usr/local/cuda-14.0 /usr/local/cuda-* )
    for candidate in "${candidates[@]}"; do
        if [ -x "$candidate/bin/nvcc" ]; then
            CUDA_HOME="$candidate"
            break
        fi
    done
fi

if [ -z "${CUDA_HOME:-}" ]; then
    echo "ERROR: CUDA_HOME is not set and no supported CUDA toolkit was found."
    echo "Install CUDA 13.x and export CUDA_HOME to your actual CUDA install path."
    echo "For example: export CUDA_HOME=/usr/local/cuda-13.2"
    exit 1
fi

CUDA_VERSION="$($CUDA_HOME/bin/nvcc -V 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')"
if [ -z "${CUDA_VERSION:-}" ]; then
    echo "ERROR: Could not determine CUDA version from $CUDA_HOME/bin/nvcc"
    exit 1
fi

if printf '%s\n' "$CUDA_VERSION" "13.0" | sort -V | head -n1 | grep -qv "13.0"; then
    echo "ERROR: CUDA 13.0+ is required for pearl-gemm, but found CUDA $CUDA_VERSION"
    exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK=1
export PEARL_GEMM_ARCH="arch=compute_89,code=sm_89"
export MAX_JOBS="$JOBS"
# Override container-level NVCC_PREPEND_FLAGS that points ccache as host compiler
export NVCC_PREPEND_FLAGS=""
export CUDAHOSTCXX=/usr/bin/g++
export CXX=/usr/bin/g++
export CC=/usr/bin/gcc

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GEMM_DIR="$REPO_ROOT/miner/pearl-gemm"

echo "=== pearl-gemm build ==="
echo "CUDA_HOME  : $CUDA_HOME"
echo "MAX_JOBS   : $MAX_JOBS"
echo "GEMM_DIR   : $GEMM_DIR"

if $PULL; then
    echo ""
    echo "[1/3] Pulling latest changes..."
    cd "$REPO_ROOT"
    git pull origin main
fi

echo ""
echo "[2/3] Cleaning previous build..."
cd "$GEMM_DIR"
rm -rf build src/*.so 2>/dev/null || true

echo ""
echo "[3/3] Building..."
START=$(date +%s)
python setup.py build_ext --inplace > /tmp/pearl_build.log 2>&1 || {
    echo "BUILD FAILED. Last 40 lines of log:"
    tail -40 /tmp/pearl_build.log
    exit 1
}
END=$(date +%s)

echo ""
echo "=== Build complete in $((END - START))s ==="
find "$GEMM_DIR" -name "*.so" 2>/dev/null | head -5 && echo "Extension built successfully." || echo "WARNING: no .so found"
