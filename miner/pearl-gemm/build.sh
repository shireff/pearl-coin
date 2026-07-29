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

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK=1
export PEARL_GEMM_ARCH="arch=compute_89,code=sm_89"
export MAX_JOBS="$JOBS"

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
rm -rf build *.so 2>/dev/null || true

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
ls -lh "$GEMM_DIR"/*.so 2>/dev/null && echo "Extension built successfully." || echo "WARNING: no .so found"
