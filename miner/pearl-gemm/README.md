# pearl-gemm

CUDA kernels for Pearl GEMM with noising/denoising and PoW extraction.

## Installation

```bash
pip install -e . --no-build-isolation
python setup.py build_ext --inplace
```

Or for a regular install:

```bash
pip install . --no-build-isolation
```

Requires CUDA 13.0+, Python 3.12+, and a GPU with compute capability `sm_90a` (H100).

If you are building on an H100 system, set:

```bash
export CUDA_HOME=/usr/local/cuda-13.x
export PEARL_GEMM_ARCH="arch=compute_90,code=sm_90a"
```
