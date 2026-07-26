"""
Default compiled kernels configuration.

Minimal set of kernels for development and PR CI.
This replaces default_compiled_kernels.jsonnet with Python + pydantic.
"""

from pearl_gemm_build_utils.kernel_configs import (
    KernelCompilationGrid,
    MatmulKernelConfig,
    NoisingAKernelConfig,
    NoisingBKernelConfig,
)

# Build matmul kernels
_matmul_kernels = []

# 128x256x128, R=64/128, stage=5 (Phase 6: Hopper TMA 5-stage pipeline)
for R in [64, 128]:
    _matmul_kernels.append(
        MatmulKernelConfig(
            tile_size_m=128,
            tile_size_n=256,
            tile_size_k=128,
            R=R,
            pipeline_stages=5,
            cM=1,
            cN=1,
        )
    )

# Also keep stage=3 as fallback for devices with < 228 KB SMEM
for R in [64, 128]:
    _matmul_kernels.append(
        MatmulKernelConfig(
            tile_size_m=128,
            tile_size_n=256,
            tile_size_k=128,
            R=R,
            pipeline_stages=3,
            cM=1,
            cN=1,
        )
    )

# Noising A: 64x64, fp16/int32
_noising_a_kernels = [
    NoisingAKernelConfig(
        tile_size_m=64,
        tile_size_k=64,
        R=R,
        pipeline_stages=2,
        AxEBL_type=dtype,
    )
    for R in [64, 128]
    for dtype in ["fp16", "int32"]
]

# Noising B: 64x64, fp16/int32
_noising_b_kernels = [
    NoisingBKernelConfig(
        tile_size_n=64,
        tile_size_k=64,
        R=R,
        pipeline_stages=2,
        EARxBpEB_type=dtype,
    )
    for R in [64, 128]
    for dtype in ["fp16", "int32"]
]

KERNEL_CONFIGS = KernelCompilationGrid(
    matmul_kernels=_matmul_kernels,
    noising_a_kernels=_noising_a_kernels,
    noising_b_kernels=_noising_b_kernels,
)
