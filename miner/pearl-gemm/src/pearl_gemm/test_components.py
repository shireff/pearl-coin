"""
Test Components Module for pearl_gemm

This module provides Python bindings for testing individual components
of the pearl_gemm library, including the inner_hash function.
"""

from importlib.util import find_spec

import torch

_VLLM_AVAILABLE = find_spec("vllm") is not None
if _VLLM_AVAILABLE:
    from vllm import _custom_ops as vllm_ops


def gpu_jackpot_hash(jackpots: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """
    Compute BLAKE3 keyed hashes for jackpot arrays on GPU.

    This function computes blake3(jackpot, key) for each jackpot array,
    using the fixed CUDA BLAKE3 keyed-hash implementation.

    Args:
        jackpots (torch.Tensor): 3D tensor of uint32 jackpot arrays,
            shape (num_jobs, num_combos, 16), device: CUDA
        keys (torch.Tensor): 2D tensor of uint32 keys,
            shape (num_jobs, 8), device: CUDA

    Returns:
        torch.Tensor: 3D tensor of uint32 hashes,
            shape (num_jobs, num_combos, 8), device: CUDA
    """
    return torch.ops.pearl_gemm.gpu_jackpot_hash(jackpots, keys)


def blake3_keyed_hash(messages: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """
    Compute single-block BLAKE3 keyed hash on GPU.

    This function computes blake3(msg, key) for each 64-byte message block,
    matching the CPU blake3 crate's keyed hash mode.

    Args:
        messages (torch.Tensor): 2D tensor of uint32 messages, shape (N, 16), device: CUDA
        key (torch.Tensor): 1D tensor of uint32 key, shape (8,), device: CUDA

    Returns:
        torch.Tensor: 2D tensor of uint32 outputs, shape (N, 8), device: CUDA
    """
    return torch.ops.pearl_gemm.blake3_keyed_hash(messages, key)


def inner_hash(input_buffer: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    """
    Compute the inner hash of a uint32 buffer.

    This function performs a specialized hash computation on a uint32 array
    using the inner_hash CUDA kernel. It applies a multi-stage MAD (multiply-add) tree
    followed by a finalization step to produce a 3-element uint16 hash.

    Args:
        input_buffer (torch.Tensor): A 1D tensor of uint32 values on CUDA device.
                                   Shape: (64,), (128,), (192,), or (256,), dtype: torch.uint32, device: CUDA
        iterations (int): Number of iterations to run the hash computation. Default: 1

    Returns:
        torch.Tensor: A 1D tensor of 3 uint16 values representing the hash.
                     Shape: (3,), dtype: torch.uint16, device: CUDA


    Raises:
        RuntimeError: If input tensor is not on CUDA device, not contiguous,
                     wrong dtype, wrong shape, or wrong number of dimensions.
    """
    return torch.ops.pearl_gemm.inner_hash(input_buffer, iterations)


def tensor_hash(
    data: torch.Tensor, key: torch.Tensor, out: torch.Tensor, roots: torch.Tensor
) -> torch.Tensor:
    """
    Compute the tensor hash of a 2D tensor.
    """
    return torch.ops.pearl_gemm.tensor_hash(data, key, out, roots)  # type: ignore


def commitment_hash_from_merkle_roots(
    A_merkle_root: torch.Tensor,
    B_merkle_root: torch.Tensor,
    key: torch.Tensor,
    commitment_hash_A: torch.Tensor,
    commitment_hash_B: torch.Tensor,
    routing_root: torch.Tensor | None = None,
    offsets_hash: torch.Tensor | None = None,
):
    """
    Compute the commitment hash from merkle roots of a 2D tensor.

    routing_root and offsets_hash are optional MoE inputs (provided together)
    that fold the routing commitment into A's seed; omitting both is the dense
    case.
    """
    return torch.ops.pearl_gemm.commitment_hash_from_merkle_roots(
        A_merkle_root,
        B_merkle_root,
        key,
        commitment_hash_A,
        commitment_hash_B,
        routing_root,
        offsets_hash,
    )  # type: ignore


def vllm_gemm(A, B_t, scale_a, scale_b, out_dtype=torch.bfloat16):
    """
    vLLM's CUTLASS scaled INT8 GEMM.

    Computes: output = (A @ B_t) * scale_a * scale_b

    Requires vllm package (pip install vllm) and torch>=2.8.0.

    Args:
        A: Input matrix A, shape (M, K), dtype int8
        B_t: Transposed input matrix B, shape (K, N), dtype int8
             Pass B.t() where B has shape (N, K)
        scale_a: Per-row scales for A, shape (M,), dtype float32
        scale_b: Per-row scales for B, shape (N,), dtype float32
        out_dtype: Output dtype (default: torch.bfloat16)

    Returns:
        Output tensor, shape (M, N), dtype out_dtype
    """
    if not _VLLM_AVAILABLE:
        raise RuntimeError(
            "vLLM not available. Install with: pip install vllm (requires torch>=2.8.0)"
        )
    return vllm_ops.cutlass_scaled_mm(
        A,
        B_t,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
        bias=None,
    )


# Make the main function available at module level
__all__ = [
    "gpu_jackpot_hash",
    "blake3_keyed_hash",
    "inner_hash",
    "tensor_hash",
    "commitment_hash_from_merkle_roots",
]
