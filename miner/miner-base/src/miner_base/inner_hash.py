from dataclasses import dataclass

import numpy as np
import torch


def xor_reduction(inputs: torch.Tensor) -> np.uint32:
    """
    XOR all uint32 elements in the input tensor.
    Returns a single uint32 value.
    """
    assert inputs.dtype == torch.int32, f"Expected int32, got {inputs.dtype}"
    arr = inputs.flatten().numpy().view(np.uint32)
    return np.bitwise_xor.reduce(arr)


@dataclass
class InnerHashResult:
    hash: np.uint32
    index: tuple[int, int]


def hash_tile(
    tensor: torch.Tensor,
    index: tuple[int, int] | None = None,
) -> InnerHashResult:
    if tensor.dtype != torch.int32:
        raise ValueError(f"Tensor dtype must be int32, got {tensor.dtype}")

    final_hash = xor_reduction(tensor)

    return InnerHashResult(hash=final_hash, index=index if index is not None else (0, 0))


class InnerHasher:
    def __init__(self, tile_h: int, tile_w: int):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.num_tiles_hashed = 0

    def hash_tensor(self, tensor: torch.Tensor) -> list[InnerHashResult]:
        if tensor.dtype != torch.int32:
            raise ValueError(f"Tensor dtype must be int32, got {tensor.dtype}")

        if tensor.shape[0] < self.tile_h or tensor.shape[1] < self.tile_w:
            raise ValueError(
                f"Tensor must have shape of at least ({self.tile_h}, {self.tile_w}), got {tensor.shape}"
            )

        num_tiles_h = tensor.shape[0] // self.tile_h
        num_tiles_w = tensor.shape[1] // self.tile_w

        self.num_tiles_hashed += num_tiles_h * num_tiles_w

        H = num_tiles_h * self.tile_h
        W = num_tiles_w * self.tile_w
        tiles_flat = (
            tensor[:H, :W]
            .reshape(num_tiles_h, self.tile_h, num_tiles_w, self.tile_w)
            .permute(0, 2, 1, 3)
            .reshape(-1, self.tile_h * self.tile_w)
        )
        tiles_np = tiles_flat.numpy().view(np.uint32)
        hash_values = np.bitwise_xor.reduce(tiles_np, axis=1)

        row_idx = np.repeat(np.arange(num_tiles_h), num_tiles_w)
        col_idx = np.tile(np.arange(num_tiles_w), num_tiles_h)
        return [
            InnerHashResult(hash=np.uint32(h), index=(int(r), int(c)))
            for h, r, c in zip(hash_values, row_idx, col_idx)
        ]
