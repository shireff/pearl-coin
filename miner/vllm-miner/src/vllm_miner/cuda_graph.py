"""CUDA Graph capture for mining pipeline optimization.

Captures the GPU mining pipeline as a CUDA Graph to eliminate CPU launch
overhead for repeated jobs.
"""

from __future__ import annotations

from typing import Any

import torch

from miner_utils import get_logger

_LOGGER = get_logger("vllm.pearl_miner.cuda_graph")


class CudaGraphCapture:
    """Manages CUDA Graph capture for mining pipeline."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.graph: torch.cuda.CUDAGraph | None = None
        self.stream = torch.cuda.Stream(device=device)
        self.is_captured = False

    def capture(
        self,
        mining_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Capture the mining function as a CUDA Graph.

        Args:
            mining_fn: The mining function to capture.
            *args: Positional arguments for the mining function.
            **kwargs: Keyword arguments for the mining function.
        """
        if self.is_captured:
            _LOGGER.warning("CUDA Graph already captured, skipping")
            return

        if not torch.cuda.is_available():
            _LOGGER.warning("CUDA not available, skipping graph capture")
            return

        try:
            with torch.cuda.stream(self.stream):
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    mining_fn(*args, **kwargs)

            self.is_captured = True
            _LOGGER.info("CUDA Graph captured successfully")
        except Exception as e:
            _LOGGER.error(f"Failed to capture CUDA Graph: {e}")
            self.graph = None
            self.is_captured = False

    def replay(self) -> None:
        """Replay the captured CUDA Graph."""
        if not self.is_captured or self.graph is None:
            _LOGGER.warning("No CUDA Graph captured, skipping replay")
            return

        try:
            self.graph.replay()
        except Exception as e:
            _LOGGER.error(f"Failed to replay CUDA Graph: {e}")

    def reset(self) -> None:
        """Reset the captured CUDA Graph."""
        self.graph = None
        self.is_captured = False
