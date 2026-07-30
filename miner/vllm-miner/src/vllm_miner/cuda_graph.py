"""CUDA Graph capture for mining pipeline optimization.

Captures the fixed GPU mining pipeline (hash A/B, merkle, noise_gen,
noisy_gemm) as a CUDA Graph to eliminate CPU launch overhead for repeated
jobs.  Dynamic operations (get_mining_job, acquire_pinned_buffer,
schedule_status_check) are kept outside the graph and executed on every
replay.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from miner_utils import get_logger

_LOGGER = get_logger("vllm.pearl_miner.cuda_graph")


class CudaGraphCapture:
    """Manages CUDA Graph capture for the fixed portion of the mining pipeline.

    Only the GPU-only, shape-invariant portion is captured:
        hash_A / hash_B → merkle → noise_gen → noisy_gemm

    Dynamic host-side operations (mining job fetch, pinned buffer
    acquisition, status callbacks) must be called outside the graph and
    are not captured.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.graph: torch.cuda.CUDAGraph | None = None
        self.stream = torch.cuda.Stream(device=device)
        self.is_captured = False
        self._key: tuple | None = None

    def capture(
        self,
        fixed_fn: Callable,
        *args: Any,
        key: tuple | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture the fixed GPU pipeline as a CUDA Graph.

        Args:
            fixed_fn: A callable that runs only the GPU-fixed portion
                (hash A/B, merkle, noise_gen, noisy_gemm).  It must not
                perform any host-side dynamic operations.
            key:  A tuple that identifies the capture configuration.
                  If the key matches a previous capture the graph is
                  reused.  If the key changes the old graph is discarded
                  and a new capture is performed.
            *args / **kwargs: Passed to fixed_fn during capture and replay.
        """
        if key is not None and self._key == key and self.is_captured:
            return

        if not torch.cuda.is_available():
            _LOGGER.warning("CUDA not available, skipping graph capture")
            return

        try:
            with torch.cuda.stream(self.stream):
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    fixed_fn(*args, **kwargs)

            self._key = key
            self.is_captured = True
            _LOGGER.info("CUDA Graph captured successfully (key=%s)", key)
        except Exception as e:
            _LOGGER.error("Failed to capture CUDA Graph: %s", e)
            self.graph = None
            self.is_captured = False
            self._key = None

    def replay(self, fixed_fn: Callable, *args: Any, **kwargs: Any) -> None:
        """Replay the captured CUDA Graph.

        Dynamic host-side operations (mining job fetch, pinned buffer
        acquisition, status callbacks) must be called separately before
        and after replay.
        """
        if not self.is_captured or self.graph is None:
            _LOGGER.warning("No CUDA Graph captured, skipping replay")
            return

        try:
            self.graph.replay()
        except Exception as e:
            _LOGGER.error("Failed to replay CUDA Graph: %s", e)

    def reset(self) -> None:
        """Reset the captured CUDA Graph."""
        if self.graph is not None:
            del self.graph
        self.graph = None
        self.is_captured = False
        self._key = None
