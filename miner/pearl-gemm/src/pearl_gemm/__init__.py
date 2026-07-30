"""
pearl_gemm package

Provides CUDA kernels for Pearl GEMM with noising/denoising and PoW extraction.

The compiled CUDA extension (pearl_gemm_cuda) is built during installation.
If you see ModuleNotFoundError for 'pearl_gemm_cuda', the extension was not compiled.
Run:  pip install -e /path/to/pearl-gemm --no-build-isolation
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys
from pathlib import Path

_MISSING_EXT_MSG = (
    "The compiled CUDA extension 'pearl_gemm_cuda' was not found.\n"
    "\n"
    "If you installed with `pip install -e .`, build the extension separately:\n"
    "\n"
    "  cd /path/to/pearl-gemm && python setup.py build_ext --inplace\n"
    "\n"
    "Otherwise, reinstall normally to compile during wheel build:\n"
    "\n"
    "  pip install . --no-build-isolation\n"
    "\n"
    "Original error: {original}"
)


def _find_cuda_extension():
    """Locate and import pearl_gemm_cuda, searching common locations."""
    import glob as _glob

    candidates = [
        "pearl_gemm.pearl_gemm_cuda",
        "pearl_gemm_cuda",
    ]
    for name in candidates:
        try:
            return _importlib.import_module(name)
        except ModuleNotFoundError:
            pass

    root = Path(__file__).resolve().parent
    search_roots = [
        root,
        root.parent,
        root / "csrc",
        ROOT_DIR if "ROOT_DIR" in globals() else root.parent.parent,
    ]
    patterns = [
        "pearl_gemm_cuda*.so",
        "pearl_gemm_cuda*.pyd",
    ]
    for search_root in search_roots:
        for pattern in patterns:
            matches = _glob.glob(str(search_root / "**" / pattern), recursive=True)
            if matches:
                for match in matches:
                    match_dir = Path(match).parent
                    if str(match_dir) not in _sys.path:
                        _sys.path.insert(0, str(match_dir))
                try:
                    return _importlib.import_module("pearl_gemm_cuda")
                except ModuleNotFoundError:
                    pass

    return None


def _load_cuda_extension():
    """Import pearl_gemm_cuda and raise a clear error if missing."""
    cuda_ext = _find_cuda_extension()
    if cuda_ext is not None:
        return cuda_ext
    raise ModuleNotFoundError(
        _MISSING_EXT_MSG.format(original="No module named 'pearl_gemm_cuda'")
    )


# Fail fast and clearly if the .so is missing — before any sub-module import
# attempts the same and produces a cryptic traceback.
_cuda_ext = _load_cuda_extension()
_sys.modules.setdefault("pearl_gemm.pearl_gemm_cuda", _cuda_ext)

# ---------------------------------------------------------------------------
# Re-export pearl_gemm_cuda symbols for the public API
# ---------------------------------------------------------------------------

HostSignalStatus = _cuda_ext.HostSignalStatus
get_host_signal_header = _cuda_ext.get_host_signal_header
get_host_signal_header_size = _cuda_ext.get_host_signal_header_size
get_host_signal_sync_size = _cuda_ext.get_host_signal_sync_size
get_required_scratchpad_bytes = _cuda_ext.get_required_scratchpad_bytes
kEALScaleFactorDenoise = _cuda_ext.kEALScaleFactorDenoise
kEBRScaleFactorDenoise = _cuda_ext.kEBRScaleFactorDenoise

# ---------------------------------------------------------------------------
# Sub-module imports — safe now that pearl_gemm_cuda is confirmed present
# ---------------------------------------------------------------------------

from . import pearl_gemm_interface  # noqa: E402
from .helpers import (  # noqa: E402
    HostSignalHeaderPinnedPool,
    ProofTileIndices,
    extract_indices,
    make_pow_target_tensor,
)
from .pearl_gemm_interface import (  # noqa: E402
    commitment_hash_from_merkle_roots,
    denoise_converter,
    gemm,
    noise_A,
    noise_B,
    noise_gen,
    noisy_gemm,
    run_tensor_hash,
    tensor_hash,
)

__all__ = [
    "HostSignalHeaderPinnedPool",
    "HostSignalStatus",
    "ProofTileIndices",
    "commitment_hash_from_merkle_roots",
    "denoise_converter",
    "extract_indices",
    "gemm",
    "get_host_signal_header",
    "get_host_signal_header_size",
    "get_host_signal_sync_size",
    "get_required_scratchpad_bytes",
    "kEALScaleFactorDenoise",
    "kEBRScaleFactorDenoise",
    "make_pow_target_tensor",
    "noise_A",
    "noise_B",
    "noise_gen",
    "noisy_gemm",
    "pearl_gemm_interface",
    "run_tensor_hash",
    "tensor_hash",
]
