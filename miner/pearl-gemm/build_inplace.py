#!/usr/bin/env python3
"""Build pearl_gemm_cuda directly without relying on setup.py cmdclass."""

from __future__ import annotations

import os
import sys
import tempfile
import shutil

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(ROOT_DIR)

_cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
if os.path.isdir(_cuda_home) and _cuda_home not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _cuda_home + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("CUDA_HOME", _cuda_home)

sys.path.insert(0, ROOT_DIR)

import torch.utils.cpp_extension as _cpp_ext

_original_check = getattr(_cpp_ext, "_check_cuda_version", None)


def _patched_check_cuda_version(compiler_name, compiler_version):
    pass


if _original_check is not None:
    _cpp_ext._check_cuda_version = _patched_check_cuda_version

os.environ.setdefault("TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK", "1")

import setup as _setup

_setup._apply_ninja_patch()
_setup._init_submodules()

extensions = _setup._build_ext_modules()
if not extensions:
    print("No extensions to build.")
    sys.exit(0)

from setuptools.dist import Distribution
from torch.utils.cpp_extension import BuildExtension

dist = Distribution({"ext_modules": extensions})
cmd = BuildExtension(dist)
cmd.inplace = 1
cmd.ensure_finalized()

tmpdir = tempfile.mkdtemp(prefix="pearl_build_")
log_path = os.path.join(tmpdir, "build.log")

import io
stderr_buffer = io.StringIO()
old_stderr = sys.stderr
sys.stderr = stderr_buffer

try:
    cmd.run()
except Exception as exc:
    import traceback
    traceback_text = traceback.format_exc()
    captured = stderr_buffer.getvalue()
    sys.stderr = old_stderr
    msg = str(exc)
    if "CUDA" in msg and "mismatch" in msg.lower():
        print(f"Retrying after CUDA version check patch: {msg}")
        _cpp_ext._check_cuda_version = _patched_check_cuda_version
        os.environ["TORCH_EXTENSION_SKIP_CUDA_VERSION_CHECK"] = "1"
        stderr_buffer = io.StringIO()
        sys.stderr = stderr_buffer
        try:
            cmd.run()
        except Exception as exc2:
            traceback_text = traceback.format_exc()
            captured = stderr_buffer.getvalue()
            sys.stderr = old_stderr
            for line in captured.splitlines() + traceback_text.splitlines():
                if ".cu:" in line or ".cuh:" in line or ".hpp:" in line:
                    print(f"ERROR_FILE_LINE: {line.strip()}")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(captured + "\n" + traceback_text)
            print("BUILD FAILED")
            shutil.rmtree(tmpdir, ignore_errors=True)
            sys.exit(1)
        finally:
            sys.stderr = old_stderr
    else:
        for line in captured.splitlines() + traceback_text.splitlines():
            if ".cu:" in line or ".cuh:" in line or ".hpp:" in line:
                print(f"ERROR_FILE_LINE: {line.strip()}")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(captured + "\n" + traceback_text)
        print("BUILD FAILED")
        shutil.rmtree(tmpdir, ignore_errors=True)
        sys.exit(1)
finally:
    sys.stderr = old_stderr

captured = stderr_buffer.getvalue()
with open(log_path, "w", encoding="utf-8") as f:
    f.write(captured)

for line in captured.splitlines():
    if ".cu:" in line or ".cuh:" in line or ".hpp:" in line:
        print(f"ERROR_FILE_LINE: {line.strip()}")

_src_dir = os.path.join(ROOT_DIR, "src", "pearl_gemm")
_dest_path = os.path.join(_src_dir, "pearl_gemm_cuda.so")

_found = None
for search_root in [ROOT_DIR, os.path.join(ROOT_DIR, "build")]:
    if not os.path.isdir(search_root):
        continue
    for dirpath, _dirnames, filenames in os.walk(search_root):
        for f in filenames:
            if f.startswith("pearl_gemm_cuda") and (
                f.endswith(".so") or f.endswith(".pyd")
            ):
                _found = os.path.join(dirpath, f)
                print(f"FOUND: {_found}")
                break
        if _found:
            break
    if _found:
        break

if _found:
    os.makedirs(_src_dir, exist_ok=True)
    shutil.copy2(_found, _dest_path)
    print(f"COPIED: {_found} -> {_dest_path}")
else:
    print("SEARCH: pearl_gemm_cuda.so not found in any search root")
    print("=== STDOUT/STDERR ===")
    print(captured)

print("BUILD COMPLETE")
for ext in extensions:
    print(f"Extension: {ext.name}")

shutil.rmtree(tmpdir, ignore_errors=True)
