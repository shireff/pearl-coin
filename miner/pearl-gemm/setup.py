"""
setup.py for pearl-gemm CUDA extension.

Design constraint: this file is imported by pip during metadata collection,
BEFORE any build-time dependencies (torch, pearl_gemm_build_utils, etc.) are
installed.  Therefore:

  * No top-level heavy imports (torch, pearl_gemm_build_utils, wheel, …).
  * No module-level executable code that touches the filesystem, GPU, or
    network (subprocess, /proc/meminfo, CUDA_HOME, …).
  * Everything that needs the above lives inside functions/classes that are
    only called during the actual build step.
"""

from __future__ import annotations

import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
import warnings
from pathlib import Path

from setuptools import setup, Extension

# ---------------------------------------------------------------------------
# Paths and static constants — safe to compute at import time
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / "csrc"
GEMM_DIR = CSRC_DIR / "gemm"

PACKAGE_NAME = "pearl_gemm"

with open(ROOT_DIR / "pyproject.toml", "rb") as _f:
    PACKAGE_VERSION = tomllib.load(_f)["project"]["version"]

BASE_WHEEL_URL = ""


# ---------------------------------------------------------------------------
# Environment-flag helpers — only read env vars, no external imports needed
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).casefold() in ("t", "true", "1", "y", "yes")


FORCE_BUILD = _env_flag("PEARL_GEMM_FORCE_BUILD", "TRUE")
SKIP_CUDA_BUILD = _env_flag("PEARL_GEMM_SKIP_CUDA_BUILD", "FALSE")
FORCE_CXX11_ABI = _env_flag("PEARL_GEMM_FORCE_CXX11_ABI", "FALSE")

DISABLE_SKIP_REDUCTION = _env_flag("PEARL_GEMM_DISABLE_SKIP_REDUCTION", "FALSE")
DISABLE_SKIP_DENOISING = _env_flag("PEARL_GEMM_DISABLE_SKIP_DENOISING", "FALSE")
DISABLE_DEBUG_MODE = _env_flag("PEARL_GEMM_DISABLE_DEBUG_MODE", "FALSE")
DISABLE_R32 = _env_flag("PEARL_GEMM_DISABLE_R32", "TRUE")
DISABLE_R64 = _env_flag("PEARL_GEMM_DISABLE_R64", "FALSE")
DISABLE_R128 = _env_flag("PEARL_GEMM_DISABLE_R128", "FALSE")
SKIP_CPP_GENERATION = _env_flag("PEARL_GEMM_SKIP_CPP_GENERATION", "FALSE")

FEATURE_FLAGS: dict[str, bool] = {
    "DISABLE_SKIP_REDUCTION": DISABLE_SKIP_REDUCTION,
    "DISABLE_SKIP_DENOISING": DISABLE_SKIP_DENOISING,
    "DISABLE_DEBUG_MODE": DISABLE_DEBUG_MODE,
    "DISABLE_R32": DISABLE_R32,
    "DISABLE_R64": DISABLE_R64,
    "DISABLE_R128": DISABLE_R128,
}

R_VALUE_TOGGLES: dict[int, bool] = {32: DISABLE_R32, 64: DISABLE_R64, 128: DISABLE_R128}
ENABLED_R_VALUES: list[int] = [r for r, disabled in R_VALUE_TOGGLES.items() if not disabled]
OUTPUT_TYPES: list[str] = ["bf16"]

RAM_PER_JOB_GB = 6
CORES_PER_JOB = 1
FALLBACK_MAX_JOBS = 4
KB_PER_GB = 1024 * 1024
NVCC_THREAD_COUNT = "4"
COMPUTE_CAPABILITY = os.getenv("PEARL_GEMM_ARCH", "")


# ---------------------------------------------------------------------------
# Resource helpers — safe on all platforms; called lazily at build time
# ---------------------------------------------------------------------------


def _available_cpu_count() -> int:
    if hasattr(os, "process_cpu_count"):
        return os.process_cpu_count()
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    count = os.cpu_count()
    if not count:
        raise RuntimeError("Could not determine CPU core count")
    return count


def _linux_total_ram_kb() -> int:
    """Read available RAM from /proc/meminfo (Linux only)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if not line.startswith("MemAvailable:"):
                continue
            match = re.search(r"MemAvailable:\s+(\d+)\s+kB", line)
            if not match:
                raise RuntimeError(f"Could not parse MemAvailable line: {line.strip()}")
            return int(match.group(1))
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def _smart_max_jobs() -> int:
    """Return a safe parallel-job count, cross-platform."""
    try:
        cores = _available_cpu_count()
        if not sys.platform.startswith("linux"):
            # Cannot read /proc/meminfo on Windows/macOS — cap at CPU count
            return max(1, cores // CORES_PER_JOB)
        ram_kb = _linux_total_ram_kb()
        return int(min(cores // CORES_PER_JOB, ram_kb // (RAM_PER_JOB_GB * KB_PER_GB)))
    except Exception as e:
        warnings.warn(
            f"_smart_max_jobs: falling back to {FALLBACK_MAX_JOBS} due to error: {e!r}",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return FALLBACK_MAX_JOBS


# ---------------------------------------------------------------------------
# Ninja-file writer (monkey-patch for arch-dependent fix from FA3)
# Defined here so it is ready when BuildExtension picks it up; the actual
# patch is applied lazily inside _apply_ninja_patch().
# ---------------------------------------------------------------------------


def _ninja_escape(path: str) -> str:
    return path.replace(" ", "$ ")


def _build_ninja_config(compiler: str, with_cuda: bool, cuda_dlink_post_cflags: list[str]) -> list[str]:
    config = ["ninja_required_version = 1.3", f"cxx = {compiler}"]
    if with_cuda or cuda_dlink_post_cflags:
        # These imports are safe here because this function is only called
        # from _write_ninja_file which is only reached during build_ext.
        from torch.utils.cpp_extension import _join_cuda_home

        nvcc = _join_cuda_home("bin", "nvcc")
        nvcc_from_env = os.getenv("PYTORCH_NVCC", nvcc)
        config += [f"nvcc_from_env = {nvcc_from_env}", f"nvcc = {nvcc}"]
    return config


def _build_ninja_flags(
    cflags: list[str],
    post_cflags: list[str],
    cuda_cflags: list[str],
    cuda_post_cflags: list[str],
    cuda_dlink_post_cflags: list[str],
    ldflags: list[str],
    with_cuda: bool,
) -> list[str]:
    flags = [
        f"cflags = {' '.join(cflags)}",
        f"post_cflags = {' '.join(post_cflags)}",
    ]
    if with_cuda:
        flags += [
            f"cuda_cflags = {' '.join(cuda_cflags)}",
            f"cuda_post_cflags = {' '.join(cuda_post_cflags)}",
        ]
    flags += [
        f"cuda_dlink_post_cflags = {' '.join(cuda_dlink_post_cflags)}",
        f"ldflags = {' '.join(ldflags)}",
    ]
    return flags


def _build_cuda_compile_rule() -> list[str]:
    import torch

    rule = ["rule cuda_compile"]
    nvcc_gendeps = ""
    if (
        torch.version.cuda is not None
        and os.getenv("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "0") != "1"
    ):
        rule += ["  depfile = $out.d", "  deps = gcc"]
        nvcc_gendeps = "--generate-dependencies-with-compile --dependency-output $out.d"
    rule.append(
        f"  command = $nvcc_from_env {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags"
    )
    return rule


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    cuda_dlink_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_cuda,
    **kwargs,
) -> None:
    """Write a ninja build file for compiling and linking CUDA extensions."""
    from torch.utils.cpp_extension import _is_cuda_file, _maybe_write, get_cxx_compiler

    def _normalize_flags(flags):
        return [f.strip() for f in flags] if flags else []

    cflags = _normalize_flags(cflags)
    post_cflags = _normalize_flags(post_cflags)
    cuda_cflags = _normalize_flags(cuda_cflags)
    cuda_post_cflags = _normalize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = _normalize_flags(cuda_dlink_post_cflags)
    ldflags = _normalize_flags(ldflags)

    assert len(sources) == len(objects) > 0
    sources = [os.path.abspath(f) for f in sources]

    config = _build_ninja_config(get_cxx_compiler(), with_cuda, cuda_dlink_post_cflags)
    flags = _build_ninja_flags(
        cflags, post_cflags, cuda_cflags, cuda_post_cflags,
        cuda_dlink_post_cflags, ldflags, with_cuda,
    )

    compile_rule = [
        "rule compile",
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
    ]

    cuda_compile_rule = _build_cuda_compile_rule() if with_cuda else []

    build = []
    for src, obj in zip(sources, objects, strict=True):
        rule = "cuda_compile" if _is_cuda_file(src) and with_cuda else "compile"
        build.append(f"build {_ninja_escape(obj)}: {rule} {_ninja_escape(src)}")

    devlink_rule, devlink = [], []
    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), "dlink.o")
        devlink_rule = [
            "rule cuda_devlink",
            "  command = $nvcc $in -o $out $cuda_dlink_post_cflags",
        ]
        devlink = [f"build {devlink_out}: cuda_devlink {' '.join(objects)}"]
        objects += [devlink_out]

    link_rule, link, default = [], [], []
    if library_target is not None:
        link_rule = ["rule link", "  command = $cxx $in $ldflags -o $out"]
        link = [f"build {library_target}: link {' '.join(objects)}"]
        default = [f"default {library_target}"]

    blocks = [
        config, flags, compile_rule, cuda_compile_rule,
        devlink_rule, link_rule, build, devlink, link, default,
    ]
    content = "\n\n".join("\n".join(b) for b in blocks) + "\n"
    _maybe_write(path, content)


def _apply_ninja_patch() -> None:
    """Monkey-patch torch's _write_ninja_file. Called once at build time."""
    import torch.utils.cpp_extension as _cpp_ext

    _cpp_ext._write_ninja_file = _write_ninja_file


# ---------------------------------------------------------------------------
# Build-time helpers — only called from PearlBuildExtension.run()
# ---------------------------------------------------------------------------


def _init_submodules() -> None:
    cutlass_dir = ROOT_DIR / "third_party" / "cutlass"
    marker = cutlass_dir / "include" / "cute" / "layout.hpp"
    if marker.exists():
        print(f"cutlass_dir already exists at {cutlass_dir}")
        return

    url = "https://github.com/NVIDIA/cutlass/archive/refs/heads/main.zip"
    zip_path = ROOT_DIR / "cutlass.zip"
    extract_dir = ROOT_DIR / "third_party"

    if cutlass_dir.is_dir():
        import shutil
        shutil.rmtree(cutlass_dir)

    print(f"Downloading CUTLASS from {url}")
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(zip_path, "wb") as out:
            out.write(resp.read())
    except Exception as exc:
        raise RuntimeError(f"Failed to download CUTLASS: {exc}") from exc

    print(f"Extracting {zip_path} ...")
    import zipfile

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            top = zf.namelist()[0].split("/")[0]
            zf.extractall(extract_dir)
            extracted = extract_dir / top
            if extracted.exists() and not cutlass_dir.exists():
                extracted.rename(cutlass_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract CUTLASS: {exc}") from exc
    finally:
        if zip_path.exists():
            zip_path.unlink()

    if not marker.exists():
        raise RuntimeError(f"CUTLASS include/cute/layout.hpp missing at {marker}")


def _get_platform() -> str:
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    raise ValueError(f"Unsupported platform: {sys.platform}")


def _get_cuda_bare_metal_version(cuda_dir: str):
    from packaging.version import parse

    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])
    return raw_output, bare_metal_version


def _is_cuda_13_or_later(cuda_dir: str) -> bool:
    try:
        _, ver = _get_cuda_bare_metal_version(cuda_dir)
        return ver >= parse("13.0")
    except Exception:
        return False


def _warn_if_cuda_home_missing(cuda_home) -> None:
    if cuda_home is not None:
        return
    warnings.warn(
        "pearl_gemm was requested, but nvcc was not found. "
        "Are you sure your environment has nvcc available? "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc.",
        stacklevel=2,
    )


def _append_nvcc_threads(nvcc_extra_args: list[str]) -> list[str]:
    return nvcc_extra_args + ["--threads", NVCC_THREAD_COUNT]


def _resolve_compute_capability() -> str:
    """Return the compute capability flag string.

    Priority:
      1. PEARL_GEMM_ARCH environment variable
      2. torch.cuda.get_device_capability() if torch is available
      3. nvidia-smi parsing
      4. Fallback to sm_89 (Ada Lovelace / RTX 40xx)
    """
    env_arch = os.getenv("PEARL_GEMM_ARCH")
    if env_arch:
        return env_arch

    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"arch=compute_{major}{minor},code=sm_{major}{minor}"
    except Exception:
        pass

    try:
        import subprocess

        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            universal_newlines=True,
        )
        cap = output.strip().split("\n")[0].strip()
        if "." in cap:
            major, minor = cap.split(".")
            minor = minor[:2].ljust(2, "0")
            return f"arch=compute_{major}{minor},code=sm_{major}{minor}"
    except Exception:
        pass

    return "arch=compute_89,code=sm_89"


def _build_ext_modules():
    """
    Build and return the list of Extension objects.
    Called lazily inside PearlBuildExtension — after all build deps are installed.
    """
    import torch
    from packaging.version import parse
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension

    from pearl_gemm_build_utils.generate_instantiations import generate_instantiations
    from pearl_gemm_build_utils.write_static_switches import (
        write_matmul_switch,
        write_noising_a_switch,
        write_noising_b_switch,
    )

    # Set MAX_JOBS now that we're in the build phase
    os.environ.setdefault("MAX_JOBS", str(_smart_max_jobs()))

    # Load kernel configs
    config_module = importlib.import_module(
        "pearl_gemm_build_utils.kernel_configs.default_compiled_kernels"
    )
    kernel_configs = config_module.KERNEL_CONFIGS

    matmul_kernels = [k for k in kernel_configs.matmul_kernels if k.R in ENABLED_R_VALUES]
    noising_a_kernels = [k for k in kernel_configs.noising_a_kernels if k.R in ENABLED_R_VALUES]
    noising_b_kernels = [k for k in kernel_configs.noising_b_kernels if k.R in ENABLED_R_VALUES]

    print(f"\ntorch.__version__  = {torch.__version__}\n")
    _warn_if_cuda_home_missing(CUDA_HOME)
    _, bare_metal_version = _get_cuda_bare_metal_version(CUDA_HOME)
    print(f"cuda version = {bare_metal_version}\n")

    arch_flags = ["-gencode", _resolve_compute_capability()]

    if not SKIP_CPP_GENERATION:
        instantiations_dir = GEMM_DIR / "instantiations"
        print(f"Writing template instantiations to {instantiations_dir}")
        generate_instantiations(matmul_kernels, noising_a_kernels, noising_b_kernels, instantiations_dir)
        print(f"Writing static switches to {GEMM_DIR}")
        write_matmul_switch(GEMM_DIR / "static_switch_matmul.h", matmul_kernels)
        write_noising_a_switch(GEMM_DIR / "static_switch_noisingA.h", noising_a_kernels)
        write_noising_b_switch(GEMM_DIR / "static_switch_noisingB.h", noising_b_kernels)

    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    cutlass_dir = ROOT_DIR / "third_party" / "cutlass"

    sources = [
        "csrc/gemm/pearl_gemm_api.cpp",
        "csrc/gemm/noise_generation.cu",
        "csrc/gemm/denoise_converter.cu",
        "csrc/gemm/inner_hash_kernel.cu",
        "csrc/gemm/jackpot_eval.cu",
        "csrc/gemm/jackpot_eval_launch.cu",
        "csrc/gemm/persistent_mining.cu",
        "csrc/gemm/persistent_mining_launch.cu",
        "csrc/gemm/persistent_pipeline.cu",
        "csrc/gemm/gpu_mining_kernel.cu",
        "csrc/gemm/gpu_mining_launch.cu",
        "csrc/gemm/gpu_mining_graph.cu",
        "csrc/gemm/pearl_gemm_api_kernels.cu",
        "csrc/stark/gpu_stark_kernels.cu",
        "csrc/stark/gpu_stark_launch.cu",
        "csrc/blake3/blake3.cu",
        "csrc/tensor_hash/tensor_hash.cu",
        "csrc/moe/build_routing_data.cu",
    ]
    sources.extend(
        f"csrc/gemm/instantiations/gemm_R{cfg.R}_{out_type}_{cfg.tile_size_m}x{cfg.tile_size_n}x{cfg.tile_size_k}_{cfg.pipeline_stages}stages_cluster{cfg.cM}x{cfg.cN}.cu"
        for cfg in matmul_kernels
        for out_type in OUTPUT_TYPES
    )
    sources.extend(
        f"csrc/gemm/instantiations/noisingA_R{cfg.R}_{cfg.AxEBL_type}_{cfg.tile_size_m}x{cfg.tile_size_k}_{cfg.pipeline_stages}stages.cu"
        for cfg in noising_a_kernels
    )
    sources.extend(
        f"csrc/gemm/instantiations/noisingB_R{cfg.R}_{cfg.EARxBpEB_type}_{cfg.tile_size_n}x{cfg.tile_size_k}_{cfg.pipeline_stages}stages.cu"
        for cfg in noising_b_kernels
    )

    feature_args = [f"-D{name}" for name, enabled in FEATURE_FLAGS.items() if enabled]

    gcc_flags = ["-O3", "-std=c++20", "-fvisibility=hidden", "-fpermissive"]
    nvcc_flags = [
        "-O3", "-std=c++20",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DNDEBUG",
    ]

    if _is_cuda_13_or_later(CUDA_HOME):
        nvcc_flags += [
            "-U__CCCL_CPP_STD_NAMESPACE",
            "-Xcompiler",
            "-fpermissive",
        ]
        print("Detected CUDA 13.0+, applying CCCL namespace workaround")

    include_dirs = [
        CSRC_DIR,
        cutlass_dir / "include",
        cutlass_dir / "examples" / "common",
        cutlass_dir / "tools" / "util" / "include",
    ]
    if sys.platform.startswith("linux"):
        cuda_cccl_include_dir = (
            Path(CUDA_HOME) / "targets" / f"{platform.machine()}-linux" / "include" / "cccl"
        )
        if cuda_cccl_include_dir.exists():
            include_dirs.append(cuda_cccl_include_dir)

    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")

    return [
        CUDAExtension(
            name="pearl_gemm_cuda",
            sources=sources,
            extra_compile_args={
                "cxx": gcc_flags + feature_args,
                "nvcc": _append_nvcc_threads(nvcc_flags + arch_flags + feature_args),
            },
            extra_link_args=[f"-Wl,-rpath,{torch_lib_path}", "-Wl,-rpath,$ORIGIN"],
            include_dirs=include_dirs,
            libraries=["cuda"],
        )
    ]


# ---------------------------------------------------------------------------
# Custom build commands
# ---------------------------------------------------------------------------
# Design notes:
#
# 1. PearlBuildExtension inherits from BuildExtension when torch is available,
#    otherwise falls back to setuptools.Command.  This preserves compatibility
#    with newer setuptools versions that validate cmdclass entries as real
#    Command subclasses at parse time (metadata collection may run without
#    torch installed).
#
# 2. CachedWheelsCommand inherits from bdist_wheel when wheel is available,
#    otherwise falls back to setuptools.Command.
# ---------------------------------------------------------------------------


try:
    from torch.utils.cpp_extension import BuildExtension as _BuildExtBase
    _HAVE_BUILD_EXT = True
except ImportError:
    from setuptools import Command as _BuildExtBase
    _HAVE_BUILD_EXT = False


class PearlBuildExtension(_BuildExtBase):
    """BuildExtension that applies ninja patch and initializes submodules."""

    def build_extensions(self):
        import os as _os
        marker = _os.path.join(ROOT_DIR, ".pearl_build_ext_marker")
        with open(marker, "w", encoding="utf-8") as _mf:
            _mf.write("build_extensions_called\n")
        print("DEBUG: PearlBuildExtension.build_extensions() called", flush=True)
        if not _HAVE_BUILD_EXT:
            raise RuntimeError(
                "pearl-gemm build requires torch. "
                "Install torch before building the CUDA extension."
            )
        _apply_ninja_patch()
        _init_submodules()
        if not SKIP_CUDA_BUILD:
            self.extensions = _build_ext_modules()
        print(f"DEBUG: extensions count = {len(self.extensions or [])}", flush=True)
        for ext in (self.extensions or []):
            print(f"DEBUG: extension name = {ext.name}", flush=True)
        super().build_extensions()
        print("DEBUG: super().build_extensions() returned", flush=True)

        if not SKIP_CUDA_BUILD:
            for ext in (self.extensions or []):
                full_path = self.get_ext_fullpath(ext.name)
                print(f"VERIFY_BUILD: expected output path = {full_path}")
                if os.path.exists(full_path):
                    print(f"VERIFY_BUILD: found = {full_path}")
                    continue
                import glob as _glob
                search_roots = [
                    os.path.dirname(full_path),
                    self.build_lib or "",
                    self.build_temp or "",
                    str(ROOT_DIR / "src" / "pearl_gemm"),
                    str(ROOT_DIR / "miner" / "pearl-gemm" / "src" / "pearl_gemm"),
                ]
                found_any = False
                for search_root in search_roots:
                    if not search_root:
                        continue
                    candidates = _glob.glob(
                        os.path.join(search_root, "**", "*pearl_gemm_cuda*"), recursive=True
                    )
                    if candidates:
                        print(f"VERIFY_BUILD: nearby candidates in {search_root} = {candidates}")
                        found_any = True
                        break
                if not found_any:
                    print("VERIFY_BUILD: pearl_gemm_cuda output not found")


try:
    from wheel.bdist_wheel import bdist_wheel as _BdistWheelBase
    _HAVE_WHEEL = True
except ImportError:
    from setuptools import Command as _BdistWheelBase
    _HAVE_WHEEL = False


class CachedWheelsCommand(_BdistWheelBase):
    """bdist_wheel that tries to download a prebuilt wheel first."""

    def run(self):
        if not _HAVE_WHEEL or FORCE_BUILD:
            return super().run()
        wheel_url, wheel_filename = _get_wheel_url()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)
            if self.dist_dir and not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)
            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = (
                f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"
            )
            if self.dist_dir:
                wheel_path = os.path.join(
                    self.dist_dir, archive_basename + ".whl"
                )
                print("Raw wheel path", wheel_path)
                shutil.move(wheel_filename, wheel_path)
        except urllib.error.HTTPError:
            print("Precompiled wheel not found. Building from source...")
            super().run()


def _get_wheel_url() -> tuple[str, str]:
    """Called only at bdist_wheel time — torch is guaranteed to be present."""
    import torch
    from packaging.version import parse

    torch_cuda_version = parse(torch.version.cuda)
    torch_version_raw = parse(torch.__version__)
    MIN_CUDA_VERSION = parse("13.0")
    if torch_cuda_version < MIN_CUDA_VERSION:
        raise RuntimeError(
            f"CUDA >= {MIN_CUDA_VERSION} is required, but torch was built with CUDA {torch_cuda_version}"
        )
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = _get_platform()
    package_version = PACKAGE_VERSION
    cuda_version = f"{torch_cuda_version.major}{torch_cuda_version.minor}"
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    wheel_filename = (
        f"{PACKAGE_NAME}-{package_version}+cu{cuda_version}torch{torch_version}"
        f"cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    )
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{package_version}", wheel_name=wheel_filename)
    return wheel_url, wheel_filename


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup(
        ext_modules=[Extension("pearl_gemm_cuda", sources=[])],
        cmdclass={
            "bdist_wheel": CachedWheelsCommand,
            "build_ext": PearlBuildExtension,
        },
    )
