"""Detect host CPU features and pick the best prebuilt llama-cpp-python
wheel from AIencoder/TurboCpp_Wheels.

NOTE: the dataset is initially empty; populate it with
   HF_TOKEN=hf_… python scripts/mirror_wheels.py
before relying on `pip install $(turbocpp pick-wheel)`. PyPI's stock
`pip install llama-cpp-python` works as a fall-back without any HF
mirror — `pick-wheel` is for choosing a CPU-feature-tuned variant.

Usage:
    from turboquant.cpu_features import best_wheel_url
    url = best_wheel_url()           # auto for current host

Each variant is a real binary at /datasets/AIencoder/TurboCpp_Wheels.
We climb a feature ladder and pick the most aggressive variant the host
actually supports — same idea as `gcc -march=native` at install time
instead of compile time.

Ladder (from least to most aggressive):
    basic_avx          → any x86-64
    basic_avx2         → +AVX2 (Haswell, 2013+)
    basic_avx2_fma_f16c → +FMA, +F16C
    basic_avx512       → +AVX-512 (Skylake-X / Cascade Lake / Zen4)
    basic_avx512_fma_f16c
    basic_avx512_fma_f16c_vnni       → +VNNI (Cooper Lake / Zen4)
    basic_avx512_fma_f16c_vnni_vbmi
    basic_avx512_fma_f16c_vnni_vbmi_bf16_amx → Sapphire Rapids
"""
from __future__ import annotations

import platform
import sys
from typing import List

LLAMA_CPP_VERSION = "0.3.16"
WHEEL_BASE = (
    "https://huggingface.co/datasets/AIencoder/TurboCpp_Wheels/resolve/main/"
    "llama_cpp_python-{ver}+{variant}-cp{py}-cp{py}-{plat}.whl"
)


def _cpu_flags() -> set[str]:
    """Best-effort CPU feature detection across Linux / Windows / macOS."""
    flags: set[str] = set()

    # Linux: read /proc/cpuinfo (the canonical source).
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags") or line.startswith("Features"):
                        for f_ in line.split(":", 1)[1].split():
                            flags.add(f_.lower())
                        break
        except OSError:
            pass

    # Windows / macOS: try py-cpuinfo if installed; otherwise nothing.
    if not flags:
        try:
            import cpuinfo  # type: ignore
            info = cpuinfo.get_cpu_info()
            for f_ in info.get("flags", []):
                flags.add(f_.lower())
        except Exception:
            pass

    return flags


def detect_variant() -> str:
    """Return the most aggressive AIencoder/TurboCpp_Wheels variant tag
    the host CPU supports."""
    flags = _cpu_flags()
    has = flags.__contains__

    # Walk the ladder top-down — pick the topmost match.
    if has("amx_tile") and has("avx512_bf16") and has("avx512_vbmi") and \
       has("avx512_vnni") and has("avx512f") and has("fma") and has("f16c"):
        return "basic_avx512_fma_f16c_vnni_vbmi_bf16_amx"
    if has("avx512_vbmi") and has("avx512_vnni") and has("avx512f") and \
       has("fma") and has("f16c"):
        return "basic_avx512_fma_f16c_vnni_vbmi"
    if has("avx512_vnni") and has("avx512f") and has("fma") and has("f16c"):
        return "basic_avx512_fma_f16c_vnni"
    if has("avx512f") and has("fma") and has("f16c"):
        return "basic_avx512_fma_f16c"
    if has("avx512f"):
        return "basic_avx512"
    if has("avx2") and has("fma") and has("f16c"):
        return "basic_avx2_fma_f16c"
    if has("avx2"):
        return "basic_avx2"
    if has("avx"):
        return "basic_avx"
    return "basic_avx2_fma_f16c"          # safe default for unknown


def _platform_tag() -> str:
    """Return the manylinux/macosx/win platform tag we have wheels for."""
    if sys.platform.startswith("linux"):
        return "manylinux_2_31_x86_64"
    if sys.platform == "darwin":
        return "macosx_11_0_arm64" if platform.machine() == "arm64" \
                                    else "macosx_10_9_x86_64"
    if sys.platform == "win32":
        return "win_amd64"
    return "manylinux_2_31_x86_64"


def best_wheel_url(variant: str | None = None,
                    py_version: str | None = None,
                    version: str = LLAMA_CPP_VERSION) -> str:
    """Compose a URL to the prebuilt llama-cpp-python wheel that matches
    the host's CPU + Python version. Override `variant` to force a tag.
    """
    if variant is None:
        variant = detect_variant()
    if py_version is None:
        py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    return WHEEL_BASE.format(
        ver=version,
        variant=variant,
        py=py_version,
        plat=_platform_tag(),
    )


GPU_VARIANTS = ("cuda12", "cuda11", "vulkan", "rocm", "sycl", "opencl")


def gpu_wheel_url(backend: str, version: str = LLAMA_CPP_VERSION,
                  py_version: str | None = None) -> str:
    """URL for a GPU-accelerated llama-cpp-python wheel from
    AIencoder/TurboCpp_Wheels. backend ∈ {cuda12, cuda11, vulkan, rocm,
    sycl, opencl}. The dataset hosts these for cp310/cp311/cp312 on
    manylinux + win_amd64; not all combos exist for every version."""
    if backend not in GPU_VARIANTS:
        raise ValueError(f"unknown GPU backend {backend!r}; "
                         f"choose from {GPU_VARIANTS}")
    if py_version is None:
        py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    return WHEEL_BASE.format(
        ver=version,
        variant=backend,
        py=py_version,
        plat=_platform_tag(),
    )


def candidate_urls() -> List[str]:
    """Return the full ladder, top-supported variant first, then weaker
    fall-backs. Useful for `pip install --index-url …` style retry logic.
    """
    chosen = detect_variant()
    ladder = [
        "basic_avx512_fma_f16c_vnni_vbmi_bf16_amx",
        "basic_avx512_fma_f16c_vnni_vbmi",
        "basic_avx512_fma_f16c_vnni",
        "basic_avx512_fma_f16c",
        "basic_avx512",
        "basic_avx2_fma_f16c",
        "basic_avx2",
        "basic_avx",
    ]
    # Slice from the chosen variant downward.
    if chosen in ladder:
        ladder = ladder[ladder.index(chosen):]
    return [best_wheel_url(v) for v in ladder]


if __name__ == "__main__":
    print("variant:    ", detect_variant())
    print("platform:   ", _platform_tag())
    print("wheel url:  ", best_wheel_url())
    print("\nfallback ladder:")
    for u in candidate_urls():
        print(" ", u)
