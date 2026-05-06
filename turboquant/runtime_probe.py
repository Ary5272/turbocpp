"""Shared environment probes used by `turbocpp info` and `turbocpp doctor`.

Centralizing these here so the two commands report identical data and
adding a new probe (e.g. ARM SVE detection, llama.cpp image digest) is
a one-line change instead of a two-place change."""

from __future__ import annotations

import platform
import shutil
import sys
from typing import Any


def detect_gpu() -> str | None:
    """Return a short human-readable GPU vendor string, or None."""
    if shutil.which("nvidia-smi"):
        return "nvidia (nvidia-smi)"
    if shutil.which("rocminfo"):
        return "amd (rocminfo)"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "apple silicon (Metal)"
    return None


def llama_cpp_status() -> dict[str, Any]:
    """Probe the installed llama-cpp-python (or note its absence)."""
    out: dict[str, Any] = {}
    try:
        import llama_cpp
    except ImportError:
        return {"installed": False}
    out["installed"] = True
    out["version"] = getattr(llama_cpp, "__version__", "?")
    try:
        from llama_cpp import llama_supports_gpu_offload  # type: ignore

        out["gpu_offload"] = bool(llama_supports_gpu_offload())
    except Exception:
        out["gpu_offload"] = None
    return out


def collect_runtime_topology() -> dict[str, Any]:
    """One-stop snapshot consumed by `info` (JSON dump) and `doctor`
    (one-line-per-check). Keep keys stable — downstream scripts may
    parse this output."""
    from . import __version__
    from .cpu_features import best_wheel_url, detect_variant
    from .llama_docker import DEFAULT_IMAGE, docker_available, image_present

    return {
        "turbocpp": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu_variant": detect_variant(),
        "best_wheel_url": best_wheel_url(),
        "docker_present": docker_available(),
        "llama_image": DEFAULT_IMAGE,
        "llama_image_pulled": (image_present(DEFAULT_IMAGE) if docker_available() else False),
        "llama_cpp": llama_cpp_status(),
        "gpu": detect_gpu(),
    }
