"""Thin wrapper around ggml-org/llama.cpp's official Docker image.

Replaces the previous git-submodule pin: instead of vendoring a fixed
commit of llama.cpp, we pull the upstream image on first use. The image
ships every llama.cpp binary already built (llama-cli, llama-server,
llama-quantize, llama-perplexity, llama-imatrix, llama-bench, …) plus
the Python conversion scripts (convert_hf_to_gguf.py, etc.).

Trade-offs vs the submodule:
  + always tracks the latest llama.cpp release (no rebase chore)
  + zero compile time (~2 GB image vs ~5 min C++ build)
  + reproducible across hosts (same binaries everywhere)
  + GPU images available (ghcr.io/ggml-org/llama.cpp:full-cuda etc.)
  − requires Docker installed
  − first call pays the image-pull cost (cached afterwards)

Tag selection follows ggml-org's own scheme:
  full       — every tool, CPU only
  full-cuda  — every tool, CUDA backend
  light      — llama-cli only (smaller)
  server     — llama-server only

We default to `full` for parity with the historical submodule build.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_IMAGE = "ghcr.io/ggml-org/llama.cpp:full"

# All tools available inside the `full` image. Adding to this list also
# auto-exposes them as `turbocpp <tool>` CLI subcommands.
LLAMA_TOOLS = (
    "llama-cli",
    "llama-server",
    "llama-quantize",
    "llama-perplexity",
    "llama-imatrix",
    "llama-bench",
    "llama-tokenize",
    "llama-embedding",
    "llama-export-lora",
    "llama-gguf-split",
    "llama-batched-bench",
    "llama-speculative",
    "convert_hf_to_gguf.py",
)


def docker_available() -> bool:
    return shutil.which("docker") is not None


def image_present(image: str) -> bool:
    """Return True if `docker images` lists this tag locally."""
    if not docker_available():
        return False
    r = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def pull_image(image: str = DEFAULT_IMAGE, *, quiet: bool = False) -> int:
    """Run `docker pull <image>`. Returns docker's exit code."""
    if not docker_available():
        raise RuntimeError(
            "docker not found on PATH — install Docker Desktop or the docker CLI"
        )
    cmd = ["docker", "pull", image]
    return subprocess.call(cmd, stdout=subprocess.DEVNULL if quiet else None)


def run_tool(
    tool: str,
    args: Sequence[str],
    *,
    image: str = DEFAULT_IMAGE,
    mounts: Mapping[str, str] | None = None,
    ports: Mapping[int, int] | None = None,
    interactive: bool = False,
    auto_pull: bool = True,
    extra_docker_args: Sequence[str] = (),
) -> int:
    """Invoke `tool` from the llama.cpp image, forwarding `args`.

    `mounts` is host_path → container_path (always read-write). The
    user's home-models directory is auto-bound to /models if not already
    in `mounts`.

    Returns the tool's exit code.
    """
    if tool not in LLAMA_TOOLS:
        raise ValueError(f"unknown llama.cpp tool: {tool!r}")
    if not docker_available():
        raise RuntimeError("docker not found on PATH")
    if auto_pull and not image_present(image):
        print(f"[turbocpp] pulling {image} (first-run, ~2 GB)...", file=sys.stderr)
        if pull_image(image) != 0:
            return 1

    if mounts is None:
        mounts = {}
    # Default model mount
    if "/models" not in mounts.values() and "MODELS_DIR" in os.environ:
        mounts[os.environ["MODELS_DIR"]] = "/models"

    cmd: list[str] = ["docker", "run", "--rm"]
    if interactive:
        cmd += ["-it"]
    for host, ctr in mounts.items():
        cmd += ["-v", f"{host}:{ctr}"]
    for host_p, ctr_p in (ports or {}).items():
        cmd += ["-p", f"{host_p}:{ctr_p}"]
    cmd += list(extra_docker_args)
    cmd += [image, tool, *args]

    return subprocess.call(cmd)


# Quoting helper for users who want to copy/paste the docker invocation.
def render_docker_cmd(tool: str, args: Sequence[str], **kwargs) -> str:
    """Return the equivalent `docker run …` string without executing it."""
    img = kwargs.get("image", DEFAULT_IMAGE)
    mounts = kwargs.get("mounts") or {}
    parts = ["docker", "run", "--rm"]
    for host, ctr in mounts.items():
        parts += ["-v", f"{host}:{ctr}"]
    parts += [img, tool, *args]
    return " ".join(shlex.quote(p) for p in parts)
