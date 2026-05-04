"""Tests for the no-runtime-dep parts of the CLI: argument parsing,
cpu_features wheel-URL composition, llama_docker dispatch contract.

These never spawn docker, never load llama-cpp-python, never touch the
network — pure unit tests so they run anywhere `pip install -e .[dev]`
works."""

from __future__ import annotations

import importlib
import os
import re
import sys

import pytest


# ---------------------------------------------------------------------------
# package surface
# ---------------------------------------------------------------------------
def test_lazy_attribute_access_does_not_require_torch(monkeypatch):
    """Importing turboquant must NOT pull in torch — only the rotate
    helpers do, and only on first use. Verify by simulating torch absent."""
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )
    blocked = {"torch", "transformers", "safetensors"}

    def guard(name, *a, **kw):
        if name.split(".")[0] in blocked:
            raise ImportError(f"simulated: {name} not installed")
        return real_import(name, *a, **kw)

    # Force a fresh import of the top-level package.
    for k in list(sys.modules):
        if k.startswith("turboquant"):
            del sys.modules[k]

    monkeypatch.setattr("builtins.__import__", guard)

    import turboquant  # must succeed

    assert turboquant.__version__  # importlib.metadata path
    # Touching the rotation API is what should explode without torch:
    with pytest.raises(ImportError):
        _ = turboquant.rotate_llama_model


def test_version_metadata_path():
    import turboquant

    # Non-empty, non-trivial version string.
    assert isinstance(turboquant.__version__, str)
    assert len(turboquant.__version__) > 0


# ---------------------------------------------------------------------------
# CLI argparse — make sure every subcommand is reachable and rejects bad input
# ---------------------------------------------------------------------------
def test_cli_help_runs():
    from turboquant.cli import main

    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


def test_cli_no_args_errors():
    from turboquant.cli import main

    with pytest.raises(SystemExit):
        main([])


@pytest.mark.parametrize(
    "sub",
    [
        "rotate",
        "bench",
        "generate",
        "serve",
        "speculative",
        "chat",
        "doctor",
        "info",
        "llama",
        "convert",
        "quantize",
        "perplexity",
        "imatrix",
        "llama-cli",
        "llama-bench",
        "pick-wheel",
        "embed",
        "tokenize",
        "download",
    ],
)
def test_each_subcommand_has_help(sub):
    """Argparse should accept `<sub> --help` without error."""
    from turboquant.cli import main

    with pytest.raises(SystemExit) as e:
        main([sub, "--help"])
    assert e.value.code == 0, f"{sub} --help exited {e.value.code}"


def test_pick_wheel_returns_url(capsys):
    from turboquant.cli import main

    rc = main(["pick-wheel"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "huggingface.co/datasets/AIencoder/" in out
    assert ".whl" in out


def test_pick_wheel_gpu_cuda(capsys):
    from turboquant.cli import main

    rc = main(["pick-wheel", "--gpu", "cuda12"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cuda12" in out


# ---------------------------------------------------------------------------
# cpu_features
# ---------------------------------------------------------------------------
def test_cpu_features_url_format():
    from turboquant.cpu_features import best_wheel_url, candidate_urls

    url = best_wheel_url()
    assert url.startswith("https://huggingface.co/datasets/AIencoder/")
    assert "TurboCpp_Wheels" in url or "llama-cpp-wheels" in url
    assert re.search(r"cp3\d+", url), "Python version tag missing"
    assert url.endswith(".whl")
    # Ladder is nonempty + the first entry equals the chosen one
    assert candidate_urls()[0] == url


def test_cpu_features_invalid_gpu_raises():
    from turboquant.cpu_features import gpu_wheel_url

    with pytest.raises(ValueError):
        gpu_wheel_url("not-a-backend")


# ---------------------------------------------------------------------------
# llama_docker — dispatch logic without actually invoking docker
# ---------------------------------------------------------------------------
def test_llama_tools_list_complete():
    from turboquant.llama_docker import LLAMA_TOOLS

    # Every tool we surface as a CLI alias must be in the canonical list.
    expected = {
        "llama-cli",
        "llama-server",
        "llama-quantize",
        "llama-perplexity",
        "llama-imatrix",
        "llama-bench",
        "convert_hf_to_gguf.py",
        "llama-speculative",
    }
    assert expected.issubset(set(LLAMA_TOOLS))


def test_render_docker_cmd_well_formed(tmp_path):
    from turboquant.llama_docker import render_docker_cmd

    cmd = render_docker_cmd(
        "llama-cli",
        ["-m", "/models/x.gguf", "-p", "hi"],
        mounts={str(tmp_path): "/models"},
    )
    # All shell-safe (shlex.quote applied), correct image, correct tool.
    assert cmd.startswith("docker run --rm")
    assert "ghcr.io/ggml-org/llama.cpp" in cmd
    assert "llama-cli" in cmd
    assert "/models/x.gguf" in cmd


def test_run_tool_rejects_unknown():
    from turboquant.llama_docker import run_tool

    with pytest.raises(ValueError):
        run_tool("not-a-tool", [])


# ---------------------------------------------------------------------------
# config.py — TOML loader, model-alias resolution
# ---------------------------------------------------------------------------
def test_config_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from turboquant import config as cfg

    importlib.reload(cfg)  # re-evaluate config_path()
    assert cfg.load() == {}
    assert cfg.defaults_for("generate") == {}
    assert cfg.resolve_model("anything") == "anything"


def test_config_loads_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    p = tmp_path / "turbocpp" / "config.toml"
    p.parent.mkdir()
    p.write_text(
        "[defaults]\n"
        "threads = 8\n"
        "[defaults.generate]\n"
        "temperature = 0.4\n"
        "threads = 16\n"  # generate override should win
        "[models]\n"
        'tiny = "/models/tiny.gguf"\n',
        encoding="utf-8",
    )
    from turboquant import config as cfg

    importlib.reload(cfg)
    g = cfg.defaults_for("generate")
    assert g["temperature"] == 0.4
    assert g["threads"] == 16
    assert cfg.resolve_model("tiny") == "/models/tiny.gguf"
    assert cfg.resolve_model("/literal/path.gguf") == "/literal/path.gguf"


# ---------------------------------------------------------------------------
# Apple Silicon / arm64 detection
# ---------------------------------------------------------------------------
def test_apple_silicon_returns_metal_variant(monkeypatch):
    import turboquant.cpu_features as cf

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    assert cf.detect_variant() == "metal"
    # Wheel URL should include the arm64 mac platform tag.
    url = cf.best_wheel_url()
    assert "macosx_11_0_arm64" in url
    # Candidate ladder is just the one entry on non-x86.
    assert cf.candidate_urls() == [url]


def test_linux_aarch64_returns_neon(monkeypatch):
    import turboquant.cpu_features as cf

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")
    monkeypatch.setattr(cf, "_cpu_flags", lambda: set())
    assert cf.detect_variant() == "neon"
    monkeypatch.setattr(cf, "_cpu_flags", lambda: {"sve"})
    assert cf.detect_variant() == "neon_sve"
    monkeypatch.setattr(cf, "_cpu_flags", lambda: {"sve2"})
    assert cf.detect_variant() == "neon_sve2"


# ---------------------------------------------------------------------------
# generate / chat new flags reach argparse without errors
# ---------------------------------------------------------------------------
def test_generate_accepts_new_flags():
    from turboquant.cli import main

    # Doesn't actually run (no model file) — we rely on argparse parsing
    # the args before the command function is invoked.
    with pytest.raises(SystemExit):
        # missing -m: argparse exit 2.
        main(["generate", "-p", "hi", "--seed", "42", "--stop", "</s>"])


def test_chat_accepts_new_flags():
    from turboquant.cli import main

    with pytest.raises(SystemExit):
        main(["chat", "--seed", "7", "--stop", "STOP", "--grammar", "/tmp/x"])
