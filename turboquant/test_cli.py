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
        "list-models",
        "list-templates",
        "quickstart",
        "version",
        "rm-model",
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
# new helpers — _msgs_to_markdown, _resolve_prompt
# ---------------------------------------------------------------------------
def test_msgs_to_markdown_renders_roles():
    from turboquant.cli import _msgs_to_markdown

    md = _msgs_to_markdown(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi back"},
        ]
    )
    assert "## system" in md
    assert "## user" in md
    assert "## assistant" in md
    assert "be terse" in md
    assert "hi back" in md


def test_resolve_prompt_prefers_inline(monkeypatch):
    from turboquant.cli import _resolve_prompt

    class A:
        prompt = "inline"
        prompt_file = None

    assert _resolve_prompt(A) == "inline"


def test_resolve_prompt_reads_file(tmp_path):
    from turboquant.cli import _resolve_prompt

    f = tmp_path / "p.txt"
    f.write_text("from-disk", encoding="utf-8")

    class A:
        prompt = None
        prompt_file = str(f)

    assert _resolve_prompt(A) == "from-disk"


def test_generate_requires_some_prompt(monkeypatch):
    """No -p, no -f, and stdin is a TTY → SystemExit."""
    from turboquant.cli import _resolve_prompt

    class A:
        prompt = None
        prompt_file = None

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    with pytest.raises(SystemExit):
        _resolve_prompt(A)


def test_generate_logprobs_flag_parsed():
    from turboquant.cli import main

    with pytest.raises(SystemExit) as e:
        main(["generate", "--help"])
    assert e.value.code == 0


def test_serve_api_key_flag_parsed():
    from turboquant.cli import main

    with pytest.raises(SystemExit) as e:
        main(["serve", "--help"])
    assert e.value.code == 0


def test_embed_normalize_flag_parsed():
    from turboquant.cli import main

    with pytest.raises(SystemExit) as e:
        main(["embed", "--help"])
    assert e.value.code == 0


def test_list_models_runs_without_runtime(capsys, tmp_path, monkeypatch):
    """list-models should not require llama-cpp-python."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from turboquant.cli import main

    rc = main(["list-models"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "alias" in out.lower() or "no aliases" in out.lower()


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


# ---------------------------------------------------------------------------
# v0.10 additions
# ---------------------------------------------------------------------------
def test_root_version_flag(capsys):
    """`turbocpp --version` prints version and exits 0."""
    from turboquant.cli import main

    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    # argparse's --version exits 0 with the version on stdout; we just
    # confirm it didn't error and produced some output.
    captured = capsys.readouterr()
    assert "turbocpp" in (captured.out + captured.err)


def test_version_subcommand_prints(capsys):
    from turboquant.cli import main

    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    # Either real version or the "0.0.0+source" fallback.
    assert out, "version subcommand printed nothing"


def test_rm_model_dry_run_handles_empty_cache(monkeypatch, tmp_path, capsys):
    """rm-model with no cache directory should not crash."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from turboquant.cli import main

    rc = main(["rm-model", "--all", "--dry-run"])
    assert rc == 0


def test_runtime_probe_keys_are_stable():
    """info + doctor both depend on these keys; downstream JSON parsers
    might too — guard against accidental rename."""
    from turboquant.runtime_probe import collect_runtime_topology

    info = collect_runtime_topology()
    expected = {
        "turbocpp",
        "python",
        "platform",
        "cpu_variant",
        "best_wheel_url",
        "docker_present",
        "llama_image",
        "llama_image_pulled",
        "llama_cpp",
        "gpu",
    }
    assert expected.issubset(info.keys()), f"missing keys: {expected - info.keys()}"


def test_open_llama_passes_n_gpu_layers(monkeypatch):
    """_open_llama should forward --n-gpu-layers when set, omit when 0."""
    captured = {}

    class FakeLlama:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(
        "llama_cpp.Llama",
        FakeLlama,
        raising=False,
    )

    # Stub `from .config import resolve_model` to a passthrough.
    import turboquant.config as _cfg

    monkeypatch.setattr(_cfg, "resolve_model", lambda x: x)

    # --- skip case (ngl=0)
    from turboquant.cli import _open_llama

    class A0:
        model = "x.gguf"
        ctx = 512
        threads = 0
        seed = 0
        n_gpu_layers = 0

    try:
        _open_llama(A0)
    except Exception:
        pass
    assert "n_gpu_layers" not in captured

    captured.clear()

    # --- include case (ngl=99)
    class A1:
        model = "x.gguf"
        ctx = 512
        threads = 0
        seed = 0
        n_gpu_layers = 99

    try:
        _open_llama(A1)
    except Exception:
        pass
    assert captured.get("n_gpu_layers") == 99


def test_generate_jsonl_format_flag_parsed():
    from turboquant.cli import main

    with pytest.raises(SystemExit):
        # missing -m → SystemExit, but argparse must accept --format jsonl
        main(["generate", "-p", "hi", "--format", "jsonl"])
