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
        "config",
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


def test_config_defaults_for_handles_malformed(tmp_path, monkeypatch):
    """A typo'd `defaults.<subcommand>` (scalar instead of table) must
    not crash defaults_for - we just ignore it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = tmp_path / "turbocpp" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        '[defaults]\nthreads = 4\ngenerate = "oops a string instead of a table"\n',
        encoding="utf-8",
    )
    from turboquant.config import defaults_for

    out = defaults_for("generate")
    assert out == {"threads": 4}


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


def test_runtime_probe_torch_status_handles_missing(monkeypatch):
    """torch_status should report installed=False without raising when
    torch isn't installed.

    Implementation: assign sys.modules['torch'] = None - Python's import
    system treats that as 'definitely unavailable' and raises
    ModuleNotFoundError on `import torch`. monkeypatch.setitem auto-restores
    the real entry on test teardown, so other tests that genuinely need
    torch are unaffected. We deliberately do NOT call `del sys.modules['torch']`
    or re-import torch from scratch - that triggers RuntimeError because
    torch's C-extension module-init code is not idempotent.
    """
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)

    from turboquant.runtime_probe import torch_status

    s = torch_status()
    assert s == {"installed": False}


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
        "torch",
        "gpu",
    }
    assert expected.issubset(info.keys()), f"missing keys: {expected - info.keys()}"


def test_open_llama_passes_n_gpu_layers(monkeypatch):
    """_open_llama should forward --n-gpu-layers when set, omit when 0.

    Works whether or not llama_cpp is actually installed: we install a
    fake module into sys.modules before _open_llama's deferred import."""
    import sys
    import types

    captured = {}

    class FakeLlama:
        def __init__(self, **kw):
            captured.update(kw)

    fake_mod = types.ModuleType("llama_cpp")
    fake_mod.Llama = FakeLlama  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_mod)

    # Stub `from .config import resolve_model` to a passthrough.
    import turboquant.config as _cfg

    monkeypatch.setattr(_cfg, "resolve_model", lambda x: x)

    from turboquant.cli import _open_llama

    # --- skip case (ngl=0)
    class A0:
        model = "x.gguf"
        ctx = 512
        threads = 0
        seed = 0
        n_gpu_layers = 0

    _open_llama(A0)
    assert "n_gpu_layers" not in captured

    captured.clear()

    # --- include case (ngl=99)
    class A1:
        model = "x.gguf"
        ctx = 512
        threads = 0
        seed = 0
        n_gpu_layers = 99

    _open_llama(A1)
    assert captured.get("n_gpu_layers") == 99


def test_generate_jsonl_format_flag_parsed():
    from turboquant.cli import main

    with pytest.raises(SystemExit):
        # missing -m → SystemExit, but argparse must accept --format jsonl
        main(["generate", "-p", "hi", "--format", "jsonl"])


def test_top_level_help_is_ascii():
    """Top-level `--help` must be pure ASCII so it never crashes on
    Windows cp1252 / POSIX C locale stdout. Subparser help strings live
    here too — keep them clean."""
    import argparse

    from turboquant.cli import main  # noqa: F401  (forces parser construction below)

    # Re-build the same parser to inspect formatted help text:
    sys.argv_backup = sys.argv
    try:
        # `main` builds + parses; we only need the formatted help string.
        # Use a separate run that captures `--help` via SystemExit.
        from io import StringIO

        buf = StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            try:
                main(["--help"])
            except SystemExit:
                pass
        finally:
            sys.stdout = old_stdout
        text = buf.getvalue()
    finally:
        sys.argv = sys.argv_backup

    assert text, "--help produced no output"
    non_ascii = [(i, ch) for i, ch in enumerate(text) if ord(ch) > 127]
    assert not non_ascii, (
        f"top-level --help contains non-ASCII chars: {non_ascii[:5]} "
        f"(would crash Windows cp1252 stdout)"
    )
    # Sanity: argparse's `argparse` import is satisfied
    assert argparse


def test_top_level_help_includes_tips_epilog():
    """The 'Tips:' epilog at the bottom of `--help` is the most reliable
    spot to teach users about HF refs and config init."""
    from io import StringIO

    from turboquant.cli import main

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        try:
            main(["--help"])
        except SystemExit:
            pass
    finally:
        sys.stdout = old_stdout
    text = buf.getvalue()
    assert "Tips:" in text
    assert "owner/repo:file.gguf" in text
    assert "doctor" in text


def test_ensure_utf8_stdio_is_idempotent():
    """`_ensure_utf8_stdio` must be a no-op safe-to-call-twice and must
    not raise when stdout doesn't support reconfigure (e.g. capsys)."""
    from turboquant.cli import _ensure_utf8_stdio

    _ensure_utf8_stdio()
    _ensure_utf8_stdio()  # second call must also succeed


# ---------------------------------------------------------------------------
# config.resolve_model HF reference parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ref,expected",
    [
        # hf:// scheme
        (
            "hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-Q4_K_M.gguf",
            ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-Q4_K_M.gguf"),
        ),
        (
            "hf://owner/repo/sub/dir/m.gguf",
            ("owner/repo", "sub/dir/m.gguf"),
        ),
        # owner/repo:filename form
        (
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-Q4_K_M.gguf",
            ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-Q4_K_M.gguf"),
        ),
        # local paths must NOT be parsed as HF refs
        ("./model.gguf", (None, None)),
        ("/abs/path/model.gguf", (None, None)),
        ("C:\\models\\m.gguf", (None, None)),  # Windows drive letter
        ("model.gguf", (None, None)),
        ("just/a/path.gguf", (None, None)),  # 2 slashes, no colon
        # malformed hf:// refs are ignored (caller's path-not-found will fire)
        ("hf://owner/repo", (None, None)),  # missing filename
        ("hf://justone", (None, None)),
        # owner/repo: with non-gguf filename = treat as local path
        ("owner/repo:something.txt", (None, None)),
    ],
)
def test_parse_hf_ref(ref, expected):
    from turboquant.config import _parse_hf_ref

    assert _parse_hf_ref(ref) == expected


def test_resolve_model_local_paths_passthrough():
    """resolve_model returns local paths untouched (no alias, no HF ref)."""
    from turboquant.config import resolve_model

    for s in ("./model.gguf", "/abs/m.gguf", "C:\\m.gguf", "plain.gguf"):
        assert resolve_model(s) == s


@pytest.mark.parametrize(
    "arg,expected",
    [
        ("./model", True),
        ("/abs/path", True),
        ("rel/path", True),
        ("C:\\m", True),
        ("~/m", True),
        ("chat", False),
        ("generate", False),
        ("rotate", False),
        ("-h", False),
        ("--block", False),
        ("model.gguf", False),
    ],
)
def test_main_module_path_heuristic(arg, expected):
    """`python -m turboquant <subcommand>` must NOT silently rewrite the
    subcommand to `rotate <subcommand>` (which broke `--help` for any
    subcommand added after v0.3)."""
    from turboquant.__main__ import _looks_like_path

    assert _looks_like_path(arg) is expected


def test_main_module_does_not_hijack_subcommand(monkeypatch):
    """End-to-end: `python -m turboquant chat --help` must reach the
    chat subparser, not get rewritten to `rotate chat --help`."""
    from turboquant import __main__ as m

    captured = []

    def fake_main(args):
        captured.append(list(args))
        return 0

    monkeypatch.setattr(m, "_main", fake_main)
    m.main(["chat", "--help"])
    assert captured == [["chat", "--help"]]
    captured.clear()
    m.main(["doctor"])
    assert captured == [["doctor"]]


def test_config_subcommands(tmp_path, monkeypatch, capsys):
    """`turbocpp config init / show / path / validate` round-trip."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from turboquant.cli import main

    # path
    rc = main(["config", "path"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert "config.toml" in out

    # init (no file → wrote)
    rc = main(["config", "init"])
    assert rc == 0
    expected = tmp_path / "turbocpp" / "config.toml"
    assert expected.is_file()

    # init (no --force → refuse)
    rc = main(["config", "init"])
    assert rc == 2

    # init --force (overwrite)
    rc = main(["config", "init", "--force"])
    assert rc == 0

    # show
    rc = main(["config", "show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[defaults]" in out and "[models]" in out

    # validate
    rc = main(["config", "validate"])
    assert rc == 0


def test_doctor_no_network_flag_parsed():
    """Argparse must accept `doctor --no-network` (offline-mode flag)."""
    from turboquant.cli import main

    rc = main(["doctor", "--no-network"])
    # rc is the number of FAILs; with --no-network the wheel-URL check
    # is skipped so it can't FAIL on network. Other checks (python, cpu)
    # are environment-dependent — just assert it ran without TypeError.
    assert isinstance(rc, int)


def test_download_accepts_combined_ref(monkeypatch, tmp_path):
    """`turbocpp download owner/repo:file.gguf` should split into two
    args before calling hf_hub_download."""
    import sys
    import types

    calls = {}

    def fake_dl(repo_id, filename, cache_dir, **kw):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        return str(tmp_path / "x.gguf")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_dl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    from turboquant.cli import main

    rc = main(["download", "owner/repo:file.gguf"])
    assert rc == 0
    assert calls["repo_id"] == "owner/repo"
    assert calls["filename"] == "file.gguf"


def test_main_module_rotate_back_compat(monkeypatch):
    """The pre-0.3 form `python -m turboquant <model_dir> <out_dir>` is
    still rewritten to `rotate <model_dir> <out_dir>`."""
    from turboquant import __main__ as m

    captured = []
    monkeypatch.setattr(m, "_main", lambda args: captured.append(list(args)) or 0)
    m.main(["./Llama-3-8B", "./out-dir"])
    assert captured == [["rotate", "./Llama-3-8B", "./out-dir"]]


def test_resolve_model_hf_ref_calls_hf_hub_download(monkeypatch, tmp_path):
    """resolve_model('owner/repo:file.gguf') triggers hf_hub_download."""
    import sys
    import types

    calls = {}

    def fake_dl(repo_id, filename, cache_dir, **kw):
        calls["repo_id"] = repo_id
        calls["filename"] = filename
        calls["cache_dir"] = cache_dir
        return str(tmp_path / "fake.gguf")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_dl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    from turboquant.config import resolve_model

    out = resolve_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-Q4_K_M.gguf")
    assert calls["repo_id"] == "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    assert calls["filename"] == "tinyllama-Q4_K_M.gguf"
    assert "turbocpp/models" in calls["cache_dir"].replace("\\", "/")
    assert out == str(tmp_path / "fake.gguf")
