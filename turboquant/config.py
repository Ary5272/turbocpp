"""Optional config file. Lives at:

    $XDG_CONFIG_HOME/turbocpp/config.toml   (default: ~/.config/turbocpp/)

Read on every CLI invocation; values feed argparse defaults so explicit
flags still win. Missing file = empty config = previous behaviour.

Example:

    [defaults]
    threads = 8
    ctx     = 4096

    [defaults.generate]
    temperature = 0.6
    top_p       = 0.9

    [defaults.chat]
    system = "You are a concise assistant."
    n_predict = 1024

    [models]
    # short alias → GGUF path (resolved by `turbocpp generate -m tiny`)
    tiny    = "~/models/tinyllama-1.1b-chat.Q4_K_M.gguf"
    llama3  = "~/models/Llama-3-8B-Q4_K_M.gguf"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib  # py>=3.11
except ImportError:  # py3.10
    import tomli as tomllib  # type: ignore[no-redef]


def config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "turbocpp" / "config.toml"


def cache_dir() -> Path:
    """`~/.cache/turbocpp/models/` (or $XDG_CACHE_HOME/turbocpp/models/).

    Used by every command that reads or writes a cached GGUF, so all
    five callers stay in sync. Does NOT create the directory; callers
    that need it created should call .mkdir(parents=True, exist_ok=True)."""
    return state_dir() / "models"


def state_dir() -> Path:
    """`~/.cache/turbocpp/` (or $XDG_CACHE_HOME/turbocpp/).

    Sibling to `cache_dir()` for non-model state (e.g. chat history)."""
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "turbocpp"


def load() -> dict[str, Any]:
    """Return parsed config or empty dict on absent / invalid file."""
    p = config_path()
    if not p.is_file():
        return {}
    try:
        with p.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def defaults_for(subcommand: str) -> dict[str, Any]:
    """Merge `[defaults]` + `[defaults.<subcommand>]`. Subcommand-specific
    keys override the global ones. A non-dict at `defaults.<subcommand>`
    (user typo, e.g. `defaults.generate = "x"` instead of a table) is
    silently ignored - we never crash on a malformed config."""
    cfg = load()
    defaults = cfg.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        return {}
    # Strip nested subsections AND any key matching a subcommand name -
    # those are not global scalar defaults regardless of type.
    base = {k: v for k, v in defaults.items() if not isinstance(v, dict) and k != subcommand}
    sub = defaults.get(subcommand, {}) or {}
    if isinstance(sub, dict):
        base.update(sub)
    return base


def resolve_model(name: str) -> str:
    """Resolve a model identifier to a local GGUF path.

    Resolution order:
    1. Alias match in `[models]` table (with ~/$VAR expansion).
    2. `hf://owner/repo/path/to/file.gguf` or `owner/repo:filename.gguf` —
       fetch from HuggingFace Hub into ~/.cache/turbocpp/models/ if not
       already cached. Requires `huggingface_hub` to be installed.
    3. Otherwise, return `name` untouched (treated as a local path).

    Raises FileNotFoundError if HF download is requested but the package
    isn't installed."""
    cfg = load().get("models", {})
    if name in cfg:
        return os.path.expandvars(os.path.expanduser(str(cfg[name])))

    repo, filename = _parse_hf_ref(name)
    if repo is not None and filename is not None:
        return _ensure_hf_cached(repo, filename)
    return name


def _parse_hf_ref(name: str) -> tuple[str | None, str | None]:
    """Recognize HF references. Returns (repo_id, filename) or (None, None).

    Forms accepted:
        hf://owner/repo/path/to/file.gguf       -> ("owner/repo", "path/to/file.gguf")
        owner/repo:filename.gguf                -> ("owner/repo", "filename.gguf")
    A bare local path like ``./model.gguf`` or ``C:\\models\\m.gguf`` is
    rejected (returns (None, None))."""
    if name.startswith("hf://"):
        rest = name[len("hf://") :]
        parts = rest.split("/", 2)
        if len(parts) < 3:
            return (None, None)
        return (f"{parts[0]}/{parts[1]}", parts[2])
    if ":" in name:
        # `owner/repo:filename.gguf` form. Reject Windows drive letters
        # (`C:\models\m.gguf` has no slash before the colon) and POSIX
        # absolute paths with a stray colon (filename must end with .gguf).
        repo, _, filename = name.partition(":")
        if repo.count("/") == 1 and filename.endswith(".gguf"):
            return (repo, filename)
    return (None, None)


def _ensure_hf_cached(repo: str, filename: str) -> str:
    """Download from HF Hub into ~/.cache/turbocpp/models/ if not cached."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise FileNotFoundError(
            f"cannot resolve {repo}:{filename}: huggingface_hub not installed. "
            f"run: pip install 'huggingface_hub<2.0'"
        ) from e
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo_id=repo, filename=filename, cache_dir=str(cache))
