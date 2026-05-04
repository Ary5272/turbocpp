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
    keys override the global ones."""
    cfg = load()
    base = dict(cfg.get("defaults", {}) or {})
    sub = cfg.get("defaults", {}).get(subcommand, {}) or {}
    # Strip nested subsections from `base` (they're tables, not values).
    base = {k: v for k, v in base.items() if not isinstance(v, dict)}
    base.update(sub)
    return base


def resolve_model(name: str) -> str:
    """If `name` matches a key under `[models]`, return the resolved GGUF
    path (with ~/$VAR expansion). Otherwise return `name` untouched."""
    cfg = load().get("models", {})
    if name in cfg:
        return os.path.expandvars(os.path.expanduser(str(cfg[name])))
    return name
