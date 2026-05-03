"""TurboQuant — offline Hadamard-rotation preprocessor for LLM weights.

Heavy deps (torch, transformers) are imported lazily by the rotate_*
helpers, so `import turboquant` and the CLI's bench / generate / serve /
chat / info / pick-wheel / llama-* paths work without torch installed.
Only `turbocpp rotate` actually needs torch.
"""

from __future__ import annotations

# Top-level public API. Functions are re-exported as thin shims that
# import their heavy deps on first call.
__all__ = [
    "hadamard_matrix",
    "block_hadamard_inplace",
    "rotate_llama_model",
    "fuse_norms_into_next",
    "apply_residual_rotation",
    "rotate_kv_for_cache_quant",
]


def __getattr__(name):
    """Lazy attribute resolution — defers torch import until first use."""
    if name in {"hadamard_matrix", "block_hadamard_inplace"}:
        from . import hadamard as _h

        return getattr(_h, name)
    if name in {"rotate_llama_model", "fuse_norms_into_next", "apply_residual_rotation"}:
        from . import turboquant as _tq

        return getattr(_tq, name)
    if name == "rotate_kv_for_cache_quant":
        from . import kvcache as _kv

        return _kv.rotate_kv_for_cache_quant
    raise AttributeError(f"module 'turboquant' has no attribute {name!r}")


# Single source of truth: pyproject.toml's [project].version.
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("turbocpp")
    except PackageNotFoundError:  # editable / source tree
        __version__ = "0.0.0+source"
except ImportError:  # py <3.8 (unsupported)
    __version__ = "0.0.0+legacy"
