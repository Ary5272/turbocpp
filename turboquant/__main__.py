"""`python -m turboquant <subcommand>` — alias for the unified CLI in cli.py.

Back-compat shim: a bare `python -m turboquant <model_dir> <out_dir>` (the
pre-0.3 invocation) still works — we forward to `rotate <args>` if the
first positional looks like a path rather than a CLI subcommand.
"""

from __future__ import annotations

import sys

from .cli import main as _main


def _looks_like_path(arg: str) -> bool:
    """Heuristic: pre-0.3's first positional is a HF model directory."""
    return (
        "/" in arg
        or "\\" in arg
        or arg.startswith(".")
        or arg.startswith("~")
        or (len(arg) >= 2 and arg[1] == ":")  # Windows drive-letter path
    )


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    # Only forward to `rotate` if the first arg PLAINLY looks like a path
    # AND there's a second arg that also looks like one (matching the old
    # `python -m turboquant <model_dir> <out_dir>` shape). Anything else
    # falls through to argparse - which gives a helpful "invalid choice"
    # error or correct subparser dispatch.
    if (
        len(args) >= 2
        and _looks_like_path(args[0])
        and _looks_like_path(args[1])
        and not args[0].startswith("-")
    ):
        args = ["rotate", *args]
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
