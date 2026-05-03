"""`python -m turboquant <subcommand>` — alias for the unified CLI in cli.py.

Back-compat shim: a bare `python -m turboquant <model_dir> <out_dir>` (the
pre-0.3 invocation) still works — we forward to `rotate <args>` if the
first positional looks like a path rather than a known subcommand name.
"""

from __future__ import annotations

import sys

from .cli import main as _main

_KNOWN = {"rotate", "bench", "generate", "serve", "-h", "--help"}


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args[0] not in _KNOWN:
        # Pre-0.3 form: `python -m turboquant <model_dir> <out_dir> [...]`
        # Forward by inserting "rotate" as the subcommand.
        args = ["rotate", *args]
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
