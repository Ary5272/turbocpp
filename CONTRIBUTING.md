# Contributing to turbocpp

Thanks for the interest. The bar for contributions is "doesn't break the
test suite, doesn't add a heavy dep, and the user-facing change is
documented." Read on for the practicalities.

## Quick setup

```bash
git clone https://github.com/Ary5272/turbocpp
cd turbocpp
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e '.[dev,runtime]'
```

`[dev]` pulls `pytest`, `ruff`, `mypy`, and the build/twine tooling.
`[runtime]` adds `llama-cpp-python` so you can exercise inference paths.

## Tests, lint, types - the same gates CI runs

```bash
ruff format turboquant scripts
ruff check  turboquant scripts
mypy turboquant --ignore-missing-imports
pytest -q turboquant/
```

All four must pass before pushing. CI re-runs them on Linux, macOS, and
Windows across Python 3.10 / 3.11 / 3.12.

## Repo layout

| path | what's there |
|---|---|
| `turboquant/cli.py`           | every CLI subcommand handler + the argparse setup |
| `turboquant/runtime_probe.py` | shared probes for `info` / `doctor` (single source of truth) |
| `turboquant/config.py`        | `~/.config/turbocpp/config.toml` parser + HF-ref resolution |
| `turboquant/cpu_features.py`  | CPU feature detection -> wheel URL |
| `turboquant/llama_docker.py`  | dispatcher for `turbocpp llama <tool>` (delegates to ggml-org's image) |
| `turboquant/turboquant.py`    | the actual Hadamard-rotation math (used by `turbocpp rotate`) |
| `turboquant/test_cli.py`      | no-runtime-dep CLI tests (CI runs these on every push) |
| `turboquant/test_turboquant.py` | rotation-math tests (need torch) |
| `extras/standalone/`          | parallel from-scratch C++17 engine (kept as a study reference) |
| `.github/workflows/`          | ci / docker / release / security pipelines |

## Adding a CLI subcommand

1. Write the handler in `cli.py` as `def _cmd_<name>(args) -> int:`.
2. Register it inside `main()` via `sub.add_parser(...)` + `set_defaults(func=...)`.
3. Add a row to the `test_each_subcommand_has_help` parametrize at the
   top of `turboquant/test_cli.py` so help-output never silently breaks.
4. If the subcommand opens a `Llama(...)`, call `_open_llama(args, ...)`
   instead of constructing one inline - `_open_llama` is the single
   place that wires up `-ngl`, `--n-batch`, `--rope-*`, HF-ref
   resolution, and any future shared knobs.

## Releasing

Tags drive everything. To cut a release:

```bash
# 1. Bump pyproject.toml version, commit, push.
# 2. Tag and push the tag:
git tag v0.X.Y -m "v0.X.Y - one-line summary"
git push --tags
# 3. CI builds the wheel + sdist, attaches SLSA build provenance,
#    creates the GitHub Release, publishes to PyPI, and mirrors to the
#    HF wheel dataset (if PYPI_API_TOKEN / HF_TOKEN are set).
# 4. `bash scripts/gen_changelog.sh` regenerates CHANGELOG.md from
#    `git tag` + commit subjects. Commit and push that.
```

The Docker image at `ghcr.io/ary5272/turbocpp` is rebuilt on every
`main` push and every `v*` tag.

## What gets accepted

- Bugfixes with a regression test.
- Small CLI / UX improvements that don't grow the dep tree.
- Performance work backed by `turbocpp bench --format json` numbers
  before / after.
- Documentation that fixes something concretely wrong (broken example,
  stale flag).

## What gets rejected (or asked to slim down)

- Heavy new dependencies. The base install is meant to stay small;
  optional features go behind an `[extra]`.
- Reformatting churn unrelated to the change you're proposing.
- Features that overlap upstream llama.cpp's existing knobs (we delegate
  to `ggml-org/llama.cpp:full` for those - don't reimplement).
- Changes to `extras/standalone/*` - that subtree is intentionally
  frozen as a study reference; new work belongs in the Python package.

## Reporting security issues

See [SECURITY.md](SECURITY.md). Use private vulnerability reports, not
public issues.
