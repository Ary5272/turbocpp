# Changelog

All notable changes per release. The "Unreleased" section accumulates
work on `main` since the last tag; everything below is auto-generated
from `git tag` + commit subjects (see `scripts/gen_changelog.sh`).

## [Unreleased]

## [v0.29.0] - 2026-05-07

_build_grammar conflict warning + tests

## [v0.28.0] - 2026-05-07

pick-wheel --py / --variant overrides

## [v0.27.0] - 2026-05-07

chat readline up-arrow + ctrl-R search

## [v0.26.1] - 2026-05-07

re-release with format-clean main (v0.26.0 wheel was fine but main ci was red)

## [v0.26.0] - 2026-05-07

split torch/transformers into [rotate] extra (base install ~5MB)

## [v0.25.0] - 2026-05-07

info --field for scriptable scalar extraction

## [v0.24.0] - 2026-05-07

_stop_list helper, _human_size, list-models KB/MB/GB

## [v0.23.0] - 2026-05-07

model-not-found friendly errors in _open_llama + serve pre-flight check

## [v0.22.0] - 2026-05-07

rotate --overwrite guard against accidental clobber

## [v0.21.0] - 2026-05-07

doctor README example, build-arg fix, sha256 paste-tolerance, chat-history test

## [v0.20.0] - 2026-05-07

serve config defaults + sampling-test extension

## [v0.19.0] - 2026-05-07

--top-k / --min-p / --repeat-penalty for generate+chat (via _sampling_kwargs)

## [v0.18.0] - 2026-05-06

cache_dir/state_dir refactor + CPU-only -ngl warning + ci timeouts

## [v0.17.0] - 2026-05-06

chat /help+/tokens, /save+/load polish, serve startup URL, speculative -ngl, rotate help, doctor color toggle

## [v0.16.0] - 2026-05-06

download hf://, speculative pre-flight, CI per-job timeouts, Dockerfile libopenblas0 + sanity probe

## [v0.15.0] - 2026-05-06

TURBOCPP_LLAMA_IMAGE override + doctor color/no-color + embed/tokenize TTY guard + speculative HF-ref help + grammar errors + Dockerfile sanity + docs

## [v0.14.0] - 2026-05-06

bench --format json + chat /help + epilog tips + chat system mismatch warn + bench scripts modernized + CONTRIBUTING + docker-compose fix + defaults_for hardening

## [v0.13.0] - 2026-05-06

turbocpp config subcommand + torch in runtime probe + serve --api-key generate + better docker-missing UX + llama list

## [v0.12.0] - 2026-05-06

speculative HF refs, /help, doctor --no-network, generate --n-batch/--rope-*/--flash-attn, download combined ref

## [v0.11.4] - 2026-05-06

chore: regen CHANGELOG.md for v0.11.4

## [v0.11.3] - 2026-05-06

chore: regen CHANGELOG.md for v0.11.3

## [v0.11.2] - 2026-05-06

chore: regen CHANGELOG.md for v0.11.2

## [v0.11.1] - 2026-05-06

chore: regen CHANGELOG.md for v0.11.1

## [v0.11.0] - 2026-05-06

chore: regen CHANGELOG.md for v0.11.0

## [v0.10.1] - 2026-05-06

fix Windows cp1252 crash on `turbocpp --help`

## [v0.10.0] - 2026-05-06

runtime_probe, version+rm-model, --n-gpu-layers, JSONL, default Q4_K_M, badges, CHANGELOG

## [v0.9.0] - 2026-05-05

better generate/chat/embed/serve, list-models/templates, quickstart

## [v0.8.0] - 2026-05-04

embed/tokenize/download cmds, grammar+stop+seed, config.toml, arm64 detect

## [v0.7.0] - 2026-05-03

doctor, persistent chat, lazy imports, ruff+mypy CI, real tests

## [v0.6.1] - 2026-05-03

— Dockerfile uses PyPI defaults; add mirror_wheels.py

## [v0.6.0] - 2026-05-03

HF mirror swap + real subcommands + correct speculative

## [v0.5.0] - 2026-05-02

fix(workflows): use real action SHAs instead of made-up ones

## [v0.4.0] - 2026-05-02

speculative decoding, CPU-tier auto-pick, GPU wheel path, ggml patch

## [v0.3.1] - 2026-05-02

release.yml: detect PYPI_API_TOKEN via a step output (GH Actions parse error)

## [v0.3.0] - 2026-04-30

release workflow + GitHub Releases install path

