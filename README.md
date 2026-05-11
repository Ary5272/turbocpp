# turbocpp

[![PyPI](https://img.shields.io/pypi/v/turbocpp?label=pypi)](https://pypi.org/project/turbocpp/)
[![Python](https://img.shields.io/pypi/pyversions/turbocpp.svg)](https://pypi.org/project/turbocpp/)
[![CI](https://github.com/Ary5272/turbocpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Ary5272/turbocpp/actions/workflows/ci.yml)
[![Docker](https://github.com/Ary5272/turbocpp/actions/workflows/docker.yml/badge.svg)](https://github.com/Ary5272/turbocpp/actions/workflows/docker.yml)
[![security](https://github.com/Ary5272/turbocpp/actions/workflows/security.yml/badge.svg)](https://github.com/Ary5272/turbocpp/actions/workflows/security.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![SLSA](https://img.shields.io/badge/SLSA-build%20provenance-blue)](https://slsa.dev/)

> **The unified llama.cpp CLI + offline Hadamard-rotation preprocessor.**
> A drop-in `turbocpp <subcommand>` for every llama.cpp workflow
> (generate, chat, serve, embed, quantize, perplexity, imatrix,
> speculative decoding, …), plus an orthogonal weight rotation that
> meaningfully reduces quantization error at **zero inference cost**.

| | |
|---|---|
| 🚀 **Live demo** | [huggingface.co/spaces/AIencoder/turboquant-visualizer](https://huggingface.co/spaces/AIencoder/turboquant-visualizer) |
| 📦 **PyPI** | `pip install 'turbocpp[runtime]'` ([PyPI](https://pypi.org/project/turbocpp/)) |
| 🐳 **GHCR** | `docker pull ghcr.io/ary5272/turbocpp:cpu` (also `:server`, `:turboquant`) |
| 🔧 **Wheel mirror** | [datasets/AIencoder/TurboCpp_Wheels](https://huggingface.co/datasets/AIencoder/TurboCpp_Wheels) — prebuilt `llama-cpp-python` for every CPU feature combo |
| 📜 **License** | MIT |

---

## Table of contents

1. [TL;DR](#tldr)
2. [Install](#install)
3. [Doctor (verify install)](#doctor-verify-install)
4. [Subcommand reference](#subcommand-reference)
5. [Model identifiers (`-m`)](#model-identifiers--m)
6. [Configuration file](#configuration-file)
7. [Constrained output (GBNF + JSON schema)](#constrained-output-gbnf--json-schema)
8. [HTTP server](#http-server)
9. [Speculative decoding](#speculative-decoding)
10. [Docker images](#docker-images)
11. [The TurboQuant story (quality vs speed)](#the-turboquant-story-quality-vs-speed)
12. [The math in one block](#the-math-in-one-block)
13. [Tests + CI](#tests--ci)
14. [Security](#security)
15. [Contributing](#contributing)
16. [Related work](#related-work)
17. [License](#license)

---

## TL;DR

```bash
# 1. Install (PyPI; ~5 MB base + 100-300 MB for the inference wheel)
pip install 'turbocpp[runtime]'

# 2. Verify (downloads TinyLlama, runs a sample completion)
turbocpp quickstart

# 3. Inference from a HuggingFace ref (auto-downloads + caches)
turbocpp generate \
    -m TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -p "Explain Hadamard rotation in one sentence:" -n 100

# 4. Multi-turn chat with persistent history (one /help away)
turbocpp chat -m TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# 5. OpenAI-compatible HTTP server on :8080 (drop-in for any OpenAI SDK)
turbocpp serve -m model.gguf --host 0.0.0.0 --port 8080

# 6. TurboQuant: same quality at smaller bit-width (the speed path)
turbocpp rotate    ~/models/Llama-3-8B  ~/models/Llama-3-8B-tq
turbocpp convert   ~/models/Llama-3-8B-tq  --outfile Llama-3-8B-tq.gguf
turbocpp quantize  Llama-3-8B-tq.gguf  Llama-3-8B-tq-Q3_K_M.gguf  Q3_K_M
```

---

## Install

The base install is intentionally tiny (just `huggingface_hub`). Extras
opt you into heavier deps so a rotation-only or CLI-only install doesn't
have to pull `torch` or `llama-cpp-python`:

| install command | what you get | size |
|---|---|---|
| `pip install turbocpp`                | CLI tools only — `pick-wheel`, `info`, `doctor`, `download`, `list-models`, `llama list`, `config`. No inference, no rotation. | ~5 MB |
| `pip install 'turbocpp[runtime]'`     | Adds `llama-cpp-python` — `generate`, `chat`, `serve`, `embed`, `tokenize`, `quickstart`. | ~5 MB + 100-300 MB depending on wheel |
| `pip install 'turbocpp[rotate]'`      | Adds `torch + transformers + safetensors` — the offline `rotate` pipeline. | ~5 MB + ~2 GB |
| `pip install 'turbocpp[all]'`         | Both runtime and rotate. | ~2.3 GB |

```bash
# Most users want this:
pip install 'turbocpp[runtime]'

# If your CPU/OS lacks a build toolchain, install llama-cpp-python from
# a prebuilt wheel instead of the [runtime] source build:
pip install turbocpp
pip install $(turbocpp pick-wheel)

# Force a specific CPU variant or GPU backend:
pip install $(turbocpp pick-wheel --gpu cuda12)
pip install $(turbocpp pick-wheel --variant basic_avx512_fma_f16c_vnni)

# Plan installs for another host (different Python version):
turbocpp pick-wheel --py 3.10 --variant basic_avx2_fma_f16c

# From the GitHub Release (always points at the latest tag — useful in
# environments where PyPI is mirrored / blocked):
pip install https://github.com/Ary5272/turbocpp/releases/latest/download/turbocpp-py3-none-any.whl
```

Supported platforms: **Python 3.10 / 3.11 / 3.12** on Linux + macOS +
Windows. CI runs the full test matrix on all three OSes every commit.

---

## Doctor (verify install)

`turbocpp doctor` walks a checklist and prints PASS / WARN / FAIL per
item. Exit code = number of FAILs, so it's safe to drop into a shell
script.

```bash
$ turbocpp doctor
turbocpp 1.0.0 doctor — linux
  [PASS]  python ≥ 3.10                          3.11.9
  [PASS]  cpu feature variant                    basic_avx2_fma_f16c
  [PASS]  llama-cpp-python                       0.3.16
  [PASS]  llama-cpp-python GPU offload           yes
  [PASS]  docker on PATH                         /usr/bin/docker
  [PASS]  image ghcr.io/ggml-org/llama.cpp:full  cached locally
  [PASS]  GPU                                    nvidia (nvidia-smi)
  [PASS]  torch (rotate)                         2.4.0 (cuda)
  [PASS]  HF wheel URL reachable                 https://huggingface.co/...
```

Flags:

- `--no-color` — strip ANSI escapes (pipe-friendly).
- `--no-network` — skip the wheel-URL HEAD probe (offline / air-gapped).

For a machine-readable view of the same data, use `turbocpp info`
(prints JSON). With `--field KEY[.NESTED]` it extracts one scalar so
shell scripts don't need `jq`:

```bash
$ turbocpp info --field cpu_variant
basic_avx2_fma_f16c

$ turbocpp info --field llama_cpp.version
0.3.16
```

---

## Subcommand reference

`turbocpp --help` is the canonical list. Run any subcommand with
`--help` for its flags. The big ones:

| subcommand | purpose | notes |
|---|---|---|
| `rotate` | Apply Walsh-Hadamard rotation to a HF model directory | needs `[rotate]` extra |
| `bench` | Synthetic rotation/quant MSE microbench (no model needed) | `--format json` for machine-readable |
| `generate` | One-shot inference | streams tokens; `--format jsonl` for one JSON-per-token |
| `chat` | Multi-turn REPL with persistent per-model history | `/help` for slash commands, `--history PATH` to override file |
| `serve` | OpenAI-compatible HTTP server (chat, completions, embeddings) | `--api-key generate` mints a fresh Bearer token |
| `speculative` | Speculative decoding (small draft proposes, big target verifies) | delegates to upstream `llama-speculative` |
| `embed` | Sentence embeddings | `--format json/jsonl/tsv` + `--normalize` |
| `tokenize` | Token count / IDs / pieces for a prompt | helpful for context-budget estimation |
| `download` | Fetch a GGUF from HuggingFace into `~/.cache/turbocpp/models/` | accepts `owner/repo file.gguf`, `owner/repo:file.gguf`, or `hf://owner/repo/file.gguf` |
| `quickstart` | Download TinyLlama + run a sample completion | proves the install works end-to-end |
| `doctor` | Environment health check (deps, wheels, docker, GPU, torch) | exit code = number of FAILs |
| `info` | Runtime topology as JSON | `--field KEY` extracts one value |
| `version` | Bare version string | also `turbocpp -V` |
| `list-models` | Show configured aliases + cached GGUFs (with sizes) | `--format json` for scripts |
| `list-templates` | Chat templates llama-cpp-python knows about | |
| `rm-model` | Delete cached GGUFs | `--all`, `--dry-run`, glob support |
| `config` | Scaffold / show / validate `~/.config/turbocpp/config.toml` | `init / show / path / validate` |
| `pick-wheel` | Print best prebuilt `llama-cpp-python` wheel URL for this host | `--gpu`, `--py`, `--variant`, `--all` |
| `convert` | HF model → GGUF (delegates to llama.cpp Docker image) | needs Docker |
| `quantize` | GGUF → quantized GGUF (defaults to `Q4_K_M` if no type given) | needs Docker |
| `perplexity` | Perplexity on a corpus | needs Docker |
| `imatrix` | Importance matrix for K-quants | needs Docker |
| `llama-cli` | Raw `llama-cli` passthrough | needs Docker |
| `llama-bench` | Official llama.cpp tok/s microbench | needs Docker |
| `llama <tool>` | Generic passthrough to any llama.cpp binary | `turbocpp llama list` enumerates available tools |

### Sampling knobs (`generate`, `chat`)

`-t / --temperature`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`,
`--stop SEQ` (repeatable), `--seed`, `--n-predict N`.

### Performance knobs (`generate`, `chat`, `serve`, `embed`)

`-ngl / --n-gpu-layers N` (offload N transformer layers to GPU; `-1` = all),
`--ctx N` (context window), `--threads N`, `--n-batch N`,
`--rope-freq-base`, `--rope-freq-scale`, `--flash-attn`.

`turbocpp` warns when `-ngl` is set but the installed `llama-cpp-python`
wheel is CPU-only — that combination silently ignores the flag, which
is confusing the first time you hit it.

### JSONL output (`generate`)

`--format jsonl` emits one JSON object per token (`{token, index,
finish_reason}`), perfect for piping into `jq` / a downstream parser:

```bash
turbocpp generate -m m.gguf -p "Tell a story." -n 32 --format jsonl \
  | jq -c '{i: .index, t: .token}'
```

---

## Model identifiers (`-m`)

`-m` accepts four forms, resolved in this order:

1. **Local path** — `./model.gguf`, `/abs/path/m.gguf`, `C:\models\m.gguf`
2. **Config alias** — short names from `[models]` in `~/.config/turbocpp/config.toml`
3. **HF colon ref** — `owner/repo:filename.gguf` (auto-fetched + cached)
4. **HF URL ref** — `hf://owner/repo/filename.gguf` (same semantics, alternate syntax)

The HF forms download into `~/.cache/turbocpp/models/` (overridable via
`$XDG_CACHE_HOME`) on first use; subsequent runs hit the cache. They
work everywhere a model path is accepted — `generate`, `chat`, `serve`,
`embed`, `tokenize`, `speculative`. `download` accepts the same forms
as a single arg:

```bash
turbocpp download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
turbocpp download hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
turbocpp download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf  # classic two-arg form
turbocpp download owner/repo:file.gguf --sha256 abc123...  # verify after download
```

Missing-model errors are friendly (path + resolution chain + hint to
`turbocpp list-models`):

```
error: model not found: ~/models/missing.gguf
  (tried: 'missing' -> '~/models/missing.gguf')
  hint: `turbocpp list-models` shows aliases + cached GGUFs
```

---

## Configuration file

`turbocpp config init` scaffolds an example `~/.config/turbocpp/config.toml`
(or `$XDG_CONFIG_HOME/turbocpp/config.toml`):

```toml
[defaults]
threads = 8
ctx     = 4096

[defaults.generate]
temperature = 0.6
top_p       = 0.9

[defaults.chat]
system    = "You are a concise assistant."
n_predict = 1024

[defaults.serve]
host = "127.0.0.1"
port = 8080

[models]
# short alias -> GGUF path (resolved by `turbocpp generate -m tiny`)
tiny    = "~/models/tinyllama-1.1b-chat.Q4_K_M.gguf"
llama3  = "~/models/Llama-3-8B-Q4_K_M.gguf"
```

| command | what it does |
|---|---|
| `turbocpp config path`     | print the resolved config path |
| `turbocpp config show`     | print the file contents (or note absence) |
| `turbocpp config init`     | scaffold a starter file (refuses to overwrite without `--force`) |
| `turbocpp config validate` | parse and fail non-zero on malformed TOML |

Explicit CLI flags always win over config defaults. A typo'd
`defaults.<subcommand>` (scalar instead of table) is silently ignored
rather than crashing.

---

## Constrained output (GBNF + JSON schema)

```bash
# GBNF grammar (any llama.cpp grammar file):
turbocpp generate -m m.gguf -p "Spell a date:" --grammar date.gbnf

# JSON schema (constrains output to valid JSON matching the schema):
turbocpp generate -m m.gguf -p "Emit a Person:" --json-schema person.json --format jsonl
```

`turbocpp` validates the file exists and is well-formed before handing
it to llama-cpp-python — the upstream "failed to parse grammar at
offset 0" error for a missing file becomes a clear `--grammar: file
not found: <path>`. If you accidentally pass both `--grammar` and
`--json-schema`, you get a warning and `--grammar` wins (so you don't
silently get the wrong constraint).

The chat REPL accepts the same flags. Inside the REPL, `/help` lists
the slash commands:

```
/quit, /exit          end the session (history saved)
/reset                clear conversation
/history              dump conversation so far
/tokens               report token count of current history
/system <TEXT>        replace the system prompt (no TEXT clears it)
/save <PATH>          save to PATH (.md or .json by extension)
/load <PATH>          load JSON conversation from PATH
/multi                next message reads until a line saying EOF
```

History persists per-model at `~/.cache/turbocpp/chat-<sha1>.json`.
Override with `--history PATH`, or disable resume entirely with
`--no-resume`. Readline gives you up-arrow recall and `Ctrl-R` search
on POSIX (no-op on Windows).

---

## HTTP server

`turbocpp serve` wraps llama-cpp-python's FastAPI server. It speaks the
OpenAI API, so any OpenAI SDK works out-of-the-box:

```bash
turbocpp serve -m model.gguf --host 0.0.0.0 --port 8080 \
               --ctx 4096 --threads 8 -ngl 99
# [serve] http://0.0.0.0:8080  (OpenAI-compatible: /v1/chat/completions,
#         /v1/completions, /v1/embeddings)
```

Bearer auth is optional:

```bash
turbocpp serve -m model.gguf --api-key sk-my-secret
# Or have turbocpp mint one for you:
turbocpp serve -m model.gguf --api-key generate
# [serve] generated API key: T8fXKj_qLp9...
# [serve] auth required: 'Authorization: Bearer T8fXKj...' (truncated)
```

`TURBOCPP_API_KEY` env var works as a fallback. The `--api-key generate`
sentinel mints a 32-char URL-safe token at startup — handy when you
want auth but don't have a secret manager in front of you.

The server pre-flights the model path (so you get
`error: model not found: <path>` instead of uvicorn's stack trace),
honors all the same `-ngl` / `--n-batch` / config knobs as `generate`,
and is what `ghcr.io/ary5272/turbocpp:server` runs by default.

---

## Speculative decoding

Speculative decoding compounds with the TurboQuant memory-bandwidth
win. A smaller "draft" model proposes K tokens; the bigger "target"
model verifies them in a single forward pass. Typical end-to-end
speedup: **1.5–3×** on memory-bound CPUs, more with bigger model gaps.

```bash
turbocpp speculative \
    -m Llama-3-8B-Q4_K_M.gguf \
    -d Llama-3-8B-Q2_K.gguf \
    -p "Explain Hadamard rotation in one paragraph." \
    -n 256 -k 4 --ctx 4096 -ngl 99
```

We delegate to upstream's tested `llama-speculative` binary (the older
in-process Python harness mutated private `Llama.n_tokens` to roll
back the KV cache, which broke silently across llama-cpp-python
versions). Docker is required because `llama-speculative` ships only
in the official `ggml-org/llama.cpp:full` image — `turbocpp` mounts
both model files in and prints a clear error if Docker is missing.

```bash
# Full 4-way head-to-head: baseline single-model, baseline speculative,
# TurboQuant single-model, TurboQuant speculative. ~5 minutes per pass.
./scripts/bench_speculative.sh /path/to/HF/Llama-3-8B
```

---

## Docker images

Three GHCR images, all install `llama-cpp-python` from a **prebuilt
wheel** at `AIencoder/TurboCpp_Wheels` — no source compile,
~30s image build (vs ~10 min for source). Runs on any x86_64 host with
AVX2 + FMA + F16C (effectively every cloud / CI / consumer CPU since
2013).

| image | what's inside | size |
|---|---|---|
| `ghcr.io/ary5272/turbocpp:cpu`        | turbocpp CLI + llama-cpp-python (prebuilt wheel) | ~500 MB |
| `ghcr.io/ary5272/turbocpp:server`     | inherits `:cpu`, ENTRYPOINT = `turbocpp serve` on `:8080` | ~500 MB |
| `ghcr.io/ary5272/turbocpp:turboquant` | inherits `:cpu`, adds CPU-only PyTorch for `rotate` | ~2.0 GB |

```bash
# Inference runtime + unified CLI
docker run --rm -v ~/models:/models ghcr.io/ary5272/turbocpp:cpu \
       generate -m /models/m.gguf -p "Hello" -n 64

# OpenAI-compatible HTTP server on :8080
docker run --rm -p 8080:8080 -v ~/models:/models ghcr.io/ary5272/turbocpp:server \
       -m /models/m.gguf

# Offline Hadamard rotation pipeline
docker run --rm -v ~/models:/models ghcr.io/ary5272/turbocpp:turboquant \
       rotate /models/Llama-3-8B /models/Llama-3-8B-tq
```

A new image is pushed to GHCR on every `main` commit and every `v*` tag
([docker.yml](.github/workflows/docker.yml)). Build locally pointing
at a different prebuilt wheel (CUDA / AVX-512 / Sapphire Rapids / …):

```bash
docker build --target cpu \
    --build-arg LLAMA_CPP_WHEEL_URL="$(turbocpp pick-wheel --gpu cuda12)" \
    -t turbocpp:cpu-cuda12 .
```

Override the upstream llama.cpp tool image (the one we mount tools
out of for `convert / quantize / perplexity / speculative / …`) via
the `TURBOCPP_LLAMA_IMAGE` env var:

```bash
TURBOCPP_LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda turbocpp quantize ...
```

The Dockerfile sanity check runs `turbocpp doctor --no-network` at
build time, so any wheel that doesn't dynamically link or that fails
to import `llama_cpp` / `huggingface_hub` breaks the build *before*
ship, not at the user's terminal.

```
   ┌───────────────────────────────────────────────────────────────┐
   │ HF model ──► turbocpp rotate ──► turbocpp convert + quantize  │
   │                                                ▼              │
   │                            standard GGUF, runs anywhere       │
   │                            llama.cpp does — every backend,    │
   │                            every architecture, every sampler  │
   └───────────────────────────────────────────────────────────────┘
```

---

## The TurboQuant story (quality vs speed)

### What llama.cpp already ships

- **Architectures**: LLaMA 1/2/3, Mistral, Mixtral (MoE), Qwen 1/2/2.5, Phi 1/2/3, Gemma 1/2, Falcon, MPT, BLOOM, GPT-2, GPT-NeoX, StableLM, Baichuan, Yi, RWKV, Mamba, …
- **Quantization**: Q2_K, Q3_K_S/M/L, Q4_0/1, Q4_K_S/M, Q5_0/1, Q5_K_S/M, Q6_K, Q8_0, Q8_K, IQ1_S/M, IQ2_XXS/XS/S/M, IQ3_XXS/S/M, IQ4_XS/NL, BF16, F16, F32
- **Backends**: CPU (AVX/AVX2/AVX-512/NEON/AMX), CUDA, Metal, Vulkan, SYCL, ROCm, Kompute, OpenCL, RPC, BLAS
- **Sampling**: greedy, temperature, top-k, top-p, min-p, typical-p, tail-free, locally-typical, dynatemp, mirostat v1+v2, repetition penalty, frequency penalty, presence penalty, logit bias, GBNF grammar, JSON mode, classifier-free guidance, beam search, speculative decoding, lookahead decoding
- **Runtime**: continuous batching, parallel sequences, prompt caching, KV-cache shifting/defrag, embeddings, reranking, LoRA hotswap, multi-modal (LLaVA, Phi-3-vision, MiniCPM-V), tools/function-calling, chat templates for every major model
- **Server**: `llama-server` (OpenAI-compatible HTTP API), web UI

TurboQuant adds: a **~100-line Python module** that rotates the
model's weight matrices in-place using Walsh-Hadamard transforms. The
rotation cancels through the residual-stream linear pieces (it's
orthogonal) so the model is fp32-bit-identical, but the per-weight-block
max-abs that drives Q4 / Q4_K rounding error drops 3–5×, which
translates to **0.3–0.5 perplexity points back at Q4_K_M** on
LLaMA-2-7B (bigger gains at lower bit-widths).

### Does this run faster than stock llama.cpp?

Honest answer has two halves.

**Same bit-width: NO.** Quantizing a TurboQuant-rotated model at Q4_K_M
and running on stock llama.cpp gives the same tokens/sec as a non-rotated
Q4_K_M. Same bytes per weight, same kernels, same memory layout. What
you get is **better quality at the same speed** — 0.3–0.5 perplexity
back on LLaMA-2-7B.

**Drop a bit-width tier: YES.**

| recipe | bytes/weight | quality | wall-clock decode |
|---|---|---|---|
| baseline Q4_K_M (no rotation)    | 4.625 | reference         | reference |
| TurboQuant Q4_K_M                | 4.625 | **better**        | same |
| **TurboQuant Q3_K_M**            | 3.5   | ≈ baseline Q4_K_M | **~1.20–1.30× faster** on memory-bound CPUs |
| TurboQuant Q2_K (aggressive)     | 2.6   | usable for some tasks | **~1.5× faster** |

The speedup comes from memory bandwidth: decoding is bandwidth-bound on
nearly all consumer CPUs (and on Sapphire Rapids when AMX tiles aren't
saturated). Fewer bytes per weight read each step = fewer cycles waiting
on DRAM. Combine with `pick-wheel`'s CPU-tier auto-pick (~10–30% over
the AVX2 default on Sapphire Rapids / Zen4) and speculative decoding
(1.5–3× on top) and you can stack **2–4× over a stock
`pip install llama-cpp-python` flow**.

### KV cache: also YES (long context)

`turboquant.kvcache.rotate_kv_for_cache_quant()` Hadamard-rotates the
attention output projection so K and V live in a Gaussianized frame
**inside** the KV cache. Combine with llama.cpp's
`--cache-type-k q4_0 --cache-type-v q4_0` and you get usable quality
at half the KV bandwidth — meaningful at 8K+ context where KV reads
dominate.

### Reproduce the numbers

```bash
# Synthetic micro (1 second, no model needed):
turbocpp bench --format json

# End-to-end on your machine, real GGUF (~3 min CPU, less w/ GPU):
./scripts/bench_e2e.sh /path/to/HF/Llama-3-8B

# 4-way head-to-head with speculative decoding:
./scripts/bench_speculative.sh /path/to/HF/Llama-3-8B
```

---

## The math in one block

For each linear layer `y = W x` in the residual stream, with `H` an
orthogonal block-Hadamard:

```
W' = H · W           (output axis rotated)         ← producers
W' = W · Hᵀ          (input axis rotated)          ← consumers
```

We pair every producer with its consumer:

```
producers (write to residual stream):  tok_embed, W_o,    W_down
consumers (read from residual stream): W_q, W_k, W_v, W_gate, W_up, lm_head
```

Since `H · Hᵀ = I`, the rotations cancel through the network. Forward
pass in fp32 is bit-identical. But quantization noise is computed on
the ROTATED weights, whose per-block distribution is near-Gaussian
thanks to the central-limit theorem — and Gaussian distributions
quantize well, while heavy-tailed real LLM weights don't.

RMSNorm is rotation-equivariant only if its `γ` vector is uniform.
Pass 1 absorbs each `γ` into the FOLLOWING linear (`W ← W · diag(γ)`)
and then sets `γ ← 1`, after which the rotation is safe. See
[`turboquant/turboquant.py`](turboquant/turboquant.py) — ~100 lines —
and [`turboquant/test_turboquant.py`](turboquant/test_turboquant.py)
for the invariance proofs (orthogonality, dot-product preservation,
QK^T preservation in the KV-rotation variant).

---

## Tests + CI

```bash
pytest -q turboquant/                       # ~120 tests (CLI + rotation math)
ctest --test-dir extras/standalone/build    # standalone-engine kernels
```

CI runs on Linux + Windows + macOS across Python 3.10 / 3.11 / 3.12
(9 OS × Python combinations) for every push and PR, plus:

- `ruff format --check` + `ruff check` (no drift slips past CI)
- `mypy turboquant --ignore-missing-imports` (gating)
- The C++17 standalone engine builds and runs its unit tests on all 3 OSes
- Per-job timeouts (lint=5min, cli-tests=10min, math=15min, docker=20min)
  so a stuck job can't burn an hour of runner time
- `gitleaks` secret scan + GitHub CodeQL static analysis (weekly + on push)
- `Dockerfile` builds three multi-stage images (`:cpu`, `:server`,
  `:turboquant`) with a `turbocpp doctor --no-network` sanity gate

---

## Security

- All `.github/workflows/*.yml` actions are pinned to commit SHAs (immutable);
  Dependabot keeps them current.
- Wheels and Docker images carry [SLSA build provenance attestations](https://slsa.dev/)
  signed by GitHub's sigstore-backed OIDC identity — verify with
  `gh attestation verify <file> --owner Ary5272`.
- `actions/checkout` runs with `persist-credentials: false` so the
  `GITHUB_TOKEN` is never written to `.git/config` on the runner.
- Workflow `permissions:` default to `contents: read`; each step
  grants only the minimum scope it needs.
- Weekly [`gitleaks`](https://github.com/gitleaks/gitleaks) +
  GitHub CodeQL scans on `main`.
- Dependabot is on for `pip`, `docker`, and `github-actions`.
- See [`SECURITY.md`](SECURITY.md) for vulnerability reporting (use
  GitHub's private vulnerability report; we respond within 72h).

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, lint/test gates,
the architecture map, the release flow, and what does / doesn't get
accepted. TL;DR:

```bash
git clone https://github.com/Ary5272/turbocpp
cd turbocpp
pip install -e '.[dev,runtime]'
ruff format turboquant scripts && ruff check turboquant scripts
mypy turboquant --ignore-missing-imports
pytest -q turboquant/
```

When adding a new CLI subcommand, route any `Llama(...)` construction
through `_open_llama(args, **overrides)` so `-ngl`, sampling knobs,
RoPE knobs, and HF-ref resolution stay wired up in exactly one place.

---

## Related work

- **QuaRot** — Ashkboos et al., 2024. Same core idea (Hadamard rotation
  before quantization). TurboQuant is the tiniest possible implementation
  of the residual-stream rotation; QuaRot also does
  activation-quantization aware modifications.
- **SpinQuant** — Liu et al., 2024. Learns the rotation matrix instead
  of using a fixed Hadamard. Complementary; trainable counterpart.
- **GPTQ** — Frantar et al., 2022. Calibration-based per-layer
  quantization; complementary, can be combined with rotation.
- **AWQ** — Lin et al., 2023. Activation-aware scaling; complementary.
- **HQQ** — half-quadratic quantization; orthogonal direction.

---

## License

- TurboQuant Python code + CLI: **MIT** ([LICENSE](LICENSE))
- llama.cpp (pulled at runtime via Docker, no submodule): **MIT**
  ([upstream LICENSE](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE))
- All third-party Python deps retain their own licenses; the project
  ships no vendored copies of anything beyond a `extras/standalone/`
  C++17 reference engine that's MIT under this repo's LICENSE.
