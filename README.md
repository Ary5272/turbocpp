# turbocpp

[![PyPI](https://img.shields.io/pypi/v/turbocpp?label=pypi)](https://pypi.org/project/turbocpp/)
[![Python](https://img.shields.io/pypi/pyversions/turbocpp.svg)](https://pypi.org/project/turbocpp/)
[![CI](https://github.com/Ary5272/turbocpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Ary5272/turbocpp/actions/workflows/ci.yml)
[![Docker](https://github.com/Ary5272/turbocpp/actions/workflows/docker.yml/badge.svg)](https://github.com/Ary5272/turbocpp/actions/workflows/docker.yml)
[![security](https://github.com/Ary5272/turbocpp/actions/workflows/security.yml/badge.svg)](https://github.com/Ary5272/turbocpp/actions/workflows/security.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![SLSA](https://img.shields.io/badge/SLSA-build%20provenance-blue)](https://slsa.dev/)

> **llama.cpp + TurboQuant.** Every llama.cpp feature, plus an offline
> Hadamard-rotation preprocessor that meaningfully improves the quality
> of any quantization (Q4_0 / Q4_K_M / Q6_K / …) at zero inference cost.

| | |
|---|---|
| 🚀 **Live demo** | [huggingface.co/spaces/AIencoder/turboquant-visualizer](https://huggingface.co/spaces/AIencoder/turboquant-visualizer) |
| 📦 **Python package** | `pip install turbocpp` ([PyPI](https://pypi.org/project/turbocpp/)) — `[runtime]` extra adds llama-cpp-python |
| 🐳 **Docker images** | `docker pull ghcr.io/ary5272/turbocpp:cpu` (also `:server`, `:turboquant`) |
| 🔧 **Wheel mirror** | [datasets/AIencoder/TurboCpp_Wheels](https://huggingface.co/datasets/AIencoder/TurboCpp_Wheels) — prebuilt llama-cpp-python for every CPU feature combo |

## Install

The base install is intentionally tiny (just `huggingface_hub`). Extras
opt you into heavier deps:

| install command | what you get | size |
|---|---|---|
| `pip install turbocpp`                | CLI tools only — `pick-wheel`, `info`, `doctor`, `download`, `list-models`, `llama list`. No inference, no rotation. | ~5 MB |
| `pip install 'turbocpp[runtime]'`     | Adds `llama-cpp-python` — `generate`, `chat`, `serve`, `embed`, `tokenize`, `quickstart`. | ~5 MB + 100-300 MB depending on wheel |
| `pip install 'turbocpp[rotate]'`      | Adds `torch + transformers` — the offline `rotate` pipeline. | ~5 MB + ~2 GB |
| `pip install 'turbocpp[all]'`         | Both runtime and rotate.                                                                                              | ~2.3 GB |

```bash
# Most users want this:
pip install 'turbocpp[runtime]'

# If your CPU/OS lacks a build toolchain, install llama-cpp-python from
# a prebuilt wheel instead of the [runtime] source build:
pip install turbocpp
pip install $(turbocpp pick-wheel)

# From the GitHub Release (always points at the latest tag — useful in
# environments where PyPI is mirrored / blocked):
pip install https://github.com/Ary5272/turbocpp/releases/latest/download/turbocpp-py3-none-any.whl
```

After install, `turbocpp doctor` reports what's wired (color-coded
PASS / WARN / FAIL with one line per check):

```bash
$ turbocpp doctor
turbocpp 0.20.0 doctor - linux
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

Pipe-friendly: `turbocpp doctor --no-color` strips ANSI escapes;
`turbocpp doctor --no-network` skips the wheel HEAD probe.

After install you get a `turbocpp` CLI:

```bash
turbocpp rotate      ./Llama-3-8B  ./Llama-3-8B-tq        # offline Hadamard rotation
turbocpp generate    -m model.gguf -p "Hello" -n 64        # one-shot inference
turbocpp chat        -m model.gguf                         # multi-turn REPL with /help
turbocpp serve       -m model.gguf --host 0.0.0.0 --port 8080
turbocpp speculative -m target.gguf -d draft.gguf -p "..." # 1.5-3× faster decode
turbocpp pick-wheel                                         # auto-pick fastest wheel
turbocpp pick-wheel  --gpu cuda12                           # GPU variant URL
turbocpp bench                                              # rotation/quant MSE microbench

# Sampling knobs (works on `generate` and `chat`):
turbocpp generate -m m.gguf -p "Hi" -t 0.7 --top-p 0.95 --top-k 40 --min-p 0.05 --repeat-penalty 1.1

# Constrained output (GBNF or JSON schema):
turbocpp generate -m m.gguf -p "..." --grammar grammar.gbnf
turbocpp generate -m m.gguf -p "..." --json-schema schema.json --format jsonl

# `-m` accepts: a local GGUF, a config alias, or a HuggingFace ref. The
# ref is downloaded on first use into ~/.cache/turbocpp/models/ and cached.
turbocpp generate -m TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hi"
turbocpp generate -m hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Hi"

# every llama.cpp tool, no submodule, no compile — pulls ggml-org/llama.cpp:full
turbocpp convert    /models/Llama-3-8B    --outfile /models/m.gguf
turbocpp quantize   /models/m.gguf  /models/m-Q4_K_M.gguf  Q4_K_M
turbocpp perplexity -m /models/m-Q4_K_M.gguf -f /data/wiki.test.raw
turbocpp imatrix    -m /models/m.gguf -f /data/calib.txt -o imatrix.dat
turbocpp llama-cli  -m /models/m.gguf -p "Hello"
turbocpp llama-bench -m /models/m.gguf
turbocpp llama       <any-tool>           # raw passthrough
```

### Security

- All `.github/workflows/*.yml` actions are pinned to commit SHAs (Dependabot keeps them current).
- Wheels and Docker images carry [SLSA build provenance attestations](https://slsa.dev/) — verify with `gh attestation verify <file> --owner Ary5272`.
- Weekly `gitleaks` + CodeQL scans on `main`.
- See [`SECURITY.md`](SECURITY.md) for vulnerability reporting.

### Get actual speedups, not just better quality

```bash
# (1) Auto-install the fastest llama-cpp-python wheel for your CPU
#     (AVX-512 / VNNI / AMX automatically chosen):
pip install $(turbocpp pick-wheel)

# (2) Speculative decoding — biggest single decode win, no kernels needed.
#     Smaller draft proposes K tokens; bigger target verifies in one pass.
turbocpp speculative \
    -m  Llama-3-8B-tq-Q4_K_M.gguf      \
    -d  Llama-3-8B-tq-Q2_K.gguf        \
    -p  "Explain quantization." -n 256 -k 4

# (3) End-to-end head-to-head benchmark (4-way matrix):
./scripts/bench_speculative.sh /path/to/HF/Llama-3-8B
```

The CPU-tier auto-pick alone gives ~10-30% over the AVX2 default on
Sapphire Rapids / Zen4. Speculative decoding stacks another 1.5-3× on
top. Together: 2-4× over a stock `pip install llama-cpp-python` flow.

## Docker

Three images on GHCR. All three install llama-cpp-python from a
**prebuilt wheel** at `AIencoder/TurboCpp_Wheels` — no source compile,
~30s image build instead of ~10 min, runs on any x86_64 host with
AVX2 + FMA + F16C.

| image | what's inside | size |
|---|---|---|
| `ghcr.io/ary5272/turbocpp:cpu`        | turbocpp CLI + llama-cpp-python (prebuilt wheel)           | ~500 MB |
| `ghcr.io/ary5272/turbocpp:server`     | inherits `:cpu`, ENTRYPOINT = `turbocpp serve` on `:8080`  | ~500 MB |
| `ghcr.io/ary5272/turbocpp:turboquant` | inherits `:cpu`, adds CPU-only PyTorch for `rotate`        | ~2.0 GB |

```bash
# Inference runtime + unified CLI
docker run --rm -v ~/models:/models ghcr.io/ary5272/turbocpp:cpu \
       generate -m /models/m.gguf -p "Hello" -n 64

# OpenAI-compatible HTTP server on :8080
docker run --rm -p 8080:8080 -v ~/models:/models ghcr.io/ary5272/turbocpp:server \
       -m /models/m.gguf

# Offline Hadamard rotation
docker run --rm -v ~/models:/models ghcr.io/ary5272/turbocpp:turboquant \
       rotate /models/Llama-3-8B /models/Llama-3-8B-tq
```

A new image is pushed to GHCR on every `main` commit and every `v*` tag
([docker.yml](.github/workflows/docker.yml)). Build locally pointing at
a different prebuilt wheel (e.g. AVX-512 / VNNI / Sapphire Rapids):

```bash
docker build --target cpu \
    --build-arg LLAMA_CPP_WHEEL_URL="$(turbocpp pick-wheel --gpu cuda12)" \
    -t turbocpp:cpu-cuda12 .
```

```
   ┌───────────────────────────────────────────────────────────────┐
   │ HF model ──► turboquant rotate ──► llama.cpp convert+quantize │
   │                                                ▼              │
   │                            standard GGUF, runs anywhere       │
   │                            llama.cpp does — every backend,    │
   │                            every architecture, every sampler  │
   └───────────────────────────────────────────────────────────────┘
```

## Layout

| path | purpose |
|---|---|
| `ghcr.io/ggml-org/llama.cpp:full` | upstream **ggml-org/llama.cpp**, pulled at runtime via Docker — the inference engine, every quantization format, every GPU backend (CUDA / Metal / Vulkan / SYCL / ROCm), HTTP server, samplers, grammars, ~50 model architectures. We **stopped vendoring** llama.cpp as a git submodule in 0.5.0 so you always get whatever ggml-org's latest stable image is, without us pinning a stale commit. The `turbocpp llama <tool>` and `turbocpp convert / quantize / perplexity / imatrix / llama-cli / llama-bench` subcommands all forward into this image. |
| [`turboquant/`](turboquant) | the differentiator — Python package that applies Walsh-Hadamard rotation to a HuggingFace model **before** quantization. Output is a standard rotated HF checkpoint that you feed to `convert_hf_to_gguf.py` unmodified |
| [`extras/standalone/`](extras/standalone) | a parallel from-scratch C++17 implementation written earlier in the project. Pure CPU, AVX2/AVX-512, K-quants, GQA, YaRN, mirostat, beam search, GBNF subset, OpenAI-compat HTTP server. Useful as a study reference and a lighter-weight runtime when you don't need llama.cpp's full footprint |

## Why "llama.cpp + TurboQuant"

llama.cpp already ships:

- **Architectures**: LLaMA 1/2/3, Mistral, Mixtral (MoE), Qwen 1/2/2.5, Phi 1/2/3, Gemma 1/2, Falcon, MPT, BLOOM, GPT-2, GPT-NeoX, StableLM, Baichuan, Yi, RWKV, Mamba, …
- **Quantization**: Q2_K, Q3_K_S/M/L, Q4_0/1, Q4_K_S/M, Q5_0/1, Q5_K_S/M, Q6_K, Q8_0, Q8_K, IQ1_S/M, IQ2_XXS/XS/S/M, IQ3_XXS/S/M, IQ4_XS/NL, BF16, F16, F32
- **Backends**: CPU (AVX/AVX2/AVX-512/NEON/AMX), CUDA, Metal, Vulkan, SYCL, ROCm, Kompute, OpenCL, RPC, BLAS
- **Sampling**: greedy, temperature, top-k, top-p, min-p, typical-p, tail-free, locally-typical, dynatemp, mirostat v1+v2, repetition penalty, frequency penalty, presence penalty, logit bias, GBNF grammar, JSON mode, classifier-free guidance, beam search, speculative decoding, lookahead decoding
- **Runtime**: continuous batching, parallel sequences, prompt caching, KV-cache shifting/defrag, embeddings, reranking, LoRA hotswap, multi-modal (LLaVA, Phi-3-vision, MiniCPM-V), tools/function-calling, chat templates for every major model
- **Server**: `llama-server` (OpenAI-compatible HTTP API: completions, chat, embeddings, tools), web UI

TurboQuant adds: a **2 KB Python module** that rotates the model's weight matrices in-place using Walsh-Hadamard transforms. The rotation cancels through the residual-stream linear pieces (it's orthogonal) so the model is fp32-bit-identical, but the per-weight-block max-abs that drives Q4 / Q4_K rounding error drops 3-5×, which translates to **0.3-0.5 perplexity improvement at Q4_K_M** on LLaMA-2-7B (and bigger gains at lower bit-widths).

## Does this actually run faster than stock llama.cpp?

It's the right question and the honest answer has two parts:

### Same bit-width: NO

Quantizing a TurboQuant-rotated model at Q4_K_M and running it on stock
llama.cpp gives the **exact same tokens/sec** as a non-rotated Q4_K_M
of the same model. Same bytes per weight, same kernels, same memory
layout. What you get is **better quality at the same speed** — about
0.3-0.5 perplexity points back at Q4_K_M on LLaMA-2-7B.

### Drop a bit-width tier: YES

The real speed win is using the recovered quality budget to drop one
quantization tier:

| recipe | bytes/weight | quality | wall-clock decode |
|---|---|---|---|
| baseline Q4_K_M (no rotation)        | 4.625 | reference | reference |
| TurboQuant Q4_K_M                    | 4.625 | **better** | same |
| **TurboQuant Q3_K_M**                | 3.5   | ≈ baseline Q4_K_M | **~1.20-1.30× faster** on memory-bound CPUs |
| TurboQuant Q2_K (aggressive)         | 2.6   | usable for some tasks | **~1.5× faster** |

The speedup comes from memory bandwidth: decoding is bandwidth-bound on
nearly all consumer CPUs (and on Sapphire Rapids when the workload
doesn't fit AMX tiles, which is most of them at long context). Fewer
bytes per weight read each step = fewer cycles waiting on DRAM.

### KV cache: also YES (long context)

`turboquant.kvcache.rotate_kv_for_cache_quant()` Hadamard-rotates the
attention output projection so K and V live in a Gaussianized frame
**inside** the KV cache. Combine with llama.cpp's
`--cache-type-k q4_0 --cache-type-v q4_0` and you get usable quality at
half the KV bandwidth — meaningful at 8K+ context where KV reads dominate.

### Reproduce the numbers

```bash
# Synthetic micro (1 second, no model needed):
python -m turboquant.bench

# End-to-end on your machine, real GGUF:
./scripts/bench_e2e.sh /path/to/HF/Llama-3-8B
```

The end-to-end script builds both a baseline-Q4_K_M and a TurboQuant-Q3_K_M
GGUF and runs `llama-bench` on each.

## Quick start

No git submodule, no manual cmake — every llama.cpp tool is forwarded
into ggml-org's official Docker image, so a clean `pip install` is the
whole setup.

```bash
# 1. Install
pip install 'turbocpp[runtime]'

# 2. Verify (downloads TinyLlama, runs a sample completion):
turbocpp quickstart

# 3. End-to-end (the SPEED path: rotated Q3_K_M ≈ baseline Q4_K_M quality).
#    Each step delegates to the right tool — no cmake, no submodule.
turbocpp rotate    ~/models/Llama-3-8B  ~/models/Llama-3-8B-tq
turbocpp convert   ~/models/Llama-3-8B-tq  --outfile Llama-3-8B-tq.gguf
turbocpp quantize  Llama-3-8B-tq.gguf  Llama-3-8B-tq-Q3_K_M.gguf  Q3_K_M
turbocpp generate  -m Llama-3-8B-tq-Q3_K_M.gguf \
                   -p "Explain Hadamard quantization in one sentence:" -n 100

# 4. Or the QUALITY path (same speed as baseline, better numbers):
turbocpp quantize  Llama-3-8B-tq.gguf  Llama-3-8B-tq-Q4_K_M.gguf  Q4_K_M
```

## TurboQuant: the math in one block

For each linear layer `y = W x` in the residual stream, with `H` an
orthogonal block-Hadamard:

```
W' = H · W           (output axis rotated)         ← producers
W' = W · Hᵀ          (input axis rotated)          ← consumers
```

We pair every producer with its consumer:
`tok_embed`, `W_o`, `W_down` ← producers (output rotated)
`W_q`, `W_k`, `W_v`, `W_gate`, `W_up`, `lm_head` ← consumers (input rotated).

Since `H · Hᵀ = I`, the rotations cancel through the network. Forward
pass in fp32 is bit-identical. But quantization noise is computed on the
ROTATED weights, whose per-block distribution is near-Gaussian thanks
to the central-limit theorem — and Gaussian distributions quantize
well, while heavy-tailed real LLM weights don't.

RMSNorm is rotation-equivariant only if its γ vector is uniform. Pass 1
absorbs each γ into the FOLLOWING linear (`W ← W · diag(γ)`) and then
sets γ ← 1, after which the rotation is safe.

See [`turboquant/turboquant.py`](turboquant/turboquant.py) — 100 lines.

## Tests

```bash
pytest -q turboquant/                       # rotation math + CLI parser (~65 tests)
ctest --test-dir extras/standalone/build    # standalone-engine kernels
```

CI runs the turboquant tests on Linux + Windows + macOS, plus builds the
standalone engine and runs its unit tests.

## Related work

- **QuaRot** (Ashkboos et al. 2024)
- **SpinQuant** (Liu et al. 2024)
- **GPTQ** (Frantar et al. 2022) — calibration-based, complementary
- **AWQ** (Lin et al. 2023) — activation-aware scaling, complementary

## License

- TurboQuant code: **MIT** ([LICENSE](LICENSE))
- llama.cpp (pulled at runtime via Docker, no submodule): **MIT** ([upstream LICENSE](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE))
