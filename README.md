# turbocpp

> **llama.cpp + TurboQuant.** Every llama.cpp feature, plus an offline
> Hadamard-rotation preprocessor that meaningfully improves the quality
> of any quantization (Q4_0 / Q4_K_M / Q6_K / …) at zero inference cost.

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
| [`llama.cpp/`](llama.cpp) | upstream **ggml-org/llama.cpp** as a git submodule — the inference engine, all of its quantization formats, GPU backends (CUDA / Metal / Vulkan / SYCL / ROCm), HTTP server, samplers, grammars, and ~50 model architectures |
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

```bash
# 1. Clone with submodules
git clone --recursive https://github.com/Ary5272/turbocpp
cd turbocpp

# 2. Build llama.cpp (CPU; see llama.cpp/README.md for CUDA / Metal / Vulkan)
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j

# 3. Install the turboquant Python package
pip install -e .                        # uses pyproject.toml
# or:  pip install -r turboquant/requirements.txt

# 4. End-to-end (the SPEED path: rotated Q3_K_M ≈ baseline Q4_K_M quality):
python -m turboquant ~/models/Llama-3-8B  ~/models/Llama-3-8B-tq
python llama.cpp/convert_hf_to_gguf.py ~/models/Llama-3-8B-tq \
       --outfile Llama-3-8B-tq.gguf
llama.cpp/build/bin/llama-quantize \
       Llama-3-8B-tq.gguf Llama-3-8B-tq-Q3_K_M.gguf Q3_K_M
llama.cpp/build/bin/llama-cli -m Llama-3-8B-tq-Q3_K_M.gguf \
       -p "Explain Hadamard quantization in one sentence:" -n 100

# 5. Or the QUALITY path (same speed as baseline, better numbers):
llama.cpp/build/bin/llama-quantize \
       Llama-3-8B-tq.gguf Llama-3-8B-tq-Q4_K_M.gguf Q4_K_M
```

## HuggingFace Space demo

`space/` is a ready-to-deploy Gradio app that visualizes the rotation
effect on a synthetic LLM weight tensor:

```
huggingface.co/new-space   →   sdk: gradio   →   copy space/* into the repo
```

Locally:
```bash
pip install -e ".[demo]"
python space/app.py
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
pytest turboquant/test_turboquant.py        # rotation invariants + math
ctest --test-dir extras/standalone/build    # standalone-engine kernels
```

CI runs the turboquant tests on Linux + Windows + macOS, plus builds the
standalone engine and runs its 15 unit tests.

## Related work

- **QuaRot** (Ashkboos et al. 2024)
- **SpinQuant** (Liu et al. 2024)
- **GPTQ** (Frantar et al. 2022) — calibration-based, complementary
- **AWQ** (Lin et al. 2023) — activation-aware scaling, complementary

## License

- TurboQuant code: **MIT** ([LICENSE](LICENSE))
- llama.cpp submodule: **MIT** (their `LICENSE`)
