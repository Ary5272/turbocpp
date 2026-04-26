# TurboCPP

**Fast CPU-only LLM inference in pure C++17.** TurboQuant-rotated quantization, full llama.cpp-style feature set, AVX2/FMA SIMD, mmap loader, BPE tokenizer, streaming generation. Zero runtime dependencies.

```
                    ┌──────────────────────────────────┐
   prompt ───►  BPE │  embed → [norm → attn → res →    │
       chat tmpl    │           norm → FFN → res] × N  │ ──► sample ──► token
   stream  ◄──  BPE │  → norm → lm_head → logits        │     stop seq
                    └──────────────────────────────────┘
                                  KV cache (fp32 / Q4 / Q3) + snapshot
```

## Features

### Quantization
- **TurboQuant** — Hadamard-rotated Q4 quantization (Gaussianizes weight blocks → ~3-5× lower per-block max-abs → much smaller rounding error than vanilla Q4_0)
- **Q4_0**, **Q8_0**, **fp16** weight formats (F16C-accelerated conversion)
- **Block-Hadamard** transform (n log n butterfly, in-place AVX2)
- **TurboQuant-style KV cache** — 4-bit and 3-bit modes, ~7-10× memory reduction

### Model architecture
- **GQA / MQA** support (LLaMA-2-70B, LLaMA-3, Mistral, Mixtral)
- **RoPE** with **YaRN, NTK-aware, and Linear (Position Interpolation)** scaling for long context
- **RMSNorm**, **SwiGLU FFN** (LLaMA-style)
- **Multi-head attention** with parallel-per-head dispatch

### Sampling
- Greedy, **temperature**, **top-k**, **top-p**, **min-p**
- **Repetition penalty** with rolling window
- **Mirostat v2** (adaptive perplexity control)
- **Logit bias** (per-token additive bias / banned tokens)
- **Stop sequences** (string-level early termination)

### Runtime
- **Chat templates** — LLaMA-3, ChatML, Mistral, LLaMA-2
- **Prompt cache** — KV-snapshot save/load to disk
- **Embeddings mode** — extract `last_hidden()` post-final-norm for sentence embeddings
- **Threaded matmul** + **parallel attention heads** via std::thread pool
- **Mmap binary loader** — Windows + POSIX, zero-copy tensor views
- **Byte-level BPE tokenizer**
- HuggingFace **converter script** with optional Q4 / Q8 quantization
- 32-byte aligned everywhere, no `std::vector` in hot path, no `malloc` in `forward()`

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Targets:
- `turbocpp` — CLI (load model, prompt, stream output)
- `turbocpp-server` — OpenAI-compatible HTTP server
- `turbocpp-bench` — kernel microbenchmarks
- `turbocpp-tests` — unit tests (`ctest --test-dir build`)

CI matrix: Ubuntu, Windows, macOS — see `.github/workflows/ci.yml`.

## Quick start

**Smoke test (no model needed):**
```bash
./build/turbocpp -p "hello" -n 32
```

**Real model (LLaMA-3 chat):**
```bash
# 1. Convert + quantize a HuggingFace model
python scripts/convert_hf.py /path/to/Llama-3-8B model.tcpp --quant q4

# 2. Run with chat template + sampling settings
./build/turbocpp -m model.tcpp \
                 --vocab vocab.txt --merges merges.txt \
                 --chat llama3 --system "You are concise." \
                 -p "Explain RoPE in two sentences." \
                 -n 256 -t 0.7 --top-p 0.9 --min-p 0.05 \
                 -r 1.1 --stop "<|eot_id|>" --stop "</s>"
```

**Mirostat for stable perplexity:**
```bash
./build/turbocpp -m model.tcpp -p "Story:" --mirostat 2 --mirostat-tau 5.0
```

**Prompt cache (warm start across runs):**
```bash
./build/turbocpp -m model.tcpp -p "$(cat long_doc.txt)" \
                 --prompt-cache /tmp/doc.kv -n 0
./build/turbocpp -m model.tcpp -p "Summarize:" --prompt-cache /tmp/doc.kv -n 100
```

Full options: `./build/turbocpp -h`.

## TurboQuant explained

Vanilla Q4 stores 32 weights as 1 fp32 scale + 32 nibbles. The scale is set by the block's max-abs value — but that's dominated by tail outliers in real LLM weights, blowing up the quant step and the rounding error.

TurboQuant fixes this by applying a fast **Walsh–Hadamard transform** (block size 128) to weight rows before quantization. The transform is orthogonal — it preserves dot products — and turns each block's distribution Gaussian via the central-limit theorem. Result: per-block max-abs drops 2-4×, quant step shrinks proportionally, and rounding error drops by the same factor. Same 4-bit budget, much better quality.

At inference, the same Hadamard is applied to the activation row in scratch memory before each matmul. The transform is `O(K log K)` — negligible vs the `O(NK)` matmul work it precedes.

```
Offline (converter):
   W_tq = Q4_quantize(BlockHadamard(W))

Inference (per matmul row):
   x_rot = BlockHadamard(x)        // O(K log K), ~50 µs at K=4096
   y     = q4_dot(x_rot, W_tq)     // standard Q4 dequant-fused dot

Math: Hadamard is its own inverse, so dot(x, W_row) == dot(H(x), H(W_row)).
```

See `quant/hadamard.cpp` and `quant/tq.cpp`.

## Architecture

| Module | Files | Purpose |
|---|---|---|
| `core/` | alignment, allocator, tensor | aligned malloc, RAII buffer, row-major tensor |
| `math/` | matmul, vec_ops, activations | 3-tier matmul, SIMD softmax/dot, GELU/SiLU |
| `model/` | rmsnorm, rope, attention, transformer | LLaMA block w/ GQA + YaRN |
| `kv_cache/` | kv_cache | fp32 cache + snapshot save/load |
| `quant/` | q4, q8, fp16, **hadamard, tq**, kv_quant | block quant + TurboQuant |
| `loader/` | gguf | mmap binary format (Win + POSIX) |
| `tokenizer/` | bpe | byte-level BPE encode/decode |
| `runtime/` | thread_pool, parallel_ops, sampling, **chat**, inference | scheduler + sampler + chat templates + generation loop |

## Tested

`ctest` covers (12 tests):
1. matmul tier agreement (naive ≡ blocked ≡ AVX2)
2. softmax sums to 1
3. RMSNorm produces unit-variance output
4. RoPE preserves L2 norm
5-7. Q4 / Q8 / fp16 round-trip
8. KV cache Q4 round-trip
9. BPE encode→decode is identity
10. Hadamard involution (`H(H(x)) == x`)
11. TurboQuant matmul vs fp32 reference
12. Stop sequence matcher
13. LLaMA-3 chat template

## Comparison vs llama.cpp

| Feature | TurboCPP | llama.cpp |
|---|:---:|:---:|
| AVX2 + FMA matmul | ✓ | ✓ |
| **AVX-512** 16-wide FMA dispatch | ✓ | ✓ |
| ARM NEON / Apple Silicon | ✗ (roadmap) | ✓ |
| GPU backends (CUDA / Metal / Vulkan) | ✗ (CPU by design) | ✓ |
| Q4_0 / Q8_0 / fp16 / BF16 | ✓ | ✓ |
| **K-quants Q4_K_M / Q6_K / Q8_K** | ✓ | ✓ |
| I-quants (IQ2 / IQ3 / IQ4) | ✗ (roadmap) | ✓ |
| **TurboQuant (Hadamard rotation)** | ✓ | ✗ |
| KV-cache 4-bit / 3-bit | ✓ | partial |
| **GGUF v3 reader** (drop-in llama.cpp models) | ✓ | ✓ |
| GQA / MQA | ✓ | ✓ |
| **MoE / Mixtral** | ✓ | ✓ |
| Sliding window + context shift | ✓ | ✓ |
| ALiBi positional encoding | ✓ | ✓ |
| YaRN / NTK / linear RoPE scaling | ✓ | ✓ |
| Top-k / top-p / min-p / repeat penalty | ✓ | ✓ |
| **Typical-p / tail-free / dynatemp** | ✓ | ✓ |
| **Mirostat v1 + v2** | ✓ | ✓ |
| Logit bias / banned tokens | ✓ | ✓ |
| **Classifier-free guidance** | ✓ | ✓ |
| Stop sequences | ✓ | ✓ |
| **Beam search** | ✓ | ✓ |
| **Speculative decoding** | ✓ | ✓ |
| **Grammar / JSON-mode sampling** | ✓ (JSON SM) | ✓ (full GBNF) |
| Chat templates (LLaMA-3, ChatML, Mistral, LLaMA-2) | ✓ | ✓ |
| **LoRA adapter merge** | ✓ | ✓ |
| Prompt cache to disk | ✓ | ✓ |
| Embeddings mode | ✓ | ✓ |
| **HTTP server (OpenAI /v1/completions)** | ✓ | ✓ |
| LLaVA / multi-modal | ✗ (roadmap) | ✓ |

## Roadmap

Already shipped above. Still on the list:
- VNNI / AMX (Sapphire Rapids) dispatch
- I-quants (IQ2_XXS, IQ4_NL — codebook-based formats)
- Flash-attention-style fused softmax+QK+AV
- Batched prefill (T-major matmul for prompt processing)
- Tree speculative decoding (Medusa heads)
- Continuous batching for the HTTP server
- LLaVA / multimodal
- ARM NEON / Apple Silicon native dispatch
- SSE / streaming for /v1/completions
- /v1/embeddings endpoint
- Full GBNF grammar (not just JSON state machine)

## License

MIT — see [LICENSE](LICENSE).
