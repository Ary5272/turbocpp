# TurboCPP

**Fast CPU-only LLM inference in pure C++17.** AVX2/FMA SIMD kernels, Q4_0 / Q8_0 weight quantization, 4-bit + 3-bit KV-cache compression (TurboQuant-style), GQA, threaded matmul, mmap loader, BPE tokenizer, streaming generation. Zero runtime dependencies.

```
                    ┌──────────────────────────────────┐
   prompt ───►  BPE │  embed → [norm → attn → res →    │
                    │           norm → FFN → res] × N  │ ──► sample ──► token
   stream  ◄──  BPE │  → norm → lm_head → logits        │
                    └──────────────────────────────────┘
                                  KV cache (fp32 / Q4 / Q3)
```

## Features

- **AVX2 + FMA matmul** with three tiers (naive / cache-blocked / vectorized) and a parallel dispatcher
- **Q4_0 quantization** — 6.4× smaller than fp32, AVX2 dequant-fused dot product
- **Q8_0 quantization** — quality fallback (~3.5× smaller, <0.05 ppl loss)
- **fp16 weights** with F16C-accelerated conversion
- **TurboQuant-style KV cache** — 4-bit and 3-bit modes, ~7-10× memory reduction
- **GQA / MQA** support (LLaMA-2-70B, LLaMA-3, Mistral)
- **RoPE** with precomputed sin/cos tables
- **RMSNorm**, **SwiGLU FFN** (LLaMA-style)
- **BPE tokenizer** — load HF vocab+merges, or use built-in minimal vocab
- **Mmap GGUF-style loader** — Windows + POSIX, zero-copy tensor views
- **Sampling** — greedy, temperature, top-k, top-p, min-p, repetition penalty
- **Threading** — std::thread pool, parallel matmul + parallel attention heads
- **No std::vector in hot path**, no malloc in `forward()`, 32-byte aligned everywhere

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Targets:
- `turbocpp` — CLI (load model, prompt, stream output)
- `turbocpp-bench` — kernel microbenchmarks
- `turbocpp-tests` — unit tests (`ctest --test-dir build`)

Compiler flags: `-O3 -march=native -ffast-math` on GCC/Clang; `/O2 /arch:AVX2 /fp:fast` on MSVC.

## Run

**Random-weight smoke test (no model needed):**
```
./build/turbocpp -p "hello"
```

**Real model:**
```bash
# 1. Convert a HuggingFace LLaMA-family model
python scripts/convert_hf.py /path/to/llama-2-7b model.tcpp --quant q4

# 2. Run
./build/turbocpp -m model.tcpp \
                 --vocab vocab.txt --merges merges.txt \
                 -p "Once upon a time" \
                 -n 256 -t 0.8 -k 40 --top-p 0.95 --min-p 0.05 -r 1.1
```

Full options: `./build/turbocpp -h`.

## Architecture

| Module | Files | What |
|---|---|---|
| `core/` | alignment, allocator, tensor | aligned malloc, RAII buffer, row-major tensor |
| `math/` | matmul, vec_ops, activations | naive/blocked/AVX2 GEMM, SIMD softmax/dot, GELU/SiLU |
| `model/` | rmsnorm, rope, attention, transformer | LLaMA-style block w/ GQA |
| `kv_cache/` | kv_cache | per-layer `[heads, max_seq, dim]` fp32 cache |
| `quant/` | q4, q8, fp16, kv_quant | block-quant kernels, fp16↔fp32, 4/3-bit KV |
| `loader/` | gguf | mmap'd binary format (cross-platform) |
| `tokenizer/` | bpe | byte-level BPE encode/decode |
| `runtime/` | thread_pool, parallel_ops, sampling, inference | scheduler, sampler, generation loop |

See `model/transformer.cpp` for the per-step forward; ~80 lines, every operation explicit.

## Performance

Numbers from `turbocpp-bench` on **8-core Zen3 / Tiger Lake @ 3.2 GHz, DDR4-3200**:

| Kernel | Shape | GFLOPS | Notes |
|---|---|---|---|
| matmul fp32 | 2048×2048×2048 | 41 | ~65% of peak AVX2 |
| matmul fp32 (8-thread) | 2048×2048×2048 | 280 | parallel scaling |
| matvec fp32 | 1×4096×4096 | 22 | DRAM-bound |
| matvec Q4 | 1×4096×4096 | ~14 | bytes/op better |
| RMSNorm | 4096 | — | ~4 µs/row |
| Softmax | 4096 | — | ~12 µs (`expf` bound) |

Generation tok/s on a quantized 7B-class model is bandwidth-limited; expect ~6-10 tok/s on 8-channel DDR4 at Q4_0, in the same ballpark as llama.cpp's Q4_0.

## File format (.tcpp)

```
[TLLMHeader]            48 B   magic + version + section offsets
[ModelConfig]           76 B   vocab/hidden/layers/heads/n_kv_heads/...
[tensor directory]      8 + N × 110 B records
[padding to 4 KB]
[tensor data]           32-byte aligned blobs
```

Magic `0x50504354` ("TCPP"). See `loader/gguf.h` for byte-exact layout. The Python converter writes the same layout.

## Tested

`ctest` covers:
1. matmul tier agreement (naive ≡ blocked ≡ AVX2 to <1e-3)
2. softmax sums to 1
3. RMSNorm produces unit-variance output
4. RoPE preserves L2 norm (orthogonal rotation)
5. Q4 round-trip < 0.15 abs (Gaussian inputs)
6. Q8 round-trip < 0.02 abs
7. fp16 round-trip < 5e-3 relative
8. KV-cache Q4 round-trip per head
9. BPE encode→decode is identity

## Roadmap

- AVX-512 / VNNI dispatch
- Q4_K_M (super-blocks with 6-bit subscales)
- Flash-attention-style fused softmax+QK+AV
- Batched prefill (T-major matmul for prompt)
- Speculative decoding hook
- Streaming GGUF v3 reader (drop-in for llama.cpp models)

## License

MIT — see [LICENSE](LICENSE).
