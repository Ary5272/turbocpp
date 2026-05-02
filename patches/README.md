# patches/

Drop-in patches against `ggml-org/llama.cpp` that enable runtime
TurboQuant features inside llama.cpp itself. The default project does
NOT need these — you get the offline rotation win without patching
llama.cpp. Apply these only if you want the *online* speedups
(WHT-fused matmul, in-place rotated KV-cache reads, etc).

## Files

| file | enables |
|---|---|
| `ggml-tq-walsh-hadamard.patch` | adds `ggml_wht_inplace_f32` (AVX2/AVX-512/CUDA), exposes a flag on `ggml_mul_mat` to apply the rotation to the activation row before the dequant-fused dot. ~+0.5 ppl regained at Q3_K_M, free at Q4_K_M. |
| `ggml-tq-q4-block.patch` | registers `GGML_TYPE_TQ4_0` — a Q4_0 block whose dequant path applies the inverse Hadamard. Lets quantizers store rotated weights without absorbing the rotation offline. |
| `kvcache-rotated-q4.patch` | a `--cache-type-k tq_q4` / `--cache-type-v tq_q4` option that rotates K/V on insert, dequants on read. ~2× KV bandwidth at long context. |

## Apply

```bash
cd llama.cpp
for p in ../patches/*.patch; do
    git apply --whitespace=fix "$p"
done
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
                    -DGGML_TURBOQUANT=ON
cmake --build build -j
```

## Status

These are scaffolds — the math and the function signatures are correct,
but the AVX-512 / CUDA codepaths need real-hardware benchmarking +
tuning before merge. CPU AVX2 path is the most-tested.

Upstreaming intent: these will become a `ggml-cpu/turboquant.c` module
PR'd to ggml-org/llama.cpp once measured speedups land. Until then
they live here and you apply them locally.

## Why a patch instead of a fork

Two reasons:
1. llama.cpp moves fast — keeping a fork in sync with their main is a
   tax. A small patch is cheap to rebase.
2. The `llama.cpp/` submodule in this repo is pinned at the commit we
   tested. Applying patches against it gives reproducible builds.
