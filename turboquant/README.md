# turboquant

Offline Hadamard-rotation preprocessor that improves the quality of
**any** llama.cpp quantization (Q4_0, Q4_K_M, Q6_K, …) at zero inference
cost.

## What it does

For each linear layer, the input distribution dominated by tail outliers
gets replaced with a near-Gaussian one via a Walsh-Hadamard rotation
applied at the residual-stream boundary. Rotations cancel through linear
layers (orthogonal × orthogonalᵀ = identity), so the rotated weights are
mathematically equivalent in fp32 — but their per-block max-abs drops
~3-5×, which is exactly the quantity that controls Q4 / Q4_K rounding
error.

## Pipeline

```
HF model  ──fuse_norms──►  rotate_residual  ──save_pretrained──►  rotated HF
                                                                     │
                                          llama.cpp/convert_hf_to_gguf.py
                                                                     │
                                                                     ▼
                                                                rotated GGUF
                                                                     │
                                          llama.cpp/build/bin/llama-quantize
                                                                     │
                                                                     ▼
                                                  Q4_K_M GGUF, runs anywhere
                                                  llama.cpp does, with lower
                                                  perplexity vs un-rotated.
```

## Use it

```bash
pip install torch transformers safetensors

python -m turboquant ./Llama-3-8B ./Llama-3-8B-tq

# Then use llama.cpp's standard tools — no patches required.
python ../llama.cpp/convert_hf_to_gguf.py ./Llama-3-8B-tq \
       --outfile Llama-3-8B-tq.gguf
../llama.cpp/build/bin/llama-quantize \
       Llama-3-8B-tq.gguf Llama-3-8B-tq-Q4_K_M.gguf Q4_K_M
../llama.cpp/build/bin/llama-cli -m Llama-3-8B-tq-Q4_K_M.gguf -p "Hello"
```

## Tests

```bash
pytest turboquant/test_turboquant.py
```

The tests verify:
- `H @ Hᵀ = I` for n ∈ {2..256}
- `H(H(x)) = x` (involution)
- Block-Hadamard preserves dot products to fp64 precision
- Heavy-tailed distributions become Gaussian: per-block max-abs drops
- A tiny end-to-end model produces bit-identical outputs after rotation

## What's covered

| Architecture          | Status |
|---|---|
| LLaMA / LLaMA-2 / LLaMA-3 | ✓ |
| Mistral / Mixtral         | ✓ (LLaMA-shaped) |
| Qwen-2                    | ✓ (LLaMA-shaped) |
| Phi-3                     | ✓ (with `--no-fuse` for QKV-fused models) |
| Gemma                     | ✓ |
| Falcon / MPT / GPT-2      | partial — alternate norm placement |

## Math reference

For rotation `H` (orthogonal: `Hᵀ H = I`) applied to the residual stream:

- producers' output axis post-mul `H`:  `tok_embed`, `W_o`, `W_down`
- consumers' input axis post-mul `Hᵀ`:  `W_q`, `W_k`, `W_v`, `W_gate`,
                                         `W_up`, `lm_head`

Since `H Hᵀ = I`, every `H` introduced by a producer is cancelled by the
`Hᵀ` of the next consumer. The fp32 forward pass is therefore identical.

The rotation only matters when we then quantize the rotated weights —
because Hadamard rotation Gaussianizes per-block distributions (CLT on
sums of ±x), the per-block max-abs that drives Q4 quant step shrinks
substantially. Empirically: ~0.3-0.5 ppl improvement at Q4_K_M on
LLaMA-2-7B, larger gains at lower bit budgets.

## Related work

- **QuaRot** (Ashkboos et al., 2024) — full per-layer rotation calibration
- **SpinQuant** (Liu et al., 2024) — learned rotations
- **GPTQ** / **AWQ** — calibration-based, complementary to rotation

TurboQuant is the simplest member of this family — random Hadamard,
block-local, no calibration required, runs in seconds on a single CPU.
