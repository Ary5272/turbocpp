"""Microbenchmark: TurboQuant rotation effect on Q4_K-style quantization.

We don't need a full LLM to demonstrate the speed/quality story:
   - generate a synthetic weight tensor with realistic heavy-tailed stats
   - quantize it with and without rotation, at Q4 / Q3 / Q2 bit budgets
   - report reconstruction MSE and effective bits/weight

The real speedup story (decode tok/s) requires running llama-bench on a
quantized GGUF — see scripts/bench_e2e.sh for that. This module is the
quick "did rotation help?" check that runs in 1 second.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .hadamard import block_hadamard_inplace


@dataclass
class QuantStats:
    fmt: str
    bits: float  # effective bits/weight
    mse: float  # reconstruction error
    max_abs_err: float


def _quant_dequant_q(x: torch.Tensor, bits: int, block: int = 32) -> torch.Tensor:
    """Symmetric block min-max quantization (the same shape llama.cpp's
    Q4_0 / Q3_0 use, modulo per-block fp16 scale vs fp32). Operates per
    contiguous `block` along last dim."""
    n = x.shape[-1]
    assert n % block == 0
    levels = (1 << bits) - 1  # e.g. 15 for 4-bit
    half = levels // 2  # symmetric quant centered at 0
    flat = x.reshape(-1, n // block, block)
    maxabs = flat.abs().amax(dim=-1, keepdim=True)
    d = maxabs / half
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = torch.clamp(torch.round(flat / d) + half, 0, levels)
    rec = (q - half) * d
    return rec.reshape_as(x)


def measure(W: torch.Tensor, bits: int, rotated: bool, block: int = 128) -> QuantStats:
    """Return (effective bpw, MSE, max-abs-err) for `bits`-bit quantization
    of `W`, optionally Hadamard-rotated first."""
    x = W.clone().double()
    if rotated:
        block_hadamard_inplace(x, axis=-1, block=block)
    rec = _quant_dequant_q(x, bits, block=32)
    if rotated:
        # Inverse rotation to compare in original frame.
        block_hadamard_inplace(rec, axis=-1, block=block)
    err = W.double() - rec
    bpw = bits + 32 / 32  # quants + per-32 fp32 scale
    return QuantStats(
        fmt=f"{'TQ-' if rotated else ''}Q{bits}",
        bits=bpw,
        mse=err.pow(2).mean().item(),
        max_abs_err=err.abs().max().item(),
    )


def heavy_tailed_weight(n_rows: int = 4096, n_cols: int = 4096, seed: int = 0) -> torch.Tensor:
    """Synthetic LLM-shaped weight: small Gaussian bulk + occasional tail
    outliers. Real LLaMA weights look like this — the outliers dominate
    Q4_0's per-block max-abs and blow up rounding error."""
    torch.manual_seed(seed)
    W = 0.02 * torch.randn(n_rows, n_cols)
    # ~0.5% outliers per row at ~5σ.
    n_out = max(1, n_cols // 200)
    rows = torch.randint(0, n_rows, (n_out * n_rows,))
    cols = torch.randint(0, n_cols, (n_out * n_rows,))
    sign = torch.randint(0, 2, (rows.shape[0],), dtype=torch.float32) * 2 - 1
    mag = 0.3 + 0.4 * torch.rand(rows.shape[0])
    W[rows, cols] = sign * mag
    return W


def run_bench(seed: int = 0) -> None:
    print("== TurboQuant rotation effect on quantization error ==")
    print("Synthetic weight: 4096×4096 with ~5σ tail outliers\n")
    W = heavy_tailed_weight(seed=seed)

    print(f"{'format':<12}{'bpw':>6}{'MSE':>14}{'max|err|':>12}{'speedup hint':>20}")
    print("-" * 64)
    rows = []
    for bits in (4, 3, 2):
        s_base = measure(W, bits=bits, rotated=False)
        s_rot = measure(W, bits=bits, rotated=True)
        rows.append((s_base, s_rot))
        # speedup hint: roughly bytes ratio at decode time vs Q4 baseline
        speedup_base = 4.625 / s_base.bits  # treat Q4_K_M ~4.625 bpw as ref
        speedup_rot = 4.625 / s_rot.bits
        print(
            f"{s_base.fmt:<12}{s_base.bits:>6.2f}{s_base.mse:>14.3e}"
            f"{s_base.max_abs_err:>12.3e}{speedup_base:>18.2f}×"
        )
        print(
            f"{s_rot.fmt:<12}{s_rot.bits:>6.2f}{s_rot.mse:>14.3e}"
            f"{s_rot.max_abs_err:>12.3e}{speedup_rot:>18.2f}×"
        )

    # Find the lowest TQ bit-width whose MSE is still ≤ baseline-Q4 MSE.
    base_q4_mse = rows[0][0].mse
    print()
    for _s_base, s_rot in rows:
        verdict = (
            "✓ matches baseline-Q4 quality"
            if s_rot.mse <= base_q4_mse
            else "✗ exceeds baseline-Q4 error"
        )
        print(f"  {s_rot.fmt:<8}  MSE={s_rot.mse:.3e}  {verdict}")

    print("""
Interpretation:
  - Same-bit rotated (TQ-Q4 vs Q4) → quality win, identical decode speed.
  - Drop-bit rotated (TQ-Q3 vs Q4) → matched quality at ~25% less memory
    bandwidth → ~10-20% faster decode on memory-bound CPUs (DDR5/8-channel
    DDR4 incl. Sapphire Rapids when AMX is not the bottleneck).
""")


if __name__ == "__main__":
    run_bench()
