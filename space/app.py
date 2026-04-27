"""TurboQuant Visualizer — HuggingFace Space (Gradio).

Interactive demo showing what the Hadamard rotation actually does to a
weight tensor's quantization-error distribution. Three side-by-side
plots:

   1. raw weight histogram (heavy tail)
   2. rotated weight histogram (Gaussianized)
   3. per-block max-abs before vs after rotation

Plus a numeric summary: MSE at Q4 / Q3 / Q2, with and without rotation,
and the implied "drop a tier and run faster" speed-up estimate.
"""
import io

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from turboquant.bench import heavy_tailed_weight, measure
from turboquant.hadamard import block_hadamard_inplace


def _plot(W_raw: torch.Tensor, W_rot: torch.Tensor, block: int) -> "PIL.Image":
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    raw = W_raw.flatten().numpy()
    rot = W_rot.flatten().numpy()

    bins = np.linspace(-0.5, 0.5, 121)
    axes[0].hist(raw, bins=bins, color="#888", alpha=0.85)
    axes[0].set_title("Raw weights — heavy-tailed")
    axes[0].set_xlim(-0.5, 0.5); axes[0].set_yscale("log")

    axes[1].hist(rot, bins=bins, color="#3B82F6", alpha=0.85)
    axes[1].set_title("After block-Hadamard — Gaussianized")
    axes[1].set_xlim(-0.5, 0.5); axes[1].set_yscale("log")

    raw_blkmax = W_raw.reshape(-1, block).abs().amax(dim=-1).numpy()
    rot_blkmax = W_rot.reshape(-1, block).abs().amax(dim=-1).numpy()
    axes[2].hist(raw_blkmax, bins=40, alpha=0.6, label="raw",     color="#888")
    axes[2].hist(rot_blkmax, bins=40, alpha=0.6, label="rotated", color="#3B82F6")
    axes[2].set_title(f"per-{block} block max|w|  (drives Q4 quant step)")
    axes[2].legend()

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)


def run(rows: int, cols: int, block: int, seed: int):
    W = heavy_tailed_weight(n_rows=int(rows), n_cols=int(cols), seed=int(seed))
    W_rot = W.clone().double()
    block_hadamard_inplace(W_rot, axis=-1, block=int(block))

    # Quantization MSE
    bench_lines = []
    for bits in (4, 3, 2):
        s_base = measure(W, bits=bits, rotated=False, block=int(block))
        s_rot  = measure(W, bits=bits, rotated=True,  block=int(block))
        bench_lines.append(
            f"  Q{bits}      raw MSE = {s_base.mse:.3e}    "
            f"TQ MSE = {s_rot.mse:.3e}    "
            f"× {s_base.mse/max(s_rot.mse,1e-30):.1f} better"
        )

    # MSE-matched speed estimate.
    base_q4 = measure(W, bits=4, rotated=False, block=int(block)).mse
    speed_msg = "needs a deeper drop"
    for bits in (3, 2):
        s = measure(W, bits=bits, rotated=True, block=int(block))
        if s.mse <= base_q4:
            ratio = 4.625 / (bits + 1.0)
            speed_msg = (f"TQ-Q{bits} matches baseline-Q4 quality at "
                         f"~{ratio:.2f}× less memory bandwidth → faster decode")
            break

    summary = (
        f"weight shape = {rows}×{cols}, block_size = {block}\n"
        f"per-block max|w|  raw mean  = {W.reshape(-1, int(block)).abs().amax(dim=-1).mean():.3f}\n"
        f"per-block max|w|  rot mean  = {W_rot.reshape(-1, int(block)).abs().amax(dim=-1).mean():.3f}\n\n"
        + "\n".join(bench_lines)
        + "\n\nSpeed: " + speed_msg
    )

    return _plot(W, W_rot, int(block)), summary


demo = gr.Interface(
    fn=run,
    title="TurboQuant — Hadamard Rotation Visualizer",
    description=(
        "Drag the sliders to see how Walsh-Hadamard rotation reshapes a "
        "heavy-tailed LLM-style weight distribution. The rotation is "
        "orthogonal so model fp32 output is unchanged — but quantization "
        "error drops 3-5× because every block sees a near-Gaussian input. "
        "[github.com/Ary5272/turbocpp](https://github.com/Ary5272/turbocpp)"
    ),
    inputs=[
        gr.Slider(64,  4096, value=1024, step=64,  label="rows"),
        gr.Slider(64,  4096, value=4096, step=64,  label="cols"),
        gr.Slider(32,   256, value=128,  step=32,  label="Hadamard block size"),
        gr.Slider(0,   1000, value=0,    step=1,   label="seed"),
    ],
    outputs=[
        gr.Image(type="pil", label="distributions"),
        gr.Textbox(label="quant-error report", lines=10),
    ],
    examples=[[1024, 4096, 128, 0], [4096, 4096, 64, 7]],
)


if __name__ == "__main__":
    demo.launch()
