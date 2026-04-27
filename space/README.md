---
title: TurboQuant Visualizer
emoji: 🌀
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Visualize how Hadamard rotation Gaussianizes LLM weights
---

# TurboQuant Visualizer

Interactive demo of the offline weight-rotation step at the heart of
[turbocpp](https://github.com/Ary5272/turbocpp). Drag the sliders to see
how a Walsh-Hadamard transform reshapes a heavy-tailed LLM weight
distribution into a near-Gaussian one — which is the exact distribution
shape that Q4 / Q4_K / Q3 quantization handles best.

## What you're looking at

| panel | what |
|---|---|
| left   | raw synthetic weight (Gaussian bulk + ~5σ outliers — typical of LLaMA-style weights) |
| middle | same weight after block-Hadamard rotation; bulk is preserved, tails collapse into the Gaussian |
| right  | per-block max-abs distributions overlaid — the rotation makes each block's max-abs smaller and tighter, which is exactly what controls Q4 rounding error |

The text panel reports MSE at Q4 / Q3 / Q2 with and without rotation,
plus the implied "drop a tier and run faster" speed estimate.

## How to deploy this Space

1. Create a new Space at https://huggingface.co/new-space (Gradio SDK).
2. Copy `app.py`, `requirements.txt`, and this `README.md` into the
   Space's repo.
3. Also copy `turboquant/hadamard.py` and `turboquant/bench.py` (or run
   `pip install git+https://github.com/Ary5272/turbocpp` from inside
   the Space's `requirements.txt`).
4. Push — HF builds the image automatically.

## Local

```bash
pip install -e ".[demo]"
python -m space.app
```
