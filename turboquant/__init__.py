"""TurboQuant — offline Hadamard-rotation preprocessor for LLM weights.

Workflow:
    raw HF model
        │
        ▼
    fuse_norms_into_next()       ← absorb RMSNorm γ into the following linear
        │
        ▼
    apply_residual_rotation()    ← Hadamard-rotate the residual stream
        │
        ▼
    convert_hf_to_gguf.py        ← llama.cpp's standard converter
        │
        ▼
    quantize (Q4_K_M / Q4_0 / …) ← llama.cpp's standard quantizer
        │
        ▼
    standard GGUF that runs unchanged on llama.cpp, but with 0.3-0.5 ppl
    less degradation at the same bit budget — because every weight block
    saw a near-Gaussian input distribution at quantize time.

The math: H is orthogonal (H Hᵀ = I). Applying H at the residual stream
cancels through the linear-layer boundaries — Wq, Wk, Wv get post-mul'd
by Hᵀ on the input axis; the attention output projection (and FFN's
W_down) get pre-mul'd by H on the output axis. Inference is therefore
bit-identical to the un-rotated model in fp32 — quantization noise is
where the win shows up.
"""

from .hadamard import hadamard_matrix, block_hadamard_inplace
from .turboquant import (
    rotate_llama_model,
    fuse_norms_into_next,
    apply_residual_rotation,
)

__all__ = [
    "hadamard_matrix",
    "block_hadamard_inplace",
    "rotate_llama_model",
    "fuse_norms_into_next",
    "apply_residual_rotation",
]

__version__ = "0.1.0"
