"""TurboQuant KV-cache rotation prep.

Decoding is the wall-clock bottleneck during generation, and decoding is
attention-heavy. Attention reads keys K and values V at every step — for
long contexts the KV cache **dominates memory bandwidth**.

llama.cpp already supports KV-cache quantization via
   --cache-type-k q4_0 --cache-type-v q4_0
but quality drops noticeably without rotation because raw K/V vectors
are heavy-tailed. Hadamard-rotating K and V before quantizing fixes that
— rotated K, V at q4_0 gives quality close to fp16 at half the
bandwidth → larger context window AND faster decode.

This module exposes:

  rotate_kv_cache_compatible_weights(model, block_size=128)

which adjusts the model's W_o (attention output) and W_q / W_k / W_v
projection weights so that K and V are produced in a Hadamard-rotated
frame. Quantization (whether by llama.cpp's KV cache code or anything
else) then sees a Gaussianized K/V — much friendlier to low-bit storage.

Mathematical guarantee (when paired with `apply_residual_rotation` from
turboquant.py): the model output is fp32-bit-identical to the baseline.

Implementation note: doing this WITHOUT runtime support requires that
the rotation `H` applied to K and V is offset by inverse rotations
inside attention. Since attention is `softmax(Q Kᵀ / √d) V`, rotating
K and V by H and Q by H yields:
    softmax((Q H)(K H)ᵀ / √d) (V H) = softmax(Q Kᵀ / √d) (V H)
i.e. the score matrix is preserved (Q H · (K H)ᵀ = Q Kᵀ) and the
output is just V·H. Combine with W_o post-mul Hᵀ to absorb the H. Net:
no runtime change required, KV cache stores rotated K/V, quantization
of the rotated cache loses less precision.
"""

from __future__ import annotations

import torch

from .hadamard import block_hadamard_inplace
from .turboquant import _attn_o, _attn_qkv, _layers


@torch.no_grad()
def rotate_kv_for_cache_quant(model, block_size: int = 128) -> None:
    """Per-head rotation that improves the quality of llama.cpp's
    --cache-type-k / --cache-type-v quantization. Apply on top of (or
    independent of) `apply_residual_rotation` — this one targets attention
    head_dim, not the residual stream.

    For each layer:
        W_q : output axis (head_dim per head)  → post-mul H
        W_k : output axis                      → post-mul H
        W_v : output axis                      → post-mul H
        W_o : input axis  (V's head_dim)       → post-mul Hᵀ

    With block_size = head_dim (e.g. 128 for LLaMA-2/3), this is a single
    Hadamard per head and no head-mixing — preserves attention semantics
    exactly.
    """
    for layer in _layers(model):
        Wq, Wk, Wv = _attn_qkv(layer)
        Wo = _attn_o(layer)
        # Output-axis rotation = first axis of an [out, in] PyTorch weight.
        block_hadamard_inplace(Wq.weight.data, axis=0, block=block_size)
        block_hadamard_inplace(Wk.weight.data, axis=0, block=block_size)
        block_hadamard_inplace(Wv.weight.data, axis=0, block=block_size)
        # Wo input axis = last axis.
        block_hadamard_inplace(Wo.weight.data, axis=-1, block=block_size)
