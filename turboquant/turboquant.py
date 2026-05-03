"""Offline TurboQuant preprocessor for LLaMA-architecture HF models.

Public entry point: rotate_llama_model(model, block_size=128).

Implementation walks the model two passes:

  1. fuse_norms_into_next():
       Each RMSNorm has a per-channel γ. We absorb γ into the FOLLOWING
       linear's input columns and then set γ ← 1. After this, every
       RMSNorm is "vanilla" (just x / ||x||) and is rotation-equivariant
       — i.e. H · norm(x) = norm(H · x). Necessary because RMSNorm with
       γ ≠ 1 is NOT equivariant under non-diagonal rotations.

  2. apply_residual_rotation():
       Walk every linear that reads from or writes to the residual stream
       and apply Hadamard rotation in pairs:
         - W_q / W_k / W_v / W_gate / W_up : input axis  (post-mul Hᵀ)
         - W_o (attn output) / W_down       : output axis (pre-mul H)
         - tok_embed.weight                 : output axis (pre-mul H)
         - lm_head.weight                   : input axis  (post-mul Hᵀ)
       The two rotations cancel through the residual stream, so model
       output is fp32-identical to the un-rotated model. Quantization
       error drops because the rotated weights see a near-Gaussian input
       distribution at quantize time.

Result: a rotated state_dict that you can save and feed straight into
llama.cpp's convert_hf_to_gguf.py + llama-quantize. NO inference-time
changes, NO custom GGML type required.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .hadamard import block_hadamard_inplace


# ---------------------------------------------------------------------------
# Helpers to find pieces of a HF LLaMA / Mistral / Qwen-2 model.
# We use duck-typing on attribute names rather than isinstance checks so
# the function works across HF transformer revisions.
# ---------------------------------------------------------------------------
def _layers(model) -> Iterable:
    """Yield each transformer block."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("unknown architecture (cannot find transformer blocks)")


def _norm_input_attn(layer):
    return (
        getattr(layer, "input_layernorm", None)
        or getattr(layer, "ln_1", None)
        or getattr(layer, "attention_norm", None)
    )


def _norm_post_attn(layer):
    return (
        getattr(layer, "post_attention_layernorm", None)
        or getattr(layer, "ln_2", None)
        or getattr(layer, "ffn_norm", None)
    )


def _attn_qkv(layer):
    a = layer.self_attn if hasattr(layer, "self_attn") else layer.attention
    return a.q_proj, a.k_proj, a.v_proj


def _attn_o(layer):
    a = layer.self_attn if hasattr(layer, "self_attn") else layer.attention
    return a.o_proj if hasattr(a, "o_proj") else a.wo


def _mlp(layer):
    return layer.mlp if hasattr(layer, "mlp") else layer.feed_forward


def _ffn_gate_up(layer):
    m = _mlp(layer)
    if hasattr(m, "gate_proj") and hasattr(m, "up_proj"):
        return m.gate_proj, m.up_proj
    return m.w1, m.w3  # LLaMA-1 names


def _ffn_down(layer):
    m = _mlp(layer)
    return m.down_proj if hasattr(m, "down_proj") else m.w2


# ---------------------------------------------------------------------------
# Pass 1 — fuse RMSNorm γ into the following linear.
#
# For y = (γ ⊙ norm(x)) → linear(y) = W (γ ⊙ norm(x)) = (W diag(γ)) norm(x).
# So we set W ← W diag(γ) (i.e. multiply each column j by γ[j]) and γ ← 1.
# ---------------------------------------------------------------------------
@torch.no_grad()
def fuse_norms_into_next(model) -> None:
    for layer in _layers(model):
        norm1 = _norm_input_attn(layer)
        if norm1 is not None and getattr(norm1, "weight", None) is not None:
            g = norm1.weight.data
            for w in _attn_qkv(layer):
                w.weight.data.mul_(g)  # broadcast on input dim
            norm1.weight.data.fill_(1.0)

        norm2 = _norm_post_attn(layer)
        if norm2 is not None and getattr(norm2, "weight", None) is not None:
            g = norm2.weight.data
            for w in _ffn_gate_up(layer):
                w.weight.data.mul_(g)
            norm2.weight.data.fill_(1.0)

    # Final norm before lm_head (if present).
    final_norm = getattr(getattr(model, "model", None), "norm", None) or getattr(
        model, "final_layer_norm", None
    )
    lm_head = getattr(model, "lm_head", None)
    if (
        final_norm is not None
        and lm_head is not None
        and getattr(final_norm, "weight", None) is not None
    ):
        g = final_norm.weight.data
        lm_head.weight.data.mul_(g)
        final_norm.weight.data.fill_(1.0)


# ---------------------------------------------------------------------------
# Pass 2 — block-Hadamard rotate the residual stream.
#
# After Pass 1, RMSNorm is rotation-equivariant. We rotate every weight
# that reads from or writes to the residual stream:
#
#   producers (write):  tok_embed (output axis), W_o (out), W_down (out)
#   consumers (read):   W_q/k/v (in), W_gate/up (in), lm_head (in)
#
# Rotation pair: post-mul producer's OUTPUT axis by H, post-mul
# consumer's INPUT axis by Hᵀ. Since H = Hᵀ for our normalized symmetric
# Hadamard, both reduce to multiplying along the same axis by H.
# ---------------------------------------------------------------------------
@torch.no_grad()
def apply_residual_rotation(model, block_size: int = 128) -> None:
    # Rotate token embeddings (output axis = last dim).
    tok_embed = getattr(getattr(model, "model", None), "embed_tokens", None) or getattr(
        model, "wte", None
    )
    if tok_embed is not None:
        block_hadamard_inplace(tok_embed.weight.data, axis=-1, block=block_size)

    for layer in _layers(model):
        # Consumers — input axis (last dim of weight in PyTorch [out, in])
        for w in (*_attn_qkv(layer), *_ffn_gate_up(layer)):
            block_hadamard_inplace(w.weight.data, axis=-1, block=block_size)

        # Producers — output axis (first dim of weight)
        block_hadamard_inplace(_attn_o(layer).weight.data, axis=0, block=block_size)
        block_hadamard_inplace(_ffn_down(layer).weight.data, axis=0, block=block_size)

    # lm_head: input axis.
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        block_hadamard_inplace(lm_head.weight.data, axis=-1, block=block_size)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
@torch.no_grad()
def rotate_llama_model(model, block_size: int = 128, fuse_norms: bool = True):
    """Rotate `model` in place. Returns it for fluent chaining."""
    if fuse_norms:
        fuse_norms_into_next(model)
    apply_residual_rotation(model, block_size=block_size)
    return model
