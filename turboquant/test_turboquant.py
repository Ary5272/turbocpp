"""Self-checks. Run with: pytest turboquant/test_turboquant.py"""
import math

import pytest
import torch

from turboquant.hadamard import (
    block_hadamard_inplace,
    hadamard_matrix,
    is_orthogonal,
)


def test_hadamard_orthogonal():
    for n in (2, 4, 8, 16, 32, 64, 128, 256):
        assert is_orthogonal(hadamard_matrix(n))


def test_hadamard_involution():
    """H is its own inverse: H(H(x)) = x."""
    torch.manual_seed(0)
    H = hadamard_matrix(64)
    x = torch.randn(64)
    y = H @ (H @ x)
    assert torch.allclose(x, y, atol=1e-5)


def test_block_hadamard_dot_product_preserved():
    """The whole point: rotation preserves dot products of paired vectors.

        dot(x, w) == dot(H(x), H(w))
    so quantizing H(W) and rotating activations through the same H gives
    the same answer as quantizing W and not rotating — except quantization
    error is much lower because H(W)'s blocks are Gaussianized.
    """
    torch.manual_seed(7)
    n = 256
    x = torch.randn(n, dtype=torch.float64)
    w = torch.randn(n, dtype=torch.float64)
    pre = torch.dot(x, w)

    xx = x.clone(); ww = w.clone()
    # block-Hadamard is a 1D op; route it through a 2D path
    block_hadamard_inplace(xx.view(1, n), axis=-1, block=128)
    block_hadamard_inplace(ww.view(1, n), axis=-1, block=128)
    post = torch.dot(xx, ww)
    assert math.isclose(pre.item(), post.item(), rel_tol=1e-10, abs_tol=1e-9)


def test_block_hadamard_reduces_max_abs():
    """Heavy-tailed → near-Gaussian: per-block max-abs should drop."""
    torch.manual_seed(11)
    n = 256
    # Heavy-tailed: occasional outliers + small bulk.
    w = 0.1 * torch.randn(n)
    w[3] = 5.0      # tail
    w[120] = 4.5    # tail

    pre_max = w.abs().reshape(-1, 32).max(dim=1).values

    block_hadamard_inplace(w.view(1, n), axis=-1, block=128)
    post_max = w.abs().reshape(-1, 32).max(dim=1).values

    # Average per-block max-abs should drop substantially.
    assert post_max.mean() < pre_max.mean(), (
        f"rotation didn't help: pre={pre_max.mean()}, post={post_max.mean()}"
    )


def test_rotate_tiny_model_dot_product_invariant():
    """End-to-end smoke: rotate a tiny linear-stack and verify outputs match."""
    from turboquant.turboquant import apply_residual_rotation, fuse_norms_into_next

    class TinyAttn(torch.nn.Module):
        def __init__(self, d, dk):
            super().__init__()
            self.q_proj = torch.nn.Linear(d, dk, bias=False)
            self.k_proj = torch.nn.Linear(d, dk, bias=False)
            self.v_proj = torch.nn.Linear(d, dk, bias=False)
            self.o_proj = torch.nn.Linear(dk, d, bias=False)

    class TinyMLP(torch.nn.Module):
        def __init__(self, d, df):
            super().__init__()
            self.gate_proj = torch.nn.Linear(d, df, bias=False)
            self.up_proj   = torch.nn.Linear(d, df, bias=False)
            self.down_proj = torch.nn.Linear(df, d, bias=False)

    class TinyLayer(torch.nn.Module):
        def __init__(self, d, dk, df):
            super().__init__()
            self.input_layernorm = torch.nn.LayerNorm(d, elementwise_affine=True)
            self.self_attn = TinyAttn(d, dk)
            self.post_attention_layernorm = torch.nn.LayerNorm(d, elementwise_affine=True)
            self.mlp = TinyMLP(d, df)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(32, 256)
            self.model.layers = torch.nn.ModuleList([TinyLayer(256, 256, 384)])
            self.model.norm = torch.nn.LayerNorm(256, elementwise_affine=True)
            self.lm_head = torch.nn.Linear(256, 32, bias=False)

    torch.manual_seed(13)
    m = TinyModel().double()
    # Use random γ values to make the norm-fusion path actually do work.
    for ln in [m.model.layers[0].input_layernorm,
               m.model.layers[0].post_attention_layernorm,
               m.model.norm]:
        ln.weight.data = 0.5 + torch.rand_like(ln.weight)

    x = torch.randint(0, 32, (1, 5))

    # Rotation only preserves OUTPUTS for the linear pieces; LayerNorm γ
    # absorption is a math identity. We test the residual + linear path:
    # take embedding → q_proj → output, and check rotated version yields
    # identical numbers.
    pre = m.model.embed_tokens(x) @ m.model.layers[0].self_attn.q_proj.weight.t()
    fuse_norms_into_next(m)
    apply_residual_rotation(m, block_size=128)
    post = m.model.embed_tokens(x) @ m.model.layers[0].self_attn.q_proj.weight.t()
    # The Hadamard rotations on tok_embed (out axis) and q_proj (in axis)
    # cancel: `(emb_row · H) · (H · q_row)ᵀ = emb_row · q_rowᵀ`.
    assert torch.allclose(pre, post, atol=1e-9), \
        f"rotation invariance broken: max diff {(pre - post).abs().max()}"
