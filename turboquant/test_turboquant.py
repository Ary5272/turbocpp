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


def _build_tiny_model_unit_norms():
    """Tiny LLaMA-shaped model with γ=1 everywhere — used for the
    pure-rotation invariance test where fuse_norms is a no-op."""
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
            # γ-only norm placeholder; γ pre-set to 1 so fuse_norms is a no-op.
            self.input_layernorm = torch.nn.LayerNorm(d, elementwise_affine=True, bias=False)
            self.self_attn = TinyAttn(d, dk)
            self.post_attention_layernorm = torch.nn.LayerNorm(d, elementwise_affine=True, bias=False)
            self.mlp = TinyMLP(d, df)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(32, 256)
            self.model.layers = torch.nn.ModuleList([TinyLayer(256, 256, 384)])
            self.model.norm = torch.nn.LayerNorm(256, elementwise_affine=True, bias=False)
            self.lm_head = torch.nn.Linear(256, 32, bias=False)

    torch.manual_seed(13)
    m = TinyModel().double()
    # γ = 1 everywhere → fuse_norms is a no-op → the test isolates rotation.
    for ln in [m.model.layers[0].input_layernorm,
               m.model.layers[0].post_attention_layernorm,
               m.model.norm]:
        ln.weight.data.fill_(1.0)
    return m


def test_rotation_preserves_embed_qproj_dot():
    """With γ=1, rotating tok_embed (out axis) and q_proj (in axis) cancels:
        (emb_row · H) · (q_row · H)ᵀ = emb_row · q_rowᵀ
    """
    from turboquant.turboquant import apply_residual_rotation
    m = _build_tiny_model_unit_norms()

    x = torch.randint(0, 32, (1, 5))
    pre  = m.model.embed_tokens(x) @ m.model.layers[0].self_attn.q_proj.weight.t()
    apply_residual_rotation(m, block_size=128)
    post = m.model.embed_tokens(x) @ m.model.layers[0].self_attn.q_proj.weight.t()
    assert torch.allclose(pre, post, atol=1e-9), \
        f"rotation invariance broken: max diff {(pre - post).abs().max()}"


def test_norm_fusion_preserves_attn_input_path():
    """fuse_norms_into_next must preserve the value of `q_proj(γ ⊙ x)`:
        before: y = q_proj(γ ⊙ x)
        after:  y = (q_proj · diag(γ))(x), with γ ← 1
    Both should give bit-identical fp64 results.
    """
    from turboquant.turboquant import fuse_norms_into_next

    torch.manual_seed(17)
    m = _build_tiny_model_unit_norms()
    # Now scramble γ so fusion has real work to do.
    for ln in [m.model.layers[0].input_layernorm,
               m.model.layers[0].post_attention_layernorm,
               m.model.norm]:
        ln.weight.data = 0.5 + torch.rand_like(ln.weight).double()

    x = torch.randn(1, 5, 256, dtype=torch.float64)
    g = m.model.layers[0].input_layernorm.weight.data
    W = m.model.layers[0].self_attn.q_proj.weight.data
    pre = (g * x) @ W.t()      # γ ⊙ x then linear

    fuse_norms_into_next(m)

    g2 = m.model.layers[0].input_layernorm.weight.data
    W2 = m.model.layers[0].self_attn.q_proj.weight.data
    post = (g2 * x) @ W2.t()    # γ should now be 1; linear absorbed γ
    assert torch.allclose(pre, post, atol=1e-12), \
        f"fuse_norms_into_next changed output: max diff {(pre - post).abs().max()}"
    assert torch.allclose(g2, torch.ones_like(g2)), "γ wasn't reset to 1"
