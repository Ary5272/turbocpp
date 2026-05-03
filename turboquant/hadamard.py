"""Walsh-Hadamard transform helpers.

We use:
  - hadamard_matrix(n)            for arbitrary power-of-2 n
  - block_hadamard_inplace()      to apply n×n WHT to fixed-size blocks of
                                  a longer vector / row of a matrix
"""

from __future__ import annotations

import math

import numpy as np
import torch


def hadamard_matrix(n: int, dtype=torch.float32) -> torch.Tensor:
    """Return the n×n NORMALIZED Walsh-Hadamard matrix.

    H is its own inverse (H H = I) so quantization rotations cancel under
    H @ Wᵀ ··· W @ Hᵀ ≡ identity. n must be a power of 2.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of 2, got {n}")
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    return H / math.sqrt(n)


def block_hadamard_inplace(W: torch.Tensor, axis: int = -1, block: int = 128) -> None:
    """Apply n×n Hadamard to every contiguous `block`-sized slice along `axis`.

    Used when the full dim isn't a power of 2 (e.g. ffn_dim=11008). Block
    size 128 fits comfortably in L1 and is a power of 2.
    """
    n = W.shape[axis]
    if n % block != 0:
        raise ValueError(f"axis dim {n} not divisible by block {block}")
    H = hadamard_matrix(block, dtype=W.dtype).to(W.device)
    # Reshape axis -> (n//block, block), apply H on the last dim, reshape back.
    moved = W.transpose(axis, -1)  # bring axis to last
    shape = moved.shape
    g = shape[-1] // block
    moved = moved.reshape(*shape[:-1], g, block)
    moved = moved @ H  # last-axis matmul; H is symmetric
    moved = moved.reshape(*shape)
    out = moved.transpose(axis, -1)
    W.copy_(out)


def is_orthogonal(H: torch.Tensor, tol: float = 1e-5) -> bool:
    """Self-check: H @ Hᵀ ≈ I."""
    n = H.shape[0]
    err = (H @ H.t() - torch.eye(n, dtype=H.dtype)).abs().max().item()
    return err < tol


# numpy convenience for tooling that doesn't want a torch dep
def hadamard_matrix_np(n: int) -> np.ndarray:
    return hadamard_matrix(n, dtype=torch.float64).numpy()
