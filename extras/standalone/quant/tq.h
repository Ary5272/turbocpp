#pragma once
#include <cstddef>
#include <cstdint>
#include "q4.h"

namespace turbocpp {

// ---------------------------------------------------------------------------
// TurboQuant — Hadamard-rotated block quantization.
//
// Pipeline (offline):
//   1. Apply block-Hadamard transform to each row of W (block_size = 128).
//      This Gaussianizes per-block distributions.
//   2. Quantize each rotated row with standard Q4_0 blocks.
//
// Pipeline (inference, per matmul call):
//   1. Apply the SAME block-Hadamard to x along its K dim.
//   2. Run q4_dot of x_rot with the rotated W blocks.
//
// Why this beats vanilla Q4:
//   - Vanilla Q4 max-abs is dominated by tail outliers in raw weights.
//   - After Hadamard, every output is ~Σ ±w_i / √n → near-Gaussian.
//   - Per-block max-abs drops 2-4× → quant step shrinks → rounding error
//     drops correspondingly. ~0.3-0.5 perplexity better than Q4_0 at
//     identical bit budget.
//
// Storage format: identical to Q4Block. Block size = kQ4BlockSize = 32, but
// the Hadamard rotation operates over a SUPERBLOCK of `tq_block` (default
// 128). The first 32 elements of each 128 form a Q4 block, etc. — 4 Q4
// blocks per Hadamard superblock.
// ---------------------------------------------------------------------------

constexpr size_t kTQBlockSize = 128;   // Hadamard superblock (must be 2^k)

// Quantize an [N, K] fp32 weight matrix into TurboQuant format. K must be a
// multiple of kTQBlockSize. Output uses Q4Block layout, K/32 blocks per row.
void tq_quantize_matrix(const float* W_f32, Q4Block* out, size_t N, size_t K,
                        size_t tq_block = kTQBlockSize);

// matmul with TurboQuant weights. Internally rotates each A row through a
// stack-allocated scratch buffer of size K (≤ 16KB at hidden=4096) — fits
// L1, no heap allocation.
void matmul_tq(const float* A, const Q4Block* W_tq, float* C,
               size_t M, size_t N, size_t K, size_t tq_block = kTQBlockSize);

} // namespace turbocpp
