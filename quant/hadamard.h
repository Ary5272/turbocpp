#pragma once
#include <cstddef>
#include <cstdint>

namespace turbocpp {

// ---------------------------------------------------------------------------
// Walsh–Hadamard transform — the core of TurboQuant.
//
// The H matrix is +/-1 and orthogonal: H^T H = n·I (or I when normalized).
// Applying it to a heavy-tailed (e.g. LLaMA weight) distribution has the
// effect of Gaussianizing it — the central-limit theorem kicks in on each
// output, since each output is a sum/diff of all inputs.
//
// Why it matters for quantization: max-abs scaling is dominated by tail
// outliers; a Gaussian-distributed vector with the same L2 norm has a
// MUCH smaller max-abs. Result: per-block quant step shrinks ~3-5× → far
// less rounding error at the same bit budget.
//
// Algorithm: Cooley-Tukey-style butterfly (a, b) → (a+b, a-b), iterated
// log2(n) times. Cost: n log2(n) adds.
//
// Limitations: n must be a power of two. For non-power-of-two dims we
// apply BLOCK Hadamard with block size 128 (works for 4096 = 32·128 and
// for 11008 = 86·128, the typical LLaMA shapes).
// ---------------------------------------------------------------------------

// In-place WHT on n floats. n MUST be a power of 2. Normalized so that
// the transform is orthogonal (i.e. its own inverse).
void hadamard_inplace(float* x, size_t n);

// Block-Hadamard: applies hadamard_inplace to each block of `block_size`
// elements in [0, n). n must be a multiple of block_size. block_size must
// be power-of-2 (default 128 — fits L1, log2=7 passes).
void hadamard_block_inplace(float* x, size_t n, size_t block_size = 128);

// Apply block-Hadamard to every row of an [N, K] matrix in place (operates
// on the K dimension — the contraction axis of matmul).
void hadamard_rows_inplace(float* W, size_t N, size_t K, size_t block_size = 128);

} // namespace turbocpp
