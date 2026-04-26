#pragma once
#include <cstddef>

namespace turbocpp {

// GELU (tanh approximation used by GPT-2/LLaMA-alt):
//   0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// Applied in-place over y[0..n).
void gelu_inplace(float* y, size_t n);

// SiLU (aka Swish): y = y * sigmoid(y) = y / (1 + exp(-y)).
// Used by LLaMA FFN (SwiGLU = SiLU(gate) * up).
void silu_inplace(float* y, size_t n);

// Fused SwiGLU-style: out[i] = silu(gate[i]) * up[i]. All three buffers
// size n. `out` may alias `gate` or `up`.
void silu_mul(float* out, const float* gate, const float* up, size_t n);

} // namespace turbocpp
