#pragma once
#include <cstddef>

namespace turbocpp {

// RMSNorm (Zhang & Sennrich 2019). Used by LLaMA in place of LayerNorm.
//   y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
//
// No bias, no mean subtraction (cheaper, numerically similar on trained
// weights). Operates per-row: input `x` is [rows, dim], weight is [dim].
// `y` may alias `x`.
void rmsnorm(const float* x, const float* weight, float* y,
             size_t rows, size_t dim, float eps);

// Single-row convenience — avoids the outer loop when we know rows=1
// (the generation hot path).
void rmsnorm_row(const float* x, const float* weight, float* y,
                 size_t dim, float eps);

} // namespace turbocpp
