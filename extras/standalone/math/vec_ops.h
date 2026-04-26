#pragma once
#include <cstddef>

namespace turbocpp {

// Element-wise y[i] += x[i] for i in [0, n). Pointers need not be aligned.
void vec_add_inplace(float* y, const float* x, size_t n);

// Element-wise y[i] = x[i] * v[i].
void vec_mul(float* y, const float* x, const float* v, size_t n);

// Element-wise y[i] *= s.
void vec_scale(float* y, float s, size_t n);

// Dot product of x and y. SIMD-accelerated. Returns scalar sum.
float vec_dot(const float* x, const float* y, size_t n);

// Numerically stable softmax in-place: x[i] = exp(x[i] - max) / sum.
// Input and output point at the same buffer.
void softmax_inplace(float* x, size_t n);

// Find argmax over x[0..n). Returns index. n==0 returns 0.
size_t argmax(const float* x, size_t n);

// Mean of squares: (1/n) * sum(x[i]^2). Used by RMSNorm.
float mean_sq(const float* x, size_t n);

} // namespace turbocpp
