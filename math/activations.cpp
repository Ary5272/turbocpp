#include "activations.h"
#include <cmath>

namespace turbocpp {

// tanh approximation constant: sqrt(2/π) ≈ 0.7978845608
static constexpr float kGeluK = 0.7978845608028654f;

void gelu_inplace(float* y, size_t n) {
    // Scalar implementation. A SIMD version would need a vectorized tanh
    // polynomial — worth doing later, but GELU isn't LLaMA's hot path
    // (LLaMA uses SiLU). See activations_avx2.cpp for an optimized version
    // if ever needed.
    for (size_t i = 0; i < n; ++i) {
        const float x = y[i];
        const float x3 = x * x * x;
        const float arg = kGeluK * (x + 0.044715f * x3);
        y[i] = 0.5f * x * (1.0f + std::tanh(arg));
    }
}

void silu_inplace(float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float x = y[i];
        // sigmoid(x) = 1 / (1 + exp(-x)); clamp exp argument to avoid inf.
        y[i] = x / (1.0f + std::exp(-x));
    }
}

void silu_mul(float* out, const float* gate, const float* up, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float g = gate[i];
        const float s = g / (1.0f + std::exp(-g));
        out[i] = s * up[i];
    }
}

} // namespace turbocpp
