#pragma once
#include <cstddef>
#include <cstdint>

namespace turbocpp {

// IEEE 754 half-precision (binary16). Stored as raw uint16_t — F16C
// intrinsics convert in 1 cycle; software fallback uses bit hacks.
using fp16_t = uint16_t;

// Vectorized fp16↔fp32 conversion. Arrays do not need to be aligned.
void fp16_to_fp32(const fp16_t* src, float* dst, size_t n);
void fp32_to_fp16(const float* src, fp16_t* dst, size_t n);

// Single-element scalar variants (for tests + the slow path).
fp16_t f32_to_f16_scalar(float f);
float  f16_to_f32_scalar(fp16_t h);

// Matmul with fp16 weights and fp32 activations. Same convention as
// matmul(): C[M,N] = A[M,K] @ W[N,K]^T. Weights upconverted on-the-fly
// inside an L1-resident scratch row to avoid a full materialize.
void matmul_f16(const float* A, const fp16_t* W, float* C,
                size_t M, size_t N, size_t K);

} // namespace turbocpp
