#pragma once
#include <cstddef>
#include <cstdint>

namespace turbocpp {

// Q8_0 weight quantization. 32-element blocks, fp32 scale + 32 int8 quants.
// Bytes/block = 4 + 32 = 36; compression ≈ 3.55× vs fp32.
//
// Why Q8 when we have Q4? Q8 is the quality-floor: <0.05 perplexity loss
// on LLaMA-7B vs fp16, used for embeddings and lm_head where Q4 hurts.

constexpr size_t kQ8BlockSize = 32;

#pragma pack(push, 1)
struct Q8Block {
    float  d;                  // scale
    int8_t q[kQ8BlockSize];    // signed quants
};
#pragma pack(pop)
static_assert(sizeof(Q8Block) == 36, "Q8Block must be 36 bytes");

inline size_t q8_num_blocks(size_t n) { return n / kQ8BlockSize; }

void q8_quantize(const float* x, Q8Block* out, size_t n);
void q8_dequantize(const Q8Block* blocks, float* out, size_t n);
void q8_quantize_matrix(const float* W_f32, Q8Block* out, size_t N, size_t K);

// matmul with Q8 weights and fp32 activations.
void matmul_q8(const float* A, const Q8Block* W_q, float* C,
               size_t M, size_t N, size_t K);

} // namespace turbocpp
