#pragma once
#include <cstddef>
#include <cstdint>

namespace turbocpp {

// ---------------------------------------------------------------------------
// K-quants — llama.cpp's quality-tier superblock formats.
//
// Each super-block contains 256 elements split into 16-element (Q6_K) or
// 32-element (Q4_K, Q5_K) sub-blocks. Per-sub-block scales let the
// quantizer track local distribution shifts that vanilla Q4_0 misses.
//
// We implement Q4_K_M (4.5 bpw, the most popular size/quality tier), Q6_K
// (6.5 bpw, near-fp16 quality), and Q8_K (8 bpw, used for embeddings).
// ---------------------------------------------------------------------------

constexpr size_t kKBlockSize = 256;   // super-block

// ---------------------------------------------------------------------------
// Q6_K — 6 bits/weight. 256 elems split into 16 sub-blocks of 16.
// Format (simplified vs llama.cpp; uses fp32 d instead of fp16 for clarity):
//   d:        fp32                                  4 B
//   sub_d:    int8[16]   (signed sub-scales)       16 B
//   q_low:    uint8[128] (lower 4 bits per weight)128 B
//   q_high:   uint8[64]  (upper 2 bits, packed)    64 B
// total = 212 B / super-block ≈ 6.625 bpw.
//
// Dequant: w_i = d * sub_d[i/16] * (q_i - 32)  where q_i ∈ [0, 63].
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct Q6KBlock {
    float   d;
    int8_t  sub_d[16];
    uint8_t q_low[128];
    uint8_t q_high[64];
};
#pragma pack(pop)
static_assert(sizeof(Q6KBlock) == 212, "Q6KBlock = 212 bytes");

// ---------------------------------------------------------------------------
// Q4_K_M — 4.5 bits/weight. 256 elems split into 8 sub-blocks of 32.
// Per-sub-block: 6-bit scale + 6-bit min (12 bits packed → 1.5 B). Plus
// 32 × 4-bit quants (16 B per sub-block). Total per sub-block: 17.5 B.
// Plus master d, dmin (fp32 each = 8 B).
// total = 8 + 12 + 128 = 148 B / super-block ≈ 4.625 bpw.
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct Q4KBlock {
    float   d;          // master scale of scales
    float   dmin;       // master scale of mins
    uint8_t scales[12]; // 8×(6+6) = 96 bits packed
    uint8_t qs[128];    // 256×4-bit
};
#pragma pack(pop)
static_assert(sizeof(Q4KBlock) == 148, "Q4KBlock = 148 bytes");

// ---------------------------------------------------------------------------
// Q8_K — 8 bits/weight + per-superblock fp32 scale.
// total = 4 + 256 = 260 B / super-block ≈ 8.125 bpw. Used for embeddings
// + lm_head where Q4 hurts disproportionately.
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
struct Q8KBlock {
    float  d;
    int8_t qs[256];
};
#pragma pack(pop)
static_assert(sizeof(Q8KBlock) == 260, "Q8KBlock = 260 bytes");

// API
size_t qk_num_blocks(size_t n);  // n / 256

void q6k_quantize(const float* x, Q6KBlock* out, size_t n);
void q6k_dequantize(const Q6KBlock* blocks, float* out, size_t n);
void matmul_q6k(const float* A, const Q6KBlock* W, float* C,
                size_t M, size_t N, size_t K);

void q4k_quantize(const float* x, Q4KBlock* out, size_t n);
void q4k_dequantize(const Q4KBlock* blocks, float* out, size_t n);
void matmul_q4k(const float* A, const Q4KBlock* W, float* C,
                size_t M, size_t N, size_t K);

void q8k_quantize(const float* x, Q8KBlock* out, size_t n);
void q8k_dequantize(const Q8KBlock* blocks, float* out, size_t n);
void matmul_q8k(const float* A, const Q8KBlock* W, float* C,
                size_t M, size_t N, size_t K);

} // namespace turbocpp
