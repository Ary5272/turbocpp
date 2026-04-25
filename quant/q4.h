#pragma once
#include <cstddef>
#include <cstdint>

namespace turbocpp {

// ---------------------------------------------------------------------------
// Q4_0-style block-wise 4-bit weight quantization
// ---------------------------------------------------------------------------
// Block size: 32 elements
// Per-block header: fp32 scale `d`
// Per-block payload: 16 bytes = 32 × 4-bit unsigned quants
//
// Dequantize: x_i = d * (q_i - 8)
// The -8 centering recovers a signed range ~[-8d, 7d]; symmetric around 0
// which matches typical weight distributions.
//
// Compression: 32 × 4B = 128B fp32 → 20B Q4 ≈ 6.4× smaller.
//
// Real llama.cpp Q4_0 uses fp16 scale (18B/block ≈ 7.1×). We use fp32 scale
// for clarity and to avoid the fp16 conversion path on non-F16C CPUs.
// ---------------------------------------------------------------------------

constexpr size_t kQ4BlockSize = 32;

#pragma pack(push, 1)
struct Q4Block {
    float   d;           // block scale
    uint8_t q[16];       // two 4-bit values packed per byte (low nibble = even index)
};
#pragma pack(pop)

static_assert(sizeof(Q4Block) == 20, "Q4Block must be 20 bytes");

// Number of blocks needed to store `n` fp32 elements (must have n % 32 == 0).
inline size_t q4_num_blocks(size_t n) { return n / kQ4BlockSize; }
inline size_t q4_bytes_for(size_t n)  { return q4_num_blocks(n) * sizeof(Q4Block); }

// Quantize a row of `n` floats (n multiple of 32) into pre-allocated blocks.
// Uses symmetric max-abs scaling per block.
void q4_quantize(const float* x, Q4Block* out, size_t n);

// Inverse — useful for tests / verification.
void q4_dequantize(const Q4Block* blocks, float* out, size_t n);

// Matmul with Q4 weights. Weight matrix W_q is stored as N rows, each row
// being q4_num_blocks(K) blocks. Input/output are fp32.
//
//   C[M, N] = A[M, K] @ W[N, K]^T   where W is materialized from W_q.
//
// K must be divisible by 32 (no partial tail blocks).
void matmul_q4(const float* A, const Q4Block* W_q, float* C,
               size_t M, size_t N, size_t K);

// Quantize a full [N, K] fp32 weight matrix into row-stored Q4 blocks.
// Caller pre-allocates `out` with N * q4_num_blocks(K) Q4Block entries.
void q4_quantize_matrix(const float* W_f32, Q4Block* out,
                        size_t N, size_t K);

} // namespace turbocpp
