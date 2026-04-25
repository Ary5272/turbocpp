#include "q4.h"
#include "../utils/logging.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace turbocpp {

// ---------------------------------------------------------------------------
// Per-block quantize: find max |x| over 32 elements, derive symmetric scale
// d = maxabs / 7.5, then q = round(x / d + 8) clamped to [0, 15].
// 7.5 because we map [-7.5d, 7.5d] onto [0, 15], centered at 8.
// ---------------------------------------------------------------------------
static inline void quantize_block(const float* x, Q4Block& out) {
    float maxabs = 0.0f;
    for (size_t i = 0; i < kQ4BlockSize; ++i) {
        float a = std::fabs(x[i]);
        if (a > maxabs) maxabs = a;
    }

    // d = maxabs / 7.5; if all zeros, d = 0 and all quants become 8.
    const float d = (maxabs > 0.0f) ? (maxabs / 7.5f) : 0.0f;
    const float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    out.d = d;

    for (size_t i = 0; i < kQ4BlockSize / 2; ++i) {
        int q0 = int(std::lround(x[2 * i]     * id)) + 8;
        int q1 = int(std::lround(x[2 * i + 1] * id)) + 8;
        q0 = std::clamp(q0, 0, 15);
        q1 = std::clamp(q1, 0, 15);
        out.q[i] = uint8_t((q1 << 4) | q0);
    }
}

// ---------------------------------------------------------------------------
// Per-block dequantize.
// ---------------------------------------------------------------------------
static inline void dequantize_block(const Q4Block& blk, float* out) {
    const float d = blk.d;
    for (size_t i = 0; i < kQ4BlockSize / 2; ++i) {
        const uint8_t packed = blk.q[i];
        const int q0 = int(packed & 0x0F) - 8;
        const int q1 = int(packed >> 4)   - 8;
        out[2 * i]     = d * float(q0);
        out[2 * i + 1] = d * float(q1);
    }
}

void q4_quantize(const float* x, Q4Block* out, size_t n) {
    TCPP_CHECK(n % kQ4BlockSize == 0, "q4: n must be multiple of 32, got %zu", n);
    const size_t nb = n / kQ4BlockSize;
    for (size_t b = 0; b < nb; ++b) {
        quantize_block(x + b * kQ4BlockSize, out[b]);
    }
}

void q4_dequantize(const Q4Block* blocks, float* out, size_t n) {
    TCPP_CHECK(n % kQ4BlockSize == 0, "q4: n must be multiple of 32, got %zu", n);
    const size_t nb = n / kQ4BlockSize;
    for (size_t b = 0; b < nb; ++b) {
        dequantize_block(blocks[b], out + b * kQ4BlockSize);
    }
}

void q4_quantize_matrix(const float* W_f32, Q4Block* out, size_t N, size_t K) {
    TCPP_CHECK(K % kQ4BlockSize == 0, "q4: K must be multiple of 32, got %zu", K);
    const size_t blocks_per_row = K / kQ4BlockSize;
    for (size_t n = 0; n < N; ++n) {
        q4_quantize(W_f32 + n * K, out + n * blocks_per_row, K);
    }
}

// ---------------------------------------------------------------------------
// Q4 matmul
// ---------------------------------------------------------------------------
// Per-row dot product for one (M, N) output. Two nested loops:
//   - outer over blocks (b), inner over 32 elements.
//   - For each block, accumulate sum(q * x) then multiply by block scale.
//
// Why not pre-dequantize the full row? Memory. A 4096×4096 Q4 matrix is
// ~80MB; dequantized it's ~64MB PER MATMUL — cache-thrashing. In-place
// decode keeps the working set tiny.

#if defined(__AVX2__)

// Unpack 16 bytes of Q4 into 32 int8 values in the range [-8, 7].
// Low nibble -> even index, high nibble -> odd index.
static inline void unpack_q4_block_i8(const uint8_t* q, int8_t* out) {
    for (size_t i = 0; i < 16; ++i) {
        uint8_t packed = q[i];
        out[2 * i]     = int8_t(packed & 0x0F) - 8;
        out[2 * i + 1] = int8_t(packed >> 4)   - 8;
    }
}

static inline float q4_dot_block_avx2(const Q4Block& blk, const float* x) {
    alignas(32) int8_t vals[32];
    unpack_q4_block_i8(blk.q, vals);

    // Convert int8 -> int32 -> fp32 (two 128-bit halves, four 8-lanes).
    __m128i v_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vals));
    __m128i v_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vals + 16));
    __m256i v0i = _mm256_cvtepi8_epi32(v_lo);                   // vals[0..7]
    __m256i v1i = _mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8)); // vals[8..15]
    __m256i v2i = _mm256_cvtepi8_epi32(v_hi);                   // vals[16..23]
    __m256i v3i = _mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8)); // vals[24..31]
    __m256 w0 = _mm256_cvtepi32_ps(v0i);
    __m256 w1 = _mm256_cvtepi32_ps(v1i);
    __m256 w2 = _mm256_cvtepi32_ps(v2i);
    __m256 w3 = _mm256_cvtepi32_ps(v3i);

    __m256 a0 = _mm256_loadu_ps(x +  0);
    __m256 a1 = _mm256_loadu_ps(x +  8);
    __m256 a2 = _mm256_loadu_ps(x + 16);
    __m256 a3 = _mm256_loadu_ps(x + 24);

    __m256 acc = _mm256_setzero_ps();
#  if defined(__FMA__)
    acc = _mm256_fmadd_ps(a0, w0, acc);
    acc = _mm256_fmadd_ps(a1, w1, acc);
    acc = _mm256_fmadd_ps(a2, w2, acc);
    acc = _mm256_fmadd_ps(a3, w3, acc);
#  else
    acc = _mm256_add_ps(acc, _mm256_mul_ps(a0, w0));
    acc = _mm256_add_ps(acc, _mm256_mul_ps(a1, w1));
    acc = _mm256_add_ps(acc, _mm256_mul_ps(a2, w2));
    acc = _mm256_add_ps(acc, _mm256_mul_ps(a3, w3));
#  endif

    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    __m128 ss = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, ss);
    ss = _mm_add_ss(ss, sh);
    return blk.d * _mm_cvtss_f32(ss);
}

#endif  // __AVX2__

static inline float q4_dot_block_scalar(const Q4Block& blk, const float* x) {
    float s = 0.0f;
    for (size_t i = 0; i < 16; ++i) {
        const uint8_t packed = blk.q[i];
        const int q0 = int(packed & 0x0F) - 8;
        const int q1 = int(packed >> 4)   - 8;
        s += float(q0) * x[2 * i] + float(q1) * x[2 * i + 1];
    }
    return blk.d * s;
}

static inline float q4_dot_row(const Q4Block* w_row, const float* x_row, size_t K) {
    const size_t nb = K / kQ4BlockSize;
    float acc = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        const float* xb = x_row + b * kQ4BlockSize;
#if defined(__AVX2__)
        acc += q4_dot_block_avx2(w_row[b], xb);
#else
        acc += q4_dot_block_scalar(w_row[b], xb);
#endif
    }
    return acc;
}

void matmul_q4(const float* A, const Q4Block* W_q, float* C,
               size_t M, size_t N, size_t K) {
    TCPP_CHECK(K % kQ4BlockSize == 0, "matmul_q4: K must be multiple of 32");
    const size_t blocks_per_row = K / kQ4BlockSize;
    for (size_t m = 0; m < M; ++m) {
        const float* a_row = A + m * K;
        for (size_t n = 0; n < N; ++n) {
            const Q4Block* w_row = W_q + n * blocks_per_row;
            C[m * N + n] = q4_dot_row(w_row, a_row, K);
        }
    }
}

} // namespace turbocpp
