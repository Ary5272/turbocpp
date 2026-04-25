#include "q8.h"
#include "../utils/logging.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace turbocpp {

static inline void quantize_block_q8(const float* x, Q8Block& out) {
    float maxabs = 0.0f;
    for (size_t i = 0; i < kQ8BlockSize; ++i) {
        float a = std::fabs(x[i]);
        if (a > maxabs) maxabs = a;
    }
    // Map maxabs onto int8 range (-127..127), reserving -128.
    const float d  = (maxabs > 0.0f) ? (maxabs / 127.0f) : 0.0f;
    const float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    out.d = d;
    for (size_t i = 0; i < kQ8BlockSize; ++i) {
        int q = int(std::lround(x[i] * id));
        out.q[i] = int8_t(std::clamp(q, -127, 127));
    }
}

static inline void dequantize_block_q8(const Q8Block& blk, float* out) {
    const float d = blk.d;
    for (size_t i = 0; i < kQ8BlockSize; ++i) {
        out[i] = d * float(blk.q[i]);
    }
}

void q8_quantize(const float* x, Q8Block* out, size_t n) {
    TCPP_CHECK(n % kQ8BlockSize == 0, "q8: n must be multiple of 32");
    const size_t nb = n / kQ8BlockSize;
    for (size_t b = 0; b < nb; ++b) quantize_block_q8(x + b * kQ8BlockSize, out[b]);
}

void q8_dequantize(const Q8Block* blocks, float* out, size_t n) {
    TCPP_CHECK(n % kQ8BlockSize == 0, "q8: n must be multiple of 32");
    const size_t nb = n / kQ8BlockSize;
    for (size_t b = 0; b < nb; ++b) dequantize_block_q8(blocks[b], out + b * kQ8BlockSize);
}

void q8_quantize_matrix(const float* W_f32, Q8Block* out, size_t N, size_t K) {
    TCPP_CHECK(K % kQ8BlockSize == 0, "q8: K must be multiple of 32");
    const size_t bpr = K / kQ8BlockSize;
    for (size_t n = 0; n < N; ++n) q8_quantize(W_f32 + n * K, out + n * bpr, K);
}

// AVX2 dot of one Q8 block with 32 fp32 activations. The cvt path is
// int8 → int32 → fp32 in two halves.
#if defined(__AVX2__)
static inline float q8_dot_block(const Q8Block& blk, const float* x) {
    __m128i v_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk.q));
    __m128i v_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk.q + 16));
    __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo));
    __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8)));
    __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi));
    __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8)));

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

    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    __m128 ss = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, ss);
    ss = _mm_add_ss(ss, sh);
    return blk.d * _mm_cvtss_f32(ss);
}
#else
static inline float q8_dot_block(const Q8Block& blk, const float* x) {
    float s = 0.0f;
    for (size_t i = 0; i < kQ8BlockSize; ++i) s += float(blk.q[i]) * x[i];
    return blk.d * s;
}
#endif

void matmul_q8(const float* A, const Q8Block* W_q, float* C,
               size_t M, size_t N, size_t K) {
    TCPP_CHECK(K % kQ8BlockSize == 0, "matmul_q8: K must be multiple of 32");
    const size_t bpr = K / kQ8BlockSize;
    for (size_t m = 0; m < M; ++m) {
        const float* a_row = A + m * K;
        for (size_t n = 0; n < N; ++n) {
            const Q8Block* w_row = W_q + n * bpr;
            float acc = 0.0f;
            for (size_t b = 0; b < bpr; ++b) {
                acc += q8_dot_block(w_row[b], a_row + b * kQ8BlockSize);
            }
            C[m * N + n] = acc;
        }
    }
}

} // namespace turbocpp
