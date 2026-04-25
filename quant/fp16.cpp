#include "fp16.h"
#include "../utils/logging.h"
#include <cstring>

#if defined(__AVX2__) || defined(__F16C__) || defined(_MSC_VER)
#  include <immintrin.h>
#endif

namespace turbocpp {

// ---------------------------------------------------------------------------
// Scalar conversion (bit-hack fallback when F16C is missing)
// ---------------------------------------------------------------------------
fp16_t f32_to_f16_scalar(float f) {
#if defined(__F16C__)
    return _cvtss_sh(f, 0);
#else
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp_  = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp_ <= 0) {                       // subnormal / zero
        if (exp_ < -10) return uint16_t(sign);
        mant |= 0x800000;
        uint32_t shift = 14 - exp_;
        uint32_t round = (mant >> (shift - 1)) & 1;
        return uint16_t(sign | ((mant >> shift) + round));
    }
    if (exp_ >= 31) return uint16_t(sign | 0x7C00 | (mant ? 1 : 0));  // inf/nan
    return uint16_t(sign | (uint32_t(exp_) << 10) | (mant >> 13));
#endif
}

float f16_to_f32_scalar(fp16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    const uint32_t sign = (uint32_t(h) & 0x8000) << 16;
    uint32_t exp_  = (uint32_t(h) >> 10) & 0x1F;
    uint32_t mant  = uint32_t(h) & 0x3FF;
    uint32_t f;
    if (exp_ == 0) {
        if (mant == 0) f = sign;
        else { while (!(mant & 0x400)) { mant <<= 1; --exp_; }
               mant &= 0x3FF; exp_ += 1;
               f = sign | ((exp_ + 112) << 23) | (mant << 13); }
    } else if (exp_ == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp_ + 112) << 23) | (mant << 13);
    }
    float out; std::memcpy(&out, &f, 4); return out;
#endif
}

// ---------------------------------------------------------------------------
// Vectorized conversion
// ---------------------------------------------------------------------------
void fp16_to_fp32(const fp16_t* src, float* dst, size_t n) {
#if defined(__F16C__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f  = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; ++i) dst[i] = f16_to_f32_scalar(src[i]);
#else
    for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32_scalar(src[i]);
#endif
}

void fp32_to_fp16(const float* src, fp16_t* dst, size_t n) {
#if defined(__F16C__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256  f = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
    }
    for (; i < n; ++i) dst[i] = f32_to_f16_scalar(src[i]);
#else
    for (size_t i = 0; i < n; ++i) dst[i] = f32_to_f16_scalar(src[i]);
#endif
}

// ---------------------------------------------------------------------------
// fp16 × fp32 matmul. We process one weight row in slabs of 64 elements,
// converting to a stack-allocated fp32 scratch, then dotting with A. 64
// floats = 256B = 4 cache lines, fits L1 trivially.
// ---------------------------------------------------------------------------

void matmul_f16(const float* A, const fp16_t* W, float* C,
                size_t M, size_t N, size_t K) {
#if defined(__F16C__)
    constexpr size_t SLAB = 64;
    alignas(32) float wbuf[SLAB];
    for (size_t m = 0; m < M; ++m) {
        const float* a_row = A + m * K;
        for (size_t n = 0; n < N; ++n) {
            const fp16_t* w_row = W + n * K;
            __m256 acc = _mm256_setzero_ps();
            size_t k = 0;
            for (; k + SLAB <= K; k += SLAB) {
                fp16_to_fp32(w_row + k, wbuf, SLAB);
                for (size_t s = 0; s < SLAB; s += 8) {
                    __m256 a = _mm256_loadu_ps(a_row + k + s);
                    __m256 w = _mm256_load_ps(wbuf + s);
#  if defined(__FMA__)
                    acc = _mm256_fmadd_ps(a, w, acc);
#  else
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(a, w));
#  endif
                }
            }
            // tail (rare, K is usually a multiple of 64)
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s  = _mm_add_ps(lo, hi);
            __m128 sh = _mm_movehdup_ps(s);
            __m128 ss = _mm_add_ps(s, sh);
            sh = _mm_movehl_ps(sh, ss);
            ss = _mm_add_ss(ss, sh);
            float sum = _mm_cvtss_f32(ss);
            for (; k < K; ++k) sum += a_row[k] * f16_to_f32_scalar(w_row[k]);
            C[m * N + n] = sum;
        }
    }
#else
    // Fallback: scalar conversion + scalar dot
    for (size_t m = 0; m < M; ++m) {
        const float* a_row = A + m * K;
        for (size_t n = 0; n < N; ++n) {
            const fp16_t* w_row = W + n * K;
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) sum += a_row[k] * f16_to_f32_scalar(w_row[k]);
            C[m * N + n] = sum;
        }
    }
#endif
}

} // namespace turbocpp
