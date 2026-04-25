#include "vec_ops.h"
#include <cmath>
#include <cstddef>
#include <cstring>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace turbocpp {

void vec_add_inplace(float* y, const float* x, size_t n) {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(y + i);
        __m256 b = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, _mm256_add_ps(a, b));
    }
    for (; i < n; ++i) y[i] += x[i];
#else
    for (size_t i = 0; i < n; ++i) y[i] += x[i];
#endif
}

void vec_mul(float* y, const float* x, const float* v, size_t n) {
#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(x + i);
        __m256 b = _mm256_loadu_ps(v + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(a, b));
    }
    for (; i < n; ++i) y[i] = x[i] * v[i];
#else
    for (size_t i = 0; i < n; ++i) y[i] = x[i] * v[i];
#endif
}

void vec_scale(float* y, float s, size_t n) {
#if defined(__AVX2__)
    __m256 vs = _mm256_set1_ps(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(a, vs));
    }
    for (; i < n; ++i) y[i] *= s;
#else
    for (size_t i = 0; i < n; ++i) y[i] *= s;
#endif
}

float vec_dot(const float* x, const float* y, size_t n) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(x + i);
        __m256 b = _mm256_loadu_ps(y + i);
#  if defined(__FMA__)
        acc = _mm256_fmadd_ps(a, b, acc);
#  else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(a, b));
#  endif
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    __m128 ss = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, ss);
    ss = _mm_add_ss(ss, sh);
    float sum = _mm_cvtss_f32(ss);
    for (; i < n; ++i) sum += x[i] * y[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += x[i] * y[i];
    return sum;
#endif
}

// Numerically stable softmax. Without the max subtraction we'd overflow to
// +inf whenever a logit exceeds ~88 (log of FLT_MAX). The max pass costs
// O(n) extra memory reads but prevents NaN.
void softmax_inplace(float* x, size_t n) {
    if (n == 0) return;

    // Pass 1: max
    float maxv = x[0];
#if defined(__AVX2__)
    {
        __m256 vmax = _mm256_set1_ps(x[0]);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            vmax = _mm256_max_ps(vmax, v);
        }
        float buf[8] __attribute__((aligned(32)));
        _mm256_store_ps(buf, vmax);
        for (int j = 0; j < 8; ++j) if (buf[j] > maxv) maxv = buf[j];
        for (; i < n; ++i) if (x[i] > maxv) maxv = x[i];
    }
#else
    for (size_t i = 1; i < n; ++i) if (x[i] > maxv) maxv = x[i];
#endif

    // Pass 2: exp(x - max), sum. We use scalar expf because vectorized exp
    // requires a custom polynomial — libm's expf is SIMD-optimized on modern
    // glibc/MSVCR and good enough for softmax (which is typically small N).
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float e = std::exp(x[i] - maxv);
        x[i] = e;
        sum += e;
    }

    // Pass 3: normalize.
    const float inv = 1.0f / sum;
    vec_scale(x, inv, n);
}

size_t argmax(const float* x, size_t n) {
    if (n == 0) return 0;
    size_t best = 0;
    float bv = x[0];
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > bv) { bv = x[i]; best = i; }
    }
    return best;
}

float mean_sq(const float* x, size_t n) {
    if (n == 0) return 0.0f;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
#  if defined(__FMA__)
        acc = _mm256_fmadd_ps(v, v, acc);
#  else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
#  endif
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    __m128 ss = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, ss);
    ss = _mm_add_ss(ss, sh);
    float sum = _mm_cvtss_f32(ss);
    for (; i < n; ++i) sum += x[i] * x[i];
    return sum / float(n);
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += x[i] * x[i];
    return sum / float(n);
#endif
}

} // namespace turbocpp
