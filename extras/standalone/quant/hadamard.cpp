#include "hadamard.h"
#include "../utils/logging.h"
#include <cmath>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace turbocpp {

// One in-place butterfly pass at stride `s`: pairs (x[i], x[i+s]) → (sum, diff).
// We process 8 floats at a time with AVX2 when s ≥ 8 (no cross-lane work).
static inline void butterfly_pass(float* x, size_t n, size_t s) {
    for (size_t base = 0; base < n; base += 2 * s) {
#if defined(__AVX2__)
        if (s >= 8) {
            size_t k = 0;
            for (; k + 8 <= s; k += 8) {
                __m256 a = _mm256_loadu_ps(x + base + k);
                __m256 b = _mm256_loadu_ps(x + base + k + s);
                _mm256_storeu_ps(x + base + k,     _mm256_add_ps(a, b));
                _mm256_storeu_ps(x + base + k + s, _mm256_sub_ps(a, b));
            }
            for (; k < s; ++k) {
                float a = x[base + k];
                float b = x[base + k + s];
                x[base + k]     = a + b;
                x[base + k + s] = a - b;
            }
            continue;
        }
#endif
        for (size_t k = 0; k < s; ++k) {
            float a = x[base + k];
            float b = x[base + k + s];
            x[base + k]     = a + b;
            x[base + k + s] = a - b;
        }
    }
}

void hadamard_inplace(float* x, size_t n) {
    TCPP_CHECK((n & (n - 1)) == 0 && n > 0,
               "hadamard: n must be a power of 2, got %zu", n);
    // log2(n) butterfly passes with stride 1, 2, 4, ..., n/2.
    for (size_t s = 1; s < n; s *= 2) {
        butterfly_pass(x, n, s);
    }
    // Normalize so H · H = I (rather than nI).
    const float inv = 1.0f / std::sqrt(float(n));
#if defined(__AVX2__)
    __m256 v = _mm256_set1_ps(inv);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(a, v));
    }
    for (; i < n; ++i) x[i] *= inv;
#else
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
#endif
}

void hadamard_block_inplace(float* x, size_t n, size_t block_size) {
    TCPP_CHECK(n % block_size == 0,
               "hadamard_block: n (%zu) must be multiple of block_size (%zu)",
               n, block_size);
    TCPP_CHECK((block_size & (block_size - 1)) == 0,
               "hadamard_block: block_size (%zu) must be power of 2",
               block_size);
    const size_t nb = n / block_size;
    for (size_t b = 0; b < nb; ++b) {
        hadamard_inplace(x + b * block_size, block_size);
    }
}

void hadamard_rows_inplace(float* W, size_t N, size_t K, size_t block_size) {
    for (size_t n = 0; n < N; ++n) {
        hadamard_block_inplace(W + n * K, K, block_size);
    }
}

} // namespace turbocpp
