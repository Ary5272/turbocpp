#include "matmul.h"
#include "../core/alignment.h"
#include <algorithm>
#include <cstring>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace turbocpp {

// ---------------------------------------------------------------------------
// Tier 1: naive
// ---------------------------------------------------------------------------
//
// Access pattern: A is walked row-by-row (good), B is walked row-by-row
// (good — we indexed B[n, k], so the K loop is contiguous in B). C is written
// once at the end of the K loop. Cache failure comes from the size of B —
// inner loop over k pulls in the full row of B for each (m, n) pair. For
// K=4096 that's 16KB, which fits L1, but each change of n invalidates lines.

void matmul_naive(const float* A, const float* B, float* C,
                  size_t M, size_t N, size_t K) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            const float* a_row = A + m * K;
            const float* b_row = B + n * K;
            for (size_t k = 0; k < K; ++k) {
                acc += a_row[k] * b_row[k];
            }
            C[m * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 2: cache-blocked
// ---------------------------------------------------------------------------
//
// Tile sizes from core/alignment.h: MC=64, NC=128, KC=256. Working set per
// (mc, nc, kc) tile: A-tile MC*KC*4 = 64KB; B-tile NC*KC*4 = 128KB; C-tile
// MC*NC*4 = 32KB. A + C fit in L1 (64KB total on many x86); B sits in L2.

void matmul_blocked(const float* A, const float* B, float* C,
                    size_t M, size_t N, size_t K) {
    // Zero C up-front; inner loop accumulates.
    std::memset(C, 0, M * N * sizeof(float));

    for (size_t m0 = 0; m0 < M; m0 += kMC) {
        const size_t m_end = std::min(m0 + kMC, M);
        for (size_t n0 = 0; n0 < N; n0 += kNC) {
            const size_t n_end = std::min(n0 + kNC, N);
            for (size_t k0 = 0; k0 < K; k0 += kKC) {
                const size_t k_end = std::min(k0 + kKC, K);

                // Micro-kernel: accumulate over the current k panel.
                for (size_t m = m0; m < m_end; ++m) {
                    for (size_t n = n0; n < n_end; ++n) {
                        float acc = C[m * N + n];
                        const float* a_row = A + m * K;
                        const float* b_row = B + n * K;
                        for (size_t k = k0; k < k_end; ++k) {
                            acc += a_row[k] * b_row[k];
                        }
                        C[m * N + n] = acc;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 3: AVX2 + FMA
// ---------------------------------------------------------------------------

#if defined(__AVX2__)

// Horizontal sum of an __m256 float vector. _mm256_hadd_ps is slow on most
// uarchs; we use the 128-bit-folding trick which is ~3 cycles.
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);       // (1,1,3,3)
    __m128 ss = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, ss);            // high half of ss
    ss = _mm_add_ss(ss, sh);
    return _mm_cvtss_f32(ss);
}

// FMA is almost always available alongside AVX2 on real CPUs, but guard
// for safety. Without FMA we fall back to mul+add which the compiler may
// still fuse.
#if defined(__FMA__)
#  define TCPP_FMADD(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#  define TCPP_FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif

// Core micro-kernel: compute C[m, n0..n0+3] for one m, accumulating over
// the full K dimension. Four N accumulators share each A load — this is the
// key reuse that pulls throughput past 1 FMA/cycle.
//
// On AVX-512 hardware (Skylake-X, Ice Lake, Zen 4, Sapphire Rapids), we
// process 16 floats per FMA instead of 8 — ~2× throughput on those CPUs.
static inline void microkernel_1x4(const float* a_row, const float* B,
                                   float* c_row, size_t n0, size_t K) {
    const float* b0 = B + (n0 + 0) * K;
    const float* b1 = B + (n0 + 1) * K;
    const float* b2 = B + (n0 + 2) * K;
    const float* b3 = B + (n0 + 3) * K;

    size_t k = 0;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;

#if defined(__AVX512F__)
    // 16-wide pass first.
    __m512 a512_0 = _mm512_setzero_ps();
    __m512 a512_1 = _mm512_setzero_ps();
    __m512 a512_2 = _mm512_setzero_ps();
    __m512 a512_3 = _mm512_setzero_ps();
    for (; k + 16 <= K; k += 16) {
        __m512 a = _mm512_loadu_ps(a_row + k);
        a512_0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(b0 + k), a512_0);
        a512_1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(b1 + k), a512_1);
        a512_2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(b2 + k), a512_2);
        a512_3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(b3 + k), a512_3);
    }
    s0 = _mm512_reduce_add_ps(a512_0);
    s1 = _mm512_reduce_add_ps(a512_1);
    s2 = _mm512_reduce_add_ps(a512_2);
    s3 = _mm512_reduce_add_ps(a512_3);
#endif

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; k + 8 <= K; k += 8) {
        __m256 a = _mm256_loadu_ps(a_row + k);
        __m256 v0 = _mm256_loadu_ps(b0 + k);
        __m256 v1 = _mm256_loadu_ps(b1 + k);
        __m256 v2 = _mm256_loadu_ps(b2 + k);
        __m256 v3 = _mm256_loadu_ps(b3 + k);
        acc0 = TCPP_FMADD(a, v0, acc0);
        acc1 = TCPP_FMADD(a, v1, acc1);
        acc2 = TCPP_FMADD(a, v2, acc2);
        acc3 = TCPP_FMADD(a, v3, acc3);
    }
    s0 += hsum256(acc0);
    s1 += hsum256(acc1);
    s2 += hsum256(acc2);
    s3 += hsum256(acc3);

    // Scalar tail for K not divisible by 8.
    for (; k < K; ++k) {
        const float av = a_row[k];
        s0 += av * b0[k];
        s1 += av * b1[k];
        s2 += av * b2[k];
        s3 += av * b3[k];
    }

    c_row[n0 + 0] = s0;
    c_row[n0 + 1] = s1;
    c_row[n0 + 2] = s2;
    c_row[n0 + 3] = s3;
}

static inline void microkernel_1x1(const float* a_row, const float* b_row,
                                   float& out, size_t K) {
    __m256 acc = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 a = _mm256_loadu_ps(a_row + k);
        __m256 b = _mm256_loadu_ps(b_row + k);
        acc = TCPP_FMADD(a, b, acc);
    }
    float s = hsum256(acc);
    for (; k < K; ++k) s += a_row[k] * b_row[k];
    out = s;
}

void matmul_avx2(const float* A, const float* B, float* C,
                 size_t M, size_t N, size_t K) {
    // Fall back for degenerate K.
    if (K < 8) { matmul_naive(A, B, C, M, N, K); return; }

    const size_t N4 = (N / 4) * 4;

    for (size_t m = 0; m < M; ++m) {
        const float* a_row = A + m * K;
        float* c_row = C + m * N;

        size_t n = 0;
        for (; n < N4; n += 4) {
            microkernel_1x4(a_row, B, c_row, n, K);
        }
        // Tail for N not divisible by 4.
        for (; n < N; ++n) {
            microkernel_1x1(a_row, B + n * K, c_row[n], K);
        }
    }
}

#else  // no AVX2

void matmul_avx2(const float* A, const float* B, float* C,
                 size_t M, size_t N, size_t K) {
    matmul_blocked(A, B, C, M, N, K);
}

#endif

void matmul(const float* A, const float* B, float* C,
            size_t M, size_t N, size_t K) {
#if defined(__AVX2__)
    matmul_avx2(A, B, C, M, N, K);
#else
    matmul_blocked(A, B, C, M, N, K);
#endif
}

// ---------------------------------------------------------------------------
// Bias add: C[m, n] += b[n] for all m, n.
// ---------------------------------------------------------------------------

void add_bias(float* C, const float* b, size_t M, size_t N) {
#if defined(__AVX2__)
    const size_t N8 = (N / 8) * 8;
    for (size_t m = 0; m < M; ++m) {
        float* row = C + m * N;
        size_t n = 0;
        for (; n < N8; n += 8) {
            __m256 c = _mm256_loadu_ps(row + n);
            __m256 v = _mm256_loadu_ps(b + n);
            _mm256_storeu_ps(row + n, _mm256_add_ps(c, v));
        }
        for (; n < N; ++n) row[n] += b[n];
    }
#else
    for (size_t m = 0; m < M; ++m) {
        float* row = C + m * N;
        for (size_t n = 0; n < N; ++n) row[n] += b[n];
    }
#endif
}

} // namespace turbocpp
