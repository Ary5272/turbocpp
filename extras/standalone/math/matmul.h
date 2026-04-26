#pragma once
#include <cstddef>

namespace turbocpp {

// Convention used across the whole project:
//   C[M, N] = A[M, K] @ B[N, K]^T
//
// i.e. B is stored ROW-MAJOR with rows indexing OUTPUT FEATURES (like a
// neural-network weight matrix: W[out, in]). This matches how weights come
// out of PyTorch/llama.cpp: W is [n_out, n_in] and y = x @ W^T.
//
// A, B are const; C is overwritten by the kernel.
// Pointers should be 32-byte aligned for peak throughput (no penalty on
// modern x86 for unaligned-but-still-on-cache-line loads, but we use
// _mm256_loadu_ps for safety so misalignment is tolerated).

// Tier 1: triple-loop reference. Used for correctness checks only — the
// unblocked access pattern blows L1/L2 for any non-trivial shape.
void matmul_naive(const float* A, const float* B, float* C,
                  size_t M, size_t N, size_t K);

// Tier 2: cache-blocked (MC×NC×KC). Same ISA as naive but partitions the
// work so the active working set fits in L1/L2. 3-5× faster than naive
// for medium shapes.
void matmul_blocked(const float* A, const float* B, float* C,
                    size_t M, size_t N, size_t K);

// Tier 3: AVX2 + FMA, blocked, N-unrolled by 4. Peak ~70-85% of theoretical
// AVX2 FMA throughput on modern x86. This is the hot-path kernel.
//
// Requires K % 8 == 0 is NOT required — tail handled scalar. But K ≥ 8
// is assumed; short inner dims fall back to naive.
void matmul_avx2(const float* A, const float* B, float* C,
                 size_t M, size_t N, size_t K);

// Dispatch wrapper. Picks the best kernel for the platform at call time.
// On AVX2+FMA builds this is matmul_avx2; otherwise matmul_blocked.
void matmul(const float* A, const float* B, float* C,
            size_t M, size_t N, size_t K);

// Add bias b[N] to every row of C[M, N]. SIMD where possible.
void add_bias(float* C, const float* b, size_t M, size_t N);

} // namespace turbocpp
