// benchmark.cpp — microbenchmarks for the hot-path kernels.
//
// Run: ./turbocpp_bench [size]
//
// Reports GFLOPS for naive vs blocked vs AVX2 matmul at a given M=N=K size.
// Also measures RMSNorm, softmax, and Q4 dequant throughput.

#include "core/allocator.h"
#include "math/matmul.h"
#include "math/vec_ops.h"
#include "math/activations.h"
#include "model/rmsnorm.h"
#include "quant/q4.h"
#include "utils/timing.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace turbocpp;

static void fill_random(float* p, size_t n, uint64_t seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) p[i] = d(rng);
}

// Check that two matmul results agree to within `tol`. Returns max abs diff.
static float max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

struct Bench {
    const char* name;
    double ms;
    double gflops;
};

static Bench run_matmul(const char* name,
                        void (*fn)(const float*, const float*, float*,
                                   size_t, size_t, size_t),
                        const float* A, const float* B, float* C,
                        size_t M, size_t N, size_t K, int iters) {
    // Warmup
    fn(A, B, C, M, N, K);

    Timer t;
    for (int i = 0; i < iters; ++i) {
        fn(A, B, C, M, N, K);
    }
    double ms = t.ms() / iters;
    double gfs = matmul_gflops(M, N, K, ms / 1000.0);
    return {name, ms, gfs};
}

static void bench_matmul(size_t S) {
    std::printf("\n--- matmul (M=N=K=%zu) ---\n", S);
    AlignedBuffer<float> A(S * S), B(S * S), C_naive(S * S), C_blocked(S * S), C_avx2(S * S);
    fill_random(A.data(), S * S, 1);
    fill_random(B.data(), S * S, 2);

    const int iters = S >= 512 ? 3 : 20;

    // Naive only viable for small S.
    if (S <= 512) {
        auto r = run_matmul("naive  ", matmul_naive, A.data(), B.data(), C_naive.data(),
                            S, S, S, iters);
        std::printf("  %s  %8.2f ms   %7.2f GFLOPS\n", r.name, r.ms, r.gflops);
    }
    auto r2 = run_matmul("blocked", matmul_blocked, A.data(), B.data(), C_blocked.data(),
                         S, S, S, iters);
    std::printf("  %s  %8.2f ms   %7.2f GFLOPS\n", r2.name, r2.ms, r2.gflops);

    auto r3 = run_matmul("avx2   ", matmul_avx2, A.data(), B.data(), C_avx2.data(),
                         S, S, S, iters);
    std::printf("  %s  %8.2f ms   %7.2f GFLOPS\n", r3.name, r3.ms, r3.gflops);

    if (S <= 512) {
        // Sanity: check AVX2 vs naive agree.
        float diff = max_abs_diff(C_naive.data(), C_avx2.data(), S * S);
        std::printf("  naive↔avx2 max abs diff: %.3e\n", diff);
    }
}

static void bench_rmsnorm(size_t D) {
    std::printf("\n--- rmsnorm (dim=%zu) ---\n", D);
    AlignedBuffer<float> x(D), w(D), y(D);
    fill_random(x.data(), D, 1);
    for (size_t i = 0; i < D; ++i) w.data()[i] = 1.0f;
    const int iters = 10000;
    Timer t;
    for (int i = 0; i < iters; ++i) rmsnorm_row(x.data(), w.data(), y.data(), D, 1e-5f);
    double us = t.us() / iters;
    std::printf("  %.2f us/call  (%.1f M elem/s)\n", us, D / (us * 1e-6) / 1e6);
}

static void bench_softmax(size_t N) {
    std::printf("\n--- softmax (n=%zu) ---\n", N);
    AlignedBuffer<float> x(N);
    fill_random(x.data(), N, 1);
    const int iters = 2000;
    Timer t;
    for (int i = 0; i < iters; ++i) {
        // softmax modifies; re-fill every 50 iters to keep numerics sane.
        if (i % 50 == 0) fill_random(x.data(), N, uint64_t(i));
        softmax_inplace(x.data(), N);
    }
    double us = t.us() / iters;
    std::printf("  %.2f us/call  (%.1f M elem/s)\n", us, N / (us * 1e-6) / 1e6);
}

static void bench_q4(size_t N, size_t K) {
    std::printf("\n--- q4 matmul (M=1 N=%zu K=%zu) ---\n", N, K);
    const size_t blocks_per_row = K / kQ4BlockSize;
    AlignedBuffer<float> A(K);
    AlignedBuffer<float> W(N * K);
    AlignedBuffer<float> C_f32(N);
    AlignedBuffer<float> C_q4(N);
    std::vector<Q4Block> Wq(N * blocks_per_row);

    fill_random(A.data(), K, 1);
    fill_random(W.data(), N * K, 2);
    q4_quantize_matrix(W.data(), Wq.data(), N, K);

    // Dequantize error
    std::vector<float> dequant(N * K);
    for (size_t n = 0; n < N; ++n) {
        q4_dequantize(Wq.data() + n * blocks_per_row, dequant.data() + n * K, K);
    }
    float q_err = max_abs_diff(W.data(), dequant.data(), N * K);

    const int iters = 100;
    Timer t1;
    for (int i = 0; i < iters; ++i) matmul(A.data(), W.data(), C_f32.data(), 1, N, K);
    double ms_f32 = t1.ms() / iters;

    Timer t2;
    for (int i = 0; i < iters; ++i) matmul_q4(A.data(), Wq.data(), C_q4.data(), 1, N, K);
    double ms_q4 = t2.ms() / iters;

    float out_err = max_abs_diff(C_f32.data(), C_q4.data(), N);

    std::printf("  f32 matmul: %6.3f ms   q4 matmul: %6.3f ms  (%.2fx)\n",
           ms_f32, ms_q4, ms_f32 / ms_q4);
    std::printf("  weight quant abs err: %.3e   output abs err: %.3e\n",
           q_err, out_err);
    std::printf("  f32 bytes: %zu   q4 bytes: %zu   ratio: %.2fx\n",
           N * K * sizeof(float),
           Wq.size() * sizeof(Q4Block),
           double(N * K * sizeof(float)) / double(Wq.size() * sizeof(Q4Block)));
}

int main(int argc, char** argv) {
    size_t S = (argc > 1) ? size_t(std::atoi(argv[1])) : 512;
    std::printf("== TurboCPP benchmark ==\n");

    bench_matmul(128);
    bench_matmul(S);
    bench_rmsnorm(4096);
    bench_softmax(4096);
    bench_q4(2048, 2048);
    return 0;
}
