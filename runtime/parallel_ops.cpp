#include "parallel_ops.h"
#include "thread_pool.h"
#include "../math/matmul.h"

namespace turbocpp {

// Heuristic threshold. Parallel path has ~10-30us overhead per call on a
// typical 8-core x86 box (CV signal + cache line bouncing); below ~130k
// multiply-adds total, serial wins.
static constexpr size_t kParallelMinWork = 1 << 17;

void matmul_parallel(const float* A, const float* B, float* C,
                     size_t M, size_t N, size_t K) {
    const size_t work = M * N * K;
    if (work < kParallelMinWork) {
        matmul(A, B, C, M, N, K);
        return;
    }

    // Split over N (output rows of W / cols of C). Each worker writes a
    // disjoint slice of C — no synchronization required.
    ThreadPool& pool = global_pool();
    pool.parallel_for(N, [&](size_t n0, size_t n1) {
        const size_t n_slice = n1 - n0;
        for (size_t m = 0; m < M; ++m) {
            float* c_row = C + m * N + n0;
            matmul(A + m * K, B + n0 * K, c_row, 1, n_slice, K);
        }
    });
}

void parallel_heads(size_t n_heads, const std::function<void(size_t)>& fn) {
    ThreadPool& pool = global_pool();
    if (n_heads <= 1) { for (size_t h = 0; h < n_heads; ++h) fn(h); return; }

    pool.parallel_for(n_heads, [&](size_t h0, size_t h1) {
        for (size_t h = h0; h < h1; ++h) fn(h);
    });
}

} // namespace turbocpp
