#pragma once
#include <cstddef>
#include <functional>

namespace turbocpp {

// Parallel matmul: splits the [0, N) output-row range across threads.
// Each thread computes a contiguous slice of C. No synchronization inside
// the slice — disjoint writes.
//
// Threshold: for N * K below ~1e5 the thread-pool overhead exceeds the
// compute savings. The function dispatches to the serial kernel below that.
void matmul_parallel(const float* A, const float* B, float* C,
                     size_t M, size_t N, size_t K);

// Parallelize over attention heads. Called by transformer once per layer.
// `fn(h)` computes one head's attention slice.
void parallel_heads(size_t n_heads, const std::function<void(size_t)>& fn);

} // namespace turbocpp
