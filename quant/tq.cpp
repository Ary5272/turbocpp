#include "tq.h"
#include "hadamard.h"
#include "q4.h"
#include "../core/allocator.h"
#include "../utils/logging.h"
#include <cstring>
#include <vector>

namespace turbocpp {

void tq_quantize_matrix(const float* W_f32, Q4Block* out, size_t N, size_t K,
                        size_t tq_block) {
    TCPP_CHECK(K % tq_block == 0, "tq: K must be multiple of tq_block");
    TCPP_CHECK(tq_block % kQ4BlockSize == 0,
               "tq: tq_block must be multiple of Q4 block (32)");
    const size_t bpr = K / kQ4BlockSize;

    std::vector<float> row_buf(K);
    for (size_t n = 0; n < N; ++n) {
        std::memcpy(row_buf.data(), W_f32 + n * K, K * sizeof(float));
        hadamard_block_inplace(row_buf.data(), K, tq_block);
        q4_quantize(row_buf.data(), out + n * bpr, K);
    }
}

void matmul_tq(const float* A, const Q4Block* W_tq, float* C,
               size_t M, size_t N, size_t K, size_t tq_block) {
    TCPP_CHECK(K % tq_block == 0, "matmul_tq: K must be multiple of tq_block");
    const size_t bpr = K / kQ4BlockSize;

    // Per-call scratch for the rotated A row (one row at a time → small).
    AlignedBuffer<float> a_rot(K);

    for (size_t m = 0; m < M; ++m) {
        // Rotate A's K-dim into scratch.
        std::memcpy(a_rot.data(), A + m * K, K * sizeof(float));
        hadamard_block_inplace(a_rot.data(), K, tq_block);
        // Standard q4_dot per row of W (already in rotated space).
        for (size_t n = 0; n < N; ++n) {
            const Q4Block* w_row = W_tq + n * bpr;
            float acc = 0.0f;
            for (size_t b = 0; b < bpr; ++b) {
                // Same kernel as matmul_q4. Inlined into a local var.
                const Q4Block& blk = w_row[b];
                const float* xb = a_rot.data() + b * kQ4BlockSize;
                float s = 0.0f;
                for (size_t i = 0; i < 16; ++i) {
                    const uint8_t packed = blk.q[i];
                    const int q0 = int(packed & 0x0F) - 8;
                    const int q1 = int(packed >> 4)   - 8;
                    s += float(q0) * xb[2 * i] + float(q1) * xb[2 * i + 1];
                }
                acc += blk.d * s;
            }
            C[m * N + n] = acc;
        }
    }
}

} // namespace turbocpp
