#include "rope.h"
#include "../utils/logging.h"
#include <cmath>

namespace turbocpp {

void RopeTables::build(size_t head_dim, size_t max_seq_len, float base) {
    TCPP_CHECK(head_dim % 2 == 0, "rope: head_dim must be even, got %zu", head_dim);
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;

    const size_t half = head_dim / 2;
    cos_.resize(max_seq_len * half);
    sin_.resize(max_seq_len * half);

    // theta_i = base^(-2i/D). At position p, angle = p * theta_i.
    // Precomputing cos/sin avoids expensive trig in the hot path.
    for (size_t p = 0; p < max_seq_len; ++p) {
        for (size_t i = 0; i < half; ++i) {
            const float theta = std::pow(base, -float(2 * i) / float(head_dim));
            const float angle = float(p) * theta;
            cos_.data()[p * half + i] = std::cos(angle);
            sin_.data()[p * half + i] = std::sin(angle);
        }
    }
}

void RopeTables::apply(float* x, size_t n_heads, size_t pos) const {
    TCPP_CHECK(pos < max_seq_len_, "rope: pos %zu exceeds max_seq_len %zu", pos, max_seq_len_);
    const size_t half = head_dim_ / 2;
    const float* cos_row = cos_.data() + pos * half;
    const float* sin_row = sin_.data() + pos * half;

    // NeoX/LLaMA style: rotate pairs (x[i], x[i + D/2]) for i in [0, D/2).
    for (size_t h = 0; h < n_heads; ++h) {
        float* head = x + h * head_dim_;
        for (size_t i = 0; i < half; ++i) {
            const float c = cos_row[i];
            const float s = sin_row[i];
            const float x0 = head[i];
            const float x1 = head[i + half];
            head[i]        = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}

} // namespace turbocpp
