#include "kv_cache.h"
#include "../utils/logging.h"
#include <cstring>

namespace turbocpp {

void KVCache::init(size_t n_layers, size_t n_heads, size_t head_dim, size_t max_seq_len) {
    n_layers_ = n_layers;
    n_heads_ = n_heads;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    cur_len_ = 0;

    const size_t total = n_layers * n_heads * max_seq_len * head_dim;
    k_.resize(total);
    v_.resize(total);

    // Zero-init is important for determinism in the tail — the attention
    // loop reads only up to cur_len_, but correct initial state matters if
    // we ever run with `cur_len_ > 0` without a prior append (unusual).
    std::memset(k_.data(), 0, total * sizeof(float));
    std::memset(v_.data(), 0, total * sizeof(float));
}

void KVCache::append(size_t layer, size_t pos, const float* k_proj, const float* v_proj) {
    TCPP_CHECK(pos < max_seq_len_, "kv_cache append pos %zu exceeds max %zu", pos, max_seq_len_);

    // k_proj is laid out [h0_d0..d_{D-1}, h1_d0..d_{D-1}, ...]. We need to
    // scatter each head's head_dim slice into its own [max_seq, head_dim]
    // tile at row `pos`.
    for (size_t h = 0; h < n_heads_; ++h) {
        float* kdst = k_head(layer, h) + pos * head_dim_;
        float* vdst = v_head(layer, h) + pos * head_dim_;
        std::memcpy(kdst, k_proj + h * head_dim_, head_dim_ * sizeof(float));
        std::memcpy(vdst, v_proj + h * head_dim_, head_dim_ * sizeof(float));
    }
}

} // namespace turbocpp
