#pragma once
#include <cstddef>
#include "../core/allocator.h"
#include "../kv_cache/kv_cache.h"
#include "rope.h"

namespace turbocpp {

// Multi-head / Grouped-query attention with KV cache.
//
// Q has n_heads heads, K/V have n_kv_heads heads (≤ n_heads). Each query
// head h maps to KV head (h / heads_per_kv). When n_kv_heads == n_heads
// this degenerates to classic MHA. n_kv_heads == 1 is multi-query.
//
// Workspace owned here → no malloc in forward().
class Attention {
public:
    Attention() = default;

    void init(size_t n_heads, size_t n_kv_heads, size_t head_dim);
    void set_weights(const float* Wq, const float* Wk,
                     const float* Wv, const float* Wo);

    // Per-call modifiers. Set before forward().
    void set_sliding_window(size_t w) { sliding_window_ = w; }
    void set_alibi(bool b) { use_alibi_ = b; }

    void forward(const float* x_in, float* x_out, size_t pos, size_t layer,
                 KVCache& cache, const RopeTables& rope);

    size_t hidden() const noexcept { return n_heads_    * head_dim_; }
    size_t kv_dim() const noexcept { return n_kv_heads_ * head_dim_; }

private:
    size_t n_heads_    = 0;
    size_t n_kv_heads_ = 0;
    size_t head_dim_   = 0;

    const float* Wq_ = nullptr;
    const float* Wk_ = nullptr;
    const float* Wv_ = nullptr;
    const float* Wo_ = nullptr;

    AlignedBuffer<float> q_;
    AlignedBuffer<float> k_proj_;
    AlignedBuffer<float> v_proj_;
    AlignedBuffer<float> scores_;
    AlignedBuffer<float> attn_out_;
    size_t scores_cap_ = 0;
    size_t sliding_window_ = 0;
    bool   use_alibi_ = false;
};

} // namespace turbocpp
