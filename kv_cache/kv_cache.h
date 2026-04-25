#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "../core/allocator.h"

namespace turbocpp {

// Per-layer KV store with layout [n_heads, max_seq, head_dim].
//
// Layout choice:
//   - Outer dim n_heads → K[h, :, :] is contiguous [seq_len, head_dim],
//     making the attention matvec (Q[h] · K[h, t]) sequentially address K.
//   - Append cost is O(n_heads) memcpys of head_dim floats per layer per
//     token — negligible vs the O(seq_len × head_dim) attention compute.
//
// Pre-allocated to max_seq_len once at construction. `append_len_` tracks
// how many positions are live. No reallocation during generation.
class KVCache {
public:
    KVCache() = default;

    // n_layers × 2 (K, V) × n_heads × max_seq_len × head_dim floats.
    // For LLaMA-7B: 32 × 2 × 32 × 2048 × 128 × 4B ≈ 2 GB. Real models use
    // 4-bit KV quant (see quant/kv_quant.h) to cut this ~8×.
    void init(size_t n_layers, size_t n_heads, size_t head_dim, size_t max_seq_len);

    // Copy projected K/V for ONE token at position `pos` into cache.
    // proj_* are [n_heads * head_dim] contiguous, as produced by the linear
    // projections (one row of the K/V projection output).
    void append(size_t layer, size_t pos, const float* k_proj, const float* v_proj);

    // Get pointer to K[layer, head, 0, 0] — i.e. the start of the contiguous
    // [max_seq_len, head_dim] tile for one (layer, head). Walk it as
    // [seq_len, head_dim] during attention.
    inline float* k_head(size_t layer, size_t head) {
        return k_.data() + k_offset(layer, head);
    }
    inline const float* k_head(size_t layer, size_t head) const {
        return k_.data() + k_offset(layer, head);
    }
    inline float* v_head(size_t layer, size_t head) {
        return v_.data() + k_offset(layer, head);
    }
    inline const float* v_head(size_t layer, size_t head) const {
        return v_.data() + k_offset(layer, head);
    }

    // Reset cache (start of new sequence). Does not free memory.
    void clear() noexcept { cur_len_ = 0; }

    size_t cur_len()      const noexcept { return cur_len_; }
    void   set_cur_len(size_t n) noexcept { cur_len_ = n; }

    size_t n_layers()     const noexcept { return n_layers_; }
    size_t n_heads()      const noexcept { return n_heads_; }
    size_t head_dim()     const noexcept { return head_dim_; }
    size_t max_seq_len()  const noexcept { return max_seq_len_; }

    // Bytes of live data (K + V). Useful for memory reporting.
    size_t bytes_used() const noexcept {
        return 2 * n_layers_ * n_heads_ * cur_len_ * head_dim_ * sizeof(float);
    }
    size_t bytes_reserved() const noexcept {
        return 2 * n_layers_ * n_heads_ * max_seq_len_ * head_dim_ * sizeof(float);
    }

    // Snapshot the live portion (K, V up to cur_len) to disk along with
    // shape metadata + a token-prompt hash for re-validation. Returns true
    // on success. Meant for prompt-cache use (re-running the same long
    // prompt across invocations skips prefill entirely).
    bool save_snapshot(const std::string& path, uint64_t prompt_hash) const;
    // Returns true if the snapshot's shape and prompt_hash match. On match,
    // the cache is restored to the saved state and cur_len is set.
    bool load_snapshot(const std::string& path, uint64_t prompt_hash);

private:
    inline size_t k_offset(size_t layer, size_t head) const {
        // K storage shape: [n_layers, n_heads, max_seq_len, head_dim]
        return ((layer * n_heads_ + head) * max_seq_len_) * head_dim_;
    }

    AlignedBuffer<float> k_;
    AlignedBuffer<float> v_;
    size_t n_layers_    = 0;
    size_t n_heads_     = 0;
    size_t head_dim_    = 0;
    size_t max_seq_len_ = 0;
    size_t cur_len_     = 0;
};

} // namespace turbocpp
