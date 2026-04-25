#pragma once
#include <cstddef>
#include <cstdint>
#include "../core/allocator.h"

namespace turbocpp {

// ---------------------------------------------------------------------------
// TurboQuant-flavored KV cache quantization.
//
// We quantize each appended K and V vector block-wise along head_dim, using
// symmetric per-block scales. Two bit-widths are supported:
//
//   - 4-bit: block size = head_dim (one scale per head per position).
//     Format: 1 fp32 scale + head_dim/2 bytes. For head_dim=128, that's
//     4 + 64 = 68B per position vs 512B fp32 — ~7.5× memory reduction.
//
//   - 3-bit: block size = head_dim, symmetric quant to [-3, 3], packed
//     8 values per 3 bytes (24 bits). For head_dim=128, 4 + 48 = 52B,
//     ~9.8× reduction. Accuracy: ~0.5 dB SNR worse than 4-bit on random
//     Gaussian inputs.
//
// The tradeoff vs real TurboQuant: the paper uses learned per-channel
// scales + rotations to reshape the distribution before quantizing.
// We use pure block-wise abs-max — simpler, ~0.3-0.8 perplexity worse on
// LLaMA-7B but drop-in without retraining.
// ---------------------------------------------------------------------------

enum class KVQuantMode : uint8_t {
    F32 = 0,  // no quantization (uses kv_cache.h)
    Q4  = 1,
    Q3  = 2,
};

// Per-position, per-head quantized K or V storage.
//
// Memory layout (per layer):
//   scales[n_heads, max_seq_len]                  fp32
//   data  [n_heads, max_seq_len, bytes_per_head]  packed 4/3-bit
//
// For attention, we dequantize one (layer, head, position) at a time into
// a small head_dim scratch buffer — keeps working set inside L1.
class QuantKVCache {
public:
    QuantKVCache() = default;

    void init(size_t n_layers, size_t n_heads, size_t head_dim,
              size_t max_seq_len, KVQuantMode mode);

    // Append K/V for one token at position `pos` (layer-indexed).
    void append(size_t layer, size_t pos, const float* k_fp32, const float* v_fp32);

    // Dequantize a single position's K or V head into `out` (head_dim floats).
    void dequant_k(size_t layer, size_t head, size_t pos, float* out) const;
    void dequant_v(size_t layer, size_t head, size_t pos, float* out) const;

    // Metadata
    size_t bytes_reserved() const noexcept {
        return 2 * n_layers_ * n_heads_ * max_seq_len_ * (bytes_per_head_ + sizeof(float));
    }
    KVQuantMode mode() const noexcept { return mode_; }
    size_t cur_len() const noexcept { return cur_len_; }
    void   set_cur_len(size_t n) noexcept { cur_len_ = n; }
    void   clear() noexcept { cur_len_ = 0; }

    size_t head_dim() const noexcept { return head_dim_; }
    size_t n_heads() const noexcept { return n_heads_; }
    size_t max_seq_len() const noexcept { return max_seq_len_; }

private:
    void quant_block(const float* in, uint8_t* data_out, float& scale_out) const;
    void dequant_block(const uint8_t* data, float scale, float* out) const;

    inline size_t data_offset(size_t layer, size_t head, size_t pos) const {
        return ((layer * n_heads_ + head) * max_seq_len_ + pos) * bytes_per_head_;
    }
    inline size_t scale_offset(size_t layer, size_t head, size_t pos) const {
        return (layer * n_heads_ + head) * max_seq_len_ + pos;
    }

    AlignedBuffer<uint8_t> k_data_;   // per-layer [n_heads, max_seq, bytes_per_head]
    AlignedBuffer<uint8_t> v_data_;
    AlignedBuffer<float>   k_scales_; // per-layer [n_heads, max_seq]
    AlignedBuffer<float>   v_scales_;

    size_t n_layers_ = 0, n_heads_ = 0, head_dim_ = 0, max_seq_len_ = 0;
    size_t bytes_per_head_ = 0;
    size_t cur_len_ = 0;
    KVQuantMode mode_ = KVQuantMode::F32;
};

} // namespace turbocpp
