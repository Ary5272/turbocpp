#pragma once
#include <cstddef>
#include <cstdint>
#include "../core/allocator.h"
#include "../kv_cache/kv_cache.h"
#include "attention.h"
#include "rope.h"

namespace turbocpp {

// ---------------------------------------------------------------------------
// Model config (LLaMA-style). LLaMA-2-70B / LLaMA-3 use n_kv_heads<n_heads
// for grouped-query attention; n_kv_heads=1 is multi-query. Default keeps
// classic MHA (n_kv_heads = n_heads).
// ---------------------------------------------------------------------------
struct ModelConfig {
    size_t vocab_size   = 32000;
    size_t hidden_dim   = 256;    // = n_heads * head_dim
    size_t n_layers     = 4;
    size_t n_heads      = 8;      // query heads
    size_t n_kv_heads   = 8;      // KV heads (GQA: < n_heads, MQA: = 1)
    size_t head_dim     = 32;
    size_t ffn_dim      = 512;
    size_t max_seq_len  = 512;
    float  rms_eps      = 1e-5f;
    float  rope_base    = 10000.0f;

    // Derived
    size_t kv_dim()       const noexcept { return n_kv_heads * head_dim; }
    size_t heads_per_kv() const noexcept { return n_heads / n_kv_heads; }
};

// Packed per-layer weights. Pointers non-owning. With GQA, Wk/Wv project
// to kv_dim (= n_kv_heads*head_dim) instead of hidden_dim.
struct LayerWeights {
    const float* attn_norm = nullptr; // [hidden]
    const float* Wq = nullptr;        // [hidden,  hidden]
    const float* Wk = nullptr;        // [kv_dim,  hidden]
    const float* Wv = nullptr;        // [kv_dim,  hidden]
    const float* Wo = nullptr;        // [hidden,  hidden]
    const float* ffn_norm = nullptr;  // [hidden]
    const float* Wgate = nullptr;     // [ffn_dim, hidden]
    const float* Wup   = nullptr;     // [ffn_dim, hidden]
    const float* Wdown = nullptr;     // [hidden,  ffn_dim]
};

struct ModelWeights {
    const float* tok_embed   = nullptr;
    const float* final_norm  = nullptr;
    const float* lm_head     = nullptr;
    AlignedBuffer<LayerWeights> layers;
};

class Model {
public:
    Model() = default;

    void init(const ModelConfig& cfg);

    // Single-token forward. Updates KV cache, writes [vocab_size] logits.
    void forward(int32_t token_id, size_t pos, KVCache& cache,
                 float* logits);

    const ModelConfig& config() const noexcept { return cfg_; }
    ModelWeights& weights() noexcept { return weights_; }
    const ModelWeights& weights() const noexcept { return weights_; }

    // Embedding mode: returns the final-norm hidden state of the most
    // recent forward() (size hidden_dim). Use this to extract sentence
    // embeddings without paying the LM-head matmul. Pool across tokens
    // externally (mean / last-token / CLS).
    const float* last_hidden() const noexcept { return xb_.data(); }

private:
    ModelConfig cfg_;
    ModelWeights weights_;
    RopeTables rope_;

    AlignedBuffer<float> x_, xb_, xb2_;
    AlignedBuffer<float> gate_buf_, up_buf_, down_buf_;

    Attention attn_;
};

} // namespace turbocpp
