#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "../core/allocator.h"
#include "../kv_cache/kv_cache.h"
#include "../model/transformer.h"
#include "../tokenizer/bpe.h"
#include "sampling.h"

namespace turbocpp {

// Callback invoked after every sampled token. `tok` is the raw text of the
// token (BPE-decoded). Return false to stop generation early.
using TokenCallback = std::function<bool(int32_t token_id, const std::string& tok)>;

struct GenerateOptions {
    size_t         max_new_tokens = 128;
    bool           stop_on_eos    = true;
    SamplingParams sampling;
    TokenCallback  on_token;       // optional — for streaming UIs
};

struct GenerateStats {
    double prefill_ms = 0.0;
    double decode_ms  = 0.0;
    size_t prompt_tokens = 0;
    size_t generated_tokens = 0;

    double tokens_per_second() const noexcept {
        return generated_tokens / (decode_ms / 1000.0 + 1e-9);
    }
};

// High-level driver. Owns a KVCache + logits scratch and orchestrates
// prefill + autoregressive decoding.
class InferenceEngine {
public:
    InferenceEngine() = default;

    // Binds model+tokenizer. Does not copy — caller must keep them alive.
    void init(Model& model, BPETokenizer& tok);

    // Reset conversation state (clear KV cache).
    void reset();

    // Generate text continuing from `prompt`. Returns the generated text
    // (not including the prompt). Stats filled if `out_stats` non-null.
    std::string generate(const std::string& prompt,
                         const GenerateOptions& opts,
                         GenerateStats* out_stats = nullptr);

    const KVCache& cache() const noexcept { return cache_; }

private:
    Model*        model_ = nullptr;
    BPETokenizer* tok_   = nullptr;

    KVCache              cache_;
    AlignedBuffer<float> logits_;
    Sampler              sampler_;
};

} // namespace turbocpp
