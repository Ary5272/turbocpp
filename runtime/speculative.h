#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "../core/allocator.h"
#include "../kv_cache/kv_cache.h"
#include "../model/transformer.h"
#include "sampling.h"

namespace turbocpp {

// Speculative decoding (Leviathan et al. 2023):
//   1. Cheap "draft" model proposes K next tokens (greedy).
//   2. Target (main) model evaluates them in a single batched forward.
//   3. Accept tokens until the draft and target distributions diverge,
//      then sample a fresh token from the target's distribution at the
//      first divergence.
//
// On a 7B target + 1B draft, expected acceptance rate is ~60-80% on text,
// giving 2-3× wall-clock speedup at zero quality loss vs target alone.
//
// This implementation is "vanilla" — it uses greedy draft (top-1 from
// draft model) and accept/reject by exact-match. Tree-decoding,
// distribution-aware acceptance (Chen et al. 2023), and Medusa-style
// multi-head drafts are roadmap.

struct SpecOptions {
    size_t draft_lookahead = 5;      // tokens proposed per round
    size_t max_new_tokens  = 128;
    SamplingParams sampling;
};

struct SpecStats {
    size_t accepted = 0;
    size_t proposed = 0;
    size_t target_steps = 0;
    double total_ms = 0;
    double acceptance_rate() const noexcept {
        return proposed ? double(accepted) / double(proposed) : 0;
    }
};

// Drives speculative decoding. Both models must share the same vocabulary.
class SpeculativeRunner {
public:
    SpeculativeRunner() = default;

    void init(Model& draft, Model& target);

    // Run K rounds: draft proposes draft_lookahead tokens, target verifies.
    // `start_pos` is the absolute position of the next token to generate.
    // Returns the list of generated token ids (does NOT include the prompt).
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt,
                                  const SpecOptions& opts,
                                  SpecStats* out_stats = nullptr);

private:
    Model* draft_  = nullptr;
    Model* target_ = nullptr;
    KVCache draft_cache_, target_cache_;
    AlignedBuffer<float> draft_logits_, target_logits_;
};

} // namespace turbocpp
