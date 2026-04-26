#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "../core/allocator.h"
#include "../kv_cache/kv_cache.h"
#include "../model/transformer.h"

namespace turbocpp {

// Beam search: maintain B parallel hypotheses, expand each by top-K
// continuations, keep the B best by cumulative log-probability.
//
// Returns the highest-scoring completion. Length-normalized score
// (sum_logp / length^alpha) prevents short-bias.

struct BeamOptions {
    size_t  beam_width   = 4;
    size_t  max_new_tokens = 64;
    float   length_alpha = 0.7f;
    int32_t eos_token    = -1;
};

struct Hypothesis {
    std::vector<int32_t> tokens;
    float sum_logp = 0;
    bool  done = false;

    float score(float alpha) const {
        const float L = std::pow(float(tokens.empty() ? 1 : tokens.size()), alpha);
        return sum_logp / L;
    }
};

// Beam search driver. Owns B independent KV caches (one per beam).
class BeamSearch {
public:
    BeamSearch() = default;

    void init(Model& target, size_t beam_width);
    Hypothesis search(const std::vector<int32_t>& prompt, const BeamOptions& opts);

private:
    Model* model_ = nullptr;
    std::vector<KVCache> caches_;
    AlignedBuffer<float> logits_;
};

} // namespace turbocpp
