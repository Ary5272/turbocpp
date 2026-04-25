#pragma once
#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <vector>

namespace turbocpp {

// Sampling configuration. Defaults give greedy (temp=0).
//
// Pipeline order (matches llama.cpp):
//   1. apply repetition penalty to recent tokens
//   2. divide logits by temperature
//   3. top-k filter (set rest to -inf)
//   4. softmax → probs
//   5. min-p filter (drop p < min_p × max_p)
//   6. top-p filter (keep cumulative ≤ top_p)
//   7. multinomial sample
struct SamplingParams {
    float    temperature   = 0.0f;     // 0 = greedy (skips 2-7, takes argmax)
    int32_t  top_k         = 0;        // 0 = disabled
    float    top_p         = 1.0f;     // 1.0 = disabled
    float    min_p         = 0.0f;     // 0.0 = disabled
    float    repeat_penalty = 1.0f;    // 1.0 = disabled
    int32_t  repeat_window = 64;       // last N tokens considered
    uint64_t seed          = 0xDEADBEEFCAFEBABEull;
};

class Sampler {
public:
    Sampler() = default;
    explicit Sampler(const SamplingParams& p) : params_(p), rng_(p.seed) {}

    void set_params(const SamplingParams& p) { params_ = p; rng_.seed(p.seed); }
    const SamplingParams& params() const noexcept { return params_; }

    // Notify the sampler that `id` was the most recent token. Used by
    // repetition penalty.
    void record(int32_t id);
    void clear_history() { history_.clear(); }

    // Sample. Mutates `logits` (treat as scratch after).
    int32_t sample(float* logits, size_t vocab_size);

private:
    SamplingParams params_;
    std::mt19937_64 rng_;
    std::deque<int32_t> history_;
    std::vector<std::pair<float, int32_t>> scratch_;
};

} // namespace turbocpp
