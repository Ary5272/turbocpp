#include "speculative.h"
#include "../math/vec_ops.h"
#include "../utils/timing.h"
#include "../utils/logging.h"

namespace turbocpp {

void SpeculativeRunner::init(Model& draft, Model& target) {
    draft_ = &draft;
    target_ = &target;
    const auto& dc = draft.config();
    const auto& tc = target.config();
    TCPP_CHECK(dc.vocab_size == tc.vocab_size,
               "speculative: draft (%zu) and target (%zu) vocabs differ",
               dc.vocab_size, tc.vocab_size);
    draft_cache_.init(dc.n_layers, dc.n_kv_heads, dc.head_dim, dc.max_seq_len);
    target_cache_.init(tc.n_layers, tc.n_kv_heads, tc.head_dim, tc.max_seq_len);
    draft_logits_.resize(dc.vocab_size);
    target_logits_.resize(tc.vocab_size);
}

std::vector<int32_t> SpeculativeRunner::generate(
        const std::vector<int32_t>& prompt,
        const SpecOptions& opts,
        SpecStats* out_stats) {
    TCPP_CHECK(draft_ && target_, "speculative: init() not called");
    const size_t V = target_->config().vocab_size;

    Sampler sampler;
    sampler.set_params(opts.sampling);

    Timer t;
    SpecStats st;
    std::vector<int32_t> out;

    // Prefill BOTH caches on the prompt. Doing this twice is expensive — a
    // production engine would share caches; we keep them separate for
    // clarity since the architectures may differ.
    size_t pos = 0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        draft_->forward(prompt[i], pos, draft_cache_, draft_logits_.data());
        target_->forward(prompt[i], pos, target_cache_, target_logits_.data());
        ++pos;
    }
    st.target_steps += prompt.size();

    while (out.size() < opts.max_new_tokens) {
        // 1. Draft K tokens greedily.
        const size_t K = opts.draft_lookahead;
        std::vector<int32_t> drafted;
        drafted.reserve(K);
        for (size_t k = 0; k < K && out.size() + k < opts.max_new_tokens; ++k) {
            int32_t tok = int32_t(argmax(draft_logits_.data(), V));
            drafted.push_back(tok);
            // Advance draft cache.
            draft_->forward(tok, pos + k, draft_cache_, draft_logits_.data());
        }
        st.proposed += drafted.size();

        // 2. Verify with target. Each step: forward draft[i], then sample
        // from target's distribution. If the sampled token == drafted[i],
        // accept; else reject and use the target's sample.
        size_t accepted_this_round = 0;
        for (size_t i = 0; i < drafted.size(); ++i) {
            target_->forward(drafted[i], pos + i, target_cache_, target_logits_.data());
            ++st.target_steps;
            int32_t target_pick = sampler.sample(target_logits_.data(), V);
            if (target_pick == drafted[i]) {
                out.push_back(drafted[i]);
                ++accepted_this_round;
                ++st.accepted;
            } else {
                // Divergence: emit target_pick and rebuild from here.
                out.push_back(target_pick);
                // Rewind draft cache to position pos + i (we over-advanced it).
                draft_cache_.set_cur_len(pos + i);
                target_cache_.set_cur_len(pos + i + 1);
                // Re-run the new token through the draft so its logits
                // reflect the corrected state.
                draft_->forward(target_pick, pos + i, draft_cache_, draft_logits_.data());
                pos += i + 1;
                accepted_this_round = SIZE_MAX;  // sentinel: "broke early"
                break;
            }
        }
        if (accepted_this_round != SIZE_MAX) {
            // All drafts accepted. Sample the next-token from the last
            // target logits (the sample we did at the END of verify is at
            // position pos + K - 1 → its logits predict pos + K).
            int32_t next = sampler.sample(target_logits_.data(), V);
            out.push_back(next);
            // Sync caches to next position.
            target_cache_.set_cur_len(pos + drafted.size() + 1);
            draft_->forward(next, pos + drafted.size(), draft_cache_, draft_logits_.data());
            pos += drafted.size() + 1;
        }
        if (out.size() >= opts.max_new_tokens) break;
        if (pos + opts.draft_lookahead >= target_->config().max_seq_len) break;
    }

    st.total_ms = t.ms();
    if (out_stats) *out_stats = st;
    return out;
}

} // namespace turbocpp
