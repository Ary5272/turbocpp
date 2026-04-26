#include "beam_search.h"
#include "../math/vec_ops.h"
#include "../utils/logging.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace turbocpp {

void BeamSearch::init(Model& m, size_t beam_width) {
    model_ = &m;
    const auto& cfg = m.config();
    caches_.clear();
    caches_.resize(beam_width);
    for (auto& c : caches_)
        c.init(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, cfg.max_seq_len);
    logits_.resize(cfg.vocab_size);
}

Hypothesis BeamSearch::search(const std::vector<int32_t>& prompt,
                              const BeamOptions& opts) {
    TCPP_CHECK(model_, "beam: init() not called");
    const size_t V = model_->config().vocab_size;
    const size_t B = opts.beam_width;
    TCPP_CHECK(caches_.size() >= B, "beam: cache count < beam_width");

    std::vector<Hypothesis> beams(1);
    beams[0].sum_logp = 0;
    beams[0].tokens.reserve(prompt.size() + opts.max_new_tokens);

    // Prefill the first beam's cache; clones happen lazily when we branch.
    size_t pos = 0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        model_->forward(prompt[i], pos, caches_[0], logits_.data());
        ++pos;
    }
    // After prefill, logits_ has next-token distribution. Convert to log-probs.
    std::vector<float> base_logp(V);
    {
        float maxv = logits_.data()[0];
        for (size_t i = 1; i < V; ++i) if (logits_.data()[i] > maxv) maxv = logits_.data()[i];
        float s = 0;
        for (size_t i = 0; i < V; ++i) { base_logp[i] = logits_.data()[i] - maxv; s += std::exp(base_logp[i]); }
        const float lse = std::log(s);
        for (size_t i = 0; i < V; ++i) base_logp[i] -= lse;
    }

    // Initial expansion: pick top-B tokens.
    std::vector<std::pair<float, int32_t>> heap(V);
    for (size_t i = 0; i < V; ++i) heap[i] = {base_logp[i], int32_t(i)};
    std::nth_element(heap.begin(), heap.begin() + std::ptrdiff_t(B), heap.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    std::sort(heap.begin(), heap.begin() + std::ptrdiff_t(B),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    beams.resize(B);
    for (size_t b = 0; b < B; ++b) {
        beams[b].tokens.assign(prompt.begin(), prompt.end());
        beams[b].tokens.push_back(heap[b].second);
        beams[b].sum_logp = heap[b].first;
    }
    // Clone cache 0 → caches 1..B-1. We do this lazily by re-running prefill
    // for each clone; sharing prefilled state is a roadmap optimization.
    for (size_t b = 1; b < B; ++b) {
        caches_[b].clear();
        for (size_t i = 0; i < prompt.size(); ++i) {
            model_->forward(prompt[i], i, caches_[b], logits_.data());
        }
    }
    // Advance each beam's cache by its initial sampled token.
    for (size_t b = 0; b < B; ++b) {
        model_->forward(beams[b].tokens.back(), pos, caches_[b], logits_.data());
    }

    // Generation loop.
    size_t step = 1;
    for (; step < opts.max_new_tokens; ++step) {
        std::vector<std::tuple<float, int32_t, size_t>> cand;  // (cum_logp, tok, beam)
        cand.reserve(B * B + B);

        for (size_t b = 0; b < B; ++b) {
            if (beams[b].done) {
                cand.emplace_back(beams[b].sum_logp, -1, b);
                continue;
            }
            // logits_ currently holds the LAST forward for caches_[b]; but we
            // need to re-run forward for each beam separately (caches differ).
            // Simpler: re-forward beam b's last token here.
            model_->forward(beams[b].tokens.back(), pos + step - 1, caches_[b], logits_.data());

            float maxv = logits_.data()[0];
            for (size_t i = 1; i < V; ++i) if (logits_.data()[i] > maxv) maxv = logits_.data()[i];
            float s = 0;
            std::vector<float> lp(V);
            for (size_t i = 0; i < V; ++i) { lp[i] = logits_.data()[i] - maxv; s += std::exp(lp[i]); }
            const float lse = std::log(s);
            for (size_t i = 0; i < V; ++i) lp[i] -= lse;

            // Top-B continuations from this beam.
            std::vector<std::pair<float, int32_t>> top(V);
            for (size_t i = 0; i < V; ++i) top[i] = {lp[i], int32_t(i)};
            std::nth_element(top.begin(), top.begin() + std::ptrdiff_t(B), top.end(),
                             [](const auto& a, const auto& b2) { return a.first > b2.first; });
            for (size_t i = 0; i < B; ++i)
                cand.emplace_back(beams[b].sum_logp + top[i].first, top[i].second, b);
        }

        // Keep top-B candidates globally.
        std::sort(cand.begin(), cand.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        if (cand.size() > B) cand.resize(B);

        // Build new beams. KV caches need to be re-cloned per beam; we
        // brute-force by replaying the new tokens. (Roadmap: cache trees.)
        std::vector<Hypothesis> next(B);
        for (size_t i = 0; i < B; ++i) {
            const auto& [logp, tok, parent] = cand[i];
            next[i] = beams[parent];
            next[i].sum_logp = logp;
            if (tok >= 0) {
                next[i].tokens.push_back(tok);
                if (tok == opts.eos_token) next[i].done = true;
            }
        }
        beams = std::move(next);

        // Replay caches for the new beams: reset and re-forward each beam.
        for (size_t b = 0; b < B; ++b) {
            caches_[b].clear();
            for (size_t i = 0; i < beams[b].tokens.size(); ++i) {
                model_->forward(beams[b].tokens[i], i, caches_[b], logits_.data());
            }
        }

        if (std::all_of(beams.begin(), beams.end(),
                        [](const Hypothesis& h) { return h.done; })) break;
    }

    // Pick best by length-normalized score.
    auto best = std::max_element(
        beams.begin(), beams.end(),
        [&](const Hypothesis& a, const Hypothesis& b) {
            return a.score(opts.length_alpha) < b.score(opts.length_alpha);
        });
    return *best;
}

} // namespace turbocpp
