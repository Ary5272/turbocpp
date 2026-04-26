#include "sampling.h"
#include "../math/vec_ops.h"
#include <algorithm>
#include <cmath>

namespace turbocpp {

void Sampler::record(int32_t id) {
    history_.push_back(id);
    while (history_.size() > size_t(std::max(1, params_.repeat_window))) {
        history_.pop_front();
    }
}

// Mirostat v2 entry — runs AFTER repetition penalty + temperature.
// Algorithm:
//   1. softmax the logits.
//   2. sort descending, find smallest k s.t. probs[k] would push surprisal
//      below μ. Equivalent to k = floor(exp(μ)).
//   3. truncate to top-k, renormalize, sample, observe surprisal s.
//   4. error = s - τ; μ -= eta * error.
static int32_t mirostat_v2_sample(float* logits, size_t vocab_size,
                                  float tau, float eta, float& mu,
                                  std::mt19937_64& rng,
                                  std::vector<std::pair<float,int32_t>>& scratch) {
    // softmax in place.
    float maxv = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) if (logits[i] > maxv) maxv = logits[i];
    float sum = 0;
    for (size_t i = 0; i < vocab_size; ++i) { logits[i] = std::exp(logits[i] - maxv); sum += logits[i]; }
    for (size_t i = 0; i < vocab_size; ++i) logits[i] /= sum;

    scratch.resize(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) scratch[i] = {logits[i], int32_t(i)};
    std::sort(scratch.begin(), scratch.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // k = max ranks whose surprisal stays under μ. Surprisal of rank r ≈
    // -log(probs[r]); we want -log p ≤ μ → p ≥ e^-μ.
    const float p_min = std::exp(-mu);
    size_t k = 0;
    for (; k < vocab_size; ++k) if (scratch[k].first < p_min) break;
    if (k == 0) k = 1;

    // Renormalize the top-k.
    float s2 = 0;
    for (size_t i = 0; i < k; ++i) s2 += scratch[i].first;
    std::uniform_real_distribution<float> uni(0, 1);
    float r = uni(rng) * s2;
    float cum = 0;
    int32_t pick = scratch[0].second;
    float pick_p = scratch[0].first;
    for (size_t i = 0; i < k; ++i) {
        cum += scratch[i].first;
        if (r < cum) { pick = scratch[i].second; pick_p = scratch[i].first; break; }
    }
    // Update μ.
    const float observed = -std::log(std::max(pick_p, 1e-9f));
    mu -= eta * (observed - tau);
    if (mu < 0) mu = 0;
    return pick;
}

int32_t Sampler::sample(float* logits, size_t vocab_size) {
    if (vocab_size == 0) return 0;

    // 0. Logit bias (additive, applied first so it composes with everything
    // downstream).
    if (!params_.logit_bias.empty()) {
        for (const auto& kv : params_.logit_bias) {
            if (kv.first >= 0 && size_t(kv.first) < vocab_size) {
                logits[size_t(kv.first)] += kv.second;
            }
        }
    }

    // 1. Repetition penalty (CTRL-style: divide if positive, multiply if
    // negative — preserves sign sensibly).
    if (params_.repeat_penalty > 1.0f && !history_.empty()) {
        const float p = params_.repeat_penalty;
        for (int32_t id : history_) {
            if (id < 0 || size_t(id) >= vocab_size) continue;
            float& l = logits[size_t(id)];
            l = (l > 0.0f) ? (l / p) : (l * p);
        }
    }

    // Greedy fast-path.
    if (params_.temperature <= 0.0f) {
        return int32_t(argmax(logits, vocab_size));
    }

    // 2. Temperature.
    if (params_.temperature != 1.0f) {
        vec_scale(logits, 1.0f / params_.temperature, vocab_size);
    }

    // Mirostat v2: takes over after temperature; bypass top-k/p/min-p.
    if (params_.mirostat == 2) {
        if (mu_ <= 0.0f) mu_ = 2.0f * params_.mirostat_tau;
        return mirostat_v2_sample(logits, vocab_size,
                                  params_.mirostat_tau, params_.mirostat_eta,
                                  mu_, rng_, scratch_);
    }

    // Mirostat v1 (Basu 2020 original). Estimates Zipf parameter s from
    // top-2 ranks; picks k = (s * tau)^eta and renormalizes.
    if (params_.mirostat == 1) {
        if (mu_ <= 0.0f) mu_ = 2.0f * params_.mirostat_tau;
        // softmax then sort.
        float maxv = logits[0];
        for (size_t i = 1; i < vocab_size; ++i) if (logits[i] > maxv) maxv = logits[i];
        float ssum = 0;
        for (size_t i = 0; i < vocab_size; ++i) { logits[i] = std::exp(logits[i] - maxv); ssum += logits[i]; }
        for (size_t i = 0; i < vocab_size; ++i) logits[i] /= ssum;

        scratch_.resize(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) scratch_[i] = {logits[i], int32_t(i)};
        std::sort(scratch_.begin(), scratch_.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        // Zipf exponent estimate from top-2.
        float p1 = std::max(scratch_[0].first, 1e-9f);
        float p2 = std::max(scratch_[std::min<size_t>(1, vocab_size-1)].first, 1e-12f);
        float s = std::log(p1 / p2) / std::log(2.0f);
        size_t k = std::max<size_t>(1, size_t(std::pow(s * mu_, params_.mirostat_eta)));
        k = std::min(k, vocab_size);
        // Sample within top-k.
        float ksum = 0;
        for (size_t i = 0; i < k; ++i) ksum += scratch_[i].first;
        std::uniform_real_distribution<float> uni(0, 1);
        float r = uni(rng_) * ksum;
        float cumk = 0;
        int32_t pick = scratch_[0].second;
        float pick_p = scratch_[0].first;
        for (size_t i = 0; i < k; ++i) {
            cumk += scratch_[i].first;
            if (r < cumk) { pick = scratch_[i].second; pick_p = scratch_[i].first; break; }
        }
        const float observed = -std::log(std::max(pick_p, 1e-9f));
        mu_ -= params_.mirostat_eta * (observed - params_.mirostat_tau);
        if (mu_ < 0) mu_ = 0;
        return pick;
    }

    // Dynamic temperature: rescale temperature by softmax entropy.
    // High-entropy distribution → lower temp (more focused), low-entropy →
    // higher temp (preserves diversity). Applied AFTER static temperature.
    if (params_.dynatemp_range > 0.0f) {
        // Compute entropy in nats over a softmax(logits). Use a copy so we
        // don't mutate logits twice.
        float maxv = logits[0];
        for (size_t i = 1; i < vocab_size; ++i) if (logits[i] > maxv) maxv = logits[i];
        float s2 = 0;
        for (size_t i = 0; i < vocab_size; ++i) s2 += std::exp(logits[i] - maxv);
        const float lse = std::log(s2) + maxv;
        float H = 0;
        for (size_t i = 0; i < vocab_size; ++i) {
            float p = std::exp(logits[i] - lse);
            if (p > 0) H -= p * std::log(p);
        }
        const float Hmax = std::log(float(vocab_size));
        // Normalize entropy to [0, 1] and apply curve.
        float norm = (Hmax > 0) ? std::pow(H / Hmax, params_.dynatemp_exp) : 0;
        // Effective temp = base + range * (2*norm - 1) → in [base-range, base+range].
        const float base = params_.temperature;
        const float eff_t = std::max(0.05f, base + params_.dynatemp_range * (2 * norm - 1));
        // Re-scale logits by eff_t / base (we already divided by base in step 2).
        if (base > 0 && eff_t > 0) vec_scale(logits, base / eff_t, vocab_size);
    }

    // 3. Top-k.
    if (params_.top_k > 0 && size_t(params_.top_k) < vocab_size) {
        const size_t k = size_t(params_.top_k);
        scratch_.resize(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) scratch_[i] = {logits[i], int32_t(i)};
        std::nth_element(
            scratch_.begin(), scratch_.begin() + std::ptrdiff_t(k), scratch_.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
        const float threshold = scratch_[k - 1].first;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (logits[i] < threshold) logits[i] = -1e30f;
        }
    }

    // 4. Softmax → probs.
    softmax_inplace(logits, vocab_size);

    // 4b. Tail-free sampling (Frans 2020). Sort probs desc, take 2nd diff
    // |p[i+1] - 2*p[i] + p[i-1]| normalized cumulative; cut at z.
    if (params_.tail_free_z < 1.0f && params_.tail_free_z > 0.0f && vocab_size > 3) {
        scratch_.resize(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) scratch_[i] = {logits[i], int32_t(i)};
        std::sort(scratch_.begin(), scratch_.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<float> d2(vocab_size, 0);
        float total = 0;
        for (size_t i = 1; i + 1 < vocab_size; ++i) {
            d2[i] = std::fabs(scratch_[i+1].first - 2*scratch_[i].first + scratch_[i-1].first);
            total += d2[i];
        }
        if (total > 0) {
            float cum = 0;
            size_t cut = vocab_size;
            for (size_t i = 1; i + 1 < vocab_size; ++i) {
                cum += d2[i] / total;
                if (cum >= params_.tail_free_z) { cut = i + 1; break; }
            }
            for (size_t i = cut; i < vocab_size; ++i)
                logits[size_t(scratch_[i].second)] = 0.0f;
            float ns = 0; for (size_t i = 0; i < vocab_size; ++i) ns += logits[i];
            if (ns > 0) vec_scale(logits, 1.0f / ns, vocab_size);
        }
    }

    // 4c. Locally typical sampling (Meister et al. 2023). Filters tokens
    // whose surprisal is far from the entropy.
    if (params_.typical_p < 1.0f && params_.typical_p > 0.0f) {
        float H = 0;
        for (size_t i = 0; i < vocab_size; ++i)
            if (logits[i] > 0) H -= logits[i] * std::log(logits[i]);
        scratch_.resize(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            float surprisal = (logits[i] > 0) ? -std::log(logits[i]) : 1e30f;
            scratch_[i] = {std::fabs(surprisal - H), int32_t(i)};
        }
        std::sort(scratch_.begin(), scratch_.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        float cum = 0;
        size_t cut = vocab_size;
        for (size_t i = 0; i < vocab_size; ++i) {
            cum += logits[size_t(scratch_[i].second)];
            if (cum >= params_.typical_p) { cut = i + 1; break; }
        }
        for (size_t i = cut; i < vocab_size; ++i)
            logits[size_t(scratch_[i].second)] = 0.0f;
        float ns = 0; for (size_t i = 0; i < vocab_size; ++i) ns += logits[i];
        if (ns > 0) vec_scale(logits, 1.0f / ns, vocab_size);
    }

    // 5. Min-p (drop probs below min_p × max_p; renormalize).
    if (params_.min_p > 0.0f) {
        float maxp = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) if (logits[i] > maxp) maxp = logits[i];
        const float thresh = params_.min_p * maxp;
        float s = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            if (logits[i] < thresh) logits[i] = 0.0f;
            s += logits[i];
        }
        if (s > 0.0f) vec_scale(logits, 1.0f / s, vocab_size);
    }

    // 6. Top-p.
    if (params_.top_p < 1.0f && params_.top_p > 0.0f) {
        scratch_.resize(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) scratch_[i] = {logits[i], int32_t(i)};
        std::sort(scratch_.begin(), scratch_.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        float cum = 0.0f;
        size_t cut = vocab_size;
        for (size_t i = 0; i < vocab_size; ++i) {
            cum += scratch_[i].first;
            if (cum >= params_.top_p) { cut = i + 1; break; }
        }
        for (size_t i = cut; i < vocab_size; ++i) {
            logits[size_t(scratch_[i].second)] = 0.0f;
        }
        float s = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) s += logits[i];
        if (s > 0.0f) vec_scale(logits, 1.0f / s, vocab_size);
    }

    // 7. Multinomial.
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const float r = uni(rng_);
    float cum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cum += logits[i];
        if (r < cum) return int32_t(i);
    }
    for (size_t i = vocab_size; i-- > 0; ) {
        if (logits[i] > 0.0f) return int32_t(i);
    }
    return 0;
}

void cfg_mix_logits(float* logits_pos, const float* logits_neg,
                    float cfg_scale, size_t vocab_size) {
    // logits_pos = logits_neg + scale * (logits_pos - logits_neg)
    //            = scale*logits_pos + (1-scale)*logits_neg
    for (size_t i = 0; i < vocab_size; ++i) {
        logits_pos[i] = cfg_scale * logits_pos[i] + (1.0f - cfg_scale) * logits_neg[i];
    }
}

} // namespace turbocpp
