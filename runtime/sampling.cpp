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

int32_t Sampler::sample(float* logits, size_t vocab_size) {
    if (vocab_size == 0) return 0;

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

} // namespace turbocpp
