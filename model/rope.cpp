#include "rope.h"
#include "../utils/logging.h"
#include <algorithm>
#include <cmath>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace turbocpp {

// YaRN ramp (Peng et al. 2023, Eq. 22). Smooth interp between two
// dim indices.
static float yarn_ramp(float low, float high, size_t i, size_t /*D*/) {
    if (high - low < 1e-3f) high = low + 1e-3f;
    float t = (float(i) - low) / (high - low);
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    return 1.0f - t;
}

// Find dim index where wavelength equals orig_ctx / n_rot.
static float yarn_find_correction(float n_rot, size_t D, float base, size_t orig_ctx) {
    return float(D) * std::log(float(orig_ctx) / (n_rot * 2.0f * float(M_PI))) /
           (2.0f * std::log(base));
}

void RopeTables::build(size_t head_dim, size_t max_seq_len, float base) {
    RopeConfig cfg;
    cfg.base = base;
    build(head_dim, max_seq_len, cfg);
}

void RopeTables::build(size_t head_dim, size_t max_seq_len, const RopeConfig& cfg) {
    TCPP_CHECK(head_dim % 2 == 0, "rope: head_dim must be even");
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;

    const size_t half = head_dim / 2;
    cos_.resize(max_seq_len * half);
    sin_.resize(max_seq_len * half);

    // NTK-aware base stretch.
    float base_eff = cfg.base;
    if (cfg.mode == RopeMode::NTK && cfg.scale > 1.0f) {
        base_eff = cfg.base * std::pow(cfg.scale,
                                       float(head_dim) / float(head_dim - 2));
    }

    // YaRN: per-dim ramp + attention temperature.
    float low_dim = 0.0f, high_dim = 0.0f;
    if (cfg.mode == RopeMode::YaRN) {
        low_dim = std::max(yarn_find_correction(cfg.yarn_beta_fast, head_dim,
                                                cfg.base, cfg.orig_ctx), 0.0f);
        high_dim = std::min(yarn_find_correction(cfg.yarn_beta_slow, head_dim,
                                                 cfg.base, cfg.orig_ctx),
                            float(head_dim) - 1.0f);
        mscale_ = 0.1f * std::log(cfg.scale) + 1.0f;
    } else {
        mscale_ = 1.0f;
    }

    for (size_t p = 0; p < max_seq_len; ++p) {
        for (size_t i = 0; i < half; ++i) {
            float angle;
            switch (cfg.mode) {
                case RopeMode::Linear: {
                    float theta = std::pow(cfg.base, -float(2 * i) / float(head_dim));
                    angle = float(p) * theta / cfg.scale;
                    break;
                }
                case RopeMode::NTK: {
                    float theta = std::pow(base_eff, -float(2 * i) / float(head_dim));
                    angle = float(p) * theta;
                    break;
                }
                case RopeMode::YaRN: {
                    float ramp  = yarn_ramp(low_dim, high_dim, i, head_dim);
                    float theta_extrap = std::pow(cfg.base, -float(2*i)/float(head_dim));
                    float theta_interp = theta_extrap / cfg.scale;
                    angle = float(p) * (theta_interp * (1.0f - ramp)
                                      + theta_extrap * ramp);
                    break;
                }
                default: {
                    float theta = std::pow(cfg.base, -float(2 * i) / float(head_dim));
                    angle = float(p) * theta;
                }
            }
            cos_.data()[p * half + i] = std::cos(angle);
            sin_.data()[p * half + i] = std::sin(angle);
        }
    }
}

void RopeTables::apply(float* x, size_t n_heads, size_t pos) const {
    TCPP_CHECK(pos < max_seq_len_, "rope: pos %zu exceeds max %zu",
               pos, max_seq_len_);
    const size_t half = head_dim_ / 2;
    const float* cos_row = cos_.data() + pos * half;
    const float* sin_row = sin_.data() + pos * half;
    for (size_t h = 0; h < n_heads; ++h) {
        float* head = x + h * head_dim_;
        for (size_t i = 0; i < half; ++i) {
            const float c = cos_row[i];
            const float s = sin_row[i];
            const float x0 = head[i];
            const float x1 = head[i + half];
            head[i]        = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}

} // namespace turbocpp
