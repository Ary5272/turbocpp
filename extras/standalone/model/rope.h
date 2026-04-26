#pragma once
#include <cstddef>
#include "../core/allocator.h"

namespace turbocpp {

// Rotary Position Embedding (Su et al. 2021), LLaMA/NeoX "split-halves"
// variant: for head_dim D and position p, rotate pairs (q[i], q[i + D/2])
// by angle θ_i × p where θ_i = base^(-2i/D).
//
// LONG-CONTEXT EXTENSION:
//   Original LLaMA was trained at 2048 tokens. Going beyond requires
//   adjusting θ — three popular methods:
//
//     LINEAR (Position Interpolation, Chen et al. 2023):
//       θ'_i = θ_i / s     where s = new_ctx / orig_ctx
//       Cheap, ~1k token retrain, OK quality.
//
//     NTK-AWARE (bloc97 2023):
//       Multiply base by s^(D/(D-2)). Preserves high-freq, only stretches
//       low-freq dims. Works zero-shot for ~2× extension.
//
//     YARN (Peng et al. 2023):
//       Combines NTK with attention temperature scaling. Best quality at
//       8×+ extension. We implement the rotation portion; the temperature
//       term is folded into the softmax scale at attention time.
//
// Pick mode via RopeConfig.
enum class RopeMode { Standard, Linear, NTK, YaRN };

struct RopeConfig {
    RopeMode mode      = RopeMode::Standard;
    float    base      = 10000.0f;
    float    scale     = 1.0f;       // ctx_extension factor s (used by Linear/NTK/YaRN)
    size_t   orig_ctx  = 2048;       // training-time context (used by YaRN)
    // YaRN ramp window — frequencies inside [low_freq, high_freq] are
    // partially scaled. Defaults from the paper for LLaMA.
    float    yarn_beta_fast = 32.0f;
    float    yarn_beta_slow = 1.0f;
};

class RopeTables {
public:
    RopeTables() = default;

    // Backward-compat: standard RoPE.
    void build(size_t head_dim, size_t max_seq_len, float base = 10000.0f);

    // Full config build.
    void build(size_t head_dim, size_t max_seq_len, const RopeConfig& cfg);

    void apply(float* x, size_t n_heads, size_t pos) const;

    size_t head_dim()    const noexcept { return head_dim_; }
    size_t max_seq_len() const noexcept { return max_seq_len_; }

    // YaRN attention temperature multiplier. Returns 1.0 for non-YaRN.
    // Multiplied into the attention softmax scale: 1/sqrt(d) → mscale/sqrt(d).
    float yarn_mscale() const noexcept { return mscale_; }

private:
    AlignedBuffer<float> cos_;
    AlignedBuffer<float> sin_;
    size_t head_dim_ = 0;
    size_t max_seq_len_ = 0;
    float  mscale_ = 1.0f;
};

} // namespace turbocpp
