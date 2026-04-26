#include "moe.h"
#include "../math/matmul.h"
#include "../math/vec_ops.h"
#include "../math/activations.h"
#include "../utils/logging.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace turbocpp {

void moe_forward(const MoEConfig& cfg, const MoELayerWeights& w,
                 const float* x_in, float* x_out,
                 size_t H, size_t F,
                 float* gate, float* up, float* down,
                 float* router_logits,
                 float* scratch_out) {
    TCPP_CHECK(cfg.top_k >= 1 && cfg.top_k <= cfg.n_experts,
               "moe: top_k (%zu) must be in [1, n_experts=%zu]",
               cfg.top_k, cfg.n_experts);
    TCPP_CHECK(w.W_router, "moe: W_router missing");

    // 1. Router scores.
    matmul(x_in, w.W_router, router_logits, 1, cfg.n_experts, H);

    // 2. Top-k experts.
    std::vector<std::pair<float, size_t>> ranked(cfg.n_experts);
    for (size_t e = 0; e < cfg.n_experts; ++e) ranked[e] = {router_logits[e], e};
    std::nth_element(ranked.begin(), ranked.begin() + std::ptrdiff_t(cfg.top_k),
                     ranked.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    ranked.resize(cfg.top_k);

    // 3. Softmax over chosen experts (re-normalize so weights sum to 1).
    if (cfg.norm_topk) {
        float maxv = ranked[0].first;
        for (auto& p : ranked) if (p.first > maxv) maxv = p.first;
        float s = 0;
        for (auto& p : ranked) { p.first = std::exp(p.first - maxv); s += p.first; }
        if (s > 0) for (auto& p : ranked) p.first /= s;
    }

    // 4. Run each chosen expert and accumulate gated output.
    std::memset(scratch_out, 0, H * sizeof(float));
    for (const auto& [gate_w, e] : ranked) {
        TCPP_CHECK(e < w.W_gate.size() && e < w.W_up.size() && e < w.W_down.size(),
                   "moe: expert index %zu out of range", e);
        matmul(x_in, w.W_gate[e], gate, 1, F, H);
        matmul(x_in, w.W_up[e],   up,   1, F, H);
        silu_mul(gate, gate, up, F);
        matmul(gate, w.W_down[e], down, 1, H, F);
        // scratch += gate_w * down
        for (size_t i = 0; i < H; ++i) scratch_out[i] += gate_w * down[i];
    }
    std::memcpy(x_out, scratch_out, H * sizeof(float));
}

} // namespace turbocpp
