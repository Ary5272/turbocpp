#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "../core/allocator.h"

namespace turbocpp {

// Mixture-of-Experts FFN block (Mixtral, DeepSeek-MoE, Qwen-MoE).
//
// Per token:
//   1. Router: x @ W_router → [n_experts] logits → softmax → top-k experts.
//   2. For each selected expert e (k = top_k of them):
//        h_e = SiLU(x @ W_gate[e]^T) * (x @ W_up[e]^T) @ W_down[e]^T
//   3. Output = sum_e (gate_weight[e] * h_e), where gate_weights are
//      softmax probabilities over the chosen experts (re-normalized).
//
// Memory: each expert holds its own (W_gate, W_up, W_down). For Mixtral-8x7B
// that's 8× FFN params. We never load all experts' activations at once —
// just the top-k per token, which is the whole point.

struct MoEConfig {
    size_t n_experts = 1;       // 1 = degenerate (regular FFN)
    size_t top_k     = 1;       // top-k experts per token (Mixtral: 2)
    bool   norm_topk = true;    // re-normalize gate weights to sum to 1
};

struct MoELayerWeights {
    const float* W_router = nullptr;     // [n_experts, hidden_dim]
    // Per-expert FFN weights.
    std::vector<const float*> W_gate;    // [ffn_dim, hidden_dim]
    std::vector<const float*> W_up;
    std::vector<const float*> W_down;
};

// Forward pass: x_in [hidden_dim] → x_out [hidden_dim], non-aliasing.
// Workspace buffers `gate`, `up`, `down`, `scratch` are caller-owned to
// avoid hot-path malloc.
void moe_forward(const MoEConfig& cfg, const MoELayerWeights& w,
                 const float* x_in, float* x_out,
                 size_t hidden_dim, size_t ffn_dim,
                 float* gate_buf, float* up_buf, float* down_buf,
                 float* router_buf,
                 float* scratch_out);   // hidden_dim — accumulator

} // namespace turbocpp
