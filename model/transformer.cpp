#include "transformer.h"
#include "rmsnorm.h"
#include "embeddings.h"
#include "../math/matmul.h"
#include "../math/vec_ops.h"
#include "../math/activations.h"
#include "../runtime/parallel_ops.h"
#include "../utils/logging.h"
#include <cstring>

namespace turbocpp {

void Model::init(const ModelConfig& cfg) {
    cfg_ = cfg;
    TCPP_CHECK(cfg.hidden_dim == cfg.n_heads * cfg.head_dim,
               "model: hidden_dim (%zu) must equal n_heads*head_dim (%zu*%zu)",
               cfg.hidden_dim, cfg.n_heads, cfg.head_dim);

    rope_.build(cfg.head_dim, cfg.max_seq_len, cfg.rope_base);

    x_.resize(cfg.hidden_dim);
    xb_.resize(cfg.hidden_dim);
    xb2_.resize(cfg.hidden_dim);
    gate_buf_.resize(cfg.ffn_dim);
    up_buf_.resize(cfg.ffn_dim);
    down_buf_.resize(cfg.hidden_dim);

    TCPP_CHECK(cfg.n_kv_heads > 0 && cfg.n_heads % cfg.n_kv_heads == 0,
               "model: n_heads (%zu) must be divisible by n_kv_heads (%zu)",
               cfg.n_heads, cfg.n_kv_heads);
    attn_.init(cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);

    weights_.layers.resize(cfg.n_layers);
}

void Model::forward(int32_t token_id, size_t pos, KVCache& cache, float* logits) {
    const size_t H = cfg_.hidden_dim;
    const size_t F = cfg_.ffn_dim;
    TCPP_CHECK(weights_.tok_embed, "model: weights not set");

    // Embed.
    embed_single(weights_.tok_embed, token_id, x_.data(), H, cfg_.vocab_size);

    // Transformer blocks.
    for (size_t L = 0; L < cfg_.n_layers; ++L) {
        const LayerWeights& lw = weights_.layers.data()[L];

        // ---- Attention block ----
        // norm -> attn -> residual
        rmsnorm_row(x_.data(), lw.attn_norm, xb_.data(), H, cfg_.rms_eps);
        attn_.set_weights(lw.Wq, lw.Wk, lw.Wv, lw.Wo);
        attn_.forward(xb_.data(), xb2_.data(), pos, L, cache, rope_);
        vec_add_inplace(x_.data(), xb2_.data(), H);   // residual

        // ---- FFN block (SwiGLU) ----
        //   xb  = norm(x)
        //   g   = xb @ Wgate^T      [F]
        //   u   = xb @ Wup^T        [F]
        //   h   = silu(g) * u       [F]
        //   out = h @ Wdown^T       [H]
        //   x  += out
        rmsnorm_row(x_.data(), lw.ffn_norm, xb_.data(), H, cfg_.rms_eps);

        matmul_parallel(xb_.data(), lw.Wgate, gate_buf_.data(), 1, F, H);
        matmul_parallel(xb_.data(), lw.Wup,   up_buf_.data(),   1, F, H);
        silu_mul(gate_buf_.data(), gate_buf_.data(), up_buf_.data(), F);
        matmul_parallel(gate_buf_.data(), lw.Wdown, down_buf_.data(), 1, H, F);

        vec_add_inplace(x_.data(), down_buf_.data(), H);
    }

    // Final norm + LM head projection to vocab logits. LM head is the
    // biggest matmul in the model (V × H = 32000 × 4096 for LLaMA-7B) —
    // always benefits from threading.
    rmsnorm_row(x_.data(), weights_.final_norm, xb_.data(), H, cfg_.rms_eps);
    matmul_parallel(xb_.data(), weights_.lm_head, logits, 1, cfg_.vocab_size, H);

    // Bump cache length now that this token is fully processed.
    cache.set_cur_len(pos + 1);
}

} // namespace turbocpp
