#include "attention.h"
#include "../math/matmul.h"
#include "../math/vec_ops.h"
#include "../runtime/parallel_ops.h"
#include "../utils/logging.h"
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace turbocpp {

// score[t] = dot(q[D], K[t, D]); apply scale; softmax outside.
static inline void qk_scores(const float* q, const float* K, float* scores,
                             size_t T, size_t D, float scale) {
    matmul(q, K, scores, 1, T, D);
    vec_scale(scores, scale, T);
}

// out[d] = sum_t scores[t] * V[t, d]. T-outer, D-inner — D is contiguous.
static inline void av_accumulate(const float* scores, const float* V,
                                 float* out, size_t T, size_t D) {
    std::memset(out, 0, D * sizeof(float));

#if defined(__AVX2__)
    const size_t D8 = (D / 8) * 8;
    for (size_t t = 0; t < T; ++t) {
        const float s = scores[t];
        const __m256 vs = _mm256_set1_ps(s);
        const float* vrow = V + t * D;
        size_t d = 0;
        for (; d < D8; d += 8) {
            __m256 acc = _mm256_loadu_ps(out + d);
            __m256 v   = _mm256_loadu_ps(vrow + d);
#  if defined(__FMA__)
            acc = _mm256_fmadd_ps(vs, v, acc);
#  else
            acc = _mm256_add_ps(acc, _mm256_mul_ps(vs, v));
#  endif
            _mm256_storeu_ps(out + d, acc);
        }
        for (; d < D; ++d) out[d] += s * vrow[d];
    }
#else
    for (size_t t = 0; t < T; ++t) {
        const float s = scores[t];
        const float* vrow = V + t * D;
        for (size_t d = 0; d < D; ++d) out[d] += s * vrow[d];
    }
#endif
}

void Attention::init(size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    n_heads_    = n_heads;
    n_kv_heads_ = n_kv_heads;
    head_dim_   = head_dim;
    TCPP_CHECK(n_heads % n_kv_heads == 0,
               "attention: n_heads (%zu) must be divisible by n_kv_heads (%zu)",
               n_heads, n_kv_heads);
    q_.resize(n_heads_    * head_dim_);
    k_proj_.resize(n_kv_heads_ * head_dim_);
    v_proj_.resize(n_kv_heads_ * head_dim_);
    attn_out_.resize(n_heads_ * head_dim_);
}

void Attention::set_weights(const float* Wq, const float* Wk,
                            const float* Wv, const float* Wo) {
    Wq_ = Wq; Wk_ = Wk; Wv_ = Wv; Wo_ = Wo;
}

void Attention::forward(const float* x_in, float* x_out, size_t pos, size_t layer,
                        KVCache& cache, const RopeTables& rope) {
    const size_t H  = n_heads_    * head_dim_;
    const size_t KV = n_kv_heads_ * head_dim_;
    const size_t T  = pos + 1;
    const size_t heads_per_kv = n_heads_ / n_kv_heads_;
    TCPP_CHECK(Wq_ && Wk_ && Wv_ && Wo_, "attention: weights not set");

    if (scores_.size() < n_heads_ * cache.max_seq_len()) {
        scores_.resize(n_heads_ * cache.max_seq_len());
        scores_cap_ = cache.max_seq_len();
    }

    // (1) Q/K/V projections. Wk/Wv project to KV (= n_kv_heads*head_dim).
    matmul_parallel(x_in, Wq_, q_.data(),      1, H,  H);
    matmul_parallel(x_in, Wk_, k_proj_.data(), 1, KV, H);
    matmul_parallel(x_in, Wv_, v_proj_.data(), 1, KV, H);

    // (2) Position encoding: RoPE if not using ALiBi.
    if (!use_alibi_) {
        rope.apply(q_.data(),      n_heads_,    pos);
        rope.apply(k_proj_.data(), n_kv_heads_, pos);
    }

    // (3) Append K/V into the KV cache (sized for n_kv_heads).
    cache.append(layer, pos, k_proj_.data(), v_proj_.data());

    // (4) Per-head attention. Query head h reads from KV head (h / heads_per_kv).
    const float scale = 1.0f / std::sqrt(float(head_dim_));
    // Sliding window: only attend to the last `sliding_window_` positions.
    // start = max(0, pos - sliding_window + 1) → effective T_eff = pos - start + 1.
    const size_t start = (sliding_window_ > 0 && pos + 1 > sliding_window_)
                         ? (pos + 1 - sliding_window_) : 0;
    const size_t T_eff = T - start;

    // ALiBi slope per head: m_h = 2^(-8 h / n_heads). Bias added to score
    // BEFORE softmax: score[t] += -m_h * (pos - t).
    auto alibi_slope = [&](size_t h) -> float {
        return std::exp2(-8.0f * float(h + 1) / float(n_heads_));
    };

    parallel_heads(n_heads_, [&](size_t h) {
        const size_t kv_h = h / heads_per_kv;
        const float* q_h  = q_.data() + h * head_dim_;
        const float* K_h  = cache.k_head(layer, kv_h) + start * head_dim_;
        const float* V_h  = cache.v_head(layer, kv_h) + start * head_dim_;
        float* out_h      = attn_out_.data() + h * head_dim_;
        float* scores_buf = scores_.data()   + h * cache.max_seq_len();

        qk_scores(q_h, K_h, scores_buf, T_eff, head_dim_, scale);

        if (use_alibi_) {
            const float m = alibi_slope(h);
            for (size_t t = 0; t < T_eff; ++t) {
                const size_t abs_t = start + t;
                scores_buf[t] -= m * float(pos - abs_t);
            }
        }

        softmax_inplace(scores_buf, T_eff);
        av_accumulate(scores_buf, V_h, out_h, T_eff, head_dim_);
    });

    // (5) Output projection.
    matmul_parallel(attn_out_.data(), Wo_, x_out, 1, H, H);
}

} // namespace turbocpp
