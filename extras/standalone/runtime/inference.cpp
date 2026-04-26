#include "inference.h"
#include "chat.h"
#include "../utils/timing.h"
#include "../utils/logging.h"

namespace turbocpp {

void InferenceEngine::init(Model& model, BPETokenizer& tok) {
    model_ = &model;
    tok_   = &tok;
    const auto& cfg = model.config();
    // KV cache sized for n_kv_heads (not n_heads) — GQA savings.
    cache_.init(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, cfg.max_seq_len);
    logits_.resize(cfg.vocab_size);
}

void InferenceEngine::reset() { cache_.clear(); }

std::string InferenceEngine::generate(const std::string& prompt,
                                      const GenerateOptions& opts,
                                      GenerateStats* out_stats) {
    TCPP_CHECK(model_ && tok_, "inference: init() not called");

    sampler_.set_params(opts.sampling);
    sampler_.clear_history();

    // -----------------------------------------------------------------------
    // Encode prompt
    // -----------------------------------------------------------------------
    std::vector<int32_t> tokens = tok_->encode(prompt);
    if (tokens.empty()) {
        // Empty prompt -> start from BOS if available.
        if (tok_->bos_id >= 0) tokens.push_back(tok_->bos_id);
    }

    const auto& cfg = model_->config();
    TCPP_CHECK(tokens.size() < cfg.max_seq_len,
               "prompt length %zu exceeds max_seq_len %zu",
               tokens.size(), cfg.max_seq_len);

    std::string out_text;
    out_text.reserve(opts.max_new_tokens * 4);

    GenerateStats stats;
    stats.prompt_tokens = tokens.size();

    // -----------------------------------------------------------------------
    // Prefill: feed each prompt token one by one. We run a single-token
    // forward per step — a real engine could batch prompt tokens into a
    // single forward for better arithmetic intensity. Deferred.
    // -----------------------------------------------------------------------
    size_t pos = cache_.cur_len();
    {
        Timer t;
        for (size_t i = 0; i < tokens.size(); ++i) {
            model_->forward(tokens[i], pos + i, cache_, logits_.data());
            sampler_.record(tokens[i]);   // feed prompt into rep-penalty history
        }
        stats.prefill_ms = t.ms();
        pos += tokens.size();
    }

    // -----------------------------------------------------------------------
    // Decode loop
    // -----------------------------------------------------------------------
    StopMatcher stopper(opts.stop_sequences);
    {
        Timer t;
        for (size_t g = 0; g < opts.max_new_tokens; ++g) {
            const int32_t next = sampler_.sample(logits_.data(), cfg.vocab_size);

            if (opts.stop_on_eos && next == tok_->eos_id) break;

            const std::string& piece = tok_->id_to_token(next);
            const bool is_special = (piece.size() >= 2 &&
                                     piece.front() == '<' && piece.back() == '>');
            if (!is_special) out_text += piece;

            // Stop sequence check.
            if (!opts.stop_sequences.empty()) {
                stopper.append(piece);
                if (stopper.stopped()) {
                    // Trim out_text back to before the stop sequence text.
                    // We track only by the last cut: an upper bound that matches
                    // most cases (single-stop, single-pass).
                    const auto& seq = opts.stop_sequences[size_t(stopper.matched_seq())];
                    auto pos_in_out = out_text.rfind(seq);
                    if (pos_in_out != std::string::npos) out_text.resize(pos_in_out);
                    break;
                }
            }

            if (opts.on_token && !opts.on_token(next, piece)) break;
            ++stats.generated_tokens;

            if (pos + 1 >= cfg.max_seq_len) break;

            model_->forward(next, pos, cache_, logits_.data());
            sampler_.record(next);
            ++pos;
        }
        stats.decode_ms = t.ms();
    }

    if (out_stats) *out_stats = stats;
    return out_text;
}

} // namespace turbocpp
