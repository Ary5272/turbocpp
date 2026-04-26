#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

namespace turbocpp {

// Constrained sampling — token-level grammar enforcement.
//
// Two modes:
//
//   1. ALLOWLIST: a fixed set of allowed token ids. Logits for any other
//      id are set to -INF before sampling. Useful for force-pinning a
//      schema (e.g. "yes" or "no" only).
//
//   2. JSON: a tiny state-machine that tracks JSON depth and only allows
//      tokens whose decoded text could continue a valid JSON document.
//      Approximate (not full schema validation) but sufficient to keep
//      the model on rails for tool calls / function output.
//
// The full GBNF grammar engine from llama.cpp is much larger; this is the
// 80/20 version that handles the common cases.

class TokenAllowlist {
public:
    void allow(int32_t id) { allowed_.insert(id); }
    void allow_many(const std::vector<int32_t>& ids) {
        for (auto id : ids) allowed_.insert(id);
    }
    bool empty() const noexcept { return allowed_.empty(); }
    // Mask logits in-place: any token not in `allowed_` becomes -INF.
    void apply(float* logits, size_t vocab_size) const;

private:
    std::unordered_set<int32_t> allowed_;
};

// JSON-mode constraint. Feed each generated token's decoded text to
// observe(); query allowed_first_chars() to decide which characters can
// start the next token. Pair with the tokenizer to build a per-step
// allowlist.
class JsonGrammar {
public:
    JsonGrammar();
    void reset();

    // Update grammar state with a fresh piece of decoded text.
    void observe(const std::string& piece);

    // Returns true if the current state can accept this character as the
    // next byte. Use to filter tokens via id_to_token check.
    bool accepts(char c) const;

    bool finished() const noexcept { return depth_ == 0 && started_; }

private:
    enum class State {
        Start, AfterValue, InObjectKey, AfterKey, InString, InNumber,
        AfterColon, BeforeValue, EscapeChar, Done
    };
    State state_ = State::Start;
    int depth_ = 0;
    bool started_ = false;
    bool in_array_ = false;
    std::vector<bool> stack_array_;  // true=in array, false=in object
};

// Build a TokenAllowlist for the next position from a JsonGrammar by
// asking each vocab entry whether its first byte is accepted. O(vocab).
// `id_to_text` is invoked once per id; cheap in practice for byte-level BPE.
void grammar_token_filter(const JsonGrammar& g, size_t vocab_size,
                          const std::function<const std::string&(int32_t)>& id_to_text,
                          TokenAllowlist& out);

} // namespace turbocpp
