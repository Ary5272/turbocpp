#include "grammar.h"
#include <cctype>
#include <cmath>
#include <limits>

namespace turbocpp {

void TokenAllowlist::apply(float* logits, size_t vocab_size) const {
    if (allowed_.empty()) return;
    constexpr float NEG_INF = -1e30f;
    for (size_t i = 0; i < vocab_size; ++i) {
        if (allowed_.find(int32_t(i)) == allowed_.end()) {
            logits[i] = NEG_INF;
        }
    }
}

// ---------------------------------------------------------------------------
// JsonGrammar — a tiny scanner that approximates JSON validity.
// ---------------------------------------------------------------------------

JsonGrammar::JsonGrammar() { reset(); }

void JsonGrammar::reset() {
    state_ = State::Start;
    depth_ = 0;
    started_ = false;
    in_array_ = false;
    stack_array_.clear();
}

void JsonGrammar::observe(const std::string& piece) {
    for (char c : piece) {
        if (state_ == State::Done) return;
        // Whitespace is always allowed between tokens.
        if (std::isspace(static_cast<unsigned char>(c)) && state_ != State::InString) continue;

        switch (state_) {
            case State::Start:
                if (c == '{' || c == '[') {
                    started_ = true;
                    depth_++;
                    stack_array_.push_back(c == '[');
                    in_array_ = (c == '[');
                    state_ = (c == '{') ? State::InObjectKey : State::BeforeValue;
                } else if (c == '"') {
                    started_ = true;
                    state_ = State::InString;
                } else if (std::isdigit(static_cast<unsigned char>(c)) || c == '-') {
                    started_ = true;
                    state_ = State::InNumber;
                }
                break;

            case State::InObjectKey:
                if (c == '"') state_ = State::InString;
                else if (c == '}') {
                    depth_--;
                    if (!stack_array_.empty()) stack_array_.pop_back();
                    in_array_ = !stack_array_.empty() && stack_array_.back();
                    state_ = depth_ == 0 ? State::Done : State::AfterValue;
                }
                break;

            case State::AfterKey:
                if (c == ':') state_ = State::BeforeValue;
                break;

            case State::BeforeValue:
                if (c == '"') state_ = State::InString;
                else if (c == '{' || c == '[') {
                    depth_++;
                    stack_array_.push_back(c == '[');
                    in_array_ = (c == '[');
                    state_ = (c == '{') ? State::InObjectKey : State::BeforeValue;
                } else if (std::isdigit(static_cast<unsigned char>(c)) || c == '-') {
                    state_ = State::InNumber;
                } else if (c == 't' || c == 'f' || c == 'n') {
                    // true / false / null — minor approximation: accept.
                    state_ = State::AfterValue;
                }
                break;

            case State::InString:
                if (c == '\\') state_ = State::EscapeChar;
                else if (c == '"') {
                    // String done. If we were a key, expect colon next.
                    if (!stack_array_.empty() && !stack_array_.back()) {
                        // We're inside an object — could be key or value.
                        // Heuristic: if previous state was InObjectKey, we
                        // came from a key; else we just finished a value.
                        // We approximate by checking: if next non-ws is ':',
                        // it's a key; we'll find out and transition there.
                        state_ = State::AfterKey;
                    } else {
                        state_ = State::AfterValue;
                    }
                }
                break;

            case State::EscapeChar:
                state_ = State::InString;
                break;

            case State::InNumber:
                if (!std::isdigit(static_cast<unsigned char>(c)) &&
                    c != '.' && c != 'e' && c != 'E' && c != '+' && c != '-') {
                    state_ = State::AfterValue;
                    // Re-process this character.
                    observe(std::string(1, c));
                    return;
                }
                break;

            case State::AfterValue:
                if (c == ',') {
                    state_ = (in_array_) ? State::BeforeValue : State::InObjectKey;
                } else if (c == '}' || c == ']') {
                    depth_--;
                    if (!stack_array_.empty()) stack_array_.pop_back();
                    in_array_ = !stack_array_.empty() && stack_array_.back();
                    if (depth_ == 0) state_ = State::Done;
                }
                break;

            case State::Done: return;
        }
    }
}

bool JsonGrammar::accepts(char c) const {
    if (state_ == State::Done) return false;
    if (std::isspace(static_cast<unsigned char>(c)) && state_ != State::InString) return true;
    switch (state_) {
        case State::Start: return c == '{' || c == '[' || c == '"' ||
                                  std::isdigit(static_cast<unsigned char>(c)) || c == '-';
        case State::InObjectKey: return c == '"' || c == '}';
        case State::AfterKey: return c == ':';
        case State::BeforeValue:
            return c == '"' || c == '{' || c == '[' ||
                   std::isdigit(static_cast<unsigned char>(c)) || c == '-' ||
                   c == 't' || c == 'f' || c == 'n';
        case State::InString:    return true;   // any char inside string
        case State::EscapeChar:  return true;
        case State::InNumber:    return std::isdigit(static_cast<unsigned char>(c)) ||
                                        c == '.' || c == 'e' || c == 'E' ||
                                        c == '+' || c == '-' ||
                                        c == ',' || c == '}' || c == ']';
        case State::AfterValue:  return c == ',' || c == '}' || c == ']';
        case State::Done:        return false;
    }
    return false;
}

void grammar_token_filter(const JsonGrammar& g, size_t vocab_size,
                          const std::function<const std::string&(int32_t)>& id_to_text,
                          TokenAllowlist& out) {
    for (size_t i = 0; i < vocab_size; ++i) {
        const std::string& s = id_to_text(int32_t(i));
        if (s.empty()) continue;
        // Accept token if at least its first char is acceptable.
        if (g.accepts(s[0])) out.allow(int32_t(i));
    }
}

} // namespace turbocpp
