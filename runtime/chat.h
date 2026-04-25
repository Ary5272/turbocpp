#pragma once
#include <string>
#include <vector>

namespace turbocpp {

// Chat templates and stop sequences.
//
// Supported templates:
//   PLAIN     — concat raw turns, no markers
//   LLAMA2    — [INST] {sys}\n\n{user} [/INST] {asst} </s>
//   LLAMA3    — <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{msg}<|eot_id|>...
//   CHATML    — <|im_start|>{role}\n{msg}<|im_end|>...
//   MISTRAL   — Same as LLAMA2 but no <s>/</s> wrappers
//
// Auto-detect from a name string ("llama3", "mistral-instruct", "qwen", ...).

enum class ChatTemplate { Plain, LLaMA2, LLaMA3, ChatML, Mistral };

struct ChatMessage {
    std::string role;     // "system" | "user" | "assistant"
    std::string content;
};

ChatTemplate parse_template(const std::string& name);

// Render a list of messages into a prompt string ending at the assistant's
// turn (i.e. the model continues from there).
std::string apply_template(ChatTemplate t, const std::vector<ChatMessage>& msgs);

// Stop-sequence matcher. Tracks a rolling buffer of generated text and
// reports when any of `seqs` has been written. Use append() per generated
// piece; check stopped() / matched_seq() after.
class StopMatcher {
public:
    StopMatcher() = default;
    explicit StopMatcher(std::vector<std::string> seqs);
    void set_sequences(std::vector<std::string> seqs);
    void reset();

    // Append more text. Returns the new tail length to keep — caller can
    // truncate visible output to (visible_len - back_off) when stopped().
    void append(const std::string& piece);

    bool stopped() const noexcept { return matched_ >= 0; }
    int  matched_seq() const noexcept { return matched_; }
    // Length of accumulated text BEFORE the matched stop sequence.
    size_t cut() const noexcept { return cut_; }

private:
    std::vector<std::string> seqs_;
    std::string buf_;
    int matched_ = -1;
    size_t cut_ = 0;
    size_t max_seq_len_ = 0;
};

} // namespace turbocpp
