#include "chat.h"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace turbocpp {

static std::string lower(std::string s) {
    for (auto& c : s) c = char(std::tolower(unsigned(c)));
    return s;
}

ChatTemplate parse_template(const std::string& name_in) {
    const std::string n = lower(name_in);
    if (n.empty() || n == "plain" || n == "raw") return ChatTemplate::Plain;
    if (n.find("llama-3") != std::string::npos || n == "llama3" || n.find("llama_3") != std::string::npos)
        return ChatTemplate::LLaMA3;
    if (n.find("llama-2") != std::string::npos || n == "llama2" || n.find("llama_2") != std::string::npos)
        return ChatTemplate::LLaMA2;
    if (n.find("chatml") != std::string::npos || n.find("qwen") != std::string::npos)
        return ChatTemplate::ChatML;
    if (n.find("mistral") != std::string::npos || n.find("mixtral") != std::string::npos)
        return ChatTemplate::Mistral;
    return ChatTemplate::Plain;
}

static const std::string& role_or_empty(const std::vector<ChatMessage>& m, size_t i) {
    static const std::string empty;
    return i < m.size() ? m[i].role : empty;
}

std::string apply_template(ChatTemplate t, const std::vector<ChatMessage>& msgs) {
    std::ostringstream out;
    switch (t) {
        case ChatTemplate::LLaMA3: {
            out << "<|begin_of_text|>";
            for (const auto& m : msgs) {
                out << "<|start_header_id|>" << m.role << "<|end_header_id|>\n\n"
                    << m.content << "<|eot_id|>";
            }
            // Open assistant turn for the model to continue.
            out << "<|start_header_id|>assistant<|end_header_id|>\n\n";
            return out.str();
        }
        case ChatTemplate::ChatML: {
            for (const auto& m : msgs) {
                out << "<|im_start|>" << m.role << "\n" << m.content << "<|im_end|>\n";
            }
            out << "<|im_start|>assistant\n";
            return out.str();
        }
        case ChatTemplate::LLaMA2: {
            // <s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST] {asst}</s><s>[INST]...
            std::string sys;
            for (const auto& m : msgs) if (m.role == "system") { sys = m.content; break; }
            out << "<s>[INST] ";
            if (!sys.empty()) out << "<<SYS>>\n" << sys << "\n<</SYS>>\n\n";
            bool first = true;
            for (const auto& m : msgs) {
                if (m.role == "system") continue;
                if (m.role == "user") {
                    if (!first) out << "<s>[INST] ";
                    out << m.content << " [/INST]";
                    first = false;
                } else if (m.role == "assistant") {
                    out << " " << m.content << " </s>";
                }
            }
            return out.str();
        }
        case ChatTemplate::Mistral: {
            // [INST] sys + user [/INST] asst </s>[INST] user [/INST] ...
            std::string sys;
            for (const auto& m : msgs) if (m.role == "system") { sys = m.content; break; }
            for (size_t i = 0; i < msgs.size(); ++i) {
                const auto& m = msgs[i];
                if (m.role == "system") continue;
                if (m.role == "user") {
                    out << "[INST] ";
                    if (i == 1 && !sys.empty()) out << sys << "\n\n";
                    out << m.content << " [/INST]";
                } else if (m.role == "assistant") {
                    out << " " << m.content << "</s>";
                }
            }
            return out.str();
        }
        default: {
            for (const auto& m : msgs) out << m.role << ": " << m.content << "\n";
            out << "assistant: ";
            return out.str();
        }
    }
}

// ---------------------------------------------------------------------------
// StopMatcher
// ---------------------------------------------------------------------------

StopMatcher::StopMatcher(std::vector<std::string> seqs) {
    set_sequences(std::move(seqs));
}

void StopMatcher::set_sequences(std::vector<std::string> seqs) {
    seqs_ = std::move(seqs);
    max_seq_len_ = 0;
    for (const auto& s : seqs_) if (s.size() > max_seq_len_) max_seq_len_ = s.size();
    reset();
}

void StopMatcher::reset() {
    buf_.clear();
    matched_ = -1;
    cut_ = 0;
}

void StopMatcher::append(const std::string& piece) {
    if (matched_ >= 0) return;
    buf_ += piece;
    for (size_t i = 0; i < seqs_.size(); ++i) {
        const auto& s = seqs_[i];
        auto pos = buf_.find(s);
        if (pos != std::string::npos) {
            matched_ = int(i);
            cut_ = pos;
            return;
        }
    }
    // Trim buf_: keep only the last (max_seq_len_-1) chars — enough to
    // detect a partially-arrived stop on the next append.
    if (max_seq_len_ > 0 && buf_.size() > max_seq_len_) {
        buf_.erase(0, buf_.size() - (max_seq_len_ - 1));
    }
}

} // namespace turbocpp
