#include "bpe.h"
#include "../utils/logging.h"
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>

namespace turbocpp {

static std::string kEmptyToken;

int32_t BPETokenizer::add_token(const std::string& tok) {
    auto it = token_to_id_.find(tok);
    if (it != token_to_id_.end()) return it->second;
    int32_t id = int32_t(id_to_token_.size());
    id_to_token_.push_back(tok);
    token_to_id_.emplace(tok, id);
    return id;
}

int32_t BPETokenizer::lookup_or_unk(const std::string& tok) const {
    auto it = token_to_id_.find(tok);
    if (it != token_to_id_.end()) return it->second;
    return unk_id;
}

const std::string& BPETokenizer::id_to_token(int32_t id) const {
    if (id < 0 || size_t(id) >= id_to_token_.size()) return kEmptyToken;
    return id_to_token_[size_t(id)];
}

// ---------------------------------------------------------------------------
// Vocab/merges loading
// ---------------------------------------------------------------------------

bool BPETokenizer::load(const std::string& vocab_path,
                        const std::string& merges_path) {
    token_to_id_.clear();
    id_to_token_.clear();
    merge_rank_.clear();

    // Vocab: one per line, "token\tid".
    std::ifstream vf(vocab_path);
    if (!vf) { LOG_ERROR("cannot open vocab: %s", vocab_path.c_str()); return false; }
    std::string line;
    while (std::getline(vf, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string tok = line.substr(0, tab);
        int32_t id = std::stoi(line.substr(tab + 1));
        if (size_t(id) >= id_to_token_.size()) id_to_token_.resize(size_t(id) + 1);
        id_to_token_[size_t(id)] = tok;
        token_to_id_[tok] = id;
    }

    // Merges: "a b" per line, rank = line number.
    std::ifstream mf(merges_path);
    if (!mf) { LOG_ERROR("cannot open merges: %s", merges_path.c_str()); return false; }
    int32_t rank = 0;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string a = line.substr(0, sp);
        std::string b = line.substr(sp + 1);
        merge_rank_[a + " " + b] = rank++;
    }

    // Resolve special ids by name if present.
    auto find_id = [&](const std::string& n) -> int32_t {
        auto it = token_to_id_.find(n);
        return it == token_to_id_.end() ? -1 : it->second;
    };
    bos_id = find_id("<s>");
    eos_id = find_id("</s>");
    unk_id = find_id("<unk>");
    return true;
}

// ---------------------------------------------------------------------------
// Minimal built-in vocab: byte-level, plus a few whitespace merges so
// "hello world" produces a sensible token sequence in the demo.
// ---------------------------------------------------------------------------
void BPETokenizer::build_minimal() {
    token_to_id_.clear();
    id_to_token_.clear();
    merge_rank_.clear();

    // Reserve first 4 ids for special tokens.
    add_token("<pad>"); // 0
    bos_id = add_token("<s>");   // 1
    eos_id = add_token("</s>");  // 2
    unk_id = add_token("<unk>"); // 3

    // Byte vocab: single-char strings for bytes 0x20..0x7E (printable ASCII)
    // plus a small set of common control chars.
    for (int b = 0x20; b <= 0x7E; ++b) {
        std::string s(1, char(b));
        add_token(s);
    }
    add_token("\n");
    add_token("\t");

    // A handful of useful merges so common English text compresses a bit.
    // Rank ordering matters: earlier merges take priority.
    const char* pairs[] = {
        "t h", "h e", "i n", "e r", "a n", "o n",   // bigrams
        "re ",  "nd ",  " t", " a", " w",          // fragments
        "th e", "a nd", "o f",                     // trigrams via nested merges
    };
    int rank = 0;
    for (const char* p : pairs) {
        merge_rank_[std::string(p)] = rank++;
    }

    // Ensure every token referenced in merges exists in vocab (merge creates
    // a new symbol by concatenation).
    for (const auto& kv : merge_rank_) {
        const std::string& k = kv.first;
        auto sp = k.find(' ');
        if (sp == std::string::npos) continue;
        std::string a = k.substr(0, sp);
        std::string b = k.substr(sp + 1);
        add_token(a + b);
    }
}

// ---------------------------------------------------------------------------
// BPE encode inner loop: find the adjacent pair with lowest merge rank and
// merge it; repeat until no pair matches a merge.
// ---------------------------------------------------------------------------
void BPETokenizer::apply_bpe(std::vector<std::string>& symbols) const {
    if (symbols.size() < 2) return;

    while (true) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_i = 0;
        bool found = false;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = merge_rank_.find(symbols[i] + " " + symbols[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
                found = true;
            }
        }
        if (!found) break;

        // Merge symbols[best_i] and symbols[best_i + 1].
        symbols[best_i] = symbols[best_i] + symbols[best_i + 1];
        symbols.erase(symbols.begin() + std::ptrdiff_t(best_i + 1));
    }
}

std::vector<int32_t> BPETokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;
    if (text.empty()) return out;

    // Byte-level: one-char symbols.
    std::vector<std::string> sym;
    sym.reserve(text.size());
    for (unsigned char c : text) sym.emplace_back(1, char(c));

    apply_bpe(sym);

    out.reserve(sym.size());
    for (const auto& s : sym) {
        int32_t id = lookup_or_unk(s);
        if (id < 0 && !s.empty()) {
            // Fallback: emit each byte as its single-char token.
            for (char c : s) {
                int32_t bid = lookup_or_unk(std::string(1, c));
                out.push_back(bid >= 0 ? bid : unk_id);
            }
        } else {
            out.push_back(id);
        }
    }
    return out;
}

std::string BPETokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string s;
    s.reserve(ids.size() * 4);
    for (int32_t id : ids) {
        if (id < 0 || size_t(id) >= id_to_token_.size()) continue;
        const std::string& tok = id_to_token_[size_t(id)];
        // Skip special tokens in visible output.
        if (!tok.empty() && tok[0] == '<' && tok.back() == '>') continue;
        s += tok;
    }
    return s;
}

} // namespace turbocpp
