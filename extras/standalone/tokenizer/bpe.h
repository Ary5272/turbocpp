#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace turbocpp {

// ---------------------------------------------------------------------------
// Byte-level BPE tokenizer (GPT-2 / LLaMA family style, simplified).
//
// load_vocab/load_merges: accept tab-separated text files.
//   vocab:  "<token>\t<id>\n"
//   merges: "<a> <b>\n"         (implicit rank = line number)
//
// The encoder implements the canonical BPE algorithm:
//   1. Break text into byte-level symbols.
//   2. Repeatedly find the adjacent pair with LOWEST merge rank, merge it.
//   3. Output token ids for the resulting symbols.
//
// Performance: O(L² log L) worst case where L is the symbol list length.
// Real implementations use a heap of pair positions — left as future work.
// ---------------------------------------------------------------------------

class BPETokenizer {
public:
    BPETokenizer() = default;

    // Load from files.
    bool load(const std::string& vocab_path, const std::string& merges_path);

    // Populate a minimal ASCII-only vocab (bytes 0..255 + a few whitespace
    // merges). Intended for demos and unit tests so the binary runs without
    // an external model.
    void build_minimal();

    // Encode UTF-8 text to token ids.
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode ids back to bytes (UTF-8 string).
    std::string decode(const std::vector<int32_t>& ids) const;

    // Decode a single id (for streaming generation).
    const std::string& id_to_token(int32_t id) const;

    size_t vocab_size() const noexcept { return id_to_token_.size(); }

    // BOS/EOS/PAD/UNK ids. -1 if not set.
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t unk_id = -1;

private:
    // symbol -> id
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string>                 id_to_token_;

    // Merge priorities: lower rank = applied earlier.
    // Key: concatenation of "a" + " " + "b" (space separator unlikely to
    // appear inside byte-level tokens for printable ASCII; safe for this
    // simplified codec).
    std::unordered_map<std::string, int32_t> merge_rank_;

    int32_t add_token(const std::string& tok);
    void    apply_bpe(std::vector<std::string>& symbols) const;
    int32_t lookup_or_unk(const std::string& tok) const;
};

} // namespace turbocpp
