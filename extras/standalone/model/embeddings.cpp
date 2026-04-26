#include "embeddings.h"
#include "../utils/logging.h"
#include <cstring>

namespace turbocpp {

void embed_single(const float* embed_table, int32_t token_id,
                  float* out, size_t dim, size_t vocab_size) {
    TCPP_CHECK(token_id >= 0 && size_t(token_id) < vocab_size,
               "embed: token %d out of vocab %zu", token_id, vocab_size);
    // memcpy is essentially free for dim=4096 floats (16KB) on modern CPUs —
    // ~100ns, well within L1.
    std::memcpy(out, embed_table + size_t(token_id) * dim, dim * sizeof(float));
}

void embed_tokens(const float* embed_table, const int32_t* token_ids,
                  float* out, size_t n_tokens, size_t dim, size_t vocab_size) {
    for (size_t t = 0; t < n_tokens; ++t) {
        embed_single(embed_table, token_ids[t], out + t * dim, dim, vocab_size);
    }
}

} // namespace turbocpp
