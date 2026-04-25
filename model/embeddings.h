#pragma once
#include <cstddef>
#include <cstdint>

namespace turbocpp {

// Token embedding: for a sequence of token ids, gather the corresponding
// rows of the embedding matrix into `out`.
//
//   embed_table: [vocab_size, dim], row-major
//   token_ids:   [n_tokens]
//   out:         [n_tokens, dim]
void embed_tokens(const float* embed_table, const int32_t* token_ids,
                  float* out, size_t n_tokens, size_t dim, size_t vocab_size);

// Single-token convenience for the generation hot path.
void embed_single(const float* embed_table, int32_t token_id,
                  float* out, size_t dim, size_t vocab_size);

} // namespace turbocpp
