#pragma once
#include <cstddef>
#include "../core/allocator.h"

namespace turbocpp {

// Rotary Position Embedding (Su et al. 2021), LLaMA/NeoX "split-halves"
// variant:
//
//   For head_dim D and position p, rotate pairs (q[i], q[i + D/2]) for
//   i in [0, D/2) by angle theta_i * p where theta_i = base^(-2i/D).
//
// We precompute cos/sin tables for every (position, pair) combination up
// to max_seq_len. Tables are owned by this object for the lifetime of the
// model — one-time cost at load.
class RopeTables {
public:
    RopeTables() = default;

    // head_dim must be even (we rotate pairs). base is usually 10000.
    void build(size_t head_dim, size_t max_seq_len, float base = 10000.0f);

    // Apply rotation in-place. `x` is [n_heads, head_dim] for a single
    // token at absolute position `pos`. Used for both Q and K after their
    // projections.
    void apply(float* x, size_t n_heads, size_t pos) const;

    size_t head_dim()    const noexcept { return head_dim_; }
    size_t max_seq_len() const noexcept { return max_seq_len_; }

private:
    AlignedBuffer<float> cos_;  // [max_seq_len, head_dim/2]
    AlignedBuffer<float> sin_;  // [max_seq_len, head_dim/2]
    size_t head_dim_ = 0;
    size_t max_seq_len_ = 0;
};

} // namespace turbocpp
