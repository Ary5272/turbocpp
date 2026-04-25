#include "kv_cache.h"
#include "../utils/logging.h"
#include <cstdio>
#include <cstring>

namespace turbocpp {

void KVCache::init(size_t n_layers, size_t n_heads, size_t head_dim, size_t max_seq_len) {
    n_layers_ = n_layers;
    n_heads_ = n_heads;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    cur_len_ = 0;

    const size_t total = n_layers * n_heads * max_seq_len * head_dim;
    k_.resize(total);
    v_.resize(total);
    std::memset(k_.data(), 0, total * sizeof(float));
    std::memset(v_.data(), 0, total * sizeof(float));
}

void KVCache::append(size_t layer, size_t pos, const float* k_proj, const float* v_proj) {
    TCPP_CHECK(pos < max_seq_len_, "kv_cache append pos %zu exceeds max %zu", pos, max_seq_len_);
    for (size_t h = 0; h < n_heads_; ++h) {
        float* kdst = k_head(layer, h) + pos * head_dim_;
        float* vdst = v_head(layer, h) + pos * head_dim_;
        std::memcpy(kdst, k_proj + h * head_dim_, head_dim_ * sizeof(float));
        std::memcpy(vdst, v_proj + h * head_dim_, head_dim_ * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Snapshot format (little-endian, x86):
//   [magic:u32 = 'KVS1'][prompt_hash:u64]
//   [n_layers:u64][n_heads:u64][head_dim:u64][max_seq_len:u64][cur_len:u64]
//   [K data: n_layers * n_heads * cur_len * head_dim * f32]
//   [V data: same]
// ---------------------------------------------------------------------------
constexpr uint32_t kKVSnapMagic = 0x3153564Bu;  // "KVS1"

bool KVCache::save_snapshot(const std::string& path, uint64_t prompt_hash) const {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    auto wr = [&](const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
    bool ok = true;
    uint32_t magic = kKVSnapMagic;
    ok &= wr(&magic, sizeof magic);
    ok &= wr(&prompt_hash, sizeof prompt_hash);
    uint64_t v;
    v = n_layers_;    ok &= wr(&v, sizeof v);
    v = n_heads_;     ok &= wr(&v, sizeof v);
    v = head_dim_;    ok &= wr(&v, sizeof v);
    v = max_seq_len_; ok &= wr(&v, sizeof v);
    v = cur_len_;     ok &= wr(&v, sizeof v);

    // Per-(layer, head): write the live [cur_len, head_dim] tile rather than
    // the entire reserved [max_seq, head_dim] block.
    for (size_t L = 0; L < n_layers_; ++L) {
        for (size_t h = 0; h < n_heads_; ++h) {
            ok &= wr(k_head(L, h), cur_len_ * head_dim_ * sizeof(float));
        }
    }
    for (size_t L = 0; L < n_layers_; ++L) {
        for (size_t h = 0; h < n_heads_; ++h) {
            ok &= wr(v_head(L, h), cur_len_ * head_dim_ * sizeof(float));
        }
    }
    std::fclose(f);
    return ok;
}

bool KVCache::load_snapshot(const std::string& path, uint64_t prompt_hash) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    auto rd = [&](void* p, size_t n) { return std::fread(p, 1, n, f) == n; };
    bool ok = true;
    uint32_t magic = 0; ok &= rd(&magic, sizeof magic);
    if (!ok || magic != kKVSnapMagic) { std::fclose(f); return false; }

    uint64_t saved_hash = 0; ok &= rd(&saved_hash, sizeof saved_hash);
    if (saved_hash != prompt_hash) { std::fclose(f); return false; }

    uint64_t L, H, D, M, C;
    ok &= rd(&L, 8); ok &= rd(&H, 8); ok &= rd(&D, 8); ok &= rd(&M, 8); ok &= rd(&C, 8);
    if (!ok) { std::fclose(f); return false; }
    if (L != n_layers_ || H != n_heads_ || D != head_dim_ || M > max_seq_len_) {
        std::fclose(f); return false;
    }
    if (C > max_seq_len_) { std::fclose(f); return false; }

    for (size_t l = 0; l < n_layers_ && ok; ++l)
        for (size_t h = 0; h < n_heads_ && ok; ++h)
            ok &= rd(k_head(l, h), C * head_dim_ * sizeof(float));
    for (size_t l = 0; l < n_layers_ && ok; ++l)
        for (size_t h = 0; h < n_heads_ && ok; ++h)
            ok &= rd(v_head(l, h), C * head_dim_ * sizeof(float));

    std::fclose(f);
    if (ok) cur_len_ = size_t(C);
    return ok;
}

} // namespace turbocpp
