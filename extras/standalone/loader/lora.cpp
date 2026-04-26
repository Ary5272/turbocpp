#include "lora.h"
#include "../math/matmul.h"
#include "../math/vec_ops.h"
#include "../utils/logging.h"
#include <cstdio>
#include <cstring>
#include <vector>

namespace turbocpp {

bool LoraFile::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    uint32_t magic, version, n;
    auto rd = [&](void* p, size_t b) { return std::fread(p, 1, b, f) == b; };

    if (!rd(&magic, 4) || magic != kLoraMagic) { std::fclose(f); return false; }
    if (!rd(&version, 4) || version != 1)      { std::fclose(f); return false; }
    if (!rd(&n, 4))                            { std::fclose(f); return false; }

    adapters_.clear();
    adapters_.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        char name[64] = {0};
        uint32_t r;
        float alpha;
        uint64_t indim, outdim;
        if (!rd(name, 64))    { std::fclose(f); return false; }
        if (!rd(&r, 4))       { std::fclose(f); return false; }
        if (!rd(&alpha, 4))   { std::fclose(f); return false; }
        if (!rd(&indim, 8))   { std::fclose(f); return false; }
        if (!rd(&outdim, 8))  { std::fclose(f); return false; }

        LoraAdapter a;
        a.target = name;
        a.r = r;
        a.alpha = alpha;
        a.in_dim = indim;
        a.out_dim = outdim;
        a.A.resize(size_t(r) * size_t(indim));
        a.B.resize(size_t(outdim) * size_t(r));
        if (!rd(a.A.data(), a.A.size() * sizeof(float))) { std::fclose(f); return false; }
        if (!rd(a.B.data(), a.B.size() * sizeof(float))) { std::fclose(f); return false; }
        adapters_.push_back(std::move(a));
    }
    std::fclose(f);
    return true;
}

void lora_merge_into(float* W, const LoraAdapter& a) {
    // W[out, in] += (alpha / r) * B @ A
    // Compute the delta one row at a time to avoid allocating an [out, in]
    // scratch matrix (could be hundreds of MB for big layers).
    const size_t out_dim = size_t(a.out_dim);
    const size_t in_dim  = size_t(a.in_dim);
    const size_t r       = size_t(a.r);
    const float scale = a.alpha / float(r);

    // For each output row n: W[n, :] += scale * B[n, :] @ A
    // B[n, :] is [r], A is [r, in_dim], result is [in_dim].
    AlignedBuffer<float> row(in_dim);
    for (size_t n = 0; n < out_dim; ++n) {
        const float* b_row = a.B.data() + n * r;
        // row = b_row^T @ A, where A is [r, in_dim] row-major.
        // Equivalent to row[j] = sum_k b_row[k] * A[k, j].
        std::memset(row.data(), 0, in_dim * sizeof(float));
        for (size_t k = 0; k < r; ++k) {
            const float coef = b_row[k] * scale;
            const float* a_row = a.A.data() + k * in_dim;
            for (size_t j = 0; j < in_dim; ++j) row.data()[j] += coef * a_row[j];
        }
        // Add into W[n, :].
        vec_add_inplace(W + n * in_dim, row.data(), in_dim);
    }
}

} // namespace turbocpp
