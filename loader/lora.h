#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "../core/allocator.h"

namespace turbocpp {

// LoRA adapter (Low-Rank Adaptation, Hu et al. 2021).
//
//     W' = W + (alpha / r) * B @ A
//
// where A is [r, in_dim] and B is [out_dim, r]. r is the rank (typically
// 4-64). At inference we merge the delta into the base weights — adds
// (alpha/r) * out_dim * in_dim flops per merge but ZERO overhead per
// forward (vs the LoRA-on-the-fly path). The downside is you can't switch
// adapters without re-merging.
//
// Multi-adapter blending: caller can apply several adapters to the same
// base weight; deltas are additive.
//
// Format (.tlora):
//   [magic 'TLOR'][version u32][n u32]
//   for each adapter:
//     [target_name char[64]][r u32][alpha f32]
//     [in_dim u64][out_dim u64]
//     [A: r * in_dim * f32]
//     [B: out_dim * r * f32]

constexpr uint32_t kLoraMagic = 0x524F4C54u;     // "TLOR"

struct LoraAdapter {
    std::string target;        // tensor name in the base model (e.g. "layers.0.Wq")
    uint32_t r = 0;
    float    alpha = 1.0f;
    uint64_t in_dim = 0;
    uint64_t out_dim = 0;
    AlignedBuffer<float> A;    // [r, in_dim]
    AlignedBuffer<float> B;    // [out_dim, r]
};

class LoraFile {
public:
    LoraFile() = default;
    bool load(const std::string& path);
    const std::vector<LoraAdapter>& adapters() const noexcept { return adapters_; }
private:
    std::vector<LoraAdapter> adapters_;
};

// Merge an adapter's delta into a writable weight buffer:
//   W[out, in] += (alpha / r) * B @ A
// W is [out_dim, in_dim] row-major. A is [r, in_dim]. B is [out_dim, r].
void lora_merge_into(float* W, const LoraAdapter& a);

} // namespace turbocpp
