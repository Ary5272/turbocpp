#pragma once
#include <cstddef>

namespace turbocpp {

// 32-byte alignment supports AVX/AVX2 (256-bit) loads/stores.
// AVX-512 would need 64, but we target AVX2-first for broad compatibility.
constexpr size_t kAlign = 32;

// Cache line is 64 bytes on all x86-64 we target; used for padding between
// per-thread scratch slots to avoid false sharing.
constexpr size_t kCacheLine = 64;

// Tile parameters empirically tuned for L1 (32KB) / L2 (256KB-1MB).
// MC*KC*4 bytes fits in L2; KC*NC*4 is a panel of B in L3.
constexpr size_t kMC = 64;
constexpr size_t kNC = 128;
constexpr size_t kKC = 256;

// Rounds up x to the next multiple of `a` (a must be a power of two).
inline size_t round_up(size_t x, size_t a) noexcept {
    return (x + a - 1) & ~(a - 1);
}

} // namespace turbocpp
