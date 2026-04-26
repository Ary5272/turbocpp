#include "qk.h"
#include "../utils/logging.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace turbocpp {

size_t qk_num_blocks(size_t n) { return n / kKBlockSize; }

// ---------------------------------------------------------------------------
// Q8_K (simplest: bookend the file with it)
// ---------------------------------------------------------------------------
static void q8k_quantize_block(const float* x, Q8KBlock& out) {
    float maxabs = 0;
    for (size_t i = 0; i < kKBlockSize; ++i)
        maxabs = std::max(maxabs, std::fabs(x[i]));
    const float d  = (maxabs > 0) ? (maxabs / 127.0f) : 0;
    const float id = (d > 0) ? 1.0f / d : 0;
    out.d = d;
    for (size_t i = 0; i < kKBlockSize; ++i) {
        int q = int(std::lround(x[i] * id));
        out.qs[i] = int8_t(std::clamp(q, -127, 127));
    }
}
static void q8k_dequantize_block(const Q8KBlock& blk, float* out) {
    for (size_t i = 0; i < kKBlockSize; ++i) out[i] = blk.d * float(blk.qs[i]);
}

void q8k_quantize(const float* x, Q8KBlock* out, size_t n) {
    TCPP_CHECK(n % kKBlockSize == 0, "q8k: n must be %256");
    for (size_t b = 0; b < n / kKBlockSize; ++b)
        q8k_quantize_block(x + b * kKBlockSize, out[b]);
}
void q8k_dequantize(const Q8KBlock* blocks, float* out, size_t n) {
    TCPP_CHECK(n % kKBlockSize == 0, "q8k: n must be %256");
    for (size_t b = 0; b < n / kKBlockSize; ++b)
        q8k_dequantize_block(blocks[b], out + b * kKBlockSize);
}
void matmul_q8k(const float* A, const Q8KBlock* W, float* C,
                size_t M, size_t N, size_t K) {
    TCPP_CHECK(K % kKBlockSize == 0, "matmul_q8k: K must be %256");
    const size_t bpr = K / kKBlockSize;
    for (size_t m = 0; m < M; ++m) {
        const float* a = A + m * K;
        for (size_t n = 0; n < N; ++n) {
            const Q8KBlock* w = W + n * bpr;
            float acc = 0;
            for (size_t b = 0; b < bpr; ++b) {
                const float* ab = a + b * kKBlockSize;
                float s = 0;
                for (size_t i = 0; i < kKBlockSize; ++i) s += ab[i] * float(w[b].qs[i]);
                acc += w[b].d * s;
            }
            C[m * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Q6_K (6-bit, 16-element sub-blocks)
// ---------------------------------------------------------------------------
static void q6k_quantize_block(const float* x, Q6KBlock& out) {
    // 16 sub-blocks of 16 elements. Per-sub-block: scale by max_abs / 31.
    // Master d = max(|sub_d|) / 127; sub_d[k] = round(scale_k / d).
    float sub_scales[16];
    float maxabs_global = 0;
    for (int k = 0; k < 16; ++k) {
        float maxa = 0;
        for (int i = 0; i < 16; ++i) maxa = std::max(maxa, std::fabs(x[k * 16 + i]));
        sub_scales[k] = maxa / 31.0f;
        maxabs_global = std::max(maxabs_global, sub_scales[k]);
    }
    const float d  = (maxabs_global > 0) ? (maxabs_global / 127.0f) : 0;
    const float id = (d > 0) ? 1.0f / d : 0;
    out.d = d;
    std::memset(out.q_low, 0, sizeof out.q_low);
    std::memset(out.q_high, 0, sizeof out.q_high);
    for (int k = 0; k < 16; ++k) {
        out.sub_d[k] = int8_t(std::clamp(int(std::lround(sub_scales[k] * id)), -127, 127));
        const float sd = d * float(out.sub_d[k]);
        const float isd = (sd != 0) ? 1.0f / sd : 0;
        for (int i = 0; i < 16; ++i) {
            int q = int(std::lround(x[k * 16 + i] * isd)) + 32;
            q = std::clamp(q, 0, 63);
            const size_t idx = size_t(k * 16 + i);
            // q_low: low 4 bits, packed two-per-byte
            // q_high: high 2 bits, packed four-per-byte
            const uint8_t low4  = uint8_t(q & 0xF);
            const uint8_t high2 = uint8_t((q >> 4) & 0x3);
            if ((idx & 1) == 0) out.q_low[idx / 2]  |= low4;
            else                out.q_low[idx / 2]  |= uint8_t(low4 << 4);
            const size_t hbyte = idx / 4;
            const size_t hshift = (idx & 3) * 2;
            out.q_high[hbyte] |= uint8_t(high2 << hshift);
        }
    }
}

static void q6k_dequantize_block(const Q6KBlock& blk, float* out) {
    for (int k = 0; k < 16; ++k) {
        const float sd = blk.d * float(blk.sub_d[k]);
        for (int i = 0; i < 16; ++i) {
            const size_t idx = size_t(k * 16 + i);
            uint8_t low4 = ((idx & 1) == 0) ? (blk.q_low[idx / 2] & 0xF)
                                            : (blk.q_low[idx / 2] >> 4);
            uint8_t high2 = uint8_t((blk.q_high[idx / 4] >> ((idx & 3) * 2)) & 0x3);
            int q = int(low4 | (high2 << 4)) - 32;
            out[idx] = sd * float(q);
        }
    }
}

void q6k_quantize(const float* x, Q6KBlock* out, size_t n) {
    TCPP_CHECK(n % kKBlockSize == 0, "q6k: n must be %256");
    for (size_t b = 0; b < n / kKBlockSize; ++b)
        q6k_quantize_block(x + b * kKBlockSize, out[b]);
}
void q6k_dequantize(const Q6KBlock* blocks, float* out, size_t n) {
    TCPP_CHECK(n % kKBlockSize == 0, "q6k: n must be %256");
    for (size_t b = 0; b < n / kKBlockSize; ++b)
        q6k_dequantize_block(blocks[b], out + b * kKBlockSize);
}
void matmul_q6k(const float* A, const Q6KBlock* W, float* C,
                size_t M, size_t N, size_t K) {
    TCPP_CHECK(K % kKBlockSize == 0, "matmul_q6k: K must be %256");
    const size_t bpr = K / kKBlockSize;
    float scratch[kKBlockSize];
    for (size_t m = 0; m < M; ++m) {
        const float* a = A + m * K;
        for (size_t n = 0; n < N; ++n) {
            const Q6KBlock* w = W + n * bpr;
            float acc = 0;
            for (size_t b = 0; b < bpr; ++b) {
                q6k_dequantize_block(w[b], scratch);
                const float* ab = a + b * kKBlockSize;
                float s = 0;
                for (size_t i = 0; i < kKBlockSize; ++i) s += ab[i] * scratch[i];
                acc += s;
            }
            C[m * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_K_M (4.5 bpw)
// ---------------------------------------------------------------------------
// 8 sub-blocks of 32. Per-sub-block: 6-bit scale + 6-bit min, packed.
// Quants are 4-bit, asymmetric: x = d * scale * q + dmin * min.

// scales[12] layout (matching llama.cpp):
//   bytes 0..3 hold the low 6 bits of scales 0..3
//   bytes 4..7 hold the low 6 bits of mins   0..3
//   bytes 8..11 are the high 2 bits of scales 0..3 + mins 0..3 + scales 4..7 + mins 4..7
// Practically: pack 8 (scale, min) pairs of 6 bits each into 12 bytes (96 bits).
static inline void pack_scale_min(uint8_t* out, int idx, uint8_t s6, uint8_t m6) {
    // Use the simpler "6 bits scales[8] then 6 bits mins[8]" packing — diverges
    // from llama.cpp exact bit order but stays self-consistent.
    // Total: 8×6 + 8×6 = 96 bits = 12 bytes. We use a straight little-endian
    // bit-stream.
    auto setbits = [&](int bit_off, uint8_t val) {
        for (int b = 0; b < 6; ++b) {
            int p = bit_off + b;
            if (val & (1 << b)) out[p / 8] |= uint8_t(1 << (p % 8));
        }
    };
    setbits(idx * 6,        s6 & 0x3F);
    setbits(48 + idx * 6,   m6 & 0x3F);
}
static inline void unpack_scale_min(const uint8_t* in, int idx, uint8_t& s6, uint8_t& m6) {
    auto getbits = [&](int bit_off) -> uint8_t {
        uint8_t v = 0;
        for (int b = 0; b < 6; ++b) {
            int p = bit_off + b;
            if (in[p / 8] & (1 << (p % 8))) v |= uint8_t(1 << b);
        }
        return v;
    };
    s6 = getbits(idx * 6);
    m6 = getbits(48 + idx * 6);
}

static void q4k_quantize_block(const float* x, Q4KBlock& out) {
    // Per-sub-block: find min, max → scale = (max - min) / 15, min stored.
    float scales[8], mins[8];
    for (int k = 0; k < 8; ++k) {
        float mn = x[k * 32], mx = mn;
        for (int i = 1; i < 32; ++i) {
            float v = x[k * 32 + i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        scales[k] = (mx - mn) / 15.0f;
        mins[k]   = mn;
    }
    float maxs = 0, maxm = 0;
    for (int k = 0; k < 8; ++k) {
        maxs = std::max(maxs, scales[k]);
        maxm = std::max(maxm, std::fabs(mins[k]));
    }
    const float d    = maxs / 63.0f;
    const float dmin = maxm / 63.0f;
    out.d    = d;
    out.dmin = dmin;
    std::memset(out.scales, 0, sizeof out.scales);
    std::memset(out.qs, 0, sizeof out.qs);

    for (int k = 0; k < 8; ++k) {
        uint8_t s6 = uint8_t(std::clamp(int(std::lround(scales[k] / (d > 0 ? d : 1))), 0, 63));
        uint8_t m6 = uint8_t(std::clamp(int(std::lround(std::fabs(mins[k]) / (dmin > 0 ? dmin : 1))), 0, 63));
        pack_scale_min(out.scales, k, s6, m6);
        const float sd = d * float(s6);
        const float md = dmin * float(m6) * (mins[k] < 0 ? -1.0f : 1.0f);
        const float isd = (sd > 0) ? 1.0f / sd : 0;
        for (int i = 0; i < 32; ++i) {
            int q = int(std::lround((x[k * 32 + i] - md) * isd));
            q = std::clamp(q, 0, 15);
            const size_t idx = size_t(k * 32 + i);
            if ((idx & 1) == 0) out.qs[idx / 2] |= uint8_t(q);
            else                out.qs[idx / 2] |= uint8_t(q << 4);
        }
    }
}

static void q4k_dequantize_block(const Q4KBlock& blk, float* out) {
    for (int k = 0; k < 8; ++k) {
        uint8_t s6, m6;
        unpack_scale_min(blk.scales, k, s6, m6);
        const float sd = blk.d * float(s6);
        const float md = -blk.dmin * float(m6);  // sign convention
        for (int i = 0; i < 32; ++i) {
            const size_t idx = size_t(k * 32 + i);
            uint8_t q = ((idx & 1) == 0) ? (blk.qs[idx / 2] & 0xF) : (blk.qs[idx / 2] >> 4);
            out[idx] = sd * float(q) + md;
        }
    }
}

void q4k_quantize(const float* x, Q4KBlock* out, size_t n) {
    TCPP_CHECK(n % kKBlockSize == 0, "q4k: n must be %256");
    for (size_t b = 0; b < n / kKBlockSize; ++b)
        q4k_quantize_block(x + b * kKBlockSize, out[b]);
}
void q4k_dequantize(const Q4KBlock* blocks, float* out, size_t n) {
    TCPP_CHECK(n % kKBlockSize == 0, "q4k: n must be %256");
    for (size_t b = 0; b < n / kKBlockSize; ++b)
        q4k_dequantize_block(blocks[b], out + b * kKBlockSize);
}
void matmul_q4k(const float* A, const Q4KBlock* W, float* C,
                size_t M, size_t N, size_t K) {
    TCPP_CHECK(K % kKBlockSize == 0, "matmul_q4k: K must be %256");
    const size_t bpr = K / kKBlockSize;
    float scratch[kKBlockSize];
    for (size_t m = 0; m < M; ++m) {
        const float* a = A + m * K;
        for (size_t n = 0; n < N; ++n) {
            const Q4KBlock* w = W + n * bpr;
            float acc = 0;
            for (size_t b = 0; b < bpr; ++b) {
                q4k_dequantize_block(w[b], scratch);
                const float* ab = a + b * kKBlockSize;
                float s = 0;
                for (size_t i = 0; i < kKBlockSize; ++i) s += ab[i] * scratch[i];
                acc += s;
            }
            C[m * N + n] = acc;
        }
    }
}

} // namespace turbocpp
