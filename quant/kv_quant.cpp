#include "kv_quant.h"
#include "../utils/logging.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace turbocpp {

void QuantKVCache::init(size_t n_layers, size_t n_heads, size_t head_dim,
                       size_t max_seq_len, KVQuantMode mode) {
    n_layers_ = n_layers;
    n_heads_ = n_heads;
    head_dim_ = head_dim;
    max_seq_len_ = max_seq_len;
    mode_ = mode;
    cur_len_ = 0;

    switch (mode) {
        case KVQuantMode::Q4:
            TCPP_CHECK(head_dim % 2 == 0, "Q4 KV needs even head_dim");
            bytes_per_head_ = head_dim / 2;
            break;
        case KVQuantMode::Q3:
            TCPP_CHECK(head_dim % 8 == 0, "Q3 KV needs head_dim divisible by 8");
            bytes_per_head_ = (head_dim / 8) * 3;   // 3 bytes per 8 values
            break;
        case KVQuantMode::F32:
            bytes_per_head_ = head_dim * sizeof(float);
            break;
    }

    const size_t total_data_bytes = n_layers * n_heads * max_seq_len * bytes_per_head_;
    const size_t total_scales     = n_layers * n_heads * max_seq_len;
    k_data_.resize(total_data_bytes);
    v_data_.resize(total_data_bytes);
    k_scales_.resize(total_scales);
    v_scales_.resize(total_scales);

    std::memset(k_data_.data(), 0, total_data_bytes);
    std::memset(v_data_.data(), 0, total_data_bytes);
    std::memset(k_scales_.data(), 0, total_scales * sizeof(float));
    std::memset(v_scales_.data(), 0, total_scales * sizeof(float));
}

// ---------------------------------------------------------------------------
// Block quantize (single head_dim worth of data)
// ---------------------------------------------------------------------------
void QuantKVCache::quant_block(const float* in, uint8_t* data_out, float& scale_out) const {
    // Symmetric abs-max.
    float maxabs = 0.0f;
    for (size_t i = 0; i < head_dim_; ++i) {
        float a = std::fabs(in[i]);
        if (a > maxabs) maxabs = a;
    }

    if (mode_ == KVQuantMode::Q4) {
        // q in [0, 15], centered at 8 => signed range ~[-7.5, 7.5].
        const float d  = (maxabs > 0.0f) ? (maxabs / 7.5f) : 0.0f;
        const float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        scale_out = d;
        for (size_t i = 0; i < head_dim_ / 2; ++i) {
            int q0 = int(std::lround(in[2 * i]     * id)) + 8;
            int q1 = int(std::lround(in[2 * i + 1] * id)) + 8;
            q0 = std::clamp(q0, 0, 15);
            q1 = std::clamp(q1, 0, 15);
            data_out[i] = uint8_t((q1 << 4) | q0);
        }
    } else if (mode_ == KVQuantMode::Q3) {
        // q in [0, 7], centered at 4 => signed range ~[-3.5, 3.5].
        const float d  = (maxabs > 0.0f) ? (maxabs / 3.5f) : 0.0f;
        const float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        scale_out = d;
        // Pack 8 values into 24 bits (3 bytes). Layout:
        //   byte0 = q0 | (q1 << 3) | ((q2 & 0x3) << 6)
        //   byte1 = (q2 >> 2) | (q3 << 1) | (q4 << 4) | ((q5 & 0x1) << 7)
        //   byte2 = (q5 >> 1) | (q6 << 2) | (q7 << 5)
        const size_t groups = head_dim_ / 8;
        for (size_t g = 0; g < groups; ++g) {
            int q[8];
            for (int j = 0; j < 8; ++j) {
                int v = int(std::lround(in[g * 8 + j] * id)) + 4;
                q[j] = std::clamp(v, 0, 7);
            }
            uint8_t* o = data_out + g * 3;
            o[0] = uint8_t(q[0] | (q[1] << 3) | ((q[2] & 0x3) << 6));
            o[1] = uint8_t((q[2] >> 2) | (q[3] << 1) | (q[4] << 4) | ((q[5] & 0x1) << 7));
            o[2] = uint8_t((q[5] >> 1) | (q[6] << 2) | (q[7] << 5));
        }
    } else {
        // F32 passthrough
        scale_out = 1.0f;
        std::memcpy(data_out, in, head_dim_ * sizeof(float));
    }
}

void QuantKVCache::dequant_block(const uint8_t* data, float scale, float* out) const {
    if (mode_ == KVQuantMode::Q4) {
        for (size_t i = 0; i < head_dim_ / 2; ++i) {
            const uint8_t packed = data[i];
            const int q0 = int(packed & 0x0F) - 8;
            const int q1 = int(packed >> 4)   - 8;
            out[2 * i]     = scale * float(q0);
            out[2 * i + 1] = scale * float(q1);
        }
    } else if (mode_ == KVQuantMode::Q3) {
        const size_t groups = head_dim_ / 8;
        for (size_t g = 0; g < groups; ++g) {
            const uint8_t* b = data + g * 3;
            int q[8];
            q[0] = (b[0])        & 0x7;
            q[1] = (b[0] >> 3)   & 0x7;
            q[2] = ((b[0] >> 6) | (b[1] << 2)) & 0x7;
            q[3] = (b[1] >> 1)   & 0x7;
            q[4] = (b[1] >> 4)   & 0x7;
            q[5] = ((b[1] >> 7) | (b[2] << 1)) & 0x7;
            q[6] = (b[2] >> 2)   & 0x7;
            q[7] = (b[2] >> 5)   & 0x7;
            for (int j = 0; j < 8; ++j) out[g * 8 + j] = scale * float(q[j] - 4);
        }
    } else {
        std::memcpy(out, data, head_dim_ * sizeof(float));
    }
}

void QuantKVCache::append(size_t layer, size_t pos,
                          const float* k_fp32, const float* v_fp32) {
    TCPP_CHECK(pos < max_seq_len_, "quant_kv append: pos %zu exceeds max %zu",
               pos, max_seq_len_);
    for (size_t h = 0; h < n_heads_; ++h) {
        uint8_t* kd = k_data_.data() + data_offset(layer, h, pos);
        uint8_t* vd = v_data_.data() + data_offset(layer, h, pos);
        float& ks = k_scales_.data()[scale_offset(layer, h, pos)];
        float& vs = v_scales_.data()[scale_offset(layer, h, pos)];
        quant_block(k_fp32 + h * head_dim_, kd, ks);
        quant_block(v_fp32 + h * head_dim_, vd, vs);
    }
}

void QuantKVCache::dequant_k(size_t layer, size_t head, size_t pos, float* out) const {
    const uint8_t* d = k_data_.data()   + data_offset(layer, head, pos);
    const float s    = k_scales_.data()[scale_offset(layer, head, pos)];
    dequant_block(d, s, out);
}

void QuantKVCache::dequant_v(size_t layer, size_t head, size_t pos, float* out) const {
    const uint8_t* d = v_data_.data()   + data_offset(layer, head, pos);
    const float s    = v_scales_.data()[scale_offset(layer, head, pos)];
    dequant_block(d, s, out);
}

} // namespace turbocpp
