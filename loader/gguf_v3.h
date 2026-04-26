#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace turbocpp {

// GGUF v3 reader — drop-in for llama.cpp model files.
//
// Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
// Layout: [magic][version][n_tensors][n_kv][KV pairs][tensor infos][data].
//
// We parse:
//   - The fixed header
//   - All KV metadata pairs (model name, arch, hyperparameters, tokenizer)
//   - The tensor directory (name + dtype + shape + offset)
//
// Tensor data is mmap'd. Decoding to fp32 happens on demand via
// gguf_v3_dequant_tensor() — supported types:
//   F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K, Q8_K
// Unsupported types raise an error so the caller can skip / re-quantize.

constexpr uint32_t kGGUFMagic = 0x46554747u;   // "GGUF"

enum class GGUFType : uint32_t {
    F32   = 0,  F16   = 1,
    Q4_0  = 2,  Q4_1  = 3,
    Q5_0  = 6,  Q5_1  = 7,
    Q8_0  = 8,  Q8_1  = 9,
    Q2_K  = 10, Q3_K  = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20,
    IQ3_S = 21,   IQ2_S = 22,  IQ4_XS = 23,  I8 = 24, I16 = 25, I32 = 26,
    I64 = 27,     F64   = 28,  IQ1_M = 29,   BF16 = 30,
};

// KV metadata value. Strings owned by the reader (kept inside the file's
// mmap region, not copied).
using GGUFValue = std::variant<
    int64_t, uint64_t, double, bool, std::string,
    std::vector<int64_t>, std::vector<uint64_t>, std::vector<double>,
    std::vector<std::string>>;

struct GGUFTensorInfo {
    std::string name;
    GGUFType    dtype;
    std::vector<uint64_t> shape;     // GGUF stores in row-major
    uint64_t    offset;              // from start of data section
    uint64_t    nbytes;              // size of the on-disk blob
};

class GGUFReaderV3 {
public:
    GGUFReaderV3() = default;
    ~GGUFReaderV3();

    GGUFReaderV3(const GGUFReaderV3&) = delete;
    GGUFReaderV3& operator=(const GGUFReaderV3&) = delete;

    bool open(const std::string& path);
    void close();
    bool is_open() const noexcept { return mapped_ != nullptr; }

    // Look up metadata by GGUF key. Returns nullptr if absent.
    const GGUFValue* meta(const std::string& key) const;

    // Convenience accessors for common keys.
    std::string arch() const;          // e.g. "llama"
    std::string name() const;          // model name

    // Tensor table.
    const std::vector<GGUFTensorInfo>& tensors() const noexcept { return tensors_; }
    const GGUFTensorInfo* find_tensor(const std::string& name) const;

    // Get raw byte pointer to a tensor's on-disk blob.
    const uint8_t* tensor_data(const GGUFTensorInfo& info) const {
        return data_base_ + info.offset;
    }

    // Decode a tensor to fp32 (caller-provided buffer of nelements floats).
    // Returns false for unsupported dtypes.
    bool dequantize(const GGUFTensorInfo& info, float* out) const;

private:
    void*  mapped_ = nullptr;
    size_t mapped_size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* map_handle_  = nullptr;
#else
    int fd_ = -1;
#endif
    const uint8_t* data_base_ = nullptr;

    std::unordered_map<std::string, GGUFValue> meta_;
    std::vector<GGUFTensorInfo>                tensors_;
};

// Number of fp32 elements in a tensor (product of shape).
size_t gguf_v3_nelements(const GGUFTensorInfo& info);

// Bytes-per-element for non-block dtypes; 0 for block-quantized.
size_t gguf_v3_byte_size(const GGUFTensorInfo& info);

} // namespace turbocpp
