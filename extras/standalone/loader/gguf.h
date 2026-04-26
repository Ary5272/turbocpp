#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "../core/tensor.h"
#include "../model/transformer.h"

namespace turbocpp {

// ---------------------------------------------------------------------------
// TCPP binary model format (GGUF-inspired, simplified).
//
// The file has three sections:
//   [1] Fixed header with magic + version.
//   [2] Flat ModelConfig.
//   [3] Tensor directory: count, then (name, dtype, ndim, shape[4],
//       offset, nbytes) records.
//   [4] Padding to 4KB page boundary.
//   [5] Raw tensor data.
//
// The loader mmap()s the whole file, validates the header, and produces
// Tensor *views* into the mmap region — zero-copy. Lifetime of all Tensors
// is bound to the ModelFile that produced them.
//
// Endian: all multi-byte fields little-endian. Target x86_64 which is LE.
// ---------------------------------------------------------------------------

constexpr uint32_t kTCPPMagic = 0x50504354;   // "TCPP" little-endian
constexpr uint32_t kTCPPVersion = 1;

#pragma pack(push, 1)
struct TCPPHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t config_offset;       // from file start
    uint64_t tensor_dir_offset;
    uint64_t data_offset;
    uint64_t data_size;
};

struct TCPPTensorRecord {
    char     name[64];            // null-terminated
    uint8_t  dtype;                // DType enum cast to uint8
    uint8_t  ndim;
    uint8_t  pad[6];               // align
    uint64_t shape[4];
    uint64_t offset;               // from data section start
    uint64_t nbytes;
};
#pragma pack(pop)

// Owns the mmap. Tensors handed out are views into this buffer.
class ModelFile {
public:
    ModelFile() = default;
    ~ModelFile();

    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;

    // Open + mmap. Returns true on success.
    bool open(const std::string& path);
    void close();

    bool is_open() const noexcept { return mapped_ != nullptr; }

    const ModelConfig& config() const noexcept { return config_; }

    // Look up a tensor by name. Returns a view (does not own). Empty Tensor
    // if not found.
    Tensor get_tensor(const std::string& name) const;

    // Convenience: populate ModelWeights pointer fields from the tensor
    // directory using LLaMA-style naming:
    //   tok_embed, final_norm, lm_head, layers.L.{attn_norm, Wq, Wk, Wv, Wo,
    //   ffn_norm, Wgate, Wup, Wdown}
    bool populate(ModelWeights& weights) const;

    // Diagnostic: list tensor names and shapes.
    void print_summary() const;

private:
    void*  mapped_ = nullptr;
    size_t mapped_size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* map_handle_  = nullptr;
#else
    int fd_ = -1;
#endif

    ModelConfig config_{};
    std::unordered_map<std::string, TCPPTensorRecord> tensors_;
    const uint8_t* data_base_ = nullptr;
};

// Writer — produces TCPP files from in-memory tensors. Used by tools that
// convert PyTorch/HF weights or by tests that generate random models.
class ModelFileWriter {
public:
    bool open(const std::string& path);
    void set_config(const ModelConfig& cfg) { config_ = cfg; }
    // Record a tensor by name. `data` is copied into the file at serialize().
    void add_tensor(const std::string& name, const void* data, DType dtype,
                    const uint64_t shape[4], int ndim, size_t nbytes);
    // Flush header + directory + payload to disk. Closes the file.
    bool finalize();

private:
    std::string path_;
    ModelConfig config_{};
    struct PendingTensor {
        std::string name;
        const void* data;
        DType dtype;
        uint64_t shape[4];
        int ndim;
        size_t nbytes;
    };
    std::unordered_map<std::string, PendingTensor> pending_;  // by name
    // Insertion-ordered vector for deterministic file layout.
    std::vector<std::string> order_;
};

} // namespace turbocpp
