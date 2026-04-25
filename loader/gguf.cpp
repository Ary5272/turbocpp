#include "gguf.h"
#include "../utils/logging.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace turbocpp {

// ---------------------------------------------------------------------------
// mmap helpers (cross-platform)
// ---------------------------------------------------------------------------

ModelFile::~ModelFile() { close(); }

bool ModelFile::open(const std::string& path) {
    close();

#if defined(_WIN32)
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                               nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        LOG_ERROR("open failed: %s", path.c_str());
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hFile, &sz)) { CloseHandle(hFile); return false; }
    mapped_size_ = size_t(sz.QuadPart);

    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) { CloseHandle(hFile); return false; }

    mapped_ = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!mapped_) { CloseHandle(hMap); CloseHandle(hFile); return false; }

    file_handle_ = hFile;
    map_handle_  = hMap;
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { LOG_ERROR("open failed: %s", path.c_str()); return false; }
    struct stat st;
    if (fstat(fd_, &st) != 0) { ::close(fd_); fd_ = -1; return false; }
    mapped_size_ = size_t(st.st_size);
    mapped_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) { mapped_ = nullptr; ::close(fd_); fd_ = -1; return false; }
#endif

    // Parse header.
    if (mapped_size_ < sizeof(TCPPHeader)) {
        LOG_ERROR("file too small"); close(); return false;
    }
    const TCPPHeader* hdr = static_cast<const TCPPHeader*>(mapped_);
    if (hdr->magic != kTCPPMagic) {
        LOG_ERROR("bad magic: 0x%08x", hdr->magic); close(); return false;
    }
    if (hdr->version != kTCPPVersion) {
        LOG_ERROR("unsupported version: %u", hdr->version); close(); return false;
    }

    // Parse config.
    const uint8_t* p = static_cast<const uint8_t*>(mapped_);
    std::memcpy(&config_, p + hdr->config_offset, sizeof(ModelConfig));

    // Parse tensor directory.
    uint64_t n_tensors = *reinterpret_cast<const uint64_t*>(p + hdr->tensor_dir_offset);
    const TCPPTensorRecord* recs = reinterpret_cast<const TCPPTensorRecord*>(
        p + hdr->tensor_dir_offset + sizeof(uint64_t));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TCPPTensorRecord r = recs[i];
        // Ensure name is null-terminated.
        r.name[sizeof(r.name) - 1] = '\0';
        tensors_.emplace(std::string(r.name), r);
    }

    data_base_ = p + hdr->data_offset;
    return true;
}

void ModelFile::close() {
    if (mapped_) {
#if defined(_WIN32)
        UnmapViewOfFile(mapped_);
        if (map_handle_)  { CloseHandle(static_cast<HANDLE>(map_handle_));  map_handle_  = nullptr; }
        if (file_handle_) { CloseHandle(static_cast<HANDLE>(file_handle_)); file_handle_ = nullptr; }
#else
        munmap(mapped_, mapped_size_);
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
#endif
        mapped_ = nullptr;
        mapped_size_ = 0;
    }
    tensors_.clear();
    data_base_ = nullptr;
}

Tensor ModelFile::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return Tensor{};
    const TCPPTensorRecord& r = it->second;
    void* ptr = const_cast<uint8_t*>(data_base_ + r.offset);
    std::array<size_t, kMaxDims> shp{};
    for (int i = 0; i < r.ndim; ++i) shp[i] = size_t(r.shape[i]);
    return Tensor::view(ptr, DType(r.dtype), shp, int(r.ndim));
}

bool ModelFile::populate(ModelWeights& weights) const {
    auto fetch = [&](const std::string& n) -> const float* {
        auto it = tensors_.find(n);
        if (it == tensors_.end()) return nullptr;
        return reinterpret_cast<const float*>(data_base_ + it->second.offset);
    };

    weights.tok_embed  = fetch("tok_embed");
    weights.final_norm = fetch("final_norm");
    weights.lm_head    = fetch("lm_head");
    if (!weights.tok_embed || !weights.final_norm || !weights.lm_head) {
        LOG_ERROR("missing top-level tensors");
        return false;
    }

    for (size_t L = 0; L < config_.n_layers; ++L) {
        LayerWeights& lw = weights.layers.data()[L];
        char key[64];
        auto mk = [&](const char* suf) {
            std::snprintf(key, sizeof(key), "layers.%zu.%s", L, suf);
            return std::string(key);
        };
        lw.attn_norm = fetch(mk("attn_norm"));
        lw.Wq        = fetch(mk("Wq"));
        lw.Wk        = fetch(mk("Wk"));
        lw.Wv        = fetch(mk("Wv"));
        lw.Wo        = fetch(mk("Wo"));
        lw.ffn_norm  = fetch(mk("ffn_norm"));
        lw.Wgate     = fetch(mk("Wgate"));
        lw.Wup       = fetch(mk("Wup"));
        lw.Wdown     = fetch(mk("Wdown"));
        if (!lw.attn_norm || !lw.Wq || !lw.Wk || !lw.Wv || !lw.Wo ||
            !lw.ffn_norm  || !lw.Wgate || !lw.Wup || !lw.Wdown) {
            LOG_ERROR("missing tensors in layer %zu", L);
            return false;
        }
    }
    return true;
}

void ModelFile::print_summary() const {
    std::printf("=== ModelFile summary ===\n");
    std::printf("  vocab=%zu hidden=%zu layers=%zu heads=%zu head_dim=%zu ffn=%zu\n",
           config_.vocab_size, config_.hidden_dim, config_.n_layers,
           config_.n_heads, config_.head_dim, config_.ffn_dim);
    std::printf("  tensors: %zu\n", tensors_.size());
    for (const auto& kv : tensors_) {
        const auto& r = kv.second;
        std::printf("    %-32s dtype=%u ndim=%u shape=[%llu,%llu,%llu,%llu] bytes=%llu\n",
               kv.first.c_str(), unsigned(r.dtype), unsigned(r.ndim),
               (unsigned long long)r.shape[0], (unsigned long long)r.shape[1],
               (unsigned long long)r.shape[2], (unsigned long long)r.shape[3],
               (unsigned long long)r.nbytes);
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

bool ModelFileWriter::open(const std::string& path) {
    path_ = path;
    pending_.clear();
    order_.clear();
    return true;
}

void ModelFileWriter::add_tensor(const std::string& name, const void* data,
                                 DType dtype, const uint64_t shape[4], int ndim,
                                 size_t nbytes) {
    PendingTensor t;
    t.name = name;
    t.data = data;
    t.dtype = dtype;
    t.ndim = ndim;
    t.nbytes = nbytes;
    for (int i = 0; i < 4; ++i) t.shape[i] = (i < ndim) ? shape[i] : 0;
    if (pending_.find(name) == pending_.end()) order_.push_back(name);
    pending_[name] = t;
}

bool ModelFileWriter::finalize() {
    FILE* f = std::fopen(path_.c_str(), "wb");
    if (!f) { LOG_ERROR("cannot open %s", path_.c_str()); return false; }

    const size_t n_tensors = order_.size();

    // Compute layout.
    const size_t header_sz  = sizeof(TCPPHeader);
    const size_t config_sz  = sizeof(ModelConfig);
    const size_t dir_sz     = sizeof(uint64_t) + n_tensors * sizeof(TCPPTensorRecord);

    uint64_t cursor = header_sz + config_sz + dir_sz;
    // Align data section to 4KB for clean mmap pages.
    const uint64_t page = 4096;
    uint64_t data_offset = (cursor + page - 1) & ~(page - 1);

    TCPPHeader hdr{};
    hdr.magic = kTCPPMagic;
    hdr.version = kTCPPVersion;
    hdr.config_offset = header_sz;
    hdr.tensor_dir_offset = header_sz + config_sz;
    hdr.data_offset = data_offset;

    // Lay out tensor offsets contiguously inside data section (32-byte aligned).
    uint64_t dcur = 0;
    std::vector<TCPPTensorRecord> recs;
    recs.reserve(n_tensors);
    for (const auto& name : order_) {
        const auto& t = pending_.at(name);
        TCPPTensorRecord r{};
        std::memset(r.name, 0, sizeof(r.name));
        std::memcpy(r.name, t.name.c_str(),
                    std::min(t.name.size(), sizeof(r.name) - 1));
        r.dtype = uint8_t(t.dtype);
        r.ndim = uint8_t(t.ndim);
        for (int i = 0; i < 4; ++i) r.shape[i] = t.shape[i];
        r.offset = (dcur + 31) & ~uint64_t(31);
        r.nbytes = t.nbytes;
        dcur = r.offset + t.nbytes;
        recs.push_back(r);
    }
    hdr.data_size = dcur;

    // Write header.
    std::fwrite(&hdr, sizeof(hdr), 1, f);
    // Write config.
    std::fwrite(&config_, sizeof(config_), 1, f);
    // Write directory.
    uint64_t nt = n_tensors;
    std::fwrite(&nt, sizeof(nt), 1, f);
    std::fwrite(recs.data(), sizeof(TCPPTensorRecord), recs.size(), f);
    // Pad to data_offset.
    uint64_t zero = 0;
    while (uint64_t(std::ftell(f)) < data_offset) {
        size_t pad = std::min<uint64_t>(data_offset - std::ftell(f), sizeof(zero));
        std::fwrite(&zero, 1, pad, f);
    }
    // Write data blocks, padding between to reach each tensor's offset.
    for (size_t i = 0; i < n_tensors; ++i) {
        const auto& t = pending_.at(order_[i]);
        const uint64_t abs_off = data_offset + recs[i].offset;
        while (uint64_t(std::ftell(f)) < abs_off) {
            size_t pad = std::min<uint64_t>(abs_off - std::ftell(f), sizeof(zero));
            std::fwrite(&zero, 1, pad, f);
        }
        std::fwrite(t.data, 1, t.nbytes, f);
    }

    std::fclose(f);
    return true;
}

} // namespace turbocpp
