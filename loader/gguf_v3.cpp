#include "gguf_v3.h"
#include "../quant/fp16.h"
#include "../quant/q4.h"
#include "../quant/q8.h"
#include "../quant/qk.h"
#include "../utils/logging.h"
#include <cstdio>
#include <cstring>

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

// GGUF metadata-value type codes (different from tensor type codes).
enum class GGUF_MV : uint32_t {
    UINT8=0, INT8=1, UINT16=2, INT16=3, UINT32=4, INT32=5,
    FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10, INT64=11, FLOAT64=12
};

// Helper for streaming bytes out of the mmap region.
struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <typename T> bool read(T& v) {
        if (p + sizeof(T) > end) { ok = false; return false; }
        std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return true;
    }
    bool read_string(std::string& s) {
        uint64_t n = 0;
        if (!read(n)) return false;
        if (p + n > end) { ok = false; return false; }
        s.assign(reinterpret_cast<const char*>(p), size_t(n));
        p += n;
        return true;
    }
};

static bool read_meta_value(Cursor& c, GGUFValue& out, GGUF_MV t);

static bool read_array(Cursor& c, GGUFValue& out) {
    uint32_t et = 0; uint64_t n = 0;
    if (!c.read(et) || !c.read(n)) return false;
    GGUF_MV elt = GGUF_MV(et);
    switch (elt) {
        case GGUF_MV::INT64:
        case GGUF_MV::INT32:
        case GGUF_MV::INT16:
        case GGUF_MV::INT8: {
            std::vector<int64_t> v; v.reserve(size_t(n));
            for (uint64_t i = 0; i < n; ++i) {
                int64_t x;
                if (elt == GGUF_MV::INT64) { if (!c.read(x)) return false; }
                else if (elt == GGUF_MV::INT32) { int32_t y; if (!c.read(y)) return false; x = y; }
                else if (elt == GGUF_MV::INT16) { int16_t y; if (!c.read(y)) return false; x = y; }
                else { int8_t y; if (!c.read(y)) return false; x = y; }
                v.push_back(x);
            }
            out = std::move(v); return true;
        }
        case GGUF_MV::UINT64:
        case GGUF_MV::UINT32:
        case GGUF_MV::UINT16:
        case GGUF_MV::UINT8: {
            std::vector<uint64_t> v; v.reserve(size_t(n));
            for (uint64_t i = 0; i < n; ++i) {
                uint64_t x;
                if (elt == GGUF_MV::UINT64) { if (!c.read(x)) return false; }
                else if (elt == GGUF_MV::UINT32) { uint32_t y; if (!c.read(y)) return false; x = y; }
                else if (elt == GGUF_MV::UINT16) { uint16_t y; if (!c.read(y)) return false; x = y; }
                else { uint8_t y; if (!c.read(y)) return false; x = y; }
                v.push_back(x);
            }
            out = std::move(v); return true;
        }
        case GGUF_MV::FLOAT64: case GGUF_MV::FLOAT32: {
            std::vector<double> v; v.reserve(size_t(n));
            for (uint64_t i = 0; i < n; ++i) {
                double x;
                if (elt == GGUF_MV::FLOAT64) { if (!c.read(x)) return false; }
                else { float y; if (!c.read(y)) return false; x = y; }
                v.push_back(x);
            }
            out = std::move(v); return true;
        }
        case GGUF_MV::STRING: {
            std::vector<std::string> v; v.reserve(size_t(n));
            for (uint64_t i = 0; i < n; ++i) {
                std::string s; if (!c.read_string(s)) return false;
                v.push_back(std::move(s));
            }
            out = std::move(v); return true;
        }
        default: return false;
    }
}

static bool read_meta_value(Cursor& c, GGUFValue& out, GGUF_MV t) {
    switch (t) {
        case GGUF_MV::UINT8:  { uint8_t  v; if (!c.read(v)) return false; out = uint64_t(v); return true; }
        case GGUF_MV::INT8:   { int8_t   v; if (!c.read(v)) return false; out = int64_t(v); return true; }
        case GGUF_MV::UINT16: { uint16_t v; if (!c.read(v)) return false; out = uint64_t(v); return true; }
        case GGUF_MV::INT16:  { int16_t  v; if (!c.read(v)) return false; out = int64_t(v); return true; }
        case GGUF_MV::UINT32: { uint32_t v; if (!c.read(v)) return false; out = uint64_t(v); return true; }
        case GGUF_MV::INT32:  { int32_t  v; if (!c.read(v)) return false; out = int64_t(v); return true; }
        case GGUF_MV::UINT64: { uint64_t v; if (!c.read(v)) return false; out = v; return true; }
        case GGUF_MV::INT64:  { int64_t  v; if (!c.read(v)) return false; out = v; return true; }
        case GGUF_MV::FLOAT32:{ float    v; if (!c.read(v)) return false; out = double(v); return true; }
        case GGUF_MV::FLOAT64:{ double   v; if (!c.read(v)) return false; out = v; return true; }
        case GGUF_MV::BOOL:   { uint8_t  v; if (!c.read(v)) return false; out = bool(v); return true; }
        case GGUF_MV::STRING: { std::string v; if (!c.read_string(v)) return false; out = std::move(v); return true; }
        case GGUF_MV::ARRAY:  return read_array(c, out);
    }
    return false;
}

GGUFReaderV3::~GGUFReaderV3() { close(); }

bool GGUFReaderV3::open(const std::string& path) {
    close();
#if defined(_WIN32)
    HANDLE hF = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hF == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER sz; if (!GetFileSizeEx(hF, &sz)) { CloseHandle(hF); return false; }
    mapped_size_ = size_t(sz.QuadPart);
    HANDLE hM = CreateFileMappingA(hF, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hM) { CloseHandle(hF); return false; }
    mapped_ = MapViewOfFile(hM, FILE_MAP_READ, 0, 0, 0);
    if (!mapped_) { CloseHandle(hM); CloseHandle(hF); return false; }
    file_handle_ = hF; map_handle_ = hM;
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;
    struct stat st;
    if (fstat(fd_, &st) != 0) { ::close(fd_); fd_=-1; return false; }
    mapped_size_ = size_t(st.st_size);
    mapped_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) { mapped_=nullptr; ::close(fd_); fd_=-1; return false; }
#endif

    Cursor c{static_cast<const uint8_t*>(mapped_),
             static_cast<const uint8_t*>(mapped_) + mapped_size_};
    uint32_t magic, version; uint64_t n_tensors, n_kv;
    if (!c.read(magic) || magic != kGGUFMagic) { close(); return false; }
    if (!c.read(version) || version < 2 || version > 3) { close(); return false; }
    if (!c.read(n_tensors) || !c.read(n_kv)) { close(); return false; }

    // Read KV pairs.
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        if (!c.read_string(key)) { close(); return false; }
        uint32_t vt; if (!c.read(vt)) { close(); return false; }
        GGUFValue v;
        if (!read_meta_value(c, v, GGUF_MV(vt))) { close(); return false; }
        meta_.emplace(std::move(key), std::move(v));
    }

    // Read tensor infos.
    tensors_.reserve(size_t(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GGUFTensorInfo info;
        if (!c.read_string(info.name)) { close(); return false; }
        uint32_t ndim; if (!c.read(ndim)) { close(); return false; }
        info.shape.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            if (!c.read(info.shape[d])) { close(); return false; }
        }
        uint32_t dtype; if (!c.read(dtype)) { close(); return false; }
        info.dtype = GGUFType(dtype);
        if (!c.read(info.offset)) { close(); return false; }
        info.nbytes = gguf_v3_byte_size(info);
        tensors_.push_back(std::move(info));
    }

    // Data section is aligned. The "general.alignment" KV (default 32)
    // tells us where data starts. Round up the cursor's offset.
    size_t cur_off = size_t(c.p - static_cast<const uint8_t*>(mapped_));
    uint64_t align = 32;
    if (auto a = meta("general.alignment"); a) {
        if (auto pa = std::get_if<uint64_t>(a)) align = *pa;
    }
    size_t data_off = (cur_off + align - 1) & ~(size_t(align) - 1);
    data_base_ = static_cast<const uint8_t*>(mapped_) + data_off;
    return true;
}

void GGUFReaderV3::close() {
    if (mapped_) {
#if defined(_WIN32)
        UnmapViewOfFile(mapped_);
        if (map_handle_)  { CloseHandle((HANDLE)map_handle_); map_handle_=nullptr; }
        if (file_handle_) { CloseHandle((HANDLE)file_handle_); file_handle_=nullptr; }
#else
        munmap(mapped_, mapped_size_);
        if (fd_ >= 0) { ::close(fd_); fd_=-1; }
#endif
        mapped_ = nullptr; mapped_size_ = 0;
    }
    meta_.clear();
    tensors_.clear();
    data_base_ = nullptr;
}

const GGUFValue* GGUFReaderV3::meta(const std::string& key) const {
    auto it = meta_.find(key); return it == meta_.end() ? nullptr : &it->second;
}

std::string GGUFReaderV3::arch() const {
    if (auto v = meta("general.architecture"))
        if (auto s = std::get_if<std::string>(v)) return *s;
    return "";
}
std::string GGUFReaderV3::name() const {
    if (auto v = meta("general.name"))
        if (auto s = std::get_if<std::string>(v)) return *s;
    return "";
}

const GGUFTensorInfo* GGUFReaderV3::find_tensor(const std::string& name) const {
    for (const auto& t : tensors_) if (t.name == name) return &t;
    return nullptr;
}

size_t gguf_v3_nelements(const GGUFTensorInfo& info) {
    size_t n = 1;
    for (auto d : info.shape) n *= size_t(d);
    return n;
}

size_t gguf_v3_byte_size(const GGUFTensorInfo& info) {
    const size_t n = gguf_v3_nelements(info);
    switch (info.dtype) {
        case GGUFType::F32:    return n * 4;
        case GGUFType::F16:    return n * 2;
        case GGUFType::BF16:   return n * 2;
        case GGUFType::F64:    return n * 8;
        case GGUFType::I8:     return n;
        case GGUFType::I16:    return n * 2;
        case GGUFType::I32:    return n * 4;
        case GGUFType::I64:    return n * 8;
        case GGUFType::Q4_0:   return (n / 32) * 18;
        case GGUFType::Q4_1:   return (n / 32) * 20;
        case GGUFType::Q5_0:   return (n / 32) * 22;
        case GGUFType::Q5_1:   return (n / 32) * 24;
        case GGUFType::Q8_0:   return (n / 32) * 34;
        case GGUFType::Q8_1:   return (n / 32) * 40;
        case GGUFType::Q4_K:   return (n / 256) * 144;
        case GGUFType::Q5_K:   return (n / 256) * 176;
        case GGUFType::Q6_K:   return (n / 256) * 210;
        case GGUFType::Q8_K:   return (n / 256) * 292;
        default: return 0;  // unknown → caller skips
    }
}

// Convert IEEE bf16 (sign|8 exp|7 mantissa) to fp32 by left-shift.
static inline float bf16_to_f32(uint16_t h) {
    uint32_t x = uint32_t(h) << 16;
    float out; std::memcpy(&out, &x, 4); return out;
}

// llama.cpp Q4_0 layout (different from our internal Q4Block):
//   fp16 d (2B) + 16 bytes of nibbles. Total 18B per 32 elems.
static void llama_q4_0_dequant(const uint8_t* in, float* out, size_t n) {
    const size_t nb = n / 32;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = in + b * 18;
        uint16_t dh; std::memcpy(&dh, p, 2);
        float d = f16_to_f32_scalar(dh);
        const uint8_t* q = p + 2;
        for (size_t i = 0; i < 16; ++i) {
            int q0 = int(q[i] & 0xF) - 8;
            int q1 = int(q[i] >> 4)  - 8;
            out[b * 32 + 2 * i]     = d * float(q0);
            out[b * 32 + 2 * i + 1] = d * float(q1);
        }
    }
}

// llama.cpp Q8_0: fp16 d + 32 int8.
static void llama_q8_0_dequant(const uint8_t* in, float* out, size_t n) {
    const size_t nb = n / 32;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = in + b * 34;
        uint16_t dh; std::memcpy(&dh, p, 2);
        float d = f16_to_f32_scalar(dh);
        const int8_t* q = reinterpret_cast<const int8_t*>(p + 2);
        for (size_t i = 0; i < 32; ++i) out[b * 32 + i] = d * float(q[i]);
    }
}

// llama.cpp Q6_K: 210 bytes per 256 elems.
//   ql[128] (low 4 bits, packed two-per-byte)
//   qh[64]  (high 2 bits, packed four-per-byte)
//   scales[16] int8
//   d        fp16
static void llama_q6_k_dequant(const uint8_t* in, float* out, size_t n) {
    const size_t nb = n / 256;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = in + b * 210;
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t*  sc = reinterpret_cast<const int8_t*>(p + 192);
        uint16_t dh; std::memcpy(&dh, p + 208, 2);
        float d = f16_to_f32_scalar(dh);
        for (size_t k = 0; k < 16; ++k) {
            const float sd = d * float(sc[k]);
            for (size_t i = 0; i < 16; ++i) {
                size_t idx = k * 16 + i;
                uint8_t low4 = ((idx & 1) == 0) ? (ql[idx/2] & 0xF) : (ql[idx/2] >> 4);
                uint8_t high2 = uint8_t((qh[idx/4] >> ((idx & 3) * 2)) & 0x3);
                int q = int(low4 | (high2 << 4)) - 32;
                out[b * 256 + idx] = sd * float(q);
            }
        }
    }
}

bool GGUFReaderV3::dequantize(const GGUFTensorInfo& info, float* out) const {
    const uint8_t* p = tensor_data(info);
    const size_t n = gguf_v3_nelements(info);
    switch (info.dtype) {
        case GGUFType::F32:
            std::memcpy(out, p, n * sizeof(float)); return true;
        case GGUFType::F16:
            fp16_to_fp32(reinterpret_cast<const fp16_t*>(p), out, n); return true;
        case GGUFType::BF16:
            for (size_t i = 0; i < n; ++i) {
                uint16_t h; std::memcpy(&h, p + 2*i, 2); out[i] = bf16_to_f32(h);
            }
            return true;
        case GGUFType::Q4_0: llama_q4_0_dequant(p, out, n); return true;
        case GGUFType::Q8_0: llama_q8_0_dequant(p, out, n); return true;
        case GGUFType::Q6_K: llama_q6_k_dequant(p, out, n); return true;
        default:
            LOG_ERROR("gguf_v3: unsupported dtype %u for tensor %s",
                      unsigned(info.dtype), info.name.c_str());
            return false;
    }
}

} // namespace turbocpp
