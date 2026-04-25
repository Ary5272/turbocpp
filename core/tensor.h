#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
#include "allocator.h"

namespace turbocpp {

// Maximum tensor rank. Transformers need at most [batch, heads, seq, dim] = 4,
// plus [layers, ...] for weight stacks, so we cap at 4 and use layer-indexed
// arrays of tensors for stacked weights instead of 5-D.
constexpr int kMaxDims = 4;

enum class DType : uint8_t {
    F32  = 0,
    I32  = 1,
    Q4_0 = 2,  // 4-bit block-quantized weights (see quant/q4.h)
    U8   = 3,
};

// Row-major tensor. Strides are in ELEMENTS, not bytes. A "view" borrows
// data without owning it — used for slicing and reshaping without copies.
// No std::vector in the hot path: shape/stride are fixed-size arrays.
struct Tensor {
    void*   data      = nullptr;
    DType   dtype     = DType::F32;
    int     ndim      = 0;
    size_t  shape[kMaxDims]  = {0, 0, 0, 0};
    size_t  stride[kMaxDims] = {0, 0, 0, 0};
    size_t  nelements = 0;
    bool    owns_data = false;

    Tensor() = default;
    ~Tensor() { release(); }

    // Non-copyable; use explicit .view() or .clone() at call sites to
    // force the caller to reason about data ownership.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& o) noexcept { steal(o); }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) { release(); steal(o); }
        return *this;
    }

    // Allocates a fresh contiguous buffer. Elements are uninitialized —
    // callers must fill() or load() before reading.
    void allocate(DType t, std::array<size_t, kMaxDims> shp, int rank);

    // Borrowed view: shares the underlying data, does NOT free on destruct.
    static Tensor view(void* ptr, DType t, std::array<size_t, kMaxDims> shp, int rank);

    void release() noexcept;

    // Fill with constant (only F32 / I32 supported).
    void zero() noexcept;
    void fill_f32(float v) noexcept;

    inline float* f32() noexcept { return static_cast<float*>(data); }
    inline const float* f32() const noexcept { return static_cast<const float*>(data); }
    inline int32_t* i32() noexcept { return static_cast<int32_t*>(data); }
    inline const int32_t* i32() const noexcept { return static_cast<const int32_t*>(data); }
    inline uint8_t* u8() noexcept { return static_cast<uint8_t*>(data); }
    inline const uint8_t* u8() const noexcept { return static_cast<const uint8_t*>(data); }

    // Contiguous row-major check. All tensors created by allocate() are
    // contiguous; view()s on strided data may not be.
    bool is_contiguous() const noexcept;

    // Element-size in bytes. For Q4_0 this is NOT meaningful per-element
    // (packed format) — use quant/q4.h helpers for quantized tensors.
    size_t elem_size() const noexcept;

private:
    void steal(Tensor& o) noexcept;
    void compute_strides() noexcept;
};

} // namespace turbocpp
