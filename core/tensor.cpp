#include "tensor.h"
#include <cstring>
#include <cstdint>

namespace turbocpp {

size_t Tensor::elem_size() const noexcept {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::I32: return 4;
        case DType::U8:  return 1;
        case DType::Q4_0: return 0; // packed; see quant
    }
    return 0;
}

void Tensor::compute_strides() noexcept {
    // Row-major: last dim has stride 1, each outer dim multiplies.
    if (ndim == 0) return;
    stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
}

void Tensor::allocate(DType t, std::array<size_t, kMaxDims> shp, int rank) {
    release();
    dtype = t;
    ndim = rank;
    nelements = 1;
    for (int i = 0; i < rank; ++i) {
        shape[i] = shp[i];
        nelements *= shp[i];
    }
    compute_strides();

    size_t bytes = 0;
    if (dtype == DType::Q4_0) {
        // Caller is expected to pre-compute bytes via q4 helpers and call
        // allocate with DType::U8 for the raw byte buffer. Guard anyway.
        bytes = nelements / 2 + (nelements / 32) * 4; // rough placeholder
    } else {
        bytes = nelements * elem_size();
    }
    data = aligned_alloc_bytes(bytes, kAlign);
    owns_data = true;
}

Tensor Tensor::view(void* ptr, DType t, std::array<size_t, kMaxDims> shp, int rank) {
    Tensor out;
    out.data = ptr;
    out.dtype = t;
    out.ndim = rank;
    out.nelements = 1;
    for (int i = 0; i < rank; ++i) {
        out.shape[i] = shp[i];
        out.nelements *= shp[i];
    }
    out.compute_strides();
    out.owns_data = false;
    return out;
}

void Tensor::release() noexcept {
    if (owns_data && data) {
        aligned_free(data);
    }
    data = nullptr;
    owns_data = false;
    ndim = 0;
    nelements = 0;
    for (int i = 0; i < kMaxDims; ++i) { shape[i] = 0; stride[i] = 0; }
}

void Tensor::zero() noexcept {
    if (!data) return;
    size_t bytes = nelements * elem_size();
    std::memset(data, 0, bytes);
}

void Tensor::fill_f32(float v) noexcept {
    if (!data || dtype != DType::F32) return;
    float* p = f32();
    for (size_t i = 0; i < nelements; ++i) p[i] = v;
}

bool Tensor::is_contiguous() const noexcept {
    if (ndim == 0) return true;
    size_t expected = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (stride[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

void Tensor::steal(Tensor& o) noexcept {
    data = o.data;
    dtype = o.dtype;
    ndim = o.ndim;
    nelements = o.nelements;
    owns_data = o.owns_data;
    for (int i = 0; i < kMaxDims; ++i) { shape[i] = o.shape[i]; stride[i] = o.stride[i]; }
    o.data = nullptr;
    o.owns_data = false;
    o.nelements = 0;
    o.ndim = 0;
}

} // namespace turbocpp
