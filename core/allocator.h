#pragma once
#include <cstddef>
#include "alignment.h"

namespace turbocpp {

// Cross-platform aligned allocator. POSIX uses std::aligned_alloc (C11),
// Windows uses _aligned_malloc which requires _aligned_free.
//
// Rationale: std::aligned_alloc requires size to be a multiple of alignment;
// we round up internally so callers don't have to. Returns nullptr on failure.
void* aligned_alloc_bytes(size_t bytes, size_t alignment = kAlign) noexcept;
void  aligned_free(void* ptr) noexcept;

// Typed convenience. Does NOT construct elements — intended for POD buffers
// (floats, ints) that we zero-init explicitly when needed.
template <typename T>
inline T* aligned_alloc_n(size_t n, size_t alignment = kAlign) noexcept {
    return static_cast<T*>(aligned_alloc_bytes(n * sizeof(T), alignment));
}

// RAII guard over an aligned buffer. Non-copyable, movable. Hot-path code
// uses raw pointers handed out by .data() — the guard only manages lifetime.
template <typename T>
class AlignedBuffer {
public:
    AlignedBuffer() noexcept = default;
    explicit AlignedBuffer(size_t n, size_t alignment = kAlign)
        : ptr_(aligned_alloc_n<T>(n, alignment)), n_(n) {}

    ~AlignedBuffer() { reset(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr; o.n_ = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
        if (this != &o) { reset(); ptr_ = o.ptr_; n_ = o.n_; o.ptr_ = nullptr; o.n_ = 0; }
        return *this;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return n_; }

    void reset() noexcept {
        if (ptr_) { aligned_free(ptr_); ptr_ = nullptr; n_ = 0; }
    }

    void resize(size_t n, size_t alignment = kAlign) {
        reset();
        ptr_ = aligned_alloc_n<T>(n, alignment);
        n_ = n;
    }

private:
    T* ptr_ = nullptr;
    size_t n_ = 0;
};

} // namespace turbocpp
