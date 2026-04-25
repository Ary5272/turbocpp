#include "allocator.h"
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#  include <malloc.h>
#endif

namespace turbocpp {

void* aligned_alloc_bytes(size_t bytes, size_t alignment) noexcept {
    if (bytes == 0) return nullptr;
    // std::aligned_alloc demands size % alignment == 0.
    const size_t rounded = round_up(bytes, alignment);
#if defined(_WIN32)
    return _aligned_malloc(rounded, alignment);
#else
    void* p = nullptr;
    // posix_memalign requires alignment >= sizeof(void*) and a power of two.
    if (posix_memalign(&p, alignment, rounded) != 0) return nullptr;
    return p;
#endif
}

void aligned_free(void* ptr) noexcept {
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

} // namespace turbocpp
