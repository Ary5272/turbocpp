cmake_minimum_required(VERSION 3.16)
project(turbocpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Per-compiler flags
# ---------------------------------------------------------------------------
#
# Release: -O3 for loop/unroll aggressive opts; -march=native ties the
# binary to the build host's ISA (produces the fastest code but makes the
# binary non-portable). -ffast-math enables FP reassociation — tradeoff:
# may break strict IEEE754 (no signed zero, no NaN propagation guarantees,
# reciprocal approximation). For inference this is acceptable: softmax and
# FMA produce slight numerical differences vs reference, <1e-4 relative,
# below quantization noise. DO NOT use ffast-math for training code.

if(MSVC)
    add_compile_options(/W3 /EHsc)
    # /D NOMINMAX so windows.h doesn't poison std::min/max.
    # /wd4476/4477/4473: MSVC's format-string checker misreads `%` inside
    # stringified condition expressions in our TCPP_CHECK macros (the % is
    # the C modulo operator, not a format spec). Runtime behavior is fine.
    add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
    add_compile_options(/wd4476 /wd4477 /wd4473)
    # MSVC: /arch:AVX2 enables AVX2+FMA3 intrinsics. /fp:fast is the
    # ffast-math analog.
    add_compile_options(
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/arch:AVX2>
        $<$<CONFIG:Release>:/fp:fast>
        $<$<CONFIG:Release>:/Oi>            # intrinsics
        $<$<CONFIG:Debug>:/Od>
        $<$<CONFIG:Debug>:/Zi>
    )
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-unused-function)
    add_compile_options(
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ffast-math>
        $<$<CONFIG:Release>:-fno-plt>
        $<$<CONFIG:Release>:-funroll-loops>
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Debug>:-fsanitize=address>
        $<$<CONFIG:Debug>:-fsanitize=undefined>
    )
    add_link_options(
        $<$<CONFIG:Debug>:-fsanitize=address>
        $<$<CONFIG:Debug>:-fsanitize=undefined>
    )
    # Thread linkage
    find_package(Threads REQUIRED)
endif()

# ---------------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------------
set(TURBOCPP_SOURCES
    core/allocator.cpp
    core/tensor.cpp
    math/matmul.cpp
    math/vec_ops.cpp
    math/activations.cpp
    model/rmsnorm.cpp
    model/rope.cpp
    model/embeddings.cpp
    model/attention.cpp
    model/transformer.cpp
    model/moe.cpp
    kv_cache/kv_cache.cpp
    quant/q4.cpp
    quant/q8.cpp
    quant/qk.cpp
    quant/fp16.cpp
    quant/hadamard.cpp
    quant/tq.cpp
    quant/kv_quant.cpp
    loader/gguf.cpp
    loader/gguf_v3.cpp
    loader/lora.cpp
    tokenizer/bpe.cpp
    runtime/thread_pool.cpp
    runtime/sampling.cpp
    runtime/inference.cpp
    runtime/parallel_ops.cpp
    runtime/chat.cpp
    runtime/grammar.cpp
    runtime/speculative.cpp
    runtime/beam_search.cpp
)

add_library(turbocpp_core STATIC ${TURBOCPP_SOURCES})
target_include_directories(turbocpp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT MSVC)
    target_link_libraries(turbocpp_core PUBLIC Threads::Threads)
endif()

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(turbocpp main.cpp)
target_link_libraries(turbocpp PRIVATE turbocpp_core)

add_executable(turbocpp-bench benchmark.cpp)
target_link_libraries(turbocpp-bench PRIVATE turbocpp_core)

# HTTP server: requires sockets (Ws2_32 on Windows; nothing extra on POSIX)
add_executable(turbocpp-server server.cpp)
target_link_libraries(turbocpp-server PRIVATE turbocpp_core)
if(WIN32)
    target_link_libraries(turbocpp-server PRIVATE Ws2_32)
endif()

enable_testing()
add_executable(turbocpp-tests tests/run_tests.cpp)
target_link_libraries(turbocpp-tests PRIVATE turbocpp_core)
add_test(NAME unit COMMAND turbocpp-tests)

# ---------------------------------------------------------------------------
# Report build info
# ---------------------------------------------------------------------------
message(STATUS "turbocpp build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "turbocpp compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
