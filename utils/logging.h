#pragma once
#include <cstdio>
#include <cstdlib>

// Minimal header-only logging. No fancy streams — avoid hidden allocations
// in hot paths by keeping this as printf-style macros.
//
// LOG_TRACE is compiled out of release builds. LOG_FATAL aborts.

#ifndef TURBOCPP_LOG_LEVEL
#  define TURBOCPP_LOG_LEVEL 2   // 0=trace, 1=info, 2=warn, 3=error
#endif

#define TCPP_LOG(level, tag, fmt, ...) \
    do { if ((level) >= TURBOCPP_LOG_LEVEL) \
        std::fprintf(stderr, "[" tag "] " fmt "\n", ##__VA_ARGS__); } while (0)

#define LOG_TRACE(fmt, ...) TCPP_LOG(0, "trace", fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  TCPP_LOG(1, "info",  fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  TCPP_LOG(2, "warn",  fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) TCPP_LOG(3, "error", fmt, ##__VA_ARGS__)

#define LOG_FATAL(fmt, ...) do {                                \
    std::fprintf(stderr, "[fatal] " fmt "\n", ##__VA_ARGS__);   \
    std::abort();                                                \
} while (0)

#define TCPP_CHECK(cond, fmt, ...) do {                          \
    if (!(cond)) { LOG_FATAL("check failed: " #cond " | " fmt, ##__VA_ARGS__); } \
} while (0)
