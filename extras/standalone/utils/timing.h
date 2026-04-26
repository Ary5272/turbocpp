#pragma once
#include <chrono>
#include <cstdint>

namespace turbocpp {

// Monotonic nanosecond timer. Uses steady_clock so it's immune to wall-clock
// jumps (NTP, DST). Sufficient for benchmarking kernels — on modern x86 the
// resolution is well under a microsecond.
class Timer {
public:
    Timer() noexcept { reset(); }

    void reset() noexcept { t0_ = std::chrono::steady_clock::now(); }

    // Elapsed time in nanoseconds since the last reset().
    int64_t ns() const noexcept {
        using namespace std::chrono;
        auto dt = steady_clock::now() - t0_;
        return duration_cast<nanoseconds>(dt).count();
    }

    double us() const noexcept { return ns() / 1e3; }
    double ms() const noexcept { return ns() / 1e6; }
    double s()  const noexcept { return ns() / 1e9; }

private:
    std::chrono::steady_clock::time_point t0_;
};

// Compute GFLOPS for a matmul of M*N*K given elapsed seconds.
// Factor of 2 is 1 multiply + 1 add per inner iteration.
inline double matmul_gflops(size_t M, size_t N, size_t K, double seconds) noexcept {
    return (2.0 * double(M) * double(N) * double(K)) / (seconds * 1e9);
}

} // namespace turbocpp
