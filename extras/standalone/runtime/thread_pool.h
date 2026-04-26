#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace turbocpp {

// Fixed-size thread pool. Threads sleep on a condition variable until a
// task is enqueued. Task type is std::function<void()> — heap-allocated
// but enqueued OUTSIDE the hot path, so malloc cost is amortized.
//
// Intended for coarse-grained parallelism (one "chunk of rows" per task).
// We don't try to be a Cilk-style work-stealing pool — the overhead would
// dominate for matmul-sized problems.
class ThreadPool {
public:
    explicit ThreadPool(size_t n = 0);  // 0 => hardware_concurrency()
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t num_threads() const noexcept { return workers_.size(); }

    // Enqueue a task. Does NOT wait.
    void enqueue(std::function<void()> task);

    // Data-parallel helper: split [0, n) into contiguous ranges, one per
    // worker, and wait for all to finish. `fn(begin, end)` is called by
    // the worker. This is the main API used by matmul/attention.
    void parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn);

private:
    void worker_loop();

    std::vector<std::thread>           workers_;
    std::queue<std::function<void()>>  queue_;
    std::mutex                         mu_;
    std::condition_variable            cv_;
    std::atomic<bool>                  stop_{false};
};

// Global default pool, lazy-initialized. Used by matmul_parallel etc.
ThreadPool& global_pool();

// Override the global pool's size. Must be called BEFORE first use of
// global_pool() (i.e. before any matmul_parallel). No-op afterwards.
void set_global_pool_size(size_t n);

} // namespace turbocpp
