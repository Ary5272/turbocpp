#include "thread_pool.h"
#include <algorithm>


namespace turbocpp {

ThreadPool::ThreadPool(size_t n) {
    if (n == 0) {
        n = std::thread::hardware_concurrency();
        if (n == 0) n = 1;
    }
    workers_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_.store(true);
    }
    cv_.notify_all();
    for (auto& t : workers_) if (t.joinable()) t.join();
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stop_.load() || !queue_.empty(); });
            if (stop_.load() && queue_.empty()) return;
            task = std::move(queue_.front());
            queue_.pop();
        }
        task();
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push(std::move(task));
    }
    cv_.notify_one();
}

// ---------------------------------------------------------------------------
// parallel_for: split [0, n) into P equal chunks (P = num threads) and
// launch P-1 workers. The calling thread takes the last chunk itself so
// we don't block on a task we could have just done — saves a context switch
// on every call.
// ---------------------------------------------------------------------------
void ThreadPool::parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn) {
    const size_t P = workers_.size();
    if (P == 0 || n == 0) { if (n) fn(0, n); return; }

    // Small problems: don't bother with threading overhead.
    if (n < P * 4) { fn(0, n); return; }

    const size_t chunk = (n + P - 1) / P;

    std::mutex done_mu;
    std::condition_variable done_cv;
    std::atomic<size_t> remaining{P - 1};

    for (size_t p = 0; p < P - 1; ++p) {
        const size_t beg = p * chunk;
        const size_t end = std::min(beg + chunk, n);
        if (beg >= end) { remaining.fetch_sub(1); continue; }
        enqueue([&, beg, end] {
            fn(beg, end);
            if (remaining.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lk(done_mu);
                done_cv.notify_one();
            }
        });
    }

    // Self-run the last chunk.
    const size_t beg = (P - 1) * chunk;
    if (beg < n) fn(beg, n);

    std::unique_lock<std::mutex> lk(done_mu);
    done_cv.wait(lk, [&] { return remaining.load() == 0; });
}

// Meyers singleton — constructed on first use, thread-safe since C++11.
// Override via set_global_pool_size() BEFORE the first global_pool() call.
static size_t g_pool_override = 0;

void set_global_pool_size(size_t n) { g_pool_override = n; }

ThreadPool& global_pool() {
    static ThreadPool pool(g_pool_override);
    return pool;
}

} // namespace turbocpp
