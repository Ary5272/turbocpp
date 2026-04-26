// Minimal unit tests for the kernels that have closed-form expected values.
// No external test framework — keep dependencies tight per project rules.

#include "../core/allocator.h"
#include "../math/matmul.h"
#include "../math/vec_ops.h"
#include "../math/activations.h"
#include "../model/rmsnorm.h"
#include "../model/rope.h"
#include "../quant/q4.h"
#include "../quant/q8.h"
#include "../quant/fp16.h"
#include "../quant/hadamard.h"
#include "../quant/tq.h"
#include "../quant/qk.h"
#include "../quant/kv_quant.h"
#include "../runtime/chat.h"
#include "../runtime/grammar.h"
#include "../tokenizer/bpe.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace turbocpp;

static int g_failed = 0;

#define CHECK(cond, fmt, ...) do {                                       \
    if (!(cond)) {                                                       \
        std::fprintf(stderr, "FAIL %s:%d: " fmt "\n",                    \
                     __FILE__, __LINE__, ##__VA_ARGS__);                 \
        ++g_failed;                                                      \
    }                                                                    \
} while (0)

#define APPROX(a, b, tol) (std::fabs((a) - (b)) <= (tol))

static void test_matmul_tiers_agree() {
    const size_t M = 16, N = 32, K = 64;
    AlignedBuffer<float> A(M * K), B(N * K);
    AlignedBuffer<float> C0(M * N), C1(M * N), C2(M * N);
    std::mt19937 rng(123);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (size_t i = 0; i < M * K; ++i) A.data()[i] = d(rng);
    for (size_t i = 0; i < N * K; ++i) B.data()[i] = d(rng);

    matmul_naive  (A.data(), B.data(), C0.data(), M, N, K);
    matmul_blocked(A.data(), B.data(), C1.data(), M, N, K);
    matmul_avx2   (A.data(), B.data(), C2.data(), M, N, K);

    float max01 = 0, max02 = 0;
    for (size_t i = 0; i < M * N; ++i) {
        max01 = std::max(max01, std::fabs(C0.data()[i] - C1.data()[i]));
        max02 = std::max(max02, std::fabs(C0.data()[i] - C2.data()[i]));
    }
    // Fast-math + FMA can reorder operations; allow a relative epsilon.
    CHECK(max01 < 1e-3f, "naive vs blocked diverged (%.3e)", max01);
    CHECK(max02 < 1e-3f, "naive vs avx2 diverged (%.3e)", max02);
}

static void test_softmax_sums_to_one() {
    const size_t N = 64;
    AlignedBuffer<float> x(N);
    std::mt19937 rng(7);
    std::normal_distribution<float> d(0.0f, 5.0f);
    for (size_t i = 0; i < N; ++i) x.data()[i] = d(rng);
    softmax_inplace(x.data(), N);
    float s = 0;
    for (size_t i = 0; i < N; ++i) s += x.data()[i];
    CHECK(APPROX(s, 1.0f, 1e-5f), "softmax sum = %.6f", s);
}

static void test_rmsnorm_unit_variance() {
    const size_t D = 256;
    AlignedBuffer<float> x(D), w(D), y(D);
    std::mt19937 rng(11);
    std::normal_distribution<float> d(0.0f, 3.0f);
    for (size_t i = 0; i < D; ++i) { x.data()[i] = d(rng); w.data()[i] = 1.0f; }
    rmsnorm_row(x.data(), w.data(), y.data(), D, 1e-5f);
    float ms = 0;
    for (size_t i = 0; i < D; ++i) ms += y.data()[i] * y.data()[i];
    ms /= float(D);
    // After RMSNorm with unit weights: sqrt(mean(y²)) ≈ 1.
    CHECK(APPROX(ms, 1.0f, 1e-3f), "rmsnorm post mean_sq = %.6f", ms);
}

static void test_rope_preserves_norm() {
    const size_t H = 1, D = 32;
    RopeTables r; r.build(D, 64);
    AlignedBuffer<float> x(D), y(D);
    std::mt19937 rng(31);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (size_t i = 0; i < D; ++i) { x.data()[i] = d(rng); y.data()[i] = x.data()[i]; }
    float n0 = 0; for (size_t i = 0; i < D; ++i) n0 += x.data()[i] * x.data()[i];
    r.apply(y.data(), H, 17);
    float n1 = 0; for (size_t i = 0; i < D; ++i) n1 += y.data()[i] * y.data()[i];
    // Rotation is orthogonal -> norm preserved.
    CHECK(APPROX(n0, n1, 1e-3f * n0), "rope changed norm: %.4f -> %.4f", n0, n1);
}

static void test_q4_roundtrip() {
    const size_t N = 256;
    AlignedBuffer<float> x(N), y(N);
    std::vector<Q4Block> blocks(q4_num_blocks(N));
    std::mt19937 rng(57);
    std::normal_distribution<float> d(0.0f, 0.5f);
    for (size_t i = 0; i < N; ++i) x.data()[i] = d(rng);
    q4_quantize(x.data(), blocks.data(), N);
    q4_dequantize(blocks.data(), y.data(), N);
    float err = 0;
    for (size_t i = 0; i < N; ++i) err = std::max(err, std::fabs(x.data()[i] - y.data()[i]));
    // 4-bit on N(0, 0.5): block-max maxabs ≈ 3σ ≈ 1.5; quant step ≈
    // maxabs / 7.5 ≈ 0.2; half-LSB error ≈ 0.1. Allow some slack.
    CHECK(err < 0.15f, "q4 roundtrip error too large: %.3e", err);
}

static void test_kv_quant_roundtrip() {
    QuantKVCache kv;
    kv.init(/*layers*/1, /*heads*/2, /*head_dim*/64, /*max_seq*/8, KVQuantMode::Q4);
    AlignedBuffer<float> k(2 * 64), v(2 * 64);
    std::mt19937 rng(91);
    std::normal_distribution<float> d(0.0f, 0.3f);
    for (size_t i = 0; i < 128; ++i) { k.data()[i] = d(rng); v.data()[i] = d(rng); }
    kv.append(0, /*pos*/0, k.data(), v.data());

    AlignedBuffer<float> kdq(64), vdq(64);
    kv.dequant_k(0, /*head*/1, /*pos*/0, kdq.data());
    kv.dequant_v(0, /*head*/1, /*pos*/0, vdq.data());

    float maxe_k = 0, maxe_v = 0;
    for (size_t i = 0; i < 64; ++i) {
        maxe_k = std::max(maxe_k, std::fabs(k.data()[64 + i] - kdq.data()[i]));
        maxe_v = std::max(maxe_v, std::fabs(v.data()[64 + i] - vdq.data()[i]));
    }
    CHECK(maxe_k < 0.1f, "kv q4 K err: %.3e", maxe_k);
    CHECK(maxe_v < 0.1f, "kv q4 V err: %.3e", maxe_v);
}

static void test_bpe_round_trip() {
    BPETokenizer tok;
    tok.build_minimal();
    std::string text = "the quick brown fox";
    auto ids = tok.encode(text);
    std::string back = tok.decode(ids);
    CHECK(back == text, "bpe roundtrip: '%s' -> %zu ids -> '%s'",
          text.c_str(), ids.size(), back.c_str());
}

static void test_q8_roundtrip() {
    const size_t N = 256;
    AlignedBuffer<float> x(N), y(N);
    std::vector<Q8Block> blocks(q8_num_blocks(N));
    std::mt19937 rng(57);
    std::normal_distribution<float> d(0.0f, 0.5f);
    for (size_t i = 0; i < N; ++i) x.data()[i] = d(rng);
    q8_quantize(x.data(), blocks.data(), N);
    q8_dequantize(blocks.data(), y.data(), N);
    float err = 0;
    for (size_t i = 0; i < N; ++i) err = std::max(err, std::fabs(x.data()[i] - y.data()[i]));
    // 8-bit quantization should be << 4-bit. Half-LSB ~ maxabs/127.
    CHECK(err < 0.02f, "q8 roundtrip error too large: %.3e", err);
}

static void test_fp16_roundtrip() {
    const size_t N = 256;
    AlignedBuffer<float>  x(N), y(N);
    AlignedBuffer<fp16_t> h(N);
    std::mt19937 rng(91);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) x.data()[i] = d(rng);
    fp32_to_fp16(x.data(), h.data(), N);
    fp16_to_fp32(h.data(), y.data(), N);
    float rel = 0;
    for (size_t i = 0; i < N; ++i) {
        float e = std::fabs(x.data()[i] - y.data()[i]) / (std::fabs(x.data()[i]) + 1e-6f);
        if (e > rel) rel = e;
    }
    // fp16 has ~1e-3 relative precision.
    CHECK(rel < 5e-3f, "fp16 relative error too large: %.3e", rel);
}

// Hadamard is its own inverse (when normalized): H(H(x)) = x.
static void test_hadamard_involution() {
    const size_t N = 128;
    AlignedBuffer<float> x(N), y(N);
    std::mt19937 rng(13);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) { x.data()[i] = d(rng); y.data()[i] = x.data()[i]; }
    hadamard_inplace(y.data(), N);
    hadamard_inplace(y.data(), N);
    float err = 0;
    for (size_t i = 0; i < N; ++i) err = std::max(err, std::fabs(x.data()[i] - y.data()[i]));
    CHECK(err < 1e-4f, "hadamard involution err: %.3e", err);
}

// TurboQuant matmul should agree with fp32 matmul to within Q4 quant error.
static void test_tq_matmul() {
    const size_t M = 1, N = 64, K = 256;
    AlignedBuffer<float> A(K), W(N * K), C0(N), C1(N);
    std::mt19937 rng(101);
    std::normal_distribution<float> d(0.0f, 0.1f);
    for (size_t i = 0; i < K; ++i)     A.data()[i] = d(rng);
    for (size_t i = 0; i < N * K; ++i) W.data()[i] = d(rng);

    // Reference fp32 matmul.
    matmul(A.data(), W.data(), C0.data(), M, N, K);

    // Quantize via TurboQuant (Hadamard then Q4).
    std::vector<Q4Block> Wtq(N * (K / 32));
    tq_quantize_matrix(W.data(), Wtq.data(), N, K);
    matmul_tq(A.data(), Wtq.data(), C1.data(), M, N, K);

    float maxe = 0, maxref = 0;
    for (size_t i = 0; i < N; ++i) {
        maxe   = std::max(maxe,   std::fabs(C0.data()[i] - C1.data()[i]));
        maxref = std::max(maxref, std::fabs(C0.data()[i]));
    }
    CHECK(maxe / (maxref + 1e-6f) < 0.15f,
          "tq vs fp32 rel err too large: %.3e (ref %.3e)", maxe, maxref);
}

static void test_chat_template_llama3() {
    std::vector<ChatMessage> msgs = {
        {"system", "You are helpful."},
        {"user",   "Hi"},
    };
    std::string p = apply_template(ChatTemplate::LLaMA3, msgs);
    CHECK(p.find("<|start_header_id|>system<|end_header_id|>") != std::string::npos,
          "llama3 system header missing");
    CHECK(p.find("<|start_header_id|>assistant<|end_header_id|>") != std::string::npos,
          "llama3 assistant header missing (priming)");
}

static void test_stop_matcher() {
    StopMatcher m({"\n\n", "</s>"});
    m.append("hello");
    CHECK(!m.stopped(), "false-positive stop");
    m.append(" world\n\nrest");
    CHECK(m.stopped(), "expected stop on \\n\\n");
    CHECK(m.matched_seq() == 0, "wrong stop matched");
}

static void test_q6k_q8k_q4k_roundtrip() {
    const size_t N = 256;
    AlignedBuffer<float> x(N), y(N);
    std::mt19937 rng(202);
    std::normal_distribution<float> d(0.0f, 0.5f);
    for (size_t i = 0; i < N; ++i) x.data()[i] = d(rng);

    // Q6_K
    {
        std::vector<Q6KBlock> b(qk_num_blocks(N));
        q6k_quantize(x.data(), b.data(), N);
        q6k_dequantize(b.data(), y.data(), N);
        float err = 0;
        for (size_t i = 0; i < N; ++i) err = std::max(err, std::fabs(x.data()[i]-y.data()[i]));
        CHECK(err < 0.05f, "q6k err: %.3e", err);
    }
    // Q8_K
    {
        std::vector<Q8KBlock> b(qk_num_blocks(N));
        q8k_quantize(x.data(), b.data(), N);
        q8k_dequantize(b.data(), y.data(), N);
        float err = 0;
        for (size_t i = 0; i < N; ++i) err = std::max(err, std::fabs(x.data()[i]-y.data()[i]));
        CHECK(err < 0.02f, "q8k err: %.3e", err);
    }
    // Q4_K_M (asymmetric, more quant noise)
    {
        std::vector<Q4KBlock> b(qk_num_blocks(N));
        q4k_quantize(x.data(), b.data(), N);
        q4k_dequantize(b.data(), y.data(), N);
        float err = 0;
        for (size_t i = 0; i < N; ++i) err = std::max(err, std::fabs(x.data()[i]-y.data()[i]));
        CHECK(err < 0.25f, "q4k err: %.3e", err);
    }
}

static void test_json_grammar() {
    JsonGrammar g;
    g.observe("{\"name\":\"alice\",\"age\":30}");
    CHECK(g.finished(), "json grammar should be finished after balanced object");
}

int main() {
    test_matmul_tiers_agree();
    test_softmax_sums_to_one();
    test_rmsnorm_unit_variance();
    test_rope_preserves_norm();
    test_q4_roundtrip();
    test_q8_roundtrip();
    test_fp16_roundtrip();
    test_kv_quant_roundtrip();
    test_bpe_round_trip();
    test_hadamard_involution();
    test_tq_matmul();
    test_chat_template_llama3();
    test_stop_matcher();
    test_q6k_q8k_q4k_roundtrip();
    test_json_grammar();

    if (g_failed == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d test(s) failed\n", g_failed);
    return 1;
}
