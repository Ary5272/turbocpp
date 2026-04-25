// turbocpp — CPU LLM inference engine.
//
// Usage:
//   turbocpp [options]
//
// With no -m, runs a 4-layer random-weight demo (smoke test).
// With -m model.tcpp, loads weights and runs real inference.

#include "core/allocator.h"
#include "kv_cache/kv_cache.h"
#include "loader/gguf.h"
#include "model/transformer.h"
#include "runtime/inference.h"
#include "runtime/thread_pool.h"
#include "tokenizer/bpe.h"
#include "utils/timing.h"
#include "utils/logging.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>

using namespace turbocpp;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
struct Args {
    std::string model;
    std::string vocab;
    std::string merges;
    std::string prompt = "Once upon a time";
    int   n_predict = 128;
    float temperature = 0.8f;
    int   top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    float repeat_penalty = 1.1f;
    int   repeat_window  = 64;
    uint64_t seed = 0;
    int   threads = 0;     // 0 = auto
    bool  quiet = false;
    bool  show_help = false;
};

static void usage(const char* prog) {
    std::printf(
"%s — CPU LLM inference (TurboCPP)\n\n"
"Usage: %s [options]\n\n"
"  -m, --model PATH         model file (.tcpp). Omit to run random-demo.\n"
"      --vocab PATH         vocab file (tab-separated token<TAB>id).\n"
"      --merges PATH        BPE merges file.\n"
"  -p, --prompt TEXT        prompt (default: \"Once upon a time\")\n"
"  -n, --n-predict N        max new tokens (default: 128)\n"
"  -t, --temp FLOAT         temperature (default: 0.8; 0 = greedy)\n"
"  -k, --top-k N            top-k (default: 40; 0 = off)\n"
"      --top-p FLOAT        top-p / nucleus (default: 0.95)\n"
"      --min-p FLOAT        min-p (default: 0.05)\n"
"  -r, --repeat-penalty F   repetition penalty (default: 1.1)\n"
"      --repeat-window N    last N tokens to penalize (default: 64)\n"
"  -s, --seed N             RNG seed (default: random)\n"
"      --threads N          worker threads (default: hardware_concurrency)\n"
"  -q, --quiet              suppress timing stats\n"
"  -h, --help               this message\n",
        prog, prog);
}

static int parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if      (s == "-h" || s == "--help")            a.show_help = true;
        else if (s == "-m" || s == "--model")           a.model  = need("--model");
        else if (s == "--vocab")                        a.vocab  = need("--vocab");
        else if (s == "--merges")                       a.merges = need("--merges");
        else if (s == "-p" || s == "--prompt")          a.prompt = need("--prompt");
        else if (s == "-n" || s == "--n-predict")       a.n_predict = std::atoi(need("-n"));
        else if (s == "-t" || s == "--temp")            a.temperature = float(std::atof(need("-t")));
        else if (s == "-k" || s == "--top-k")           a.top_k = std::atoi(need("-k"));
        else if (s == "--top-p")                        a.top_p = float(std::atof(need("--top-p")));
        else if (s == "--min-p")                        a.min_p = float(std::atof(need("--min-p")));
        else if (s == "-r" || s == "--repeat-penalty") a.repeat_penalty = float(std::atof(need("-r")));
        else if (s == "--repeat-window")                a.repeat_window = std::atoi(need("--repeat-window"));
        else if (s == "-s" || s == "--seed")            a.seed = uint64_t(std::strtoull(need("-s"), nullptr, 10));
        else if (s == "--threads")                      a.threads = std::atoi(need("--threads"));
        else if (s == "-q" || s == "--quiet")           a.quiet = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", s.c_str()); return 2; }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Random demo model (no file). Used when -m is not given.
// ---------------------------------------------------------------------------
struct RandomModel {
    ModelConfig cfg;
    AlignedBuffer<float> buf;
    ModelWeights weights;
    struct LayerOff { size_t attn_norm, Wq, Wk, Wv, Wo, ffn_norm, Wgate, Wup, Wdown; };
    AlignedBuffer<LayerOff> layer_offs;
};

static RandomModel make_random_model(const ModelConfig& cfg, uint64_t seed = 42) {
    RandomModel m; m.cfg = cfg; m.layer_offs.resize(cfg.n_layers);
    const size_t H = cfg.hidden_dim, F = cfg.ffn_dim, KV = cfg.kv_dim();

    size_t off = 0;
    auto take_tok_embed = off; off += cfg.vocab_size * H;
    auto take_final     = off; off += H;
    auto take_lm_head   = off; off += cfg.vocab_size * H;
    for (size_t L = 0; L < cfg.n_layers; ++L) {
        auto& lo = m.layer_offs.data()[L];
        lo.attn_norm = off; off += H;
        lo.Wq        = off; off += H * H;
        lo.Wk        = off; off += KV * H;
        lo.Wv        = off; off += KV * H;
        lo.Wo        = off; off += H * H;
        lo.ffn_norm  = off; off += H;
        lo.Wgate     = off; off += F * H;
        lo.Wup       = off; off += F * H;
        lo.Wdown     = off; off += H * F;
    }
    m.buf.resize(off);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.02f);

    auto fill_g = [&](size_t o, size_t n) { for (size_t i=0;i<n;++i) m.buf.data()[o+i]=nd(rng); };
    auto fill_1 = [&](size_t o, size_t n) { for (size_t i=0;i<n;++i) m.buf.data()[o+i]=1.0f; };

    fill_g(take_tok_embed, cfg.vocab_size * H);
    fill_1(take_final, H);
    fill_g(take_lm_head, cfg.vocab_size * H);
    for (size_t L = 0; L < cfg.n_layers; ++L) {
        const auto& lo = m.layer_offs.data()[L];
        fill_1(lo.attn_norm, H);
        fill_g(lo.Wq, H * H);
        fill_g(lo.Wk, KV * H);
        fill_g(lo.Wv, KV * H);
        fill_g(lo.Wo, H * H);
        fill_1(lo.ffn_norm, H);
        fill_g(lo.Wgate, F * H);
        fill_g(lo.Wup,   F * H);
        fill_g(lo.Wdown, H * F);
    }

    m.weights.tok_embed  = m.buf.data() + take_tok_embed;
    m.weights.final_norm = m.buf.data() + take_final;
    m.weights.lm_head    = m.buf.data() + take_lm_head;
    m.weights.layers.resize(cfg.n_layers);
    for (size_t L = 0; L < cfg.n_layers; ++L) {
        auto& lw = m.weights.layers.data()[L];
        const auto& lo = m.layer_offs.data()[L];
        lw.attn_norm = m.buf.data() + lo.attn_norm;
        lw.Wq=m.buf.data()+lo.Wq; lw.Wk=m.buf.data()+lo.Wk;
        lw.Wv=m.buf.data()+lo.Wv; lw.Wo=m.buf.data()+lo.Wo;
        lw.ffn_norm=m.buf.data()+lo.ffn_norm;
        lw.Wgate=m.buf.data()+lo.Wgate; lw.Wup=m.buf.data()+lo.Wup;
        lw.Wdown=m.buf.data()+lo.Wdown;
    }
    return m;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args args;
    if (parse_args(argc, argv, args)) return 2;
    if (args.show_help) { usage(argv[0]); return 0; }

    // Tokenizer
    BPETokenizer tok;
    if (!args.vocab.empty() && !args.merges.empty()) {
        if (!tok.load(args.vocab, args.merges)) { LOG_FATAL("tokenizer load failed"); }
    } else {
        tok.build_minimal();
    }

    // Random seed: if seed=0, draw one from random_device.
    if (args.seed == 0) {
        std::random_device rd;
        args.seed = (uint64_t(rd()) << 32) | rd();
    }

    // Hold buffers alive for the duration of main.
    RandomModel rm;
    ModelFile mf;
    Model model;
    ModelConfig cfg;

    if (!args.model.empty()) {
        if (!mf.open(args.model)) LOG_FATAL("could not open model: %s", args.model.c_str());
        cfg = mf.config();
        model.init(cfg);
        if (!mf.populate(model.weights())) LOG_FATAL("model file missing required tensors");
        if (!args.quiet) std::printf("loaded: %s\n", args.model.c_str());
    } else {
        // Demo config
        cfg.vocab_size = tok.vocab_size();
        cfg.hidden_dim = 256;
        cfg.n_layers   = 4;
        cfg.n_heads    = 8;
        cfg.n_kv_heads = 8;
        cfg.head_dim   = 32;
        cfg.ffn_dim    = 512;
        cfg.max_seq_len = 256;
        rm = make_random_model(cfg);
        model.init(cfg);
        model.weights() = std::move(rm.weights);
        if (!args.quiet) std::printf("[demo] random %zu-layer model, vocab=%zu, hidden=%zu\n",
                                     cfg.n_layers, cfg.vocab_size, cfg.hidden_dim);
    }

    // Apply --threads BEFORE first global_pool() call. After that, the pool
    // is fixed.
    if (args.threads > 0) set_global_pool_size(size_t(args.threads));
    if (!args.quiet) std::printf("threads: %u\n",
                                 unsigned(global_pool().num_threads()));

    InferenceEngine engine;
    engine.init(model, tok);

    GenerateOptions opts;
    opts.max_new_tokens = size_t(std::max(1, args.n_predict));
    opts.sampling.temperature   = args.temperature;
    opts.sampling.top_k         = args.top_k;
    opts.sampling.top_p         = args.top_p;
    opts.sampling.min_p         = args.min_p;
    opts.sampling.repeat_penalty = args.repeat_penalty;
    opts.sampling.repeat_window  = args.repeat_window;
    opts.sampling.seed = args.seed;
    opts.on_token = [](int32_t, const std::string& piece) {
        std::printf("%s", piece.c_str());
        std::fflush(stdout);
        return true;
    };

    if (!args.quiet) std::printf("\n%s", args.prompt.c_str());
    GenerateStats st;
    engine.generate(args.prompt, opts, &st);
    std::printf("\n");

    if (!args.quiet) {
        std::printf("\n----- stats -----\n");
        std::printf("  prompt:    %zu tok\n", st.prompt_tokens);
        std::printf("  generated: %zu tok\n", st.generated_tokens);
        std::printf("  prefill:   %.1f ms (%.1f tok/s)\n", st.prefill_ms,
                    st.prompt_tokens / (st.prefill_ms / 1000.0 + 1e-9));
        std::printf("  decode:    %.1f ms (%.1f tok/s)\n", st.decode_ms,
                    st.tokens_per_second());
        std::printf("  kv used:   %.1f / %.1f MB\n",
                    engine.cache().bytes_used() / 1e6,
                    engine.cache().bytes_reserved() / 1e6);
    }
    return 0;
}
