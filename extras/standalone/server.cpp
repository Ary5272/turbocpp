// turbocpp-server — minimal HTTP server with an OpenAI-compatible
// /v1/completions endpoint. Single-threaded, raw sockets, no external deps.
//
// Request body (JSON):
//   { "prompt": "...", "max_tokens": 64, "temperature": 0.8,
//     "top_k": 40, "top_p": 0.95, "min_p": 0.05,
//     "repeat_penalty": 1.1, "stop": ["<|eot|>"] }
//
// Response body (JSON):
//   { "id": "cmpl-..", "object": "text_completion", "model": "turbocpp",
//     "choices": [{"text": "...", "finish_reason": "stop"}],
//     "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M},
//     "tokens_per_second": F }
//
// This is intentionally minimal — it's enough for cURL / OpenAI client
// SDKs that point at a custom base URL. Streaming (SSE), embeddings,
// chat-completions endpoint, and authentication are roadmap.

#include "core/allocator.h"
#include "loader/gguf.h"
#include "model/transformer.h"
#include "runtime/inference.h"
#include "runtime/thread_pool.h"
#include "tokenizer/bpe.h"
#include "utils/logging.h"

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#  define NOMINMAX
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
   typedef SOCKET sock_t;
   #define CLOSESOCK closesocket
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
   typedef int sock_t;
   #define CLOSESOCK ::close
   #define INVALID_SOCKET (-1)
#endif

using namespace turbocpp;

// ---------------------------------------------------------------------------
// Tiny JSON helpers (just what we need)
// ---------------------------------------------------------------------------
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (uint8_t(c) < 0x20) { char buf[8]; std::snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
                else out += c;
        }
    }
    return out;
}

// Extract a quoted-string field by exact key match. Returns "" if absent.
// NOT a real parser — it scans for `"<key>"` then the next `"..."`. Good
// enough for flat objects.
static std::string json_get_string(const std::string& body, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto p = body.find(needle);
    if (p == std::string::npos) return "";
    p = body.find(':', p + needle.size());
    if (p == std::string::npos) return "";
    p = body.find('"', p);
    if (p == std::string::npos) return "";
    ++p;
    std::string out;
    while (p < body.size() && body[p] != '"') {
        if (body[p] == '\\' && p + 1 < body.size()) {
            char e = body[p + 1];
            switch (e) {
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                default: out += e;
            }
            p += 2;
        } else {
            out += body[p++];
        }
    }
    return out;
}

static double json_get_number(const std::string& body, const std::string& key, double def) {
    std::string needle = "\"" + key + "\"";
    auto p = body.find(needle);
    if (p == std::string::npos) return def;
    p = body.find(':', p + needle.size());
    if (p == std::string::npos) return def;
    ++p;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) ++p;
    char* endp = nullptr;
    double v = std::strtod(body.c_str() + p, &endp);
    return endp == body.c_str() + p ? def : v;
}

static std::vector<std::string> json_get_string_array(const std::string& body,
                                                      const std::string& key) {
    std::vector<std::string> out;
    std::string needle = "\"" + key + "\"";
    auto p = body.find(needle);
    if (p == std::string::npos) return out;
    p = body.find('[', p + needle.size());
    if (p == std::string::npos) return out;
    auto end = body.find(']', p);
    if (end == std::string::npos) return out;
    size_t i = p + 1;
    while (i < end) {
        while (i < end && body[i] != '"') ++i;
        if (i >= end) break;
        ++i;
        std::string s;
        while (i < end && body[i] != '"') {
            if (body[i] == '\\' && i + 1 < end) { s += body[i + 1]; i += 2; }
            else s += body[i++];
        }
        if (i < end) ++i;
        out.push_back(std::move(s));
    }
    return out;
}

// ---------------------------------------------------------------------------
// HTTP I/O
// ---------------------------------------------------------------------------
static bool send_all(sock_t s, const char* buf, size_t n) {
    while (n > 0) {
        int sent = ::send(s, buf, int(n), 0);
        if (sent <= 0) return false;
        buf += sent; n -= sent;
    }
    return true;
}

static void send_response(sock_t s, int status, const std::string& status_text,
                          const std::string& body, const char* ctype = "application/json") {
    std::ostringstream hdr;
    hdr << "HTTP/1.1 " << status << " " << status_text << "\r\n"
        << "Content-Type: " << ctype << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Connection: close\r\n\r\n";
    std::string h = hdr.str();
    send_all(s, h.data(), h.size());
    send_all(s, body.data(), body.size());
}

// Read until headers end + body of Content-Length bytes. Returns full request.
static bool recv_request(sock_t s, std::string& out, std::string& body) {
    char buf[4096];
    out.clear(); body.clear();
    size_t header_end = std::string::npos;
    size_t content_length = 0;

    while (header_end == std::string::npos) {
        int n = ::recv(s, buf, sizeof buf, 0);
        if (n <= 0) return false;
        out.append(buf, size_t(n));
        header_end = out.find("\r\n\r\n");
        if (out.size() > 1 << 20) return false;  // 1MB header cap
    }
    auto cl_pos = out.find("Content-Length:");
    if (cl_pos == std::string::npos) cl_pos = out.find("content-length:");
    if (cl_pos != std::string::npos) {
        content_length = size_t(std::strtoull(out.c_str() + cl_pos + 15, nullptr, 10));
    }
    body = out.substr(header_end + 4);
    while (body.size() < content_length) {
        int n = ::recv(s, buf, sizeof buf, 0);
        if (n <= 0) return false;
        body.append(buf, size_t(n));
    }
    body.resize(content_length);
    return true;
}

// ---------------------------------------------------------------------------
// Server context
// ---------------------------------------------------------------------------
struct ServerCtx {
    Model* model;
    BPETokenizer* tok;
    InferenceEngine engine;
    std::atomic<uint64_t> req_id{0};
};

static std::string handle_completion(ServerCtx& ctx, const std::string& body) {
    std::string prompt = json_get_string(body, "prompt");
    int n_predict = int(json_get_number(body, "max_tokens", 64));
    if (n_predict <= 0) n_predict = int(json_get_number(body, "n_predict", 64));
    auto stops = json_get_string_array(body, "stop");

    GenerateOptions opts;
    opts.max_new_tokens = size_t(n_predict);
    opts.sampling.temperature   = float(json_get_number(body, "temperature",   0.8));
    opts.sampling.top_k         = int  (json_get_number(body, "top_k",         40));
    opts.sampling.top_p         = float(json_get_number(body, "top_p",         0.95));
    opts.sampling.min_p         = float(json_get_number(body, "min_p",         0.05));
    opts.sampling.repeat_penalty = float(json_get_number(body, "repeat_penalty", 1.1));
    opts.sampling.seed          = uint64_t(json_get_number(body, "seed", 0));
    if (opts.sampling.seed == 0) {
        std::random_device rd;
        opts.sampling.seed = (uint64_t(rd()) << 32) | rd();
    }
    opts.stop_sequences = stops;

    GenerateStats st;
    ctx.engine.reset();          // fresh KV per request — no batching yet
    std::string text = ctx.engine.generate(prompt, opts, &st);

    const uint64_t id = ctx.req_id.fetch_add(1) + 1;
    char idbuf[64]; std::snprintf(idbuf, sizeof idbuf, "cmpl-%llu", (unsigned long long)id);

    std::ostringstream r;
    r << "{"
      << "\"id\":\"" << idbuf << "\","
      << "\"object\":\"text_completion\","
      << "\"model\":\"turbocpp\","
      << "\"choices\":[{"
      <<   "\"index\":0,"
      <<   "\"text\":\"" << json_escape(text) << "\","
      <<   "\"finish_reason\":\"" << (st.generated_tokens >= opts.max_new_tokens ? "length" : "stop") << "\""
      << "}],"
      << "\"usage\":{"
      <<   "\"prompt_tokens\":" << st.prompt_tokens << ","
      <<   "\"completion_tokens\":" << st.generated_tokens << ","
      <<   "\"total_tokens\":" << st.prompt_tokens + st.generated_tokens
      << "},"
      << "\"tokens_per_second\":" << st.tokens_per_second()
      << "}";
    return r.str();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void usage(const char* prog) {
    std::printf("%s -m MODEL [--vocab V --merges M] [--port 8080] [--host 0.0.0.0]\n",
                prog);
}

int main(int argc, char** argv) {
    std::string model_path, vocab, merges, host = "0.0.0.0";
    int port = 8080;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* k){ if(i+1>=argc){std::fprintf(stderr,"need %s\n",k);std::exit(2);} return argv[++i]; };
        if (a == "-m" || a == "--model")   model_path = need("--model");
        else if (a == "--vocab")           vocab = need("--vocab");
        else if (a == "--merges")          merges = need("--merges");
        else if (a == "--port")            port = std::atoi(need("--port"));
        else if (a == "--host")            host = need("--host");
        else if (a == "-h" || a == "--help"){ usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown: %s\n", a.c_str()); return 2; }
    }

    BPETokenizer tok;
    if (!vocab.empty() && !merges.empty()) {
        if (!tok.load(vocab, merges)) { LOG_FATAL("tok load failed"); }
    } else {
        tok.build_minimal();
    }

    ModelFile mf;
    Model model;
    if (!model_path.empty()) {
        if (!mf.open(model_path)) LOG_FATAL("model open failed: %s", model_path.c_str());
        model.init(mf.config());
        if (!mf.populate(model.weights())) LOG_FATAL("populate failed");
    } else {
        std::fprintf(stderr, "Note: running without -m uses a stub model.\n");
        ModelConfig cfg;
        cfg.vocab_size = tok.vocab_size();
        cfg.hidden_dim = 256; cfg.n_layers = 4; cfg.n_heads = 8; cfg.n_kv_heads = 8;
        cfg.head_dim = 32; cfg.ffn_dim = 512; cfg.max_seq_len = 256;
        model.init(cfg);
        // Empty weights → forward will crash. Real run needs -m.
    }

    ServerCtx ctx;
    ctx.model = &model;
    ctx.tok   = &tok;
    ctx.engine.init(model, tok);

#if defined(_WIN32)
    WSADATA wsa; if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) LOG_FATAL("WSAStartup");
#endif
    sock_t srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv == INVALID_SOCKET) LOG_FATAL("socket");
    int yes = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (char*)&yes, sizeof yes);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(uint16_t(port));
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
    if (::bind(srv, (sockaddr*)&addr, sizeof addr) < 0) LOG_FATAL("bind %s:%d", host.c_str(), port);
    if (::listen(srv, 16) < 0) LOG_FATAL("listen");

    std::printf("turbocpp-server listening on %s:%d\n", host.c_str(), port);
    std::printf("  POST /v1/completions   (OpenAI-compatible)\n");

    while (true) {
        sock_t cli = ::accept(srv, nullptr, nullptr);
        if (cli == INVALID_SOCKET) continue;
        std::string req, body;
        if (!recv_request(cli, req, body)) { CLOSESOCK(cli); continue; }

        if (req.compare(0, 4, "POST") == 0 &&
            (req.find(" /v1/completions") != std::string::npos ||
             req.find(" /completion")     != std::string::npos)) {
            try {
                std::string resp = handle_completion(ctx, body);
                send_response(cli, 200, "OK", resp);
            } catch (...) {
                send_response(cli, 500, "Internal Server Error",
                              "{\"error\":\"inference failed\"}");
            }
        } else if (req.compare(0, 7, "OPTIONS") == 0) {
            send_response(cli, 204, "No Content", "");
        } else if (req.compare(0, 3, "GET") == 0 && req.find(" /") != std::string::npos) {
            send_response(cli, 200, "OK",
                          "{\"server\":\"turbocpp\",\"endpoints\":[\"POST /v1/completions\"]}");
        } else {
            send_response(cli, 404, "Not Found",
                          "{\"error\":\"unknown endpoint\"}");
        }
        CLOSESOCK(cli);
    }
}
