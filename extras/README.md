# extras/

Auxiliary code that lives alongside the llama.cpp + TurboQuant stack.

| dir | what |
|---|---|
| [`standalone/`](standalone) | a from-scratch CPU LLM inference engine in pure C++17 (the original turbocpp v0). AVX2/AVX-512 matmul, K-quants, GQA, YaRN, mirostat, beam search, GBNF subset, OpenAI-compat HTTP server. ~3500 lines, zero deps. Useful as a study reference and a low-footprint runtime when you don't need llama.cpp's full feature set. |

The "main" project is `llama.cpp/` + `turboquant/`. Everything in
`extras/` is preserved-but-decoupled: it has its own CMakeLists and
its own CI job and is not built by default.
