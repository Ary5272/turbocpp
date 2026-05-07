#!/usr/bin/env bash
#
# End-to-end speed comparison: stock llama.cpp at Q4_K_M vs TurboQuant
# rotated model at Q3_K_M (the "drop a tier and run faster" path).
#
# Usage:  ./scripts/bench_e2e.sh /path/to/HF/Llama-3-8B
#
# Reports tokens/sec for both, side by side, using llama-bench (the
# canonical microbench in llama.cpp). On memory-bound CPUs the rotated
# Q3 path should beat baseline Q4 in tok/s while matching its perplexity.
#
# Requires:
#   pip install 'turbocpp[runtime]'
#   docker  (we forward convert / quantize / llama-bench through ggml-org's image)

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <hf_model_dir>" >&2
    exit 2
fi

MODEL_DIR=$(realpath "$1")
OUT_DIR=$(mktemp -d)
trap 'rm -rf "$OUT_DIR"' EXIT

if ! command -v turbocpp >/dev/null; then
    echo "error: turbocpp not on PATH. Install with: pip install 'turbocpp[runtime]'" >&2
    exit 1
fi

echo "[1/4] Convert baseline -> fp16 GGUF"
turbocpp convert "$MODEL_DIR" --outfile "$OUT_DIR/base.f16.gguf" --outtype f16

echo "[2/4] Quantize baseline to Q4_K_M"
turbocpp quantize "$OUT_DIR/base.f16.gguf" "$OUT_DIR/base.q4km.gguf" Q4_K_M

echo "[3/4] TurboQuant-rotate then convert + quantize to Q3_K_M"
turbocpp rotate "$MODEL_DIR" "$OUT_DIR/rotated"
turbocpp convert "$OUT_DIR/rotated" --outfile "$OUT_DIR/tq.f16.gguf" --outtype f16
turbocpp quantize "$OUT_DIR/tq.f16.gguf" "$OUT_DIR/tq.q3km.gguf" Q3_K_M

echo "[4/4] llama-bench head-to-head"
echo "----- BASELINE Q4_K_M -----"
turbocpp llama-bench -m "$OUT_DIR/base.q4km.gguf" -p 128 -n 128
echo
echo "----- TURBOQUANT Q3_K_M ---"
turbocpp llama-bench -m "$OUT_DIR/tq.q3km.gguf" -p 128 -n 128
echo
echo "Compare 'tg128' (token-generation) - TQ-Q3 should be faster on"
echo "memory-bound CPUs. Compare perplexity with \`turbocpp perplexity\` if"
echo "you want quality numbers."
