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

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <hf_model_dir>" >&2
    exit 2
fi

MODEL_DIR=$1
OUT_DIR=$(mktemp -d)
trap 'rm -rf "$OUT_DIR"' EXIT

LLAMA=$(dirname "$0")/../llama.cpp/build/bin
if [[ ! -x "$LLAMA/llama-quantize" ]]; then
    echo "build llama.cpp first: cmake -S llama.cpp -B llama.cpp/build && cmake --build llama.cpp/build -j" >&2
    exit 1
fi

echo "[1/4] Convert baseline → fp16 GGUF"
python "$(dirname "$0")/../llama.cpp/convert_hf_to_gguf.py" "$MODEL_DIR" \
       --outfile "$OUT_DIR/base.f16.gguf" --outtype f16

echo "[2/4] Quantize baseline to Q4_K_M"
"$LLAMA/llama-quantize" "$OUT_DIR/base.f16.gguf" "$OUT_DIR/base.q4km.gguf" Q4_K_M

echo "[3/4] TurboQuant-rotate then convert + quantize to Q3_K_M"
python -m turboquant "$MODEL_DIR" "$OUT_DIR/rotated"
python "$(dirname "$0")/../llama.cpp/convert_hf_to_gguf.py" "$OUT_DIR/rotated" \
       --outfile "$OUT_DIR/tq.f16.gguf" --outtype f16
"$LLAMA/llama-quantize" "$OUT_DIR/tq.f16.gguf" "$OUT_DIR/tq.q3km.gguf" Q3_K_M

echo "[4/4] llama-bench head-to-head"
echo "----- BASELINE Q4_K_M -----"
"$LLAMA/llama-bench" -m "$OUT_DIR/base.q4km.gguf" -p 128 -n 128
echo
echo "----- TURBOQUANT Q3_K_M ---"
"$LLAMA/llama-bench" -m "$OUT_DIR/tq.q3km.gguf" -p 128 -n 128
echo
echo "Compare 'tg128' (token-generation) — TQ-Q3 should be faster on"
echo "memory-bound CPUs. Compare perplexity with llama-perplexity if you"
echo "want quality numbers."
