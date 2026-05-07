#!/usr/bin/env bash
# End-to-end speculative-decoding speedup benchmark.
#
# Argument: a HuggingFace LLaMA-family model directory.
# Produces: a 4-way head-to-head:
#
#    1. baseline Q4_K_M, single-model decode             (ref tok/s)
#    2. baseline Q4_K_M, speculative w/ Q2_K_M draft     (= 1.5-2.5x)
#    3. TurboQuant Q3_K_M, single-model decode           (memory-bound win)
#    4. TurboQuant Q3_K_M, speculative w/ TQ-Q2_K draft  (compounded)
#
# Usage:  ./scripts/bench_speculative.sh /path/to/HF/Llama-3-8B
#
# Requires:
#   - turbocpp:           pip install 'turbocpp[runtime]'
#   - docker  (we forward convert / quantize / llama-bench / speculative
#              into ggml-org/llama.cpp:full - no submodule needed)

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <hf_model_dir>" >&2
    exit 2
fi

MODEL_DIR=$(realpath "$1")
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

if ! command -v turbocpp >/dev/null; then
    echo "error: turbocpp not on PATH. Install with: pip install 'turbocpp[runtime]'" >&2
    exit 1
fi

# helper: prep base + rotated GGUFs at multiple quant levels
prep() {
    echo "[1/8] convert baseline -> fp16"
    turbocpp convert "$MODEL_DIR" --outfile "$OUT/base.f16.gguf" --outtype f16
    echo "[2/8] quantize Q4_K_M"
    turbocpp quantize "$OUT/base.f16.gguf" "$OUT/base.Q4_K_M.gguf" Q4_K_M
    echo "[3/8] quantize Q2_K_M (the draft)"
    turbocpp quantize "$OUT/base.f16.gguf" "$OUT/base.Q2_K_M.gguf" Q2_K_M

    echo "[4/8] TurboQuant rotate"
    turbocpp rotate "$MODEL_DIR" "$OUT/rotated"
    echo "[5/8] convert rotated -> fp16"
    turbocpp convert "$OUT/rotated" --outfile "$OUT/tq.f16.gguf" --outtype f16
    echo "[6/8] quantize TQ-Q3_K_M"
    turbocpp quantize "$OUT/tq.f16.gguf" "$OUT/tq.Q3_K_M.gguf" Q3_K_M
    echo "[7/8] quantize TQ-Q2_K (the speculative draft)"
    turbocpp quantize "$OUT/tq.f16.gguf" "$OUT/tq.Q2_K.gguf" Q2_K
    echo "[8/8] all GGUFs ready in $OUT"
    ls -lh "$OUT"/*.gguf
}

bench_single() {
    local label=$1; local model=$2
    echo
    echo "=== $label (single-model) ==="
    turbocpp llama-bench -m "$model" -p 64 -n 256 | tail -n +2
}

bench_spec() {
    local label=$1; local target=$2; local draft=$3
    echo
    echo "=== $label (speculative, draft=$(basename "$draft")) ==="
    turbocpp speculative -m "$target" -d "$draft" \
        -p "Once upon a time, in a land far away," -n 256 -k 4 \
        2>&1 | tail -1
}

prep
bench_single "baseline Q4_K_M"          "$OUT/base.Q4_K_M.gguf"
bench_spec   "baseline Q4_K_M"          "$OUT/base.Q4_K_M.gguf" "$OUT/base.Q2_K_M.gguf"
bench_single "TurboQuant Q3_K_M"        "$OUT/tq.Q3_K_M.gguf"
bench_spec   "TurboQuant Q3_K_M"        "$OUT/tq.Q3_K_M.gguf"   "$OUT/tq.Q2_K.gguf"

cat <<MSG

Read the tg128 column in llama-bench output for tok/s. Speculative rows
print "[N tok in Ts = X tok/s | accept Y%]" - that X is the wall-clock
tok/s the user actually sees. The speedup factor is the ratio against
the matching single-model row.

Expected (memory-bound CPU, e.g. Sapphire Rapids 8-core, DDR5):
  baseline single-model Q4_K_M     ~ 1.0x ref
  baseline speculative + Q2_K_M    ~ 1.6-2.4x
  TurboQuant single-model Q3_K_M   ~ 1.2x
  TurboQuant speculative + TQ-Q2_K ~ 2.0-3.0x
MSG
