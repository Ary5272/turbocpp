"""CLI: python -m turboquant <hf_model_dir> <output_dir> [--block 128]

Rotates a HF LLaMA-family model in-place and saves the result to
output_dir, where you can then point llama.cpp's convert_hf_to_gguf.py at
it. End-to-end:

    python -m turboquant ./Llama-3-8B ./Llama-3-8B-tq
    python llama.cpp/convert_hf_to_gguf.py ./Llama-3-8B-tq \\
           --outfile Llama-3-8B-tq.gguf
    ./llama.cpp/build/bin/llama-quantize Llama-3-8B-tq.gguf \\
           Llama-3-8B-tq-Q4_K_M.gguf Q4_K_M

The Q4_K_M file at the end runs on stock llama.cpp with measurably
lower perplexity than a Q4_K_M of the un-rotated model.
"""
import argparse
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit(
        "TurboQuant needs torch + transformers:\n"
        "    pip install torch transformers safetensors"
    )

from . import rotate_llama_model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="TurboQuant offline weight rotator")
    p.add_argument("model_dir", type=Path, help="HF model directory")
    p.add_argument("out_dir",   type=Path, help="output directory (will be created)")
    p.add_argument("--block",   type=int, default=128, help="Hadamard block size (power of 2)")
    p.add_argument("--no-fuse", action="store_true",
                   help="skip RMSNorm γ → linear fusion (debug only)")
    args = p.parse_args(argv)

    print(f"loading {args.model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    print(f"rotating with block_size={args.block} (fuse_norms={not args.no_fuse}) ...")
    rotate_llama_model(model, block_size=args.block, fuse_norms=not args.no_fuse)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving rotated model to {args.out_dir} ...")
    model.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)
    print("done. now run:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {args.out_dir} \\")
    print(f"         --outfile {args.out_dir.name}.gguf")
    print(f"  llama.cpp/build/bin/llama-quantize {args.out_dir.name}.gguf \\")
    print(f"         {args.out_dir.name}-Q4_K_M.gguf Q4_K_M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
