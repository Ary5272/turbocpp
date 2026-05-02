"""Unified `turbocpp` CLI.

Subcommands:
  rotate    apply Hadamard rotation to a HF model directory
  bench     run the synthetic rotation/quant MSE microbench
  generate  one-shot inference via llama-cpp-python (needs [runtime] extra)
  serve     OpenAI-compatible HTTP server via llama-cpp-python.server

Examples:
  turbocpp rotate ./Llama-3-8B ./Llama-3-8B-tq
  turbocpp bench
  turbocpp generate -m model.gguf -p "Hello" -n 64
  turbocpp serve    -m model.gguf --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# rotate (delegates to the existing Python module — no behavior change)
# ---------------------------------------------------------------------------
def _cmd_rotate(args) -> int:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("rotate needs:  pip install torch transformers safetensors")

    from . import rotate_llama_model
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
    print(
        "done. next: convert + quantize with llama.cpp:\n"
        f"  python llama.cpp/convert_hf_to_gguf.py {args.out_dir} "
        f"--outfile {args.out_dir.name}.gguf\n"
        f"  llama.cpp/build/bin/llama-quantize {args.out_dir.name}.gguf "
        f"{args.out_dir.name}-Q4_K_M.gguf Q4_K_M"
    )
    return 0


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------
def _cmd_bench(args) -> int:
    from .bench import run_bench
    run_bench(seed=args.seed)
    return 0


# ---------------------------------------------------------------------------
# generate / serve — both go through llama-cpp-python
# ---------------------------------------------------------------------------
def _import_llama_cpp():
    try:
        from llama_cpp import Llama   # noqa: F401
        return None
    except ImportError:
        return (
            "llama-cpp-python isn't installed. either:\n"
            "  pip install 'turbocpp[runtime]'\n"
            "or, on a platform where source-build fails (e.g. HF Spaces):\n"
            "  pip install https://huggingface.co/datasets/AIencoder/llama-cpp-wheels/"
            "resolve/main/llama_cpp_python-0.3.16%2Bbasic_avx2_fma_f16c-cp312-"
            "cp312-manylinux_2_31_x86_64.whl"
        )


def _cmd_generate(args) -> int:
    err = _import_llama_cpp()
    if err:
        print(err, file=sys.stderr)
        return 2
    from llama_cpp import Llama
    import time
    llm = Llama(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads or None,
        verbose=False,
    )
    t0 = time.time()
    out = llm(
        args.prompt,
        max_tokens=args.n_predict,
        temperature=args.temperature,
        top_p=args.top_p,
        echo=False,
    )
    dt = time.time() - t0
    text = out["choices"][0]["text"]
    n = out["usage"]["completion_tokens"]
    sys.stdout.write(text)
    sys.stdout.flush()
    print(
        f"\n\n[{n} tokens in {dt:.2f}s -> {n/max(dt,1e-3):.1f} tok/s]",
        file=sys.stderr,
    )
    return 0


def _cmd_speculative(args) -> int:
    err = _import_llama_cpp()
    if err:
        print(err, file=sys.stderr)
        return 2
    from llama_cpp import Llama
    from .speculative import speculative_generate

    common = dict(n_ctx=args.ctx, n_threads=args.threads or None,
                  verbose=False, logits_all=True)
    target = Llama(model_path=args.model, **common)
    draft  = Llama(model_path=args.draft,  **common)

    # Tokenize once via the target's tokenizer (must match draft's by family).
    prompt_ids = target.tokenize(args.prompt.encode("utf-8"), add_bos=True)
    eos = target.token_eos()

    def emit(_id, piece):
        sys.stdout.write(piece); sys.stdout.flush()

    out, stats = speculative_generate(
        target=target, draft=draft,
        prompt_tokens=list(prompt_ids),
        max_new_tokens=args.n_predict,
        draft_lookahead=args.lookahead,
        eos_token=eos,
        on_token=emit,
    )
    print()
    print(
        f"\n[{len(out)} tok in {stats.decode_seconds:.2f}s "
        f"= {len(out)/max(stats.decode_seconds,1e-3):.1f} tok/s | "
        f"accept {stats.accept_rate*100:.0f}% "
        f"({stats.accepted}/{stats.proposed}) | "
        f"speedup vs single-target ≈ {stats.speedup_factor:.2f}×]",
        file=sys.stderr,
    )
    return 0


def _cmd_pick_wheel(args) -> int:
    from .cpu_features import (
        best_wheel_url, candidate_urls, detect_variant, gpu_wheel_url,
    )
    if args.gpu:
        print(f"# gpu backend: {args.gpu}")
        print(gpu_wheel_url(args.gpu))
        return 0
    if args.all:
        for u in candidate_urls():
            print(u)
    else:
        print(f"# variant: {detect_variant()}")
        print(best_wheel_url())
    return 0


def _cmd_serve(args) -> int:
    err = _import_llama_cpp()
    if err:
        print(err, file=sys.stderr)
        return 2
    # llama-cpp-python ships its own OpenAI-compatible FastAPI server.
    # We just dispatch into it with the right CLI args.
    try:
        from llama_cpp.server.app import create_app
        from llama_cpp.server.settings import ServerSettings, ModelSettings
        import uvicorn
    except ImportError:
        print(
            "serve needs llama-cpp-python's server extras:\n"
            "  pip install 'llama-cpp-python[server]'",
            file=sys.stderr,
        )
        return 2

    server_settings = ServerSettings(host=args.host, port=args.port)
    model_settings = [ModelSettings(
        model=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads or 0,
    )]
    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="turbocpp",
        description="llama.cpp + TurboQuant — unified CLI",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # rotate
    pr = sub.add_parser("rotate", help="apply Hadamard rotation to a HF model")
    pr.add_argument("model_dir", type=Path, help="HF model directory")
    pr.add_argument("out_dir",   type=Path, help="output directory")
    pr.add_argument("--block",   type=int, default=128,
                    help="Hadamard block size (power of 2)")
    pr.add_argument("--no-fuse", action="store_true",
                    help="skip RMSNorm γ → linear fusion (debug)")
    pr.set_defaults(func=_cmd_rotate)

    # bench
    pb = sub.add_parser("bench", help="synthetic rotation/quant MSE microbench")
    pb.add_argument("--seed", type=int, default=0)
    pb.set_defaults(func=_cmd_bench)

    # generate
    pg = sub.add_parser("generate", help="run inference via llama-cpp-python")
    pg.add_argument("-m", "--model",      required=True, help="path to GGUF")
    pg.add_argument("-p", "--prompt",     required=True)
    pg.add_argument("-n", "--n-predict",  type=int,   default=128)
    pg.add_argument("-t", "--temperature", type=float, default=0.7)
    pg.add_argument("--top-p",           type=float, default=0.95)
    pg.add_argument("--ctx",             type=int,   default=2048)
    pg.add_argument("--threads",         type=int,   default=0)
    pg.set_defaults(func=_cmd_generate)

    # serve
    ps = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    ps.add_argument("-m", "--model",   required=True, help="path to GGUF")
    ps.add_argument("--host",          default="127.0.0.1")
    ps.add_argument("--port",          type=int, default=8080)
    ps.add_argument("--ctx",           type=int, default=4096)
    ps.add_argument("--threads",       type=int, default=0)
    ps.set_defaults(func=_cmd_serve)

    # speculative
    psp = sub.add_parser(
        "speculative",
        help="speculative decoding: small draft + big target (1.5-3x faster)"
    )
    psp.add_argument("-m", "--model",       required=True,
                     help="target GGUF (the real model)")
    psp.add_argument("-d", "--draft",       required=True,
                     help="draft GGUF (smaller/faster, same family)")
    psp.add_argument("-p", "--prompt",      required=True)
    psp.add_argument("-n", "--n-predict",   type=int, default=128)
    psp.add_argument("-k", "--lookahead",   type=int, default=4,
                     help="number of draft tokens proposed per round")
    psp.add_argument("--ctx",               type=int, default=2048)
    psp.add_argument("--threads",           type=int, default=0)
    psp.set_defaults(func=_cmd_speculative)

    # pick-wheel
    pw = sub.add_parser(
        "pick-wheel",
        help="print best prebuilt llama-cpp-python wheel URL for this host",
    )
    pw.add_argument("--all", action="store_true",
                    help="print full fallback ladder, not just the top pick")
    pw.add_argument("--gpu", choices=("cuda12", "cuda11", "vulkan",
                                      "rocm", "sycl", "opencl"),
                    help="force a GPU-accelerated variant instead of CPU")
    pw.set_defaults(func=_cmd_pick_wheel)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
