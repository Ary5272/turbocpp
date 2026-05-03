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

# argparse is imported above; re-exported under a top-level alias for the
# REMAINDER constant we reference deeper in the parser definition.


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
        from llama_cpp import Llama  # noqa: F401

        return None
    except ImportError:
        return (
            "llama-cpp-python isn't installed. either:\n"
            "  pip install 'turbocpp[runtime]'\n"
            "or, on a platform where source-build fails (e.g. HF Spaces):\n"
            "  pip install https://huggingface.co/datasets/AIencoder/TurboCpp_Wheels/"
            "resolve/main/llama_cpp_python-0.3.16%2Bbasic_avx2_fma_f16c-cp312-"
            "cp312-manylinux_2_31_x86_64.whl"
        )


def _cmd_generate(args) -> int:
    err = _import_llama_cpp()
    if err:
        print(err, file=sys.stderr)
        return 2
    import time

    from llama_cpp import Llama

    llm = Llama(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads or None,
        verbose=False,
    )
    t0 = time.time()
    n = 0
    # Stream tokens as they're generated — no buffering wait.
    for chunk in llm(
        args.prompt,
        max_tokens=args.n_predict,
        temperature=args.temperature,
        top_p=args.top_p,
        echo=False,
        stream=True,
    ):
        piece = chunk["choices"][0]["text"]
        sys.stdout.write(piece)
        sys.stdout.flush()
        n += 1
    dt = time.time() - t0
    print(
        f"\n\n[{n} tokens in {dt:.2f}s -> {n / max(dt, 1e-3):.1f} tok/s]",
        file=sys.stderr,
    )
    return 0


def _cmd_chat(args) -> int:
    """Multi-turn chat REPL with persistent history.

    History lives at ~/.cache/turbocpp/chat-<sha1(model_path)>.json so
    the next `turbocpp chat -m same.gguf` resumes the same conversation.
    Slash commands:  /reset (clear)  /save PATH  /load PATH  /quit"""
    err = _import_llama_cpp()
    if err:
        print(err, file=sys.stderr)
        return 2
    import hashlib
    import json
    import os
    from pathlib import Path

    from llama_cpp import Llama

    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "turbocpp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_id = hashlib.sha1(str(Path(args.model).resolve()).encode()).hexdigest()[:12]
    history_file = cache_dir / f"chat-{model_id}.json"

    msgs: list[dict] = []
    if not args.no_resume and history_file.exists():
        try:
            msgs = json.loads(history_file.read_text(encoding="utf-8"))
            print(f"(resumed {len(msgs)} messages from {history_file.name})", file=sys.stderr)
        except Exception:
            msgs = []
    if not msgs and args.system:
        msgs.append({"role": "system", "content": args.system})

    llm = Llama(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads or None,
        chat_format=args.chat_format or None,
        verbose=False,
    )

    def save():
        try:
            history_file.write_text(json.dumps(msgs, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"(history save failed: {e})", file=sys.stderr)

    print(
        f"turbocpp chat — Ctrl-D / /quit to exit, /reset, /save PATH, /load PATH\n"
        f"history: {history_file}",
        file=sys.stderr,
    )
    while True:
        try:
            user = input("\n› ")
        except (EOFError, KeyboardInterrupt):
            print()
            save()
            return 0
        s = user.strip()
        if not s:
            continue
        if s in ("/quit", "/exit"):
            save()
            return 0
        if s == "/reset":
            msgs = [{"role": "system", "content": args.system}] if args.system else []
            save()
            print("(history cleared)", file=sys.stderr)
            continue
        if s.startswith("/save "):
            Path(s[6:].strip()).write_text(json.dumps(msgs, indent=2), encoding="utf-8")
            print(f"(saved to {s[6:].strip()})", file=sys.stderr)
            continue
        if s.startswith("/load "):
            try:
                msgs = json.loads(Path(s[6:].strip()).read_text(encoding="utf-8"))
                print(f"(loaded {len(msgs)} messages)", file=sys.stderr)
            except Exception as e:
                print(f"(load failed: {e})", file=sys.stderr)
            continue

        msgs.append({"role": "user", "content": user})
        reply = ""
        try:
            for chunk in llm.create_chat_completion(
                messages=msgs,
                max_tokens=args.n_predict,
                temperature=args.temperature,
                stream=True,
            ):
                delta = chunk["choices"][0]["delta"].get("content", "")
                sys.stdout.write(delta)
                sys.stdout.flush()
                reply += delta
        except KeyboardInterrupt:
            print("\n(interrupted)", file=sys.stderr)
        msgs.append({"role": "assistant", "content": reply})
        save()


def _cmd_doctor(args) -> int:
    """Walk a checklist and print PASS / WARN / FAIL for each item.
    Exit code = number of FAILs."""
    import shutil
    import sys
    import urllib.request

    from . import __version__
    from .cpu_features import best_wheel_url, detect_variant
    from .llama_docker import DEFAULT_IMAGE, docker_available, image_present

    fails = 0

    def row(status, label, detail=""):
        nonlocal fails
        if status == "FAIL":
            fails += 1
        col = {
            "PASS": "\033[32mPASS\033[0m",
            "WARN": "\033[33mWARN\033[0m",
            "FAIL": "\033[31mFAIL\033[0m",
        }.get(status, status)
        print(f"  [{col}]  {label:<38} {detail}")

    print(f"turbocpp {__version__} doctor — {sys.platform}")

    # Python version
    py = sys.version_info
    if py >= (3, 10):
        row("PASS", "python ≥ 3.10", f"{py.major}.{py.minor}.{py.micro}")
    else:
        row("FAIL", "python ≥ 3.10", f"{py.major}.{py.minor}.{py.micro}")

    # CPU variant
    v = detect_variant()
    row("PASS" if "avx2" in v or "avx512" in v else "WARN", "cpu feature variant", v)

    # llama-cpp-python
    try:
        import llama_cpp

        ver = getattr(llama_cpp, "__version__", "?")
        row("PASS", "llama-cpp-python", ver)
        try:
            from llama_cpp import llama_supports_gpu_offload

            gpu = bool(llama_supports_gpu_offload())
            row(
                "PASS" if gpu else "WARN",
                "llama-cpp-python GPU offload",
                "yes" if gpu else "CPU-only build",
            )
        except Exception:
            row("WARN", "llama-cpp-python GPU offload", "couldn't probe")
    except ImportError:
        row("WARN", "llama-cpp-python", "not installed — `pip install turbocpp[runtime]`")

    # Docker (only required for the llama-* passthroughs)
    if docker_available():
        row("PASS", "docker on PATH", shutil.which("docker") or "")
        if image_present(DEFAULT_IMAGE):
            row("PASS", f"image {DEFAULT_IMAGE}", "cached locally")
        else:
            row("WARN", f"image {DEFAULT_IMAGE}", "not pulled yet (auto on first use)")
    else:
        row(
            "WARN",
            "docker on PATH",
            "missing — `turbocpp llama …` / convert / quantize unavailable",
        )

    # GPU
    if shutil.which("nvidia-smi"):
        row("PASS", "GPU", "nvidia (nvidia-smi)")
    elif shutil.which("rocminfo"):
        row("PASS", "GPU", "amd (rocminfo)")
    elif sys.platform == "darwin":
        import platform as _p

        if _p.machine() == "arm64":
            row("PASS", "GPU", "apple silicon (Metal)")
        else:
            row("WARN", "GPU", "intel mac — no GPU offload")
    else:
        row("WARN", "GPU", "none detected (CPU-only)")

    # HF wheel mirror reachability
    try:
        url = best_wheel_url()
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=5).close()
        row("PASS", "HF wheel URL reachable", url[:60] + "…")
    except Exception as e:
        row("WARN", "HF wheel URL reachable", f"{type(e).__name__}: {e}")

    print(f"\n{'OK' if fails == 0 else f'{fails} failure(s)'}.", file=sys.stderr)
    return fails


def _cmd_info(args) -> int:
    """Print the runtime topology: which wheel, which backends are
    compiled in, llama.cpp image tag, model paths, etc."""
    import json
    import platform
    import shutil

    from . import __version__
    from .cpu_features import best_wheel_url, detect_variant
    from .llama_docker import DEFAULT_IMAGE, docker_available, image_present

    info = {
        "turbocpp": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu_variant": detect_variant(),
        "best_wheel_url": best_wheel_url(),
        "docker_present": docker_available(),
        "llama_image": DEFAULT_IMAGE,
        "llama_image_pulled": image_present(DEFAULT_IMAGE) if docker_available() else False,
    }
    try:
        import llama_cpp  # noqa: F401

        info["llama_cpp_python"] = getattr(llama_cpp, "__version__", "?")
        from llama_cpp import llama_supports_gpu_offload  # type: ignore

        info["llama_gpu_offload"] = bool(llama_supports_gpu_offload())
    except ImportError:
        info["llama_cpp_python"] = None
    if shutil.which("nvidia-smi"):
        info["gpu"] = "nvidia (nvidia-smi present)"
    elif shutil.which("rocminfo"):
        info["gpu"] = "amd (rocminfo present)"
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        info["gpu"] = "apple silicon (Metal available)"
    else:
        info["gpu"] = None
    print(json.dumps(info, indent=2))
    return 0


def _cmd_speculative(args) -> int:
    """Speculative decoding via llama.cpp's own `llama-speculative` binary
    (delegated through the official Docker image). The previous in-process
    Python harness mutated `Llama.n_tokens` to roll back the KV cache —
    that's not a public API, breaks across llama-cpp-python versions, and
    silently produced wrong results when it didn't crash. This path uses
    the upstream-tested implementation."""
    from pathlib import Path

    from .llama_docker import DEFAULT_IMAGE, run_tool

    target_p = Path(args.model).resolve()
    draft_p = Path(args.draft).resolve()
    if target_p.parent != draft_p.parent:
        print("note: target and draft live in different dirs; mounting both", file=sys.stderr)

    mounts = {str(target_p.parent): "/m_target"}
    if draft_p.parent != target_p.parent:
        mounts[str(draft_p.parent)] = "/m_draft"
    target_in_ctr = f"/m_target/{target_p.name}"
    draft_in_ctr = (
        f"/m_target/{draft_p.name}"
        if draft_p.parent == target_p.parent
        else f"/m_draft/{draft_p.name}"
    )

    forwarded = [
        "-m",
        target_in_ctr,
        "-md",
        draft_in_ctr,
        "-p",
        args.prompt,
        "-n",
        str(args.n_predict),
        "--draft-max",
        str(args.lookahead),
        "-c",
        str(args.ctx),
    ]
    if args.threads:
        forwarded += ["-t", str(args.threads)]
    return run_tool(
        "llama-speculative", forwarded, image=args.image or DEFAULT_IMAGE, mounts=mounts
    )


def _cmd_llama_passthrough(args) -> int:
    """Forward to a binary in ggml-org's official llama.cpp image."""
    from .llama_docker import DEFAULT_IMAGE, LLAMA_TOOLS, run_tool

    tool = getattr(args, "_tool", None) or args.tool
    if tool not in LLAMA_TOOLS:
        print(f"unknown tool {tool!r}; choose from {LLAMA_TOOLS}", file=sys.stderr)
        return 2

    mounts = {}
    for spec in args.mount:
        if ":" not in spec:
            print(f"bad --mount {spec!r}; expected host:container", file=sys.stderr)
            return 2
        host, ctr = spec.split(":", 1)
        mounts[host] = ctr

    ports = {}
    for spec in args.port:
        if ":" not in spec:
            print(f"bad --port {spec!r}; expected host:container", file=sys.stderr)
            return 2
        h, c = spec.split(":", 1)
        ports[int(h)] = int(c)

    image = args.image or DEFAULT_IMAGE
    rest = list(args.rest or [])
    if rest and rest[0] == "--":
        rest = rest[1:]
    return run_tool(
        tool, rest, image=image, mounts=mounts, ports=ports, interactive=sys.stdin.isatty()
    )


def _cmd_pick_wheel(args) -> int:
    from .cpu_features import (
        best_wheel_url,
        candidate_urls,
        detect_variant,
        gpu_wheel_url,
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
        import uvicorn
        from llama_cpp.server.app import create_app
        from llama_cpp.server.settings import ModelSettings, ServerSettings
    except ImportError:
        print(
            "serve needs llama-cpp-python's server extras:\n"
            "  pip install 'llama-cpp-python[server]'",
            file=sys.stderr,
        )
        return 2

    server_settings = ServerSettings(host=args.host, port=args.port)
    model_settings = [
        ModelSettings(
            model=args.model,
            n_ctx=args.ctx,
            n_threads=args.threads or 0,
        )
    ]
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
    pr.add_argument("out_dir", type=Path, help="output directory")
    pr.add_argument("--block", type=int, default=128, help="Hadamard block size (power of 2)")
    pr.add_argument("--no-fuse", action="store_true", help="skip RMSNorm γ → linear fusion (debug)")
    pr.set_defaults(func=_cmd_rotate)

    # bench
    pb = sub.add_parser("bench", help="synthetic rotation/quant MSE microbench")
    pb.add_argument("--seed", type=int, default=0)
    pb.set_defaults(func=_cmd_bench)

    # generate
    pg = sub.add_parser("generate", help="run inference via llama-cpp-python")
    pg.add_argument("-m", "--model", required=True, help="path to GGUF")
    pg.add_argument("-p", "--prompt", required=True)
    pg.add_argument("-n", "--n-predict", type=int, default=128)
    pg.add_argument("-t", "--temperature", type=float, default=0.7)
    pg.add_argument("--top-p", type=float, default=0.95)
    pg.add_argument("--ctx", type=int, default=2048)
    pg.add_argument("--threads", type=int, default=0)
    pg.set_defaults(func=_cmd_generate)

    # serve
    ps = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    ps.add_argument("-m", "--model", required=True, help="path to GGUF")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8080)
    ps.add_argument("--ctx", type=int, default=4096)
    ps.add_argument("--threads", type=int, default=0)
    ps.set_defaults(func=_cmd_serve)

    # speculative
    psp = sub.add_parser(
        "speculative", help="speculative decoding: small draft + big target (1.5-3x faster)"
    )
    psp.add_argument("-m", "--model", required=True, help="target GGUF (the real model)")
    psp.add_argument(
        "-d", "--draft", required=True, help="draft GGUF (smaller/faster, same family)"
    )
    psp.add_argument("-p", "--prompt", required=True)
    psp.add_argument("-n", "--n-predict", type=int, default=128)
    psp.add_argument(
        "-k", "--lookahead", type=int, default=4, help="number of draft tokens proposed per round"
    )
    psp.add_argument("--ctx", type=int, default=2048)
    psp.add_argument("--threads", type=int, default=0)
    psp.add_argument("--image", default=None, help="override llama.cpp image tag (e.g. :full-cuda)")
    psp.set_defaults(func=_cmd_speculative)

    # chat (multi-turn REPL with auto chat template + persistent history)
    pc = sub.add_parser("chat", help="multi-turn chat REPL (auto template, persistent history)")
    pc.add_argument("-m", "--model", required=True, help="GGUF path")
    pc.add_argument("-s", "--system", default="", help="system prompt")
    pc.add_argument("-n", "--n-predict", type=int, default=512)
    pc.add_argument("-t", "--temperature", type=float, default=0.7)
    pc.add_argument("--ctx", type=int, default=4096)
    pc.add_argument("--threads", type=int, default=0)
    pc.add_argument(
        "--chat-format",
        default="",
        help="force template (llama-3, chatml, mistral-instruct, "
        "…); blank = auto from GGUF metadata",
    )
    pc.add_argument(
        "--no-resume",
        action="store_true",
        help="don't load previous conversation from ~/.cache/turbocpp/chat-*.json",
    )
    pc.set_defaults(func=_cmd_chat)

    # doctor (one-shot environment check)
    pd = sub.add_parser(
        "doctor", help="check turbocpp install health (deps, wheels, docker, GPU, …)"
    )
    pd.set_defaults(func=_cmd_doctor)

    # info
    pi = sub.add_parser("info", help="show runtime topology (wheel, backends, GPU, …)")
    pi.set_defaults(func=_cmd_info)

    # llama-cpp tool passthroughs (delegate to the official llama.cpp image)
    psp_tools = sub.add_parser(
        "llama",
        help="passthrough to a llama.cpp binary inside ggml-org's official "
        "Docker image (replaces the old git submodule)",
    )
    psp_tools.add_argument(
        "tool",
        help="binary name: llama-cli | llama-server | llama-quantize | "
        "llama-perplexity | llama-imatrix | llama-bench | llama-tokenize | "
        "llama-embedding | llama-export-lora | llama-gguf-split | "
        "llama-batched-bench | llama-speculative | convert_hf_to_gguf.py",
    )
    psp_tools.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to the tool (use -- to separate flags)",
    )
    psp_tools.add_argument(
        "--image",
        default=None,
        help="override image tag (default: ggml-org/llama.cpp:full; use :full-cuda for GPU)",
    )
    psp_tools.add_argument(
        "--mount", action="append", default=[], help="host:container bind mount, repeatable"
    )
    psp_tools.add_argument(
        "--port", action="append", default=[], help="host:container port forward, repeatable"
    )
    psp_tools.set_defaults(func=_cmd_llama_passthrough)

    # convenience aliases for the most-used tools
    for alias, tool, h in (
        ("convert", "convert_hf_to_gguf.py", "HF model → GGUF (delegates to llama.cpp image)"),
        ("quantize", "llama-quantize", "GGUF → quantized GGUF"),
        ("perplexity", "llama-perplexity", "compute perplexity on a corpus"),
        ("imatrix", "llama-imatrix", "build an importance matrix for K-quants"),
        ("llama-cli", "llama-cli", "raw llama-cli passthrough"),
        ("llama-bench", "llama-bench", "official llama.cpp microbench"),
    ):
        ap = sub.add_parser(alias, help=h)
        ap.add_argument("rest", nargs=argparse.REMAINDER)
        ap.add_argument("--image", default=None)
        ap.add_argument("--mount", action="append", default=[])
        ap.add_argument("--port", action="append", default=[])
        ap.set_defaults(func=_cmd_llama_passthrough, _tool=tool)

    # pick-wheel
    pw = sub.add_parser(
        "pick-wheel",
        help="print best prebuilt llama-cpp-python wheel URL for this host",
    )
    pw.add_argument(
        "--all", action="store_true", help="print full fallback ladder, not just the top pick"
    )
    pw.add_argument(
        "--gpu",
        choices=("cuda12", "cuda11", "vulkan", "rocm", "sycl", "opencl"),
        help="force a GPU-accelerated variant instead of CPU",
    )
    pw.set_defaults(func=_cmd_pick_wheel)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
