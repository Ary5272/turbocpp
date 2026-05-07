"""Unified `turbocpp` CLI.

Run `turbocpp --help` for the up-to-date subcommand list (it auto-renders
from the argparse setup in `main()` below; keeping a duplicate list here
just rots).

Architecture notes (orient yourself before editing):

* Every subcommand handler is `_cmd_<name>(args) -> int`. argparse
  registration lives in `main()` near the bottom of this file.
* All Llama() construction goes through `_open_llama(args, **overrides)`
  so `-ngl`, `--n-batch`, `--rope-*`, `--flash-attn`, and HF-ref
  resolution are wired up exactly once.
* `runtime_probe.collect_runtime_topology()` is the single source of
  truth shared by `info` (JSON dump) and `doctor` (human-readable
  checklist). Add new probes there, not in either command.
* Commands that need llama.cpp tools (convert / quantize / perplexity
  / imatrix / llama-cli / llama-bench / speculative) are forwarded
  through `llama_docker.run_tool()` into ggml-org's official image -
  we don't vendor llama.cpp as a submodule anymore.
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
        "done. next: convert + quantize:\n"
        f"  turbocpp convert  {args.out_dir} --outfile {args.out_dir.name}.gguf\n"
        f"  turbocpp quantize {args.out_dir.name}.gguf "
        f"{args.out_dir.name}-Q4_K_M.gguf Q4_K_M"
    )
    return 0


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------
def _cmd_bench(args) -> int:
    from .bench import run_bench

    run_bench(seed=args.seed, fmt=getattr(args, "format", "text"))
    return 0


# ---------------------------------------------------------------------------
# generate / serve — both go through llama-cpp-python
# ---------------------------------------------------------------------------
def _import_llama_cpp():
    try:
        from llama_cpp import Llama  # noqa: F401

        return None
    except ImportError:
        # `pick-wheel` returns the right URL for this CPU + Python combo,
        # which is dramatically more useful than naming a fixed variant.
        try:
            from .cpu_features import best_wheel_url

            url = best_wheel_url()
        except Exception:
            url = "https://huggingface.co/datasets/AIencoder/TurboCpp_Wheels"
        return (
            "llama-cpp-python isn't installed. either:\n"
            "  pip install 'turbocpp[runtime]'\n"
            "or, on a platform where source-build fails (e.g. HF Spaces),\n"
            "use the prebuilt wheel matched to this host:\n"
            f"  pip install {url}\n"
            "  (or run `turbocpp pick-wheel --all` for the full fallback ladder)"
        )


def _build_grammar(args):
    """Construct a llama_cpp.LlamaGrammar from --grammar (GBNF) or
    --json-schema. Returns None when neither is provided."""
    from pathlib import Path

    from llama_cpp import LlamaGrammar  # type: ignore

    if getattr(args, "grammar", None):
        text = Path(args.grammar).read_text(encoding="utf-8")
        return LlamaGrammar.from_string(text)
    if getattr(args, "json_schema", None):
        import json as _json

        schema_text = Path(args.json_schema).read_text(encoding="utf-8")
        return LlamaGrammar.from_json_schema(_json.dumps(_json.loads(schema_text)))
    return None


def _msgs_to_markdown(msgs: list[dict]) -> str:
    """Render a chat history as a Markdown transcript."""
    out: list[str] = ["# turbocpp chat transcript", ""]
    for m in msgs:
        role = m.get("role", "?")
        head = {"system": "## system", "user": "## user", "assistant": "## assistant"}.get(
            role, f"## {role}"
        )
        out += [head, "", m.get("content", ""), ""]
    return "\n".join(out)


def _open_llama(args, **overrides):
    """Construct a llama_cpp.Llama with shared kwargs from `args`. Override
    or extend with **overrides (e.g. embedding=True, logits_all=True).
    Resolves model aliases via config.resolve_model."""
    from llama_cpp import Llama  # type: ignore

    from .config import resolve_model

    kwargs = dict(
        model_path=resolve_model(args.model),
        n_ctx=getattr(args, "ctx", 2048),
        n_threads=getattr(args, "threads", 0) or None,
        seed=getattr(args, "seed", 0) if getattr(args, "seed", 0) != 0 else -1,
        verbose=False,
    )
    # GPU offload: -ngl N. -1 means "all layers", 0 = CPU only (default).
    ngl = getattr(args, "n_gpu_layers", 0)
    if ngl:
        kwargs["n_gpu_layers"] = ngl
    # Optional shared knobs (only forwarded when the parser actually had
    # them, so subcommands that don't expose them aren't affected).
    n_batch = getattr(args, "n_batch", 0)
    if n_batch:
        kwargs["n_batch"] = n_batch
    rope_freq_base = getattr(args, "rope_freq_base", 0.0)
    if rope_freq_base:
        kwargs["rope_freq_base"] = rope_freq_base
    rope_freq_scale = getattr(args, "rope_freq_scale", 0.0)
    if rope_freq_scale:
        kwargs["rope_freq_scale"] = rope_freq_scale
    flash_attn = getattr(args, "flash_attn", False)
    if flash_attn:
        kwargs["flash_attn"] = True
    kwargs.update(overrides)
    return Llama(**kwargs)


def _resolve_prompt(args) -> str:
    """generate accepts --prompt, --prompt-file, or stdin. This picks
    whichever is provided (precedence: -p > -f > stdin)."""
    if getattr(args, "prompt", None):
        return args.prompt
    if getattr(args, "prompt_file", None):
        from pathlib import Path

        return Path(args.prompt_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("error: provide -p PROMPT, -f FILE, or pipe text on stdin")


def _cmd_generate(args) -> int:
    err = _import_llama_cpp()
    if err:
        print(err, file=sys.stderr)
        return 2
    import time

    prompt = _resolve_prompt(args)
    logits_all = bool(args.logprobs and args.logprobs > 0)
    llm = _open_llama(args, logits_all=logits_all)
    grammar = _build_grammar(args)
    t0 = time.time()
    n = 0
    if args.logprobs and args.logprobs > 0:
        # Non-streaming, since llama-cpp-python only returns logprobs in
        # the full completion response.
        out = llm(
            prompt,
            max_tokens=args.n_predict,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=args.stop or None,
            grammar=grammar,
            logprobs=args.logprobs,
            echo=False,
        )
        text = out["choices"][0]["text"]
        lp = out["choices"][0].get("logprobs") or {}
        sys.stdout.write(text)
        sys.stdout.flush()
        n = out["usage"]["completion_tokens"]
        if lp.get("top_logprobs"):
            import json as _json

            sys.stderr.write("\n--- top_logprobs ---\n")
            sys.stderr.write(_json.dumps(lp["top_logprobs"], indent=2))
            sys.stderr.write("\n")
    else:
        # Stream tokens as they're generated — no buffering wait.
        import json as _json

        jsonl = getattr(args, "format", "text") == "jsonl"
        for chunk in llm(
            prompt,
            max_tokens=args.n_predict,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=args.stop or None,
            grammar=grammar,
            echo=False,
            stream=True,
        ):
            piece = chunk["choices"][0]["text"]
            if jsonl:
                sys.stdout.write(
                    _json.dumps(
                        {
                            "token": piece,
                            "index": n,
                            "finish_reason": chunk["choices"][0].get("finish_reason"),
                        }
                    )
                    + "\n"
                )
            else:
                sys.stdout.write(piece)
            sys.stdout.flush()
            n += 1
    dt = time.time() - t0
    if not args.quiet:
        print(
            f"\n\n[{n} tokens in {dt:.2f}s -> {n / max(dt, 1e-3):.1f} tok/s]",
            file=sys.stderr,
        )
    else:
        sys.stdout.write("\n")
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

    from .config import resolve_model

    args.model = resolve_model(args.model)

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
    elif msgs and args.system:
        # Resumed history may carry a stale system prompt; if the user
        # passed --system this run, surface the mismatch so they can /reset
        # or accept it as-is. Silently overriding would be worse.
        existing_sys = next((m["content"] for m in msgs if m.get("role") == "system"), None)
        if existing_sys is not None and existing_sys != args.system:
            print(
                "(note: resumed history has a different system prompt than "
                "--system; use /system <NEW> or /reset to update)",
                file=sys.stderr,
            )

    # Route through _open_llama so n_gpu_layers + future shared kwargs
    # (e.g. n_batch, rope_freq_*) only have to be wired up in one place.
    llm = _open_llama(args, chat_format=args.chat_format or None)
    grammar = _build_grammar(args)

    def save():
        try:
            history_file.write_text(json.dumps(msgs, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"(history save failed: {e})", file=sys.stderr)

    print(
        "turbocpp chat — slash cmds: /help /quit /reset /save P /load P /multi /history /system TEXT\n"
        f"history: {history_file}",
        file=sys.stderr,
    )

    def _read_user() -> str | None:
        """Returns next user message, or None on EOF/quit. /multi reads
        until a line that's just `EOF` (heredoc-style)."""
        try:
            line = input("\n› ")
        except (EOFError, KeyboardInterrupt):
            return None
        if line.strip() == "/multi":
            print("  (multi-line: end with a single line saying EOF)", file=sys.stderr)
            buf: list[str] = []
            while True:
                try:
                    nxt = input("… ")
                except (EOFError, KeyboardInterrupt):
                    break
                if nxt.strip() == "EOF":
                    break
                buf.append(nxt)
            return "\n".join(buf)
        return line

    while True:
        user = _read_user()
        if user is None:
            print()
            save()
            return 0
        s = user.strip()
        if not s:
            continue
        if s in ("/help", "/?"):
            print(
                "  /quit, /exit          end the session (history saved)\n"
                "  /reset                clear conversation\n"
                "  /history              dump conversation so far\n"
                "  /system <TEXT>        replace the system prompt\n"
                "  /save <PATH>          save to PATH (.md or .json by extension)\n"
                "  /load <PATH>          load JSON conversation from PATH\n"
                "  /multi                next message reads until a line saying EOF",
                file=sys.stderr,
            )
            continue
        if s in ("/quit", "/exit"):
            save()
            return 0
        if s == "/reset":
            msgs = [{"role": "system", "content": args.system}] if args.system else []
            save()
            print("(history cleared)", file=sys.stderr)
            continue
        if s == "/history":
            for m in msgs:
                role = m["role"]
                content = m["content"]
                tag = {"system": "[sys]", "user": "[usr]", "assistant": "[ai ]"}.get(role, role)
                print(f"{tag} {content[:300]}{'…' if len(content) > 300 else ''}", file=sys.stderr)
            continue
        if s.startswith("/system "):
            new_sys = s[len("/system ") :].strip()
            # Replace existing system message or prepend.
            msgs = [m for m in msgs if m.get("role") != "system"]
            msgs.insert(0, {"role": "system", "content": new_sys})
            save()
            print("(system prompt updated)", file=sys.stderr)
            continue
        if s.startswith("/save "):
            target = Path(s[6:].strip())
            if target.suffix.lower() == ".md":
                target.write_text(_msgs_to_markdown(msgs), encoding="utf-8")
            else:
                target.write_text(json.dumps(msgs, indent=2), encoding="utf-8")
            print(f"(saved to {target})", file=sys.stderr)
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
                stop=args.stop or None,
                grammar=grammar,
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
    Exit code = number of FAILs. Shares its data source with `info` so
    they're always consistent."""
    import shutil
    import sys
    import urllib.request

    from .runtime_probe import collect_runtime_topology

    info = collect_runtime_topology()
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

    print(f"turbocpp {info['turbocpp']} doctor — {sys.platform}")

    py = sys.version_info
    row(
        "PASS" if py >= (3, 10) else "FAIL",
        "python ≥ 3.10",
        f"{py.major}.{py.minor}.{py.micro}",
    )

    v = info["cpu_variant"]
    row("PASS" if "avx2" in v or "avx512" in v else "WARN", "cpu feature variant", v)

    lcpp = info["llama_cpp"]
    if lcpp.get("installed"):
        row("PASS", "llama-cpp-python", lcpp.get("version", "?"))
        gpu = lcpp.get("gpu_offload")
        row(
            "PASS" if gpu else "WARN",
            "llama-cpp-python GPU offload",
            "yes" if gpu else "CPU-only build" if gpu is False else "couldn't probe",
        )
    else:
        row("WARN", "llama-cpp-python", "not installed — `pip install turbocpp[runtime]`")

    if info["docker_present"]:
        row("PASS", "docker on PATH", shutil.which("docker") or "")
        row(
            "PASS" if info["llama_image_pulled"] else "WARN",
            f"image {info['llama_image']}",
            "cached locally"
            if info["llama_image_pulled"]
            else "not pulled yet (auto on first use)",
        )
    else:
        row(
            "WARN",
            "docker on PATH",
            "missing — `turbocpp llama …` / convert / quantize unavailable",
        )

    g = info["gpu"]
    row("PASS" if g else "WARN", "GPU", g or "none detected (CPU-only)")

    t = info["torch"]
    if t.get("installed"):
        bits = []
        if t.get("cuda_available"):
            bits.append("cuda")
        if t.get("mps_available"):
            bits.append("mps")
        row("PASS", "torch (rotate)", f"{t.get('version')} ({', '.join(bits) or 'cpu'})")
    else:
        row("WARN", "torch (rotate)", "not installed - `pip install torch transformers`")

    # HF wheel mirror reachability — kept in doctor only since it costs network.
    if getattr(args, "no_network", False):
        row("WARN", "HF wheel URL reachable", "skipped (--no-network)")
    else:
        try:
            url = info["best_wheel_url"]
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=5).close()
            row("PASS", "HF wheel URL reachable", url[:60] + "…")
        except Exception as e:
            row("WARN", "HF wheel URL reachable", f"{type(e).__name__}: {e}")

    print(f"\n{'OK' if fails == 0 else f'{fails} failure(s)'}.", file=sys.stderr)
    return fails


def _cmd_info(args) -> int:
    """Print the runtime topology as JSON. Single source of truth lives
    in runtime_probe.collect_runtime_topology() so info and doctor agree."""
    import json

    from .runtime_probe import collect_runtime_topology

    print(json.dumps(collect_runtime_topology(), indent=2))
    return 0


def _cmd_version(args) -> int:
    """Print just the version (matches `turbocpp --version`)."""
    from . import __version__

    print(__version__)
    return 0


def _cmd_rm_model(args) -> int:
    """Delete cached GGUFs in ~/.cache/turbocpp/models/. With --all wipes
    the whole cache; otherwise pass exact filename(s)."""
    import os
    from pathlib import Path

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "turbocpp" / "models"
    if not cache.is_dir():
        print("(no cache directory)", file=sys.stderr)
        return 0

    if args.all:
        files = list(cache.rglob("*.gguf"))
    else:
        if not args.names:
            print("error: pass filenames to delete or --all", file=sys.stderr)
            return 2
        files = []
        for n in args.names:
            files += list(cache.rglob(n))
        if not files:
            print(f"no match for {args.names}", file=sys.stderr)
            return 1

    if args.dry_run:
        for p in files:
            print(p)
        return 0
    freed = 0
    for p in files:
        try:
            sz = p.stat().st_size
            p.unlink()
            freed += sz
            print(f"rm {p}", file=sys.stderr)
        except OSError as e:
            print(f"failed to delete {p}: {e}", file=sys.stderr)
    print(f"freed {freed / 1e9:.2f} GB", file=sys.stderr)
    return 0


def _cmd_speculative(args) -> int:
    """Speculative decoding via llama.cpp's own `llama-speculative` binary
    (delegated through the official Docker image). The previous in-process
    Python harness mutated `Llama.n_tokens` to roll back the KV cache —
    that's not a public API, breaks across llama-cpp-python versions, and
    silently produced wrong results when it didn't crash. This path uses
    the upstream-tested implementation."""
    from pathlib import Path

    from .config import resolve_model
    from .llama_docker import DEFAULT_IMAGE, docker_available, run_tool

    if not docker_available():
        print(
            "error: speculative decoding delegates to llama-speculative "
            "in ggml-org/llama.cpp:full. Install Docker first.",
            file=sys.stderr,
        )
        return 2
    # HF refs (`owner/repo:file.gguf` or `hf://...`) are downloaded to the
    # cache, then bind-mounted into the container like any local path.
    target_p = Path(resolve_model(args.model)).resolve()
    draft_p = Path(resolve_model(args.draft)).resolve()
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


def _cmd_embed(args) -> int:
    """Compute sentence embeddings via llama-cpp-python's `Llama(embedding=True)`.
    Reads each input line as a separate sentence, prints `tab`-joined floats."""
    err = _import_llama_cpp()
    if err:
        print(err, file=sys.stderr)
        return 2
    import json

    if args.input == "-" or not args.input:
        sentences = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    else:
        from pathlib import Path

        sentences = [
            line.rstrip("\n")
            for line in Path(args.input).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not sentences and args.text:
        sentences = [args.text]
    if not sentences:
        print("nothing to embed (use --text or pipe lines on stdin)", file=sys.stderr)
        return 2

    llm = _open_llama(args, embedding=True)

    def _l2_normalize(v):
        s = sum(x * x for x in v) ** 0.5
        return [x / s for x in v] if s > 0 else v

    out = []
    for s in sentences:
        v = llm.create_embedding(s)["data"][0]["embedding"]
        if args.normalize:
            v = _l2_normalize(v)
        out.append({"text": s, "embedding": v})

    if args.format == "tsv":
        for row in out:
            sys.stdout.write("\t".join(f"{x:.6g}" for x in row["embedding"]) + "\n")
    elif args.format == "jsonl":
        for row in out:
            sys.stdout.write(json.dumps(row) + "\n")
    else:  # json
        sys.stdout.write(json.dumps(out, indent=2))
        sys.stdout.write("\n")
    return 0


def _cmd_tokenize(args) -> int:
    """Tokenize a prompt and report ids + count. Useful for context-budget
    estimation before a long generation."""
    err = _import_llama_cpp()
    if err:
        print(err, file=sys.stderr)
        return 2

    if args.text:
        text = args.text
    elif args.input and args.input != "-":
        from pathlib import Path

        text = Path(args.input).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    llm = _open_llama(args)
    ids = llm.tokenize(text.encode("utf-8"), add_bos=args.add_bos)
    if args.format == "ids":
        print(" ".join(str(i) for i in ids))
    elif args.format == "pieces":
        for i in ids:
            piece = llm.detokenize([i]).decode("utf-8", errors="replace")
            print(f"{i}\t{piece!r}")
    else:  # count
        print(len(ids))
    return 0


def _cmd_download(args) -> int:
    """Fetch a GGUF (or any single file) from a HuggingFace repo to
    ~/.cache/turbocpp/models/. Wraps huggingface_hub with progress + a
    sha256 integrity check on completion."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "needs: pip install 'huggingface_hub<2.0'",
            file=sys.stderr,
        )
        return 2
    import os
    from pathlib import Path

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "turbocpp" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    # Allow `turbocpp download owner/repo:file.gguf` (HF-ref form) as a
    # shorthand for `turbocpp download owner/repo file.gguf`. Lets users
    # paste the same identifier they'd use with `-m`.
    repo, filename = args.repo, args.filename
    if filename is None and ":" in repo and repo.count("/") == 1:
        repo, _, filename = repo.partition(":")
    if filename is None:
        print("error: pass FILENAME or use owner/repo:file.gguf form", file=sys.stderr)
        return 2
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        cache_dir=str(cache),
        local_dir=str(cache) if args.flat else None,
    )
    print(path)
    if args.sha256:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != args.sha256.lower():
            print(f"sha256 mismatch: expected {args.sha256}, got {actual}", file=sys.stderr)
            return 3
        print(f"sha256 ok ({actual})", file=sys.stderr)
    return 0


def _cmd_list_models(args) -> int:
    """List GGUFs available to turbocpp: aliases from config.toml + files
    in ~/.cache/turbocpp/models/."""
    import os
    from pathlib import Path

    from .config import load

    cfg = load()
    aliases = cfg.get("models", {}) or {}
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "turbocpp" / "models"
    cache_files = sorted(cache.rglob("*.gguf")) if cache.is_dir() else []

    if args.format == "json":
        import json

        print(
            json.dumps(
                {
                    "aliases": aliases,
                    "cache": [
                        {"path": str(p), "size_bytes": p.stat().st_size} for p in cache_files
                    ],
                },
                indent=2,
            )
        )
    else:
        if aliases:
            print("# aliases (from ~/.config/turbocpp/config.toml)")
            for k, v in aliases.items():
                print(f"  {k:<20}  {v}")
        else:
            print("# (no aliases configured — see SECURITY.md / config.py docstring)")
        print()
        if cache_files:
            print(f"# cached GGUFs in {cache}:")
            for p in cache_files:
                size = p.stat().st_size / 1e9
                print(f"  {size:5.2f} GB  {p}")
        else:
            print(f"# (no GGUFs in {cache} — fetch one with `turbocpp download REPO FILE`)")
    return 0


def _cmd_list_templates(args) -> int:
    """Print the chat templates llama-cpp-python knows about."""
    err = _import_llama_cpp()
    if err:
        print(err, file=sys.stderr)
        return 2
    from llama_cpp import llama_chat_format

    registry = getattr(llama_chat_format, "LlamaChatCompletionHandlerRegistry", None)
    handlers = getattr(registry, "_chat_handlers", None) if registry is not None else None
    formats = sorted(handlers.keys()) if handlers else []
    # llama-cpp-python's API for this varies; fall back to attribute probe.
    if not formats:
        formats = sorted(
            n
            for n in dir(llama_chat_format)
            if not n.startswith("_")
            and callable(getattr(llama_chat_format, n))
            and n not in {"register_chat_format", "register_chat_completion_handler"}
        )
    for n in formats:
        print(n)
    return 0


def _cmd_config(args) -> int:
    """`turbocpp config show|init|path` - inspect / scaffold the toml file."""
    from .config import config_path, load

    p = config_path()
    op = args.op
    if op == "path":
        print(p)
        return 0
    if op == "show":
        if not p.is_file():
            print(f"# (no config at {p})")
            return 0
        print(p.read_text(encoding="utf-8"), end="")
        return 0
    if op == "init":
        if p.is_file() and not args.force:
            print(f"refusing to overwrite {p} (pass --force)", file=sys.stderr)
            return 2
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "# turbocpp config - all sections optional. Re-generate any time\n"
            "# with `turbocpp config init --force`.\n\n"
            "[defaults]\n"
            "# threads = 8        # CPU threads (0 = auto)\n"
            "# ctx     = 4096     # default context window\n\n"
            "[defaults.generate]\n"
            "# temperature = 0.7\n"
            "# top_p       = 0.95\n\n"
            "[defaults.chat]\n"
            '# system    = "You are a concise assistant."\n'
            "# n_predict = 1024\n\n"
            "[models]\n"
            "# Aliases: turbocpp generate -m tiny -p 'hi'\n"
            '# tiny  = "~/models/tinyllama-Q4_K_M.gguf"\n',
            encoding="utf-8",
        )
        print(f"wrote {p}")
        return 0
    # `validate` - parse and exit non-zero if the file is malformed.
    if op == "validate":
        if not p.is_file():
            print(f"# (no config at {p})")
            return 0
        cfg = load()
        if not cfg:
            # load() swallows parse errors and returns {}; re-parse to surface them
            try:
                import tomllib  # py>=3.11
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            with p.open("rb") as f:
                tomllib.load(f)  # raises on bad toml
        print("ok")
        return 0
    raise SystemExit(f"unknown op: {op}")


def _cmd_quickstart(args) -> int:
    """Download a tiny known-good GGUF and run a one-shot generation, so
    a fresh `pip install turbocpp[runtime]` can prove it works."""
    err = _import_llama_cpp()
    if err:
        print(err, file=sys.stderr)
        return 2
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("needs: pip install 'huggingface_hub<2.0'", file=sys.stderr)
        return 2
    import os
    from pathlib import Path

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "turbocpp" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    fname = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    print(f"[1/2] downloading {fname} from {repo} ...", file=sys.stderr)
    path = hf_hub_download(repo_id=repo, filename=fname, cache_dir=str(cache))
    print(f"   → {path}", file=sys.stderr)

    print("[2/2] generating sample completion ...", file=sys.stderr)
    from llama_cpp import Llama

    llm = Llama(model_path=path, n_ctx=512, verbose=False)
    out = llm(
        "Q: What is the capital of France?\nA:",
        max_tokens=24,
        temperature=0.0,
        echo=False,
        stop=["\n"],
        stream=False,
    )
    # stream=False guarantees a Mapping rather than an iterator.
    assert isinstance(out, dict), f"expected dict, got {type(out)}"
    text = out["choices"][0]["text"].strip()
    print(f"\nmodel said: {text}\n")
    print("✓ turbocpp + llama-cpp-python work on this host.")
    return 0


def _cmd_llama_passthrough(args) -> int:
    """Forward to a binary in ggml-org's official llama.cpp image."""
    from .llama_docker import DEFAULT_IMAGE, LLAMA_TOOLS, docker_available, run_tool

    tool = getattr(args, "_tool", None) or args.tool
    if tool == "--list" or tool == "list":
        for t in LLAMA_TOOLS:
            print(t)
        return 0
    if tool not in LLAMA_TOOLS:
        print(
            f"unknown tool {tool!r}; choose from {', '.join(LLAMA_TOOLS)}",
            file=sys.stderr,
        )
        return 2
    if not docker_available():
        print(
            f"error: `turbocpp {tool}` needs Docker (we delegate to "
            f"ggml-org/llama.cpp:full). Install Docker Desktop or the "
            f"`docker` CLI, then re-run.",
            file=sys.stderr,
        )
        return 2

    # Default Q4_K_M when `turbocpp quantize IN OUT` is called with only
    # the two GGUF paths (the third positional is normally the quant type).
    if tool == "llama-quantize":
        rest = list(args.rest or [])
        if rest and rest[0] == "--":
            rest = rest[1:]
        # Heuristic: if exactly two non-flag positionals and no -f/-o/etc,
        # llama-quantize would refuse — append the default type.
        non_flag = [r for r in rest if not r.startswith("-")]
        looks_bare = len(non_flag) == 2 and len(rest) == 2
        if looks_bare:
            args.rest = rest + ["Q4_K_M"]
            print("[turbocpp] no quant type given, defaulting to Q4_K_M", file=sys.stderr)

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
        try:
            ports[int(h)] = int(c)
        except ValueError:
            print(f"bad --port {spec!r}; both sides must be integers", file=sys.stderr)
            return 2

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

    # Resolve API key from --api-key, $TURBOCPP_API_KEY, or none.
    # Special-case: --api-key generate (or env=generate) emits a fresh
    # 32-char URL-safe random key, so users can spin up an authed server
    # without first inventing a secret.
    import os as _os
    import secrets

    from .config import resolve_model

    api_key = args.api_key or _os.environ.get("TURBOCPP_API_KEY", "")
    if api_key.lower() in ("generate", "auto", "random"):
        api_key = secrets.token_urlsafe(24)
        print(f"[serve] generated API key: {api_key}", file=sys.stderr)

    ssettings_kwargs: dict = dict(host=args.host, port=args.port)
    if api_key:
        ssettings_kwargs["api_key"] = api_key
    server_settings = ServerSettings(**ssettings_kwargs)

    ms_kwargs: dict = dict(
        model=resolve_model(args.model),
        n_ctx=args.ctx,
        n_threads=args.threads or 0,
    )
    # GPU offload: --n-gpu-layers / -ngl
    ngl = getattr(args, "n_gpu_layers", 0)
    if ngl:
        ms_kwargs["n_gpu_layers"] = ngl
    model_settings = [ModelSettings(**ms_kwargs)]
    app = create_app(
        server_settings=server_settings,
        model_settings=model_settings,
    )
    if api_key and not args.quiet:
        print(
            f"[serve] API key required: clients must send "
            f"'Authorization: Bearer {api_key[:6]}…' (truncated)",
            file=sys.stderr,
        )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _defaults_for(subcommand: str) -> dict:
    """Pull subcommand defaults from ~/.config/turbocpp/config.toml. Empty
    on missing file or parse error."""
    try:
        from .config import defaults_for

        return defaults_for(subcommand)
    except Exception:
        return {}


def _ensure_utf8_stdio() -> None:
    """Force stdout/stderr to UTF-8 on Windows. Default `cp1252` blows up on
    non-ASCII chars in argparse help (e.g. the em-dash in our description),
    crashing `turbocpp --help` before the user sees anything. PEP 540
    `PYTHONUTF8=1` would fix it too, but most users won't set it."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass


def main(argv=None) -> int:
    _ensure_utf8_stdio()
    from . import __version__

    p = argparse.ArgumentParser(
        prog="turbocpp",
        description="llama.cpp + TurboQuant - unified CLI",
        epilog=(
            "Tips:\n"
            "  -m accepts: local GGUF | config alias | hf://owner/repo/file.gguf\n"
            "             | owner/repo:file.gguf  (auto-fetched from HF Hub)\n"
            "  Run `turbocpp doctor` for an environment health check.\n"
            "  Run `turbocpp config init` to scaffold ~/.config/turbocpp/config.toml.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-V", "--version", action="version", version=f"turbocpp {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # rotate
    pr = sub.add_parser("rotate", help="apply Hadamard rotation to a HF model")
    pr.add_argument("model_dir", type=Path, help="HF model directory")
    pr.add_argument("out_dir", type=Path, help="output directory")
    pr.add_argument("--block", type=int, default=128, help="Hadamard block size (power of 2)")
    pr.add_argument(
        "--no-fuse", action="store_true", help="skip RMSNorm gamma -> linear fusion (debug)"
    )
    pr.set_defaults(func=_cmd_rotate)

    # bench
    pb = sub.add_parser("bench", help="synthetic rotation/quant MSE microbench")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (json is machine-readable; text is the default report)",
    )
    pb.set_defaults(func=_cmd_bench)

    # generate
    gd = _defaults_for("generate")
    pg = sub.add_parser("generate", help="run inference via llama-cpp-python")
    pg.add_argument(
        "-m",
        "--model",
        required=True,
        help="GGUF path | alias | hf://owner/repo/file.gguf | owner/repo:file.gguf",
    )
    pg.add_argument(
        "-p", "--prompt", default=None, help="prompt text (else --prompt-file or stdin)"
    )
    pg.add_argument("-f", "--prompt-file", default=None, help="read prompt from this file")
    pg.add_argument("-n", "--n-predict", type=int, default=gd.get("n_predict", 128))
    pg.add_argument("-t", "--temperature", type=float, default=gd.get("temperature", 0.7))
    pg.add_argument("--top-p", type=float, default=gd.get("top_p", 0.95))
    pg.add_argument("--ctx", type=int, default=gd.get("ctx", 2048))
    pg.add_argument("--threads", type=int, default=gd.get("threads", 0))
    pg.add_argument(
        "--seed", type=int, default=gd.get("seed", 0), help="RNG seed (0 = random per call)"
    )
    pg.add_argument(
        "--stop", action="append", default=[], help="stop sequence; repeat for multiple"
    )
    pg.add_argument("--grammar", help="path to a GBNF grammar file")
    pg.add_argument("--json-schema", help="path to a JSON schema (constrains output)")
    pg.add_argument(
        "--logprobs",
        type=int,
        default=0,
        metavar="N",
        help="report top-N logprobs per token (disables streaming)",
    )
    pg.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress timing stats; just print the completion text",
    )
    pg.add_argument(
        "-ngl",
        "--n-gpu-layers",
        type=int,
        default=0,
        metavar="N",
        help="offload N transformer layers to GPU (-1 = all). Requires a "
        "GPU-built llama-cpp-python wheel.",
    )
    pg.add_argument(
        "--format",
        choices=("text", "jsonl"),
        default="text",
        help="output format: 'text' streams raw tokens, 'jsonl' emits one "
        "JSON line per token (good for piping into jq).",
    )
    pg.add_argument("--n-batch", type=int, default=0, metavar="N", help="logical batch size")
    pg.add_argument(
        "--rope-freq-base",
        type=float,
        default=0.0,
        metavar="F",
        help="override RoPE base frequency (advanced; 0 = use model's metadata)",
    )
    pg.add_argument(
        "--rope-freq-scale",
        type=float,
        default=0.0,
        metavar="F",
        help="RoPE frequency scale for context-length stretching",
    )
    pg.add_argument(
        "--flash-attn",
        action="store_true",
        help="enable flash attention if the build supports it",
    )
    pg.set_defaults(func=_cmd_generate)

    # serve
    ps = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    ps.add_argument(
        "-m",
        "--model",
        required=True,
        help="GGUF path | alias | hf://owner/repo/file.gguf | owner/repo:file.gguf",
    )
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8080)
    ps.add_argument("--ctx", type=int, default=4096)
    ps.add_argument("--threads", type=int, default=0)
    ps.add_argument(
        "--api-key",
        default="",
        help="require Bearer auth on every request (also reads "
        "TURBOCPP_API_KEY env var). Empty string = no auth. "
        "Pass 'generate' (or 'auto' / 'random') to mint a fresh "
        "32-char URL-safe key on startup.",
    )
    ps.add_argument("-q", "--quiet", action="store_true", help="suppress startup banner")
    ps.add_argument(
        "-ngl",
        "--n-gpu-layers",
        type=int,
        default=0,
        metavar="N",
        help="offload N transformer layers to GPU (-1 = all). Requires a "
        "GPU-built llama-cpp-python wheel.",
    )
    ps.set_defaults(func=_cmd_serve)

    # speculative
    psp = sub.add_parser(
        "speculative", help="speculative decoding: small draft + big target (1.5-3x faster)"
    )
    psp.add_argument(
        "-m",
        "--model",
        required=True,
        help="target GGUF | alias | hf://owner/repo/file.gguf | owner/repo:file.gguf",
    )
    psp.add_argument(
        "-d",
        "--draft",
        required=True,
        help="draft GGUF | alias | hf://owner/repo/file.gguf | owner/repo:file.gguf",
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
    cd = _defaults_for("chat")
    pc = sub.add_parser("chat", help="multi-turn chat REPL (auto template, persistent history)")
    pc.add_argument(
        "-m",
        "--model",
        required=True,
        help="GGUF path | alias | hf://owner/repo/file.gguf | owner/repo:file.gguf",
    )
    pc.add_argument("-s", "--system", default=cd.get("system", ""), help="system prompt")
    pc.add_argument("-n", "--n-predict", type=int, default=cd.get("n_predict", 512))
    pc.add_argument("-t", "--temperature", type=float, default=cd.get("temperature", 0.7))
    pc.add_argument("--ctx", type=int, default=cd.get("ctx", 4096))
    pc.add_argument("--threads", type=int, default=cd.get("threads", 0))
    pc.add_argument("--seed", type=int, default=cd.get("seed", 0))
    pc.add_argument(
        "--stop", action="append", default=[], help="stop sequence; repeat for multiple"
    )
    pc.add_argument("--grammar", help="path to a GBNF grammar file")
    pc.add_argument("--json-schema", help="path to a JSON schema (constrains output)")
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
    pc.add_argument(
        "-ngl",
        "--n-gpu-layers",
        type=int,
        default=0,
        metavar="N",
        help="offload N transformer layers to GPU (-1 = all)",
    )
    pc.set_defaults(func=_cmd_chat)

    # embed
    pe = sub.add_parser("embed", help="compute sentence embeddings")
    pe.add_argument(
        "-m",
        "--model",
        required=True,
        help="GGUF path | alias | hf://owner/repo/file.gguf | owner/repo:file.gguf",
    )
    pe.add_argument("--text", help="single sentence (else reads stdin/--input lines)")
    pe.add_argument("-i", "--input", help="path to a text file (one sentence per line)")
    pe.add_argument("--format", choices=("json", "jsonl", "tsv"), default="json")
    pe.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize each vector (cosine-similarity-ready)",
    )
    pe.add_argument("--ctx", type=int, default=512)
    pe.add_argument("--threads", type=int, default=0)
    pe.add_argument(
        "-ngl",
        "--n-gpu-layers",
        type=int,
        default=0,
        metavar="N",
        help="offload N transformer layers to GPU",
    )
    pe.set_defaults(func=_cmd_embed)

    # tokenize
    pt = sub.add_parser("tokenize", help="tokenize a prompt (count / ids / pieces)")
    pt.add_argument(
        "-m",
        "--model",
        required=True,
        help="GGUF path | alias | hf://owner/repo/file.gguf | owner/repo:file.gguf",
    )
    pt.add_argument("--text", help="text to tokenize")
    pt.add_argument("-i", "--input", help="file with text to tokenize")
    pt.add_argument("--format", choices=("count", "ids", "pieces"), default="count")
    pt.add_argument("--add-bos", action="store_true", help="prepend BOS token")
    pt.add_argument("--ctx", type=int, default=512)
    pt.add_argument("--threads", type=int, default=0)
    pt.set_defaults(func=_cmd_tokenize)

    # download
    pdl = sub.add_parser("download", help="fetch a GGUF (or any file) from HF")
    pdl.add_argument(
        "repo",
        help="HF repo id (e.g. TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF), or "
        "the combined `owner/repo:file.gguf` form",
    )
    pdl.add_argument("filename", nargs="?", help="exact filename inside the repo")
    pdl.add_argument("--sha256", help="if set, verify the downloaded file matches")
    pdl.add_argument(
        "--flat",
        action="store_true",
        help="dump straight into ~/.cache/turbocpp/models/ instead of HF's snapshot tree",
    )
    pdl.set_defaults(func=_cmd_download)

    # doctor (one-shot environment check)
    pd = sub.add_parser(
        "doctor", help="check turbocpp install health (deps, wheels, docker, GPU, ...)"
    )
    pd.add_argument(
        "--no-network",
        action="store_true",
        help="skip the HF wheel-URL HEAD probe (offline / air-gapped)",
    )
    pd.set_defaults(func=_cmd_doctor)

    # info
    pi = sub.add_parser("info", help="show runtime topology (wheel, backends, GPU, ...)")
    pi.set_defaults(func=_cmd_info)

    # version
    pv = sub.add_parser("version", help="print just the package version")
    pv.set_defaults(func=_cmd_version)

    # rm-model
    prm = sub.add_parser(
        "rm-model",
        help="delete cached GGUFs in ~/.cache/turbocpp/models/",
    )
    prm.add_argument("names", nargs="*", help="exact filenames to delete (glob ok)")
    prm.add_argument("--all", action="store_true", help="wipe every cached GGUF")
    prm.add_argument(
        "--dry-run", action="store_true", help="print paths that would be deleted, don't delete"
    )
    prm.set_defaults(func=_cmd_rm_model)

    # list-models
    plm = sub.add_parser("list-models", help="list configured aliases + cached GGUFs")
    plm.add_argument("--format", choices=("text", "json"), default="text")
    plm.set_defaults(func=_cmd_list_models)

    # list-templates
    plt = sub.add_parser("list-templates", help="list chat templates known to llama-cpp-python")
    plt.set_defaults(func=_cmd_list_templates)

    # config (show / init / path / validate)
    pcfg = sub.add_parser("config", help="show / init / path / validate the turbocpp config.toml")
    pcfg.add_argument("op", choices=("show", "init", "path", "validate"))
    pcfg.add_argument("--force", action="store_true", help="(init) overwrite an existing file")
    pcfg.set_defaults(func=_cmd_config)

    # quickstart — zero-config "is this thing on?"
    pq = sub.add_parser("quickstart", help="download TinyLlama + run a sample completion")
    pq.set_defaults(func=_cmd_quickstart)

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
        ("convert", "convert_hf_to_gguf.py", "HF model -> GGUF (delegates to llama.cpp image)"),
        ("quantize", "llama-quantize", "GGUF -> quantized GGUF"),
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
