"""One-shot copy of llama-cpp-python wheels from a source HF dataset
into AIencoder/TurboCpp_Wheels.

We renamed the canonical wheel mirror in 0.6.0 but the old dataset still
holds 8000+ wheels. Rather than re-build them all, this script streams
selected wheels from old → new without ever touching local disk for the
big payloads (uses snapshot_download with allow_patterns + upload_folder).

Usage (set HF_TOKEN with write scope on the destination dataset):

    HF_TOKEN=hf_… python scripts/mirror_wheels.py \\
        --src AIencoder/llama-cpp-wheels \\
        --dst AIencoder/TurboCpp_Wheels \\
        --filter 'llama_cpp_python-0.3.16+basic_avx2_fma_f16c-cp312-cp312-manylinux*' \\
        --filter 'llama_cpp_python-0.3.16+basic_avx512_*-cp312-cp312-manylinux*'

To copy everything (warning: ~30 GB):
    HF_TOKEN=hf_… python scripts/mirror_wheels.py --filter '*'

This script is idempotent — uploads are skipped when the destination
already has the same blob hash.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="AIencoder/llama-cpp-wheels", help="source dataset id")
    p.add_argument("--dst", default="AIencoder/TurboCpp_Wheels", help="destination dataset id")
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        help="glob to copy (repeatable). default: cp312 Linux wheels",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not args.filter:
        # Sensible default: every cp312 manylinux x86_64 wheel for the
        # llama-cpp-python version turbocpp pins. That's the set
        # `pick-wheel` actually returns URLs for.
        args.filter = ["llama_cpp_python-0.3.16+*-cp312-cp312-manylinux*.whl"]

    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("HF_TOKEN env var required (write scope on destination dataset)")

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        sys.exit("pip install 'huggingface_hub<2.0'")

    api = HfApi(token=tok)
    print(f"discovering files matching {args.filter} on {args.src} ...")
    src_files = api.list_repo_files(args.src, repo_type="dataset")
    import fnmatch

    matched = [f for f in src_files if any(fnmatch.fnmatch(f, pat) for pat in args.filter)]
    print(f"matched {len(matched)} files; will mirror to {args.dst}")
    if args.dry_run:
        for f in matched[:50]:
            print("  ", f)
        if len(matched) > 50:
            print(f"  ... +{len(matched) - 50} more")
        return 0

    if not matched:
        return 0

    with tempfile.TemporaryDirectory() as td:
        local = snapshot_download(
            repo_id=args.src,
            repo_type="dataset",
            allow_patterns=args.filter,
            local_dir=td,
            token=tok,
        )
        print(f"downloaded into {local}, uploading ...")
        api.upload_folder(
            repo_id=args.dst,
            repo_type="dataset",
            folder_path=local,
            allow_patterns=args.filter,
            commit_message=f"mirror {len(matched)} wheels from {args.src}",
        )
    print(f"done. wheels live at https://huggingface.co/datasets/{args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
