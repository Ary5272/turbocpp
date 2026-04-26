#!/usr/bin/env python3
"""
convert_hf.py - convert a HuggingFace LLaMA-family model to TurboCPP's .tcpp
binary format.

Usage:
    python convert_hf.py <hf_model_dir> <output.tcpp> [--quant q4|q8|f32]

Requirements: torch, safetensors (pip install torch safetensors).

Notes:
- Tested with LLaMA-2 / Mistral / TinyLlama. Other architectures may need
  attention/FFN naming tweaks (search for ATTN_KEYS / FFN_KEYS below).
- Output is a flat binary matching loader/gguf.h. We dump fp32 by default;
  use --quant q4 / q8 to compress weights at convert time.
- Tokenizer must be exported separately (vocab.txt + merges.txt). HF's
  tokenizer.json can be split with the included `dump_tokenizer` helper.
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("pip install torch")
try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


# --------------------------------------------------------------------------
# TCPP file format (mirrors loader/gguf.h)
# --------------------------------------------------------------------------
MAGIC = 0x50504354          # "TCPP"
VERSION = 1
DTYPE_F32  = 0
DTYPE_I32  = 1
DTYPE_Q4_0 = 2
DTYPE_U8   = 3

ATTN_KEYS = {
    "Wq":   ["self_attn.q_proj.weight", "attention.wq.weight"],
    "Wk":   ["self_attn.k_proj.weight", "attention.wk.weight"],
    "Wv":   ["self_attn.v_proj.weight", "attention.wv.weight"],
    "Wo":   ["self_attn.o_proj.weight", "attention.wo.weight"],
    "attn_norm": ["input_layernorm.weight", "attention_norm.weight"],
}
FFN_KEYS = {
    "Wgate": ["mlp.gate_proj.weight", "feed_forward.w1.weight"],
    "Wup":   ["mlp.up_proj.weight",   "feed_forward.w3.weight"],
    "Wdown": ["mlp.down_proj.weight", "feed_forward.w2.weight"],
    "ffn_norm": ["post_attention_layernorm.weight", "ffn_norm.weight"],
}
TOP_KEYS = {
    "tok_embed":  ["model.embed_tokens.weight", "tok_embeddings.weight"],
    "final_norm": ["model.norm.weight", "norm.weight"],
    "lm_head":    ["lm_head.weight", "output.weight"],
}


# --------------------------------------------------------------------------
# Block-wise quantization (Q4_0-like, fp32 scale variant matching quant/q4.h)
# --------------------------------------------------------------------------
def q4_quantize_row(x: np.ndarray) -> bytes:
    assert x.size % 32 == 0, f"row size {x.size} must be %32"
    out = bytearray()
    x = x.astype(np.float32).reshape(-1, 32)
    for blk in x:
        maxabs = float(np.max(np.abs(blk)))
        d = maxabs / 7.5 if maxabs > 0 else 0.0
        out += struct.pack("<f", d)
        if d == 0:
            out += bytes(16)
            continue
        q = np.clip(np.round(blk / d).astype(np.int32) + 8, 0, 15).astype(np.uint8)
        # Pack 2 nibbles per byte; even index -> low nibble, odd -> high.
        packed = (q[0::2] | (q[1::2] << 4)).astype(np.uint8)
        out += packed.tobytes()
    return bytes(out)


def q8_quantize_row(x: np.ndarray) -> bytes:
    assert x.size % 32 == 0
    out = bytearray()
    x = x.astype(np.float32).reshape(-1, 32)
    for blk in x:
        maxabs = float(np.max(np.abs(blk)))
        d = maxabs / 127.0 if maxabs > 0 else 0.0
        out += struct.pack("<f", d)
        if d == 0:
            out += bytes(32)
        else:
            q = np.clip(np.round(blk / d), -127, 127).astype(np.int8)
            out += q.tobytes()
    return bytes(out)


def quantize_matrix(W: np.ndarray, mode: str) -> tuple[bytes, int, list[int]]:
    """Return (raw_bytes, dtype_code, shape_list)."""
    W = W.astype(np.float32)
    if mode == "f32":
        return W.tobytes(), DTYPE_F32, list(W.shape)
    if mode == "q4":
        # Quantize per row (last dim).
        assert W.shape[-1] % 32 == 0, f"last dim {W.shape[-1]} not %32 for Q4"
        out = bytearray()
        for row in W.reshape(-1, W.shape[-1]):
            out += q4_quantize_row(row)
        return bytes(out), DTYPE_Q4_0, list(W.shape)
    if mode == "q8":
        assert W.shape[-1] % 32 == 0
        out = bytearray()
        for row in W.reshape(-1, W.shape[-1]):
            out += q8_quantize_row(row)
        return bytes(out), DTYPE_U8, list(W.shape)  # raw byte payload
    raise ValueError(f"unknown quant: {mode}")


# --------------------------------------------------------------------------
# HF reading
# --------------------------------------------------------------------------
def load_hf_state_dict(model_dir: Path) -> dict:
    """Returns flat name -> torch.Tensor."""
    sd = {}
    bins = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not bins:
        sys.exit(f"no .safetensors / .bin in {model_dir}")
    for f in sorted(bins):
        if f.suffix == ".safetensors":
            if safe_open is None:
                sys.exit("pip install safetensors")
            with safe_open(f, framework="pt") as h:
                for k in h.keys():
                    sd[k] = h.get_tensor(k)
        else:
            sd.update(torch.load(f, map_location="cpu", weights_only=True))
    return sd


def first_match(sd: dict, candidates: list[str], layer_idx: int | None = None):
    for name in candidates:
        key = f"model.layers.{layer_idx}.{name}" if layer_idx is not None else name
        if key in sd:
            return sd[key].cpu().numpy()
        # alt LLaMA naming: "layers.{L}.{name}"
        key_alt = f"layers.{layer_idx}.{name}" if layer_idx is not None else name
        if key_alt in sd:
            return sd[key_alt].cpu().numpy()
    return None


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------
def pack_config(cfg: dict) -> bytes:
    # ModelConfig layout (must match transformer.h byte-for-byte).
    return struct.pack(
        "<QQQQQQQQff",                      # 8×size_t + 2×float
        cfg["vocab_size"],
        cfg["hidden_dim"],
        cfg["n_layers"],
        cfg["n_heads"],
        cfg["n_kv_heads"],
        cfg["head_dim"],
        cfg["ffn_dim"],
        cfg["max_seq_len"],
        cfg["rms_eps"],
        cfg["rope_base"],
    )


def pack_tensor_record(name: str, dtype: int, ndim: int, shape, offset: int, nbytes: int) -> bytes:
    name_bytes = name.encode("ascii")[:63] + b"\0" * max(0, 64 - len(name))
    return (
        name_bytes[:64]
        + struct.pack("<BB", dtype, ndim)
        + b"\0" * 6
        + struct.pack("<QQQQ", *(list(shape) + [0] * (4 - len(shape))))
        + struct.pack("<QQ", offset, nbytes)
    )


def write_tcpp(out_path: Path, cfg: dict, tensors: list[tuple[str, np.ndarray]], quant: str):
    blobs = []
    records = []
    cursor = 0
    for name, W in tensors:
        # tok_embed and lm_head we leave as f32 by default — Q4/Q8 hurt them
        # disproportionately. Override only if explicitly using f32.
        mode = quant
        if name in ("tok_embed", "lm_head") and quant in ("q4",):
            mode = "q8"   # safer compromise
        if W.ndim == 1:
            mode = "f32"  # norms must be fp32
        raw, dtype, shape = quantize_matrix(W, mode)
        # 32-byte align
        pad = (-cursor) & 31
        cursor += pad
        blobs.append((b"\0" * pad, raw))
        records.append((name, dtype, len(shape), shape, cursor, len(raw)))
        cursor += len(raw)

    hdr_size = 48                      # TLLMHeader
    cfg_size = 8 * 8 + 4 * 2          # ModelConfig
    dir_size = 8 + len(records) * (64 + 1 + 1 + 6 + 4 * 8 + 2 * 8)
    pre_data = hdr_size + cfg_size + dir_size
    data_offset = (pre_data + 4095) & ~4095

    with open(out_path, "wb") as f:
        # TLLMHeader
        f.write(struct.pack(
            "<IIQQQQ",
            MAGIC, VERSION,
            hdr_size,                     # config_offset
            hdr_size + cfg_size,          # tensor_dir_offset
            data_offset,                  # data_offset
            cursor,                       # data_size
        ))
        # Config
        f.write(pack_config(cfg))
        # Directory
        f.write(struct.pack("<Q", len(records)))
        for r in records:
            f.write(pack_tensor_record(*r))
        # Pad
        f.seek(data_offset)
        for pad, raw in blobs:
            f.write(pad)
            f.write(raw)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--quant", choices=["f32", "q4", "q8"], default="f32")
    args = ap.parse_args()

    cfg_path = args.model_dir / "config.json"
    if not cfg_path.is_file():
        sys.exit(f"missing config.json in {args.model_dir}")
    hf_cfg = json.loads(cfg_path.read_text())

    cfg = {
        "vocab_size":  int(hf_cfg["vocab_size"]),
        "hidden_dim":  int(hf_cfg["hidden_size"]),
        "n_layers":    int(hf_cfg["num_hidden_layers"]),
        "n_heads":     int(hf_cfg["num_attention_heads"]),
        "n_kv_heads":  int(hf_cfg.get("num_key_value_heads", hf_cfg["num_attention_heads"])),
        "head_dim":    int(hf_cfg["hidden_size"] // hf_cfg["num_attention_heads"]),
        "ffn_dim":     int(hf_cfg["intermediate_size"]),
        "max_seq_len": int(hf_cfg.get("max_position_embeddings", 2048)),
        "rms_eps":     float(hf_cfg.get("rms_norm_eps", 1e-5)),
        "rope_base":   float(hf_cfg.get("rope_theta", 10000.0)),
    }
    print(f"config: {cfg}")

    print("loading state_dict...")
    sd = load_hf_state_dict(args.model_dir)

    tensors: list[tuple[str, np.ndarray]] = []
    for k, paths in TOP_KEYS.items():
        W = first_match(sd, paths)
        if W is None:
            sys.exit(f"missing {k}: tried {paths}")
        tensors.append((k, W))

    for L in range(cfg["n_layers"]):
        for k, paths in {**ATTN_KEYS, **FFN_KEYS}.items():
            W = first_match(sd, paths, L)
            if W is None:
                sys.exit(f"layer {L} missing {k}")
            tensors.append((f"layers.{L}.{k}", W))

    print(f"writing {args.output} ({args.quant})...")
    write_tcpp(args.output, cfg, tensors, args.quant)
    sz = args.output.stat().st_size
    print(f"done. {sz / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
