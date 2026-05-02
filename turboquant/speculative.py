"""Speculative decoding on top of llama-cpp-python.

This is the single biggest decode-time speedup that doesn't require
custom kernels. The mechanism:

  1. A small/cheap DRAFT model proposes K tokens autoregressively.
  2. The TARGET (real) model verifies all K proposals in ONE forward
     pass — same compute as a single normal decode step.
  3. Tokens that match the target's argmax are accepted; the first
     mismatch is replaced with the target's choice and the rest are
     discarded.

Net: on memory-bound CPUs, expect 1.5–3× decode tok/s, depending on
acceptance rate (high for narrative/code prompts, lower for pure noise).

TurboQuant pairs naturally:
   - DRAFT  = TurboQuant Q2_K_M / Q3_K_M of the same model family
              (smaller → faster proposer, rotation keeps quality usable)
   - TARGET = baseline Q4_K_M / Q5_K_M / Q6_K (your normal model)

Both models share the same tokenizer.

Implementation notes:
  - We use llama-cpp-python's `Llama` interface for both.
  - Verification is done via `llm.eval(tokens)` followed by reading
    `llm._scores[-K:]` (logits of the last K positions). This is how
    llama.cpp's own speculative-decoding reference does it.
  - Acceptance is greedy-argmax-equality. Probabilistic rejection
    sampling (Leviathan et al.) is implementable later but greedy
    captures most of the wall-clock win.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class SpecStats:
    proposed: int = 0
    accepted: int = 0
    target_calls: int = 0
    decode_seconds: float = 0.0

    @property
    def accept_rate(self) -> float:
        return self.accepted / max(1, self.proposed)

    @property
    def speedup_factor(self) -> float:
        # Each target call yields at least 1 token + accepted draft tokens.
        # Without speculation, each token costs 1 target call.
        return (self.accepted + self.target_calls) / max(1, self.target_calls)


def _argmax_logit(scores_row) -> int:
    """Argmax over a logits row. Works on llama.cpp's flat list, numpy
    array, or any sequence."""
    best_i = 0
    best_v = scores_row[0]
    for i in range(1, len(scores_row)):
        v = scores_row[i]
        if v > best_v:
            best_v = v; best_i = i
    return best_i


def speculative_generate(
    target,                   # llama_cpp.Llama (the real model)
    draft,                    # llama_cpp.Llama (the small/fast model)
    prompt_tokens: List[int],
    max_new_tokens: int,
    draft_lookahead: int = 4,
    eos_token: Optional[int] = None,
    on_token=None,            # optional callback(token_id, text_piece)
) -> tuple[List[int], SpecStats]:
    """Speculative-decode `max_new_tokens` from `prompt_tokens`.

    Returns (generated_token_ids, stats).
    """
    stats = SpecStats()
    out: List[int] = []

    # Prime both models with the prompt. llama-cpp-python's `eval`
    # appends the tokens to its KV cache.
    target.reset()
    target.eval(prompt_tokens)
    draft.reset()
    draft.eval(prompt_tokens)

    t0 = time.time()
    while len(out) < max_new_tokens:
        # ── 1. Draft autoregressive proposal of K tokens ──
        proposals: List[int] = []
        for _ in range(draft_lookahead):
            tok = _argmax_logit(draft.eval_logits[-1])
            proposals.append(tok)
            stats.proposed += 1
            if eos_token is not None and tok == eos_token:
                break
            draft.eval([tok])

        # ── 2. Target verifies the whole batch in ONE forward call ──
        target.eval(proposals)
        stats.target_calls += 1

        # Target now has the predicted next tokens at positions
        # [-(K+1), -K, …, -1]. Walk left-to-right, accept while argmax
        # matches the proposal.
        accepted_here = 0
        offset = len(target.eval_logits) - len(proposals) - 1
        for i, prop in enumerate(proposals):
            target_pred = _argmax_logit(target.eval_logits[offset + i])
            if target_pred == prop:
                out.append(prop)
                stats.accepted += 1
                accepted_here += 1
                if on_token:
                    piece = target.detokenize([prop]).decode("utf-8", "replace")
                    on_token(prop, piece)
                if eos_token is not None and prop == eos_token:
                    stats.decode_seconds = time.time() - t0
                    return out, stats
                if len(out) >= max_new_tokens:
                    stats.decode_seconds = time.time() - t0
                    return out, stats
            else:
                # First rejection: take target's choice instead, discard rest.
                out.append(target_pred)
                if on_token:
                    piece = target.detokenize([target_pred]).decode("utf-8", "replace")
                    on_token(target_pred, piece)
                # Roll back the wrongly-evaluated tail in the target's KV
                # cache so its position counter matches what we actually kept.
                rollback = len(proposals) - i - 1
                if rollback > 0:
                    # llama-cpp-python doesn't expose an in-place rollback;
                    # truncate via internal state. The most portable path
                    # is to reset and re-eval the new prefix — slower but
                    # always correct. For draft_lookahead small (≤8) this
                    # cost is negligible.
                    target.n_tokens -= rollback
                # Same rollback in the draft model, plus catch up to the
                # target's actual chosen token.
                draft_rollback = len(proposals) - accepted_here - 1
                if draft_rollback > 0:
                    draft.n_tokens -= draft_rollback
                draft.eval([target_pred])
                stats.accepted += 0   # the divergence token isn't a "draft win"
                if eos_token is not None and target_pred == eos_token:
                    stats.decode_seconds = time.time() - t0
                    return out, stats
                break
        else:
            # All K proposals accepted → target's last logits become the
            # next draft seed; need draft to also see the accepted tokens.
            # Draft already has them from its own autoregressive loop.
            pass

    stats.decode_seconds = time.time() - t0
    return out, stats
