"""Rejection-sampled SFT record: one kept win, token-exact.

  {"id", "root_id", "bucket", "source", "split": "train",
   "messages": [{"role", "content", "train": false}, ...],   # prompt side, never a loss target
   "prompt_ids": [...],        # exactly what vLLM fed the model (verified == local template)
   "completion_ids": [...],    # raw sampled ids, EOS (<|im_end|>) included; the ONLY loss target
   "completion_text": "...",   # decode(completion_ids), for reading only - never re-tokenize it
   "n_tokens": int,            # len(prompt_ids) + len(completion_ids) <= MAX_TOKENS
   "grade": {"pass": true, "detail": "..."},
   "model": "Qwen/Qwen3.5-4B@<rev>", "tokenizer_template_sha": "...",
   "sampling": {"temperature", "top_p", "max_tokens"}}

root_id is the leak key: a task's wins never straddle train and eval/holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MAX_TOKENS = 4096
REQUIRED = ("id", "root_id", "bucket", "source", "split", "messages", "prompt_ids",
            "completion_ids", "completion_text", "n_tokens", "grade", "model",
            "tokenizer_template_sha", "sampling")


def template_sha(tokenizer) -> str:
    """Hash of the chat template + vocab. If this differs between SFT data and the
    trainer/GRPO tokenizer, the stored ids no longer mean the same thing."""
    h = hashlib.sha256()
    h.update((tokenizer.chat_template or "").encode())
    h.update(json.dumps(sorted(tokenizer.get_vocab().items()), ensure_ascii=False).encode())
    return h.hexdigest()


def validate(r: dict, eos_id: int) -> None:
    missing = [k for k in REQUIRED if k not in r]
    if missing:
        raise ValueError(f"{r.get('id')}: missing {missing}")
    if r["split"] != "train":
        raise ValueError(f"{r['id']}: split={r['split']} - only train records are SFT data")
    if any(m.get("train") for m in r["messages"]):
        raise ValueError(f"{r['id']}: a prompt message is marked train")
    if not r["grade"].get("pass"):
        raise ValueError(f"{r['id']}: not a win")
    if not r["completion_ids"] or r["completion_ids"][-1] != eos_id:
        raise ValueError(f"{r['id']}: completion does not end in EOS (truncated?)")
    n = len(r["prompt_ids"]) + len(r["completion_ids"])
    if n != r["n_tokens"] or n > MAX_TOKENS:
        raise ValueError(f"{r['id']}: n_tokens={r['n_tokens']} actual={n} max={MAX_TOKENS}")


def load(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
