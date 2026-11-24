"""Shared bits for the handoff gym: served client, tasks, prompts, grading.

Reuses evals/graders.py
(grade_mhqa, full-chain) and evals/datasets/synth_multihop.py (world + chains)
by import -- nothing is copied. synth_multihop is only IMPORTED: its bare
`main()` rewrites the shared eval set, never call it.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "evals"), str(ROOT / "evals" / "datasets")]

import synth_multihop as mh  # noqa: E402
from graders import grade_mhqa  # noqa: E402

import notes  # noqa: E402  (grpo-local/notes.py: the one note call)

BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8000")   # e.g. pod tunnel :8001
DATA = HERE / "data"
NOTE_CAP = notes.NOTE_CAP       # shared note spec: grpo-local/notes.py
sys.path.append(str(ROOT))
from gym_budget import BUDGET   # noqa: E402  prompt + completion, the one constant (=8192 unless GYM_BUDGET/--budget)

_TOK = None


def check_server(base: str | None = None) -> None:
    """Fail fast if BUDGET exceeds the server's max_model_len (gym_budget.check_server)."""
    import gym_budget
    gym_budget.check_server(base or BASE, BUDGET)


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    return _TOK


def room(user: str, thinking: bool) -> int:
    """Completion tokens left inside BUDGET for this prompt."""
    ids = tokenizer().apply_chat_template([{"role": "user", "content": user}],
                                          add_generation_prompt=True, tokenize=True,
                                          enable_thinking=thinking)
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    return BUDGET - len(ids)

# --- tasks -------------------------------------------------------------------

# Big world + deep chains (6-8 hops), so an empty note fails honestly at the
# reader's normal budget. Seeds here are training-side only; mh.SEED is refused.
WORLD = dict(n_people=40, n_companies=24, n_cities=30, n_rivers=30)
# Sealed handoff eval worlds (seal.py). Never freeze/collect/train from these.
HOLDOUT_SEEDS = (9101,)


def _deep_hop(w, people, per_hop, seed, hops: int) -> list[dict]:
    """hops >= 6 (not in synth_multihop): p -> mentor x (hops-4) -> company ->
    founder -> city -> river. Distinct answers first, same rule as mh.
    A person may not repeat on the chain (mentor cycles make it ambiguous)."""
    E, F, B, R, M = w["employer"], w["founder"], w["born"], w["river"], w["mentor"]
    d = hops - 4

    def fn(p):
        c = [p]
        for _ in range(d):
            c.append(M[c[-1]])
        co = E[c[-1]]
        fo = F[co]
        ci = B[fo]
        return c + [co, fo, ci, R[ci]]

    def ok(p):
        ch = fn(p)
        return len(set(ch[: d + 1])) == d + 1

    rng = random.Random(seed + hops)
    seen, picked = set(), []
    for p in rng.sample(people, len(people)):
        if ok(p) and fn(p)[-1] not in seen:
            seen.add(fn(p)[-1])
            picked.append(p)
    whose = "{p}'s mentor"
    for _ in range(d - 1):
        whose = "the mentor of " + whose
    q = ("Which river flows through the birth city of the founder of the company that "
         f"{whose} works for?")
    kinds = ["person"] + ["mentor"] * d + ["company", "founder", "city", "river"]
    fmt = " -> ".join(f"<{k}>" for k in kinds)
    return [{"hops": hops, "person": p, "chain": fn(p), "expected": fn(p)[-1],
             "q": q.format(p=p) + f"\nGive the full chain on the last line: ANSWER: {fmt}"}
            for p in picked[:per_hop]]


def make_tasks(seed: int, hops: tuple[int, ...] = (7, 8), per_hop: int = 6,
               allow_holdout: bool = False) -> list[dict]:
    if seed == mh.SEED:
        raise SystemExit("that seed is the eval world")
    if seed in HOLDOUT_SEEDS and not allow_holdout:
        raise SystemExit(f"seed {seed} is the sealed handoff holdout (seal.py)")
    w, people, facts = mh.build_world(seed, **WORLD)
    qs = mh.make_questions(w, people, per_hop, seed, [h for h in hops if h <= 5])
    for h in hops:
        if h >= 6:
            qs += _deep_hop(w, people, per_hop, seed, h)
    out = []
    for q in qs:
        ans = q["expected"]
        out.append({
            "id": f"hg:{seed}:h{q['hops']}:{q['person'].replace(' ', '_')}",
            "root_id": f"hg:{seed}:{q['person'].replace(' ', '_')}",
            "hops": q["hops"],
            "prompt": "Facts:\n" + "\n".join(facts) + f"\n\nQuestion: {q['q']}",
            "answer": {"answer": ans, "aliases": [ans[4:]] if ans.startswith("the ") else [],
                       "chain": q["chain"]},
        })
    return out


def links_in_trace(task: dict, trace: str) -> int:
    """How many chain links (after the subject) her trace already names."""
    t = trace.lower()
    return sum(c.lower() in t for c in task["answer"]["chain"][1:])


def chain_in_trace(task: dict, trace: str) -> bool:
    """True if every hop after the subject is already named in her trace: a
    note from here would just paste the answer, so the cut is useless."""
    t = trace.lower()
    return all(c.lower() in t for c in task["answer"]["chain"][1:])


def grade(task: dict, text: str) -> tuple[bool, str]:
    return grade_mhqa(task, text)


# --- prompts -----------------------------------------------------------------

def handoff_prompt(task: dict, trace: str) -> str:
    return notes.handoff_prompt(task["prompt"], trace, "the question, the facts")


def resume_prompt(task: dict, note: str) -> str:
    return notes.resume_prompt(task["prompt"], note, "the facts")


# --- served client -----------------------------------------------------------

def chat(model: str, user: str, max_tokens: int, thinking: bool,
         temperature: float = 0.7, top_p: float = 0.95, timeout: int = 900) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "return_token_ids": True}
    err = ""
    for _ in range(3):
        try:
            r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=timeout)
            r.raise_for_status()
            d = r.json()
            ch = d["choices"][0]
            msg = ch["message"]
            return {"content": msg.get("content") or "",
                    "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                    "finish": ch.get("finish_reason", ""),
                    "ids": ch.get("token_ids") or [],
                    "prompt_ids": d.get("prompt_token_ids") or []}
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:160]}"
    return {"error": err, "content": "", "reasoning": "", "finish": "", "ids": [],
            "prompt_ids": []}


def complete_ids(model: str, prompt_ids: list[int], max_tokens: int,
                 temperature: float = 0.7, top_p: float = 0.95, timeout: int = 900) -> dict:
    """Raw completion continuing an exact token stream (budget forcing)."""
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "return_token_ids": True,
            "skip_special_tokens": False}
    err = ""
    for _ in range(3):
        try:
            r = requests.post(f"{BASE}/v1/completions", json=body, timeout=timeout)
            r.raise_for_status()
            ch = r.json()["choices"][0]
            return {"text": ch.get("text") or "", "finish": ch.get("finish_reason", ""),
                    "ids": ch.get("token_ids") or []}
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:160]}"
    return {"error": err, "text": "", "finish": "", "ids": []}


def write_note(model: str, hp: str, mode: str) -> dict:
    """One handoff note under notes.py's caps. Modes:
      off    thinking off, note only (<= NOTE_CAP).
      think  thinking on, one call of NOTE_CALL_MAX (nothing enforces the think cap).
      force  budget forcing: think up to NOTE_THINK_CAP; if </think> hasn't come,
             the harness appends it and she writes the note (<= NOTE_CAP) from there.
    Returns text, note_tokens, gen ids (her tokens + any forced close), prompt_ids,
    over_cap, forced."""
    tok = tokenizer()
    n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    T = notes.NOTE_TEMPERATURE
    if mode == "off":
        r = chat(model, hp, notes.NOTE_CAP, thinking=False, temperature=T)
        if r.get("error"):
            return {"error": r["error"]}
        nt = n(r["content"])
        return {"text": r["content"], "note_tokens": nt, "ids": r["ids"],
                "prompt_ids": r["prompt_ids"], "over_cap": notes.over_cap(r["finish"], nt),
                "forced": False}
    if mode == "think":
        r = chat(model, hp, notes.NOTE_CALL_MAX, thinking=True, temperature=T)
        if r.get("error"):
            return {"error": r["error"]}
        nt = n(r["content"])
        return {"text": r["content"], "note_tokens": nt, "ids": r["ids"],
                "prompt_ids": r["prompt_ids"], "over_cap": notes.over_cap(r["finish"], nt),
                "forced": False}
    # force
    close = tok.convert_tokens_to_ids("</think>")
    r = chat(model, hp, notes.NOTE_THINK_CAP, thinking=True, temperature=T)
    if r.get("error"):
        return {"error": r["error"]}
    ids, forced = list(r["ids"]), False
    if close in ids and r["finish"] == "stop":            # thought + note inside 600
        nt = n(r["content"])
        return {"text": r["content"], "note_tokens": nt, "ids": ids,
                "prompt_ids": r["prompt_ids"], "over_cap": nt > notes.NOTE_CAP,
                "forced": False}
    if close not in ids:                                   # still thinking at the cap
        ids += tok("\n</think>\n\n", add_special_tokens=False)["input_ids"]
        forced = True
    head = tok.decode(ids[ids.index(close) + 1:]) if close in ids else ""
    left = notes.NOTE_CAP - n(head)
    c = complete_ids(model, r["prompt_ids"] + ids, max(1, left), temperature=T)
    if c.get("error"):
        return {"error": c["error"]}
    ids += c["ids"]
    text = (head + c["text"]).replace("<|im_end|>", "").strip()
    nt = n(text)
    return {"text": text, "note_tokens": nt, "ids": ids, "prompt_ids": r["prompt_ids"],
            "over_cap": c["finish"] == "length" or nt > notes.NOTE_CAP, "forced": forced}


def full_text(res: dict) -> str:
    """What the grader sees: the answer section (reasoning stripped by the parser)."""
    return res["content"]


def jsonl_write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def jsonl_read(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
