"""Deterministic stub for common.chat / common.complete_ids (tests + branch.py --stub).

Answers the three prompt kinds branch.py sends, with real Qwen token ids:
  episode  (task prompt)   -> long reasoning that never names the chain; hits
                              max_tokens -> finish "length" (the compaction point)
  handoff  (notes.handoff_prompt) -> note i of each prompt: even = right chain,
                              odd = wrong last hop  => one pass + one fail per group
  resume   (notes.resume_prompt)  -> answers with the note's CHAIN line (or gives up)
Pass/fail is decided by the real grader (grade_mhqa), not by the stub.
"""

from __future__ import annotations

import contextlib
import re
import threading

import common as C
import branch as B


class StubModel:
    def __init__(self, tasks: list[dict], tamper_prompt_ids: bool = False):
        self.gold = {t["prompt"]: t["answer"]["chain"] for t in tasks}
        self.calls: dict[str, int] = {}
        self.tamper = tamper_prompt_ids
        self.lock = threading.Lock()
        self.log: list[str] = []

    def _task_prompt(self, user: str) -> str:
        return user.split("\n\n---\n", 1)[0]

    def _gen(self, user: str, text: str, max_tokens: int, thinking: bool) -> dict:
        tok = C.tokenizer()
        pids = B.encoding.render_chat(tok, [{"role": "user", "content": user}], thinking)
        if self.tamper:
            pids = pids[:-1]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        finish = "stop"
        if len(ids) > max_tokens:
            ids, finish = ids[:max_tokens], "length"
        out = tok.decode(ids)
        reasoning, _, content = out.partition("</think>") if thinking else ("", "", out)
        if thinking and "</think>" not in out:
            reasoning, content = out, ""
        return {"content": content.lstrip("\n"), "reasoning": reasoning, "finish": finish,
                "ids": ids + ([tok.convert_tokens_to_ids("<|im_end|>")] if finish == "stop" else []),
                "prompt_ids": pids}

    def chat(self, model, user, max_tokens, thinking, temperature=0.7, top_p=0.95,
             timeout=900):
        if "Write a HANDOFF" in user:
            task = self._task_prompt(user)
            chain = self.gold[task]
            with self.lock:
                i = self.calls.get(user, 0)
                self.calls[user] = i + 1
            good = i % 2 == 0
            ans = chain[1:] if good else chain[1:-1] + ["the Nowhere River"]
            note = (f"Goal: answer the chain question.\nDone: traced facts.\n"
                    f"Current state: CHAIN: {' -> '.join(ans)}\nHypothesis: chain above.\n"
                    f"Next step: verify and answer.\nDon't repeat: rescanning all facts.")
            self.log.append("note-pass" if good else "note-fail")
            return self._gen(user, f"Sizing up the state.\n</think>\n\n{note}", max_tokens,
                             thinking)
        if "<handoff>" in user:
            m = re.search(r"CHAIN: (.+)", user)
            ans = m.group(1).strip() if m else "unknown"
            return self._gen(user, f"Checking the note.\n</think>\n\nANSWER: {ans}",
                             max_tokens, thinking)
        filler = "".join(f"Scanning fact block {i} for the next link.\n" for i in range(2000))
        return self._gen(user, filler, max_tokens, thinking)

    def complete_ids(self, model, prompt_ids, max_tokens, **kw):
        raise NotImplementedError("stub: use note-mode think/off")

    @contextlib.contextmanager
    def installed(self):
        old = C.chat, C.complete_ids
        C.chat, C.complete_ids = self.chat, self.complete_ids
        try:
            yield self
        finally:
            C.chat, C.complete_ids = old
