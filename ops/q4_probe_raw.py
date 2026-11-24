"""Raw-reply probe: what do the Q4 arms actually send when an episode ends `empty` or `no-action`?

Same server flags, persona prompt and collection turn caps as eval_q4 (harness h3), but on TRAIN seeds
(refuses any seed found in data/holdout/*.json), so every reply can be saved. Writes all replies to
runs/pilots/probe-raw-1009/<size>-<arm>.jsonl and prints only the replies that ended an episode badly.
  python ops/q4_probe_raw.py --size 4b --arm r3sft --families repo_q:10,file_work:10,edit:5,svg_make:5
GPU: only when no scored eval is running.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_q4 as E  # noqa: E402

sys.path.insert(0, str(E.GYM))
CAPS = {"repo_q": 24, "file_work": 32, "edit": 32, "svg_make": 32, "web_q": 20,
        "doc_make": 40, "office_draft": 48}          # = collection max_turns (train.jsonl)
OUT = E.ROOT / "runs" / "pilots" / "probe-raw-1009"


def holdout_seeds() -> set[tuple[str, int]]:
    from evaluation_integrity import reserved_seeds
    supplied = os.environ.get('GRPO_HOLDOUT_MANIFEST')
    return reserved_seeds(E.GYM / 'data/holdout', Path(supplied) if supplied else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--families", default="repo_q:10,file_work:10",
                    help="family:episodes,... (one episode per train seed)")
    ap.add_argument("--seed0", type=int, default=42000)
    from run_policy import add_model_flag, require_model_opt_in
    add_model_flag(ap)
    a = ap.parse_args()
    require_model_opt_in(a.allow_model_run)
    holdout_seeds()  # fail closed before importing tokenizers or starting services
    os.chdir(E.GYM)
    import agent
    import repo_gen
    import tools as T
    from config import Config
    from encoding import render_chat
    from eval import make_task, system_for
    from runner import Runner
    from transformers import AutoTokenizer
    from workspace import Workspace

    sealed = holdout_seeds()
    tok = AutoTokenizer.from_pretrained(E.MODELS[a.size])
    raw_log: list[dict] = []
    real_chat = agent.chat_once

    def chat(*args, **kw):                       # what the server returned, untouched
        ch = real_chat(*args, **kw)
        raw_log.append(ch)
        return ch
    agent.chat_once = chat

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.size}-{a.arm}.jsonl"
    tally: dict[str, dict[str, int]] = {}
    proc = E.start_server(E.gguf_path(a.size, a.arm), 1, E.LOGS / f"q4-server-{a.size}-{a.arm}.log")
    try:
        with out.open("a", encoding="utf-8") as fh:
            for part in a.families.split(","):
                fam, n = part.split(":")
                if fam == "web_q":                   # web_q only has a fixed article list
                    import web_q
                    seeds = iter(web_q.TRAIN_SEEDS)
                else:
                    seeds = iter(range(a.seed0 + 1, a.seed0 + 100000))
                done = 0
                while done < int(n):
                    seed = next(seeds)
                    if (fam, seed) in sealed:
                        continue
                    specs = repo_gen.make_specs(seed, fam)
                    if not specs:
                        continue
                    spec = specs[done % len(specs)]  # rotate templates (specs[0] alone = only the easy one)
                    cfg = Config()
                    cfg.serve.base, cfg.serve.served_model = E.BASE, f"q4-{a.arm}"
                    cfg.episode.tool_dialect, cfg.episode.bionic_profile = "bionic", E.PROFILE
                    cfg.episode.budget, cfg.episode.max_turns = 16000, CAPS[fam]
                    cfg.episode.compact, cfg.episode.note_style = True, "facts"
                    task = make_task(spec)
                    ws = Workspace(cfg.episode.data_root / "workspaces" / f"probe-{a.arm}-{fam}", Runner("docker"))
                    ws.reset()
                    repo_gen.write_files(repo_gen.find_spec(spec["seed"], fam, spec["name"])["files"], ws.root)
                    raw_log.clear()
                    ep = agent.run_episode(cfg, task, ws, tokenizer=tok,
                                           prompt_len_fn=lambda m: len(render_chat(tok, m, True)))
                    passed = None if ep.error else bool(task.reward(ep.turns, ws) >= 0.5)
                    replies = [{"content": ch.get("content") or "", "reasoning": ch.get("reasoning") or "",
                                "finish": ch.get("finish_reason"), "completion_tokens": ch.get("usage_completion"),
                                "parsed": T.parse_action(ch.get("content") or "") is not None} for ch in raw_log]
                    fh.write(json.dumps({"arm": a.arm, "family": fam, "seed": seed, "name": spec["name"],
                                         "stop": ep.stop, "passed": passed, "error": ep.error,
                                         "system_sha_ok": system_for(cfg, spec) == (ep.initial_messages or [{}])[0].get("content"),
                                         "replies": replies}) + "\n")
                    fh.flush()
                    t = tally.setdefault(fam, {})
                    key = "pass" if passed else (ep.stop or "?")
                    t[key] = t.get(key, 0) + 1
                    done += 1
                    if ep.stop in ("empty", "no-action") and replies:
                        r = replies[-1]
                        print(f"\n===== {a.arm} {fam} seed {seed} stop={ep.stop} after {len(replies) - 1} calls "
                              f"| last reply {r['completion_tokens']} tok, finish={r['finish']}\n"
                              f"  content  : {r['content'][-600:]!r}\n  reasoning: {r['reasoning'][-600:]!r}",
                              flush=True)
                print(f"[probe] {a.arm} {fam}: {t}", flush=True)
    finally:
        E.stop_server(proc)
    print(f"[probe] done {a.arm}: {tally} -> {out}", flush=True)


if __name__ == "__main__":
    main()
