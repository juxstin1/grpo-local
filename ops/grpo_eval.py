"""Public development comparison; model execution requires explicit operator opt-in.

Default: contract_v1 outcomes and work-contract-v1 instructions. Historical families
require --legacy-diagnostic and are not qualified task-success measurements.
In-process --holdout is refused: private evaluation needs a separate authority.

Historical fixed-task behavior (diagnostic only):

Tasks: seeds 47001.. (train range, never used by grpo_q4 runs, which count up from 46001; holdout refused),
one template per seed chosen by seed, families from --families. Each task K attempts per adapter on the Q4
llama-server (base Q4_K_M + --lora), the same Bionic native loop and plain T=1.0 sampler as training.
Reports pass rate per family and overall, plus the grader reason counts.

  .venv\\Scripts\\python.exe -u ops\\grpo_eval.py --out runs/grpo-q4/r3/eval \\
      --adapters sft=runs/pilots/q4-bf16-gap-1010/lora-sft-1008.f16.gguf,step20=runs/grpo-q4/r3/lora-step20.f16.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grpo_q4 as Q  # noqa: E402
from evaluation_integrity import validate_export_policy, reserved_seeds
from run_policy import add_model_flag, require_model_opt_in
import contract_tasks
import work_contract
Q.NEED_LP[0] = False                                    # evals only need the text, not Q4 log-probs

ROOT = HERE.parent


def fixed_tasks(M, families: list[str], per_family: int, seed0: int) -> list[tuple[str, dict]]:
    out = []
    for fam in families:
        seed, got = seed0, 0
        while got < per_family:
            seed += 1
            if (fam, seed) in M["sealed"]:
                continue
            specs = M["repo_gen"].make_specs(seed, fam)
            if specs:
                out.append((fam, specs[seed % len(specs)]))
                got += 1
    return out


def holdout_tasks(families: list[str]) -> list[tuple[str, dict]]:
    """Private evaluation must run outside this development identity."""
    validate_export_policy(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", required=True, help="label=path.gguf,label=path.gguf")
    ap.add_argument("--families", default=contract_tasks.FAMILY)
    ap.add_argument("--per-family", type=int, default=12)
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=47000)
    ap.add_argument("--holdout", action="store_true", help="sealed holdout items instead of seeds from --seed0")
    ap.add_argument("--real", action="store_true", help="real Bionic fumble tasks (data/bionic-real/tasks)")
    ap.add_argument("--tasks-dir", help="with --real: another task folder (e.g. data/diag-eval/tasks)")
    ap.add_argument("--diag-ab", action="store_true",
                    help="spec 31: each adapter twice, edit diagnostics off (<label>) and on (<label>+diag)")
    ap.add_argument("--bionic-agentctx", action="store_true",
                    help="loop as the user's Bionic (fix/agent-context) runs it: 24 rounds + last-round note, "
                         "ctx 32768. Tools/prompt are still the gym mirror (see parity drift).")
    ap.add_argument("--transcripts", action="store_true", help="save each episode's decoded conversation")
    ap.add_argument("--prompt-extra", default="", help="EXPERIMENT (not Bionic): text appended to the agent "
                    "system prompt, to test a prompt fix before proposing it for Bionic")
    ap.add_argument("--prompt-patch", help="EXPERIMENT (not Bionic): JSON file of [old, new] replacements "
                    "applied to the agent system prompt (each old must be present)")
    ap.add_argument('--legacy-diagnostic', action='store_true',
                    help='Replay historical graders for debugging only; not qualified task scores.')
    ap.add_argument('--trusted-manifest', type=Path,
                    help='Externally supplied manifest for legacy task exclusion discovery.')
    ap.add_argument('--instruction-profile', choices=('work-contract-v1', 'legacy'), default='work-contract-v1')
    add_model_flag(ap)
    a = ap.parse_args()
    validate_export_policy(a.holdout, a.transcripts, task_level_exports=a.holdout)
    require_model_opt_in(a.allow_model_run)
    if a.attempts < 1 or a.per_family < 1:
        raise SystemExit('attempts and per-family must be positive')
    if not a.legacy_diagnostic and (a.real or a.families != contract_tasks.FAMILY):
        raise SystemExit('Only contract_v1 has qualified rewards; historical families require --legacy-diagnostic.')
    if not a.legacy_diagnostic and (a.instruction_profile == 'legacy' or a.prompt_patch or a.prompt_extra):
        raise SystemExit('Qualified development uses the versioned work contract; prompt experiments require --legacy-diagnostic.')
    if (a.prompt_patch or a.prompt_extra) and a.instruction_profile != 'legacy':
        raise SystemExit('Historical prompt patches require --instruction-profile legacy; they cannot silently alter another profile.')
    Q.PROMPT_PROFILE = a.instruction_profile
    Q.LEGACY_DIAGNOSTIC = a.legacy_diagnostic
    if a.bionic_agentctx:
        Q.MAX_ROUNDS, Q.CTX = 24, 32768
        Q.LAST_ROUND_NOTE = ("[Tool step limit reached for this message: tools are off for this reply. Don't write a "
                             "tool call. Say what you finished, what is left, and that the user can say \"continue\" "
                             "to keep going.]")
    out = (ROOT / a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    adapters = [(kv.split("=", 1)[0], (ROOT / kv.split("=", 1)[1]).resolve()) for kv in a.adapters.split(",")]
    for _, p in adapters:
        if not p.exists():
            raise SystemExit(f"missing adapter {p}")

    M = Q.gym()
    if a.legacy_diagnostic:
        M['sealed'] = reserved_seeds(Q.G.GYM / 'data/holdout', a.trusted_manifest)
    else:
        contract_tasks.specs()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q.G.BASE_MODEL)
    if a.prompt_extra:
        base_prompt = M["native"].system_prompt
        M["native"].system_prompt = lambda tools, *x, **k: (
            base_prompt(tools, *x, **k) + ("\n\n" + a.prompt_extra if tools else ""))
    patches = json.loads((ROOT / a.prompt_patch).read_text(encoding="utf-8")) if a.prompt_patch else []
    if patches:                                         # EXPERIMENT: reword lines of the mirrored prompt
        unpatched = M["native"].system_prompt

        def patched(tools, *x, **k):
            s = unpatched(tools, *x, **k)
            for old, new in patches:
                if tools and old not in s:
                    raise SystemExit(f"--prompt-patch: not in prompt: {old[:60]}")
                s = s.replace(old, new)
            return s
        M["native"].system_prompt = patched
    if not a.legacy_diagnostic:
        tasks = [(contract_tasks.FAMILY, s) for s in contract_tasks.specs()]
    elif a.real:                                          # real Bionic fumble tasks (diagnostic only)
        import real_tasks
        a.families = real_tasks.FAMILY
        tasks = [(real_tasks.FAMILY, s) for s in
                 real_tasks.load(*([(ROOT / a.tasks_dir).resolve()] if a.tasks_dir else []))]
    else:
        tasks = (holdout_tasks(a.families.split(",")) if a.holdout
                 else fixed_tasks(M, a.families.split(","), a.per_family, a.seed0))
    print("[tasks]", [(f, s["name"]) for f, s in tasks], flush=True)
    report = {"tasks": [(f, s["name"]) for f, s in tasks], "attempts": a.attempts, "arms": {},
              "loop": {"max_rounds": Q.MAX_ROUNDS, "ctx": Q.CTX, "last_round_note": bool(Q.LAST_ROUND_NOTE)}, "prompt_extra": a.prompt_extra, "prompt_patch": a.prompt_patch}
    report.update(sealed=False, score_kind='legacy-diagnostic' if a.legacy_diagnostic else 'qualified-development',
                  suite=contract_tasks.suite_metadata() if not a.legacy_diagnostic else None,
                  final_round_note=Q.FINAL_ROUND_NOTE, extra_last_round_note=Q.LAST_ROUND_NOTE,
                  instruction=work_contract.prompt_metadata(Q.PROMPT_PROFILE, tools=M['native'].native_tools(),
                                                            legacy_system=M['native'].system_prompt))
    import edit_diagnostics                             # gym mirror of Bionic's edit diagnostics (spec 31)
    arms = [(lbl + suffix, g, on) for lbl, g in adapters
            for suffix, on in ((("", False), ("+diag", True)) if a.diag_ab else (("", edit_diagnostics.ENABLED[0]),))]
    for label, gguf, diag_on in arms:
        edit_diagnostics.ENABLED[0] = diag_on
        edit_diagnostics._CACHE.clear()
        edit_diagnostics._SEEN_MISSING.clear()
        edit_diagnostics.STATS.update(blocks=0, with_errors=0)
        proc, t = Q.start_server(gguf, out / f"server-{label}.log")
        rows = []
        try:
            for fam, spec in tasks:
                for k in range(a.attempts):
                    e = Q.episode(M, tok, fam, spec, f"eval-{label}-{k}")
                    seqs = e.pop("seqs")
                    if a.transcripts and seqs:                 # last request = whole conversation + final output
                        td = out / "transcripts"
                        td.mkdir(exist_ok=True)
                        (td / f"{label}-{spec['name']}-{k}.txt").write_text(
                            f"reward {e['reward']:.0f} {e['reason']} stop {e['stop']} rounds {e['rounds']}\n\n"
                            + tok.decode(seqs[-1]["prompt_ids"] + seqs[-1]["gen_ids"]), encoding="utf-8")
                    rows.append(e)
                    print(f"[{label}] {spec['name']} #{k}: {e['reward']:.0f} {e['reason'][:40]} "
                          f"stop {e['stop']} rounds {e['rounds']}", flush=True)
        finally:
            Q.stop_server(proc)
        arm = {"adapter": str(gguf), "edit_diagnostics": diag_on, "diag_blocks": dict(edit_diagnostics.STATS),
               "pass": sum(r["reward"] for r in rows) / len(rows), "n": len(rows),
               "by_family": {f: sum(r["reward"] for r in rows if r["family"] == f) /
                             max(1, sum(1 for r in rows if r["family"] == f)) for f in a.families.split(",")},
               "by_task": {s["name"]: [r["reward"] for r in rows if r["name"] == s["name"]] for _, s in tasks},
               "reasons": Counter(r["reason"] for r in rows if r["reward"] == 0).most_common(),
               "stops": Counter(r["stop"] for r in rows).most_common()}
        report["arms"][label] = arm
        print(f"[arm] {label}: pass {arm['pass']:.3f} {arm['by_family']} diag {arm['diag_blocks']}", flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        with (out / "episodes.jsonl").open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"arm": label, **r}) + "\n")
    print("[done]", json.dumps({k: (v["pass"], v["by_family"]) for k, v in report["arms"].items()}), flush=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
