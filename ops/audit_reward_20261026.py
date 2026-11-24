"""Offline reward adapter audit. No model, GPU, service, or submitted-code execution.

Exit 1 denotes a reproduced reward-contract mismatch; 2 denotes an audit error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GYM = ROOT.parent / "grpo-local-wt-filetools" / "sandbox-gym"
sys.dont_write_bytecode = True
sys.path[:0] = [str(GYM), str(ROOT / "ops")]

import file_work
import graders
import office_draft
import real_tasks
import schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = [Path(m.__file__) for m in (file_work, graders, office_draft, real_tasks, schema)]
    paths += [Path(__file__), ROOT / "ops/grpo_q4.py", ROOT / "ops/q4_bf16_gap.py"]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows = []

    def turn(text, **kw):
        return schema.Turn(content=text, resp_text=text, **kw)

    with tempfile.TemporaryDirectory(prefix="reward-audit-") as tmp:
        def check(name, spec, files, turns, expected, explanation):
            root = Path(tmp) / name
            root.mkdir()
            real_tasks.write_files(files, root)
            task = schema.Task(name=spec["name"], family=spec["family"], prompt=spec["prompt"],
                               tools=spec["tools"], ground_truth=spec["ground_truth"],
                               reward=lambda *_: 0.0, meta={"native": True})
            # Same adapter dispatch as eval.make_task(native=True), used by GRPO.
            reward = graders.reward_for(task)(turns, SimpleNamespace(root=root))
            rows.append(dict(name=name, expected_reward=expected, actual_reward=reward,
                             agrees=reward == expected, rationale=explanation))

        f = file_work.specs(62001)[0]
        truth = json.loads(f["ground_truth"])
        correct = {**f["files"], truth["path"]: "Confirmed details:\n" + "\n".join(truth["required"]) + "\n"}
        answer = "ANSWER: " + truth["path"]
        check("filework_correct", f, correct, [turn(answer)], 1.0, "Correct saved facts and final answer.")
        if not args.smoke:
            check("filework_negated", f, {**f["files"], truth["path"]:
                  "None of these details is correct: " + "; ".join(truth["required"]) + ".\n"},
                  [turn(answer)], 0.0, "Output explicitly denies every required fact.")
            check("filework_blank_final", f, correct,
                  [turn(answer, action=schema.Action("read_file", {})), turn("")], 0.0,
                  "Earlier tool round names output; last assistant reply is empty.")
            check("filework_retracted_answer", f, correct,
                  [turn(answer, action=schema.Action("read_file", {})),
                   turn("I could not complete the task. Disregard my earlier answer.")], 0.0,
                  "Final reply retracts the earlier answer, which last_answer still reuses.")
            check("filework_truncation_control", f, correct, [turn(answer, truncated=True)], 0.0,
                  "Truncation must override correct saved files and answer.")
            check("filework_wrong_path_control", f, correct, [turn("ANSWER: wrong.txt")], 0.0,
                  "Wrong delivery path must fail.")
            clarify = next(s for s in file_work.specs(62001)
                           if json.loads(s["ground_truth"])["kind"] == "clarify")
            candidates = json.loads(clarify["ground_truth"])["candidates"]
            names = [Path(p).stem.split("_")[0] for p in candidates]
            check("clarify_irrelevant_question", clarify, clarify["files"],
                  [turn("Would " + " and ".join(names) + " like a sandwich?")], 0.0,
                  "Mentions candidate names but does not clarify which file to edit.")
            o = office_draft.specs(62001)[0]
            ot = json.loads(o["ground_truth"])
            ideal = office_draft.ideal_draft(o)
            ofiles = {**o["files"], ot["path"]: ideal}
            oa = "ANSWER: " + ot["path"]
            check("office_correct", o, ofiles, [turn(oa)], 1.0, "Reference draft and correct answer.")
            check("office_sent_claim", o, {**ofiles, ot["path"]: ideal +
                  "\nThis email has been sent to the recipient.\n"}, [turn(oa)], 0.0,
                  "Explicitly prohibited sent claim receives reward.")
            check("office_missing_sources", o, {ot["path"]: ideal}, [turn(oa)], 0.0,
                  "Every original source is deleted; only the draft remains.")

            # Inject a missing verifier executable, without launching a subprocess.
            # Current grade() returns False, which real_tasks.make_task maps to reward 0.
            root = Path(tmp) / "infra"
            root.mkdir()
            spec = dict(files={}, check={"run": [{"cmd": ["missing-audit-verifier"]}]})
            with patch("subprocess.run", side_effect=FileNotFoundError("audit injected missing backend")):
                ok, reason = real_tasks.grade(root, spec, [turn("Done.")])
            rows.append(dict(name="verifier_unavailable", expected_reward=None, actual_reward=float(ok),
                             agrees=False, reason=reason,
                             rationale="Unavailable verifier should invalidate the attempt, not give model reward zero."))

    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source changed during audit: " + path)
    summary = dict(cases=len(rows), mismatches=sum(not r["agrees"] for r in rows),
                   false_positive_rewards=sum(r["expected_reward"] == 0 and r["actual_reward"] == 1 for r in rows),
                   invalid_scored=sum(r["expected_reward"] is None for r in rows))
    report = dict(scope="Hand-authored counterexamples; not model behavior or a prevalence estimate",
                  gym=str(GYM), source_sha256=hashes, summary=summary, cases=rows)
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT / "runs/audits"):
            raise ValueError("Output must be under runs/audits")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary))
    for row in rows:
        print(f"{row['name']}: expected={row['expected_reward']} actual={row['actual_reward']}")
    return int(bool(summary["mismatches"]))


if __name__ == "__main__":
    raise SystemExit(main())
