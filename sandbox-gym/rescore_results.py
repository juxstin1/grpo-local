"""Re-score stored eval results from their transcripts with the correct grader.

Why: eval.py builds each episode's workspace at data/workspaces/<item>-a<k>, a path
shared by every eval run. Two evals of the same set run at once (e.g. r3 + 9B
rebaseline) wipe each other's workspace (ws.reset) between her last edit and
the grade -> a correct edit recorded as 0 (or the other model's files graded).

Here every episode is rebuilt in its own private temp dir: seed files ->
replay_edits(her transcript) -> the family grader (graders.reward_for via
eval.make_task), with Runner(<backend>) for any code execution. The recorded
final-turn `truncated` flag is carried over (transcripts don't store it).

Never touches the eval dirs: writes <out>/<slug>/raw/<fam>-rescored.jsonl and
<out>/<slug>/<fam>.summary-rescored.json (default <out> = OUT, outside data/results
so status.ps1 / raw/*.jsonl globs never see them). Each regrade workspace is
retained under <out>/<slug>/workspaces. Records without a transcript are kept
with rescored=null.

  python rescore_results.py r3-docker-edit_9001-9005 r3-docker-editv_h_9001-9005
  python rescore_results.py --glob "r3-*" --families edit,editv_h
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from agent import replay_edits
from config import Config
from eval import make_task
from graders import files_ok_ignoring_answer
from repo_gen import find_spec, write_files
from runner import BridgeError, Runner
from schema import Action, Turn
from workspace import Workspace

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "_archive" / "2026-09-30" / "rescored"


def holdout_items() -> dict[tuple[str, str], dict]:
    items = {}
    for f in sorted((HERE / "data" / "holdout").glob("*.json")):
        try:
            for it in json.loads(f.read_text(encoding="utf-8")).get("items", []):
                if "checks" in it or it.get("family", "").startswith("fix"):
                    items.setdefault((it["family"], it["name"]), it)
        except Exception:  # noqa: BLE001 -- non-set json in holdout/
            continue
    return items


def regrade(item: dict, tr: dict, rec: dict, backend: str,
            workspaces: Path) -> tuple[int, bool]:
    workspaces.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="rescore-", dir=workspaces))
    ws = Workspace(tmp / "ws", Runner(backend))
    ws.reset()
    write_files(find_spec(item["seed"], item["family"], item["name"])["files"], ws.root)
    turns = [Turn(content=t["content"], resp_text=t["content"],
                  action=Action(t["action"]["name"], t["action"]["arguments"]) if t["action"] else None,
                  obs=t["obs"]) for t in tr["turns"]]
    if turns and rec.get("truncated"):
        turns[-1].truncated = True
    replay_edits(ws, turns)
    task = make_task(item)
    passed = task.reward(turns, ws) >= 0.5
    ok_noans = (not passed and rec.get("answer") is None
                and files_ok_ignoring_answer(task, turns, ws))
    return int(passed), bool(ok_noans)


def rescore_file(raw: Path, items: dict, backend: str, out: Path = OUT) -> dict:
    slug_dir = raw.parent.parent
    dst = out / slug_dir.name
    (dst / "raw").mkdir(parents=True, exist_ok=True)
    fam = raw.stem
    out_rows, before, after, flips01, flips10, missing, errors = [], 0, 0, 0, 0, 0, 0
    for line in raw.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("passed") is None:
            continue
        old = int(bool(rec["passed"]))
        before += old
        tp = slug_dir / "transcripts" / f"{rec['name']}-a{rec['attempt']}.json"
        item = items.get((rec["family"], rec["name"]))
        new = None
        row = {**rec, "recorded_passed": rec["passed"]}
        if tp.exists() and item is not None:
            try:
                new, ok_noans = regrade(item, json.loads(tp.read_text(encoding="utf-8")),
                                         rec, backend, dst / "workspaces")
                row.update(passed=bool(new), files_ok_no_answer=ok_noans, rescored=new)
            except BridgeError as e:
                errors += 1
                row.update(rescored=None, rescore_error=str(e)[:200])
        else:
            missing += 1
            row.update(rescored=None, rescore_error="no transcript" if not tp.exists() else "no holdout item")
        if new is not None:
            after += new
            flips01 += old == 0 and new == 1
            flips10 += old == 1 and new == 0
        else:
            after += old                      # unrescorable: keep recorded verdict in the count
        out_rows.append(row)
    (dst / "raw" / f"{fam}-rescored.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    summ = {"slug": slug_dir.name, "family": fam, "n": len(out_rows), "backend": backend,
            "recorded_pass": before, "rescored_pass": after, "flips_0to1": flips01,
            "flips_1to0": flips10, "unrescorable": missing, "bridge_errors": errors}
    (dst / f"{fam}.summary-rescored.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--glob", default=None, help="results slug glob, e.g. 'r3-*'")
    ap.add_argument("--families", default="edit,editv_h")
    ap.add_argument("--runner", default="docker", help="docker (home) or remote (bridge)")
    ap.add_argument("--out", type=Path, default=OUT, help="output root (never data/results)")
    a = ap.parse_args()
    res = Config().episode.data_root / "results"
    slugs = list(a.slugs) + ([p.name for p in sorted(res.glob(a.glob))] if a.glob else [])
    fams = a.families.split(",")
    items = holdout_items()
    print(f"{'slug':42} {'fam':8} {'n':>3} {'before':>6} {'after':>6} {'0->1':>5} {'1->0':>5} {'unscored':>8}")
    for slug in slugs:
        for fam in fams:
            raw = res / slug / "raw" / f"{fam}.jsonl"
            if not raw.exists():
                continue
            s = rescore_file(raw, items, a.runner, a.out)
            print(f"{slug:42} {fam:8} {s['n']:>3} {s['recorded_pass']:>6} {s['rescored_pass']:>6} "
                  f"{s['flips_0to1']:>5} {s['flips_1to0']:>5} {s['unrescorable']:>8}"
                  + (f"  bridge-errors {s['bridge_errors']}" if s["bridge_errors"] else ""))


if __name__ == "__main__":
    main()
