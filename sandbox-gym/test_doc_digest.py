"""Checks for family doc_digest (long doc -> SUMMARY.md). No model, no GPU.
Token-length checks need the Qwen/Qwen3.5-4B tokenizer in the HF cache (skip otherwise).

  ..\\.venv\\Scripts\\python.exe -m unittest test_doc_digest -v
"""

import json
import os
import re
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DOCDIGEST_PARTIAL_LOG", "")      # tests never write the results log

import doc_digest as D  # noqa: E402
import seal  # noqa: E402
from graders import files_ok_ignoring_answer, reward_for  # noqa: E402
from repo_gen import find_spec, make_specs, write_files  # noqa: E402
from runner import Runner  # noqa: E402
from schema import Task, Turn  # noqa: E402
from workspace import Workspace  # noqa: E402

HOLDOUT_DIR = Path(__file__).parent / "data" / "holdout"
SEALED = {t: HOLDOUT_DIR / f"smoke_docdigest_t{t}_{ss[0]}-{ss[-1]}.json"
          for t, ss in D.HOLDOUT_SEEDS.items()}
TRAIN_SEEDS = [1, 2, 3, 7, 42, 123]
ALL_HOLDOUT = [s for ss in D.HOLDOUT_SEEDS.values() for s in ss]
SLACK = 0.06                                           # measured tokens within +-6% of x * budget
READ_LIMIT = D.READ_LINES


def task_for(sp):
    t = Task(name=sp["name"], family=sp["family"], prompt=sp["prompt"], tools=sp["tools"],
             ground_truth=sp["ground_truth"], reward=lambda turns, ws: 0.0,
             meta={"seed": sp["seed"], "checks": []})
    t.reward = reward_for(t)
    return t


def mutations(sp):
    """(ideal, +1 decoy, -1 key, over cap) summaries for a spec."""
    ideal, gt = sp["oracle"], sp["gt"]
    d = gt["decoys"][0]
    k0 = gt["keys"][0]
    with_decoy = ideal + f"- Note: an earlier figure was {d['text']}.\n"
    missing = "\n".join(ln for ln in ideal.split("\n") if k0["project"] not in ln) + "\n"
    over = ideal + "\n" + " ".join(["filler"] * (gt["cap"] - D.words(ideal) + 1)) + "\n"
    return ideal, with_decoy, missing, over


def _tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    except Exception:
        return None


GT = {"keys": [
    {"project": "Velmar Works", "head": "Velmar", "kind": "budget", "text": "4.2 million crowns",
     "atoms": [["num", 4200000.0]]},
    {"project": "Orlen Yard", "head": "Orlen", "kind": "headcount", "text": "418 staff",
     "atoms": [["num", 418.0]]},
    {"project": "Tessaly Bridge", "head": "Tessaly", "kind": "date", "text": "14 March 2026",
     "atoms": [["date", 3, 14, 2026]]},
    {"project": "Calby Food Hub", "head": "Calby", "kind": "partners",
     "text": "Abrams & Carrow, Ottley Brothers and Prowse Group",
     "atoms": [["name", "Abrams & Carrow"], ["name", "Ottley Brothers"], ["name", "Prowse Group"]]},
    {"project": "Dunmere Reservoir", "head": "Dunmere", "kind": "reduction", "text": "23.5 percent",
     "atoms": [["num", 23.5]]},
    {"project": "Fenwick Freight Terminal", "head": "Fenwick", "kind": "throughput",
     "text": "61,400 tonnes", "atoms": [["num", 61400.0]]},
], "decoys": [
    {"type": "draft", "text": "3.8 million crowns", "atoms": [["num", 3800000.0]]},
    {"type": "retract", "text": "460 staff", "atoms": [["num", 460.0]]},
    {"type": "hypo", "text": "9 June 2026", "atoms": [["date", 6, 9, 2026]]},
    {"type": "draft", "text": "Wardle Civil", "atoms": [["name", "Wardle Civil"]]},
], "heads": ["Velmar", "Orlen", "Tessaly", "Calby", "Dunmere", "Fenwick", "Harlow"], "cap": 150}

GOOD = """# Summary
- Velmar Works: budget 4.2 million crowns
- Orlen Yard: 418 staff
- Tessaly Bridge opens 14 March 2026
- Calby Food Hub partners: Abrams & Carrow, Ottley Brothers and Prowse Group
- Dunmere Reservoir cut water losses by 23.5 percent
- Fenwick Freight Terminal: 61,400 tonnes a year
"""


class GraderTolerance(unittest.TestCase):
    def ok(self, text):
        return D.grade_summary(text, GT)[0]

    def test_ideal_passes(self):
        self.assertTrue(self.ok(GOOD))

    def test_number_formats(self):
        for alt in ("4,200,000 crowns", "4.2M", "4.2m crowns", "CR 4.2 mn", "4.20 million",
                    "4200000", "4.2 Million", "4,200 thousand", "4.2 million"):
            self.assertTrue(self.ok(GOOD.replace("4.2 million crowns", alt)), alt)
        for alt in ("61400 tonnes", "61.4 thousand tonnes", "61.4k tonnes", "61 400 t"):
            self.assertTrue(self.ok(GOOD.replace("61,400 tonnes", alt)), alt)
        for alt in ("23.5%", "23.5 per cent", "23.50 percent"):
            self.assertTrue(self.ok(GOOD.replace("23.5 percent", alt)), alt)
        for bad in ("4.3 million", "42 million", "4.2 billion", "4.2"):
            self.assertFalse(self.ok(GOOD.replace("4.2 million crowns", bad)), bad)

    def test_dates(self):
        for alt in ("March 14, 2026", "14th March", "14 Mar 2026", "2026-03-14", "14/03/2026",
                    "on the 14th of March 2026", "Mar. 14"):
            self.assertTrue(self.ok(GOOD.replace("14 March 2026", alt)), alt)
        for bad in ("14 March 2025", "15 March 2026", "March 2026"):
            self.assertFalse(self.ok(GOOD.replace("14 March 2026", bad)), bad)

    def test_case_articles_lists(self):
        for alt in ("abrams and carrow, ottley brothers, prowse group",
                    "Abrams & Carrow; Ottley Brothers; Prowse Group",
                    "the Abrams & Carrow firm, Ottley Brothers Ltd and the Prowse Group",
                    "ABRAMS AND CARROW, OTTLEY BROTHERS AND PROWSE GROUP"):
            self.assertTrue(self.ok(GOOD.replace(
                "Abrams & Carrow, Ottley Brothers and Prowse Group", alt)), alt)
        bullets = GOOD.replace("Calby Food Hub partners: Abrams & Carrow, Ottley Brothers and Prowse Group",
                               "Calby Food Hub delivery partners:\n  * Abrams & Carrow\n  * Ottley Brothers\n"
                               "  * Prowse Group")
        self.assertTrue(self.ok(bullets))
        self.assertFalse(self.ok(GOOD.replace(", Ottley Brothers", "")))
        self.assertTrue(self.ok(GOOD.replace("Velmar Works", "the velmar works project")))

    def test_layouts(self):
        heading = ("## Velmar Works\nBudget: 4.2 million\n\n## Orlen Yard\nHeadcount: 418\n\n"
                   "## Tessaly Bridge\nOpening: 14 March 2026\n\n## Calby Food Hub\n"
                   "- Abrams & Carrow\n- Ottley Brothers\n- Prowse Group\n\n## Dunmere Reservoir\n23.5%\n\n"
                   "## Fenwick Freight Terminal\n61,400 t\n")
        self.assertTrue(self.ok(heading))
        table = ("| Project | Final figure |\n|---|---|\n| Velmar Works | 4.2M crowns |\n"
                 "| Orlen Yard | 418 staff |\n| Tessaly Bridge | 2026-03-14 |\n"
                 "| Calby Food Hub | Abrams & Carrow, Ottley Brothers, Prowse Group |\n"
                 "| Dunmere Reservoir | 23.5 % |\n| Fenwick Freight Terminal | 61,400 tonnes |\n")
        self.assertTrue(self.ok(table))
        prose = ("The approved budget of 4.2 million crowns for Velmar Works; Orlen Yard has 418 staff. "
                 "Tessaly Bridge opened on 14 March 2026.\nCalby Food Hub works with Abrams & Carrow, "
                 "Ottley Brothers and Prowse Group.\nDunmere Reservoir: 23.5 percent. Fenwick Freight "
                 "Terminal handled 61,400 tonnes.\n")
        self.assertTrue(self.ok(prose))

    def test_values_must_attach_to_their_project(self):
        swapped = GOOD.replace("budget 4.2 million crowns", "418 staff").replace(
            "Orlen Yard: 418 staff", "Orlen Yard: 4.2 million crowns")
        ok, info = D.grade_summary(swapped, GT)
        self.assertFalse(ok)
        self.assertEqual(info["missing"], ["Velmar Works", "Orlen Yard"])

    def test_decoys_cap_file(self):
        for d in ("3.8 million", "3,800,000", "460", "9 June", "June 9th, 2026", "Wardle Civil",
                  "wardle civil ltd"):
            ok, info = D.grade_summary(GOOD + f"(an early draft said {d})\n", GT)
            self.assertFalse(ok, d)
            self.assertEqual(len(info["decoys_hit"]), 1, d)
            self.assertEqual(info["found"], 6)
        ok, info = D.grade_summary(GOOD + "word " * 150, GT)
        self.assertFalse(ok)
        self.assertIn("words", info["reason"])
        ok, info = D.grade_summary(None, GT)
        self.assertFalse(ok)
        self.assertEqual(info["reason"], "no SUMMARY.md")
        ok, info = D.grade_summary(GOOD.replace("- Orlen Yard: 418 staff\n", ""), GT)
        self.assertEqual((ok, info["found"], info["n"]), (False, 5, 6))


class DocDigestTests(unittest.TestCase):
    def test_deterministic_and_dispatch(self):
        for s in (9601, 9621, 3):
            self.assertEqual(D.specs(s), D.specs(s))
            self.assertEqual(make_specs(s, "doc_digest"), D.specs(s))
        self.assertEqual([sp["tier"] for sp in D.specs(5)], [1, 2, 3])
        self.assertEqual([sp["tier"] for sp in D.specs(9612)], [2])

    def test_reserved_seed_refused(self):
        for s in (9600, 9606, 9699):
            with self.assertRaises(ValueError):
                D.specs(s)

    def test_split_by_root(self):
        hold = D._holdout_heads()
        hold_names = {sp["name"] for s in ALL_HOLDOUT for sp in D.specs(s)}
        for s in TRAIN_SEEDS:
            for sp in D.specs(s):
                self.assertNotIn(sp["name"], hold_names)
                self.assertFalse({h.lower() for h in sp["gt"]["heads"]} & hold, sp["name"])
                text = " ".join(sp["files"].values()).lower()
                for h in hold:
                    self.assertIsNone(re.search(rf"\b{h}\b", text), (sp["name"], h))

    def test_plants_spread_and_unique(self):
        for s in ALL_HOLDOUT + TRAIN_SEEDS[:3]:
            for sp in D.specs(s):
                flat = " ".join(" ".join(sp["files"][p].split()) for p in sorted(sp["files"]))
                pos = []
                for k in sp["gt"]["keys"]:
                    i = flat.find(k["text"])
                    self.assertGreaterEqual(i, 0, (sp["name"], k))
                    pos.append(flat.rfind(k["text"]) / len(flat))
                thirds = {min(2, int(p * 3)) for p in pos}
                self.assertEqual(thirds, {0, 1, 2}, (sp["name"], pos))
                # every decoy value is written only in planted sentences (1, or 2 with its erratum)
                for d in sp["gt"]["decoys"]:
                    n = sum(D.has_atom(ln, [*d["atoms"][0]]) for ln in re.split(r"(?<=[.;])\s", flat))
                    self.assertIn(n, (1, 2), (sp["name"], d))
                vals = [D._atom_key(tuple(a)) for x in sp["gt"]["keys"] + sp["gt"]["decoys"]
                        for a in x["atoms"]]
                self.assertEqual(len(vals), len(set(vals)), sp["name"])

    def test_token_lengths_exceed_budget(self):
        tok = _tok()
        if tok is None:
            self.skipTest("Qwen/Qwen3.5-4B tokenizer not available")
        for s in ALL_HOLDOUT + TRAIN_SEEDS[:2]:
            for sp in D.specs(s):
                self.assertEqual(sp["budget"], D.HOLDOUT_BUDGET if s in ALL_HOLDOUT
                                 else D.train_budget())
                n = len(tok.encode("".join(sp["files"].values())))
                want = D.TIERS[sp["tier"]]["x"] * sp["budget"]
                self.assertGreater(n, sp["budget"], sp["name"])          # never fits one window
                self.assertLess(abs(n - want) / want, SLACK, (sp["name"], n, want))

    def test_oracle_holdouts(self):
        """Per holdout item: reading the doc through the tools costs > budget and fits the
        turn cap; the ideal summary passes; +1 decoy, -1 key, over cap each fail."""
        tok = _tok()
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d) / "ws", Runner("host"))
            for s in ALL_HOLDOUT:
                sp, = D.specs(s)
                ws.reset()
                write_files(sp["files"], ws.root)
                obs, calls = [], 0
                for path in sorted(sp["files"]):
                    off = 1
                    while True:
                        out = ws.read_file(path, offset=off, limit=READ_LIMIT)
                        calls += 1
                        self.assertNotIn("chars truncated", out, (sp["name"], off))
                        obs.append(out)
                        if "more lines" not in out:
                            break
                        off += READ_LIMIT
                body = " ".join(re.sub(r"^\s*\d+: ", "", ln) for o in obs for ln in o.split("\n")[1:])
                body = " ".join(body.split())
                for x in sp["gt"]["keys"] + sp["gt"]["decoys"]:
                    self.assertIn(x["text"], body, (sp["name"], x["text"]))
                self.assertLessEqual(calls + 2, sp["turn_cap"], (sp["name"], calls))
                if tok is not None:
                    self.assertGreater(sum(len(tok.encode(o)) for o in obs), sp["budget"], sp["name"])
                ideal, with_decoy, missing, over = mutations(sp)
                self.assertTrue(D.grade_summary(ideal, sp["gt"])[0], sp["name"])
                for bad in (with_decoy, missing, over):
                    self.assertFalse(D.grade_summary(bad, sp["gt"])[0], sp["name"])
                # same through the workspace + reward_for (write_file is her only writer)
                t = task_for(sp)
                ans = [Turn(content="ANSWER: done")]
                self.assertEqual(t.reward(ans, ws), 0.0)                   # no SUMMARY.md yet
                ws.write_file(D.SUMMARY, ideal)
                self.assertEqual(t.reward(ans, ws), 1.0, sp["name"])
                self.assertEqual(t.reward([Turn(content="I wrote it.")], ws), 0.0)
                self.assertEqual(t.reward([Turn(content="ANSWER: done", truncated=True)], ws), 0.0)
                self.assertTrue(files_ok_ignoring_answer(t, [Turn(content="x")], ws))
                for bad in (with_decoy, missing, over):
                    (ws.root / D.SUMMARY).write_text(bad, encoding="utf-8")
                    self.assertEqual(t.reward(ans, ws), 0.0, sp["name"])

    def test_train_oracle(self):
        for s in TRAIN_SEEDS:
            for sp in D.specs(s):
                ideal, with_decoy, missing, over = mutations(sp)
                self.assertTrue(D.grade_summary(ideal, sp["gt"])[0], sp["name"])
                self.assertLessEqual(D.words(ideal), sp["gt"]["cap"] // 2, sp["name"])
                for bad in (with_decoy, missing, over):
                    self.assertFalse(D.grade_summary(bad, sp["gt"])[0], sp["name"])

    def test_partial_log(self):
        sp = D.specs(9601)[0]
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "p.jsonl"
            old, D.PARTIAL_LOG = D.PARTIAL_LOG, str(log)
            try:
                (Path(d) / D.SUMMARY).write_text(mutations(sp)[2], encoding="utf-8")
                ok, info = D.grade_workspace(Path(d), sp["ground_truth"], log_as="docdigest-t1-9601-a1")
            finally:
                D.PARTIAL_LOG = old
            row = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual((ok, row["found"], row["n"], row["passed"]), (False, 4, 5, False))

    def test_sealed_holdouts_match_generator(self):
        for tier, path in SEALED.items():
            ok, m = seal.verify(path)
            self.assertTrue(ok, path)
            self.assertEqual((m["family"], len(m["items"])), ("doc_digest", 5))
            for it in m["items"]:
                sp = find_spec(it["seed"], "doc_digest", it["name"])
                self.assertEqual(sp["tier"], tier)
                self.assertEqual((sp["prompt"], sp["ground_truth"]), (it["prompt"], it["ground_truth"]))


if __name__ == "__main__":
    unittest.main()
