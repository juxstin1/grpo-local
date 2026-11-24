"""Checks for family doc_make (md/PDF report from notes + csv). No model, no GPU.

  ..\\.venv\\Scripts\\python.exe -m unittest test_doc_make -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import agent
import doc_make as D
import doc_tools
import seal
import tools as T
from graders import reward_for
from repo_gen import find_spec, make_specs, write_files
from runner import Runner
from schema import Action, Task, Turn
from workspace import Workspace

HOLDOUT_DIR = Path(__file__).parent / "data" / "holdout"
SEALED = {t: HOLDOUT_DIR / f"smoke_docmake_t{t}_9401-9405.json" for t in (1, 2, 3)}
TRAIN_SEEDS = [5000, 5001, 5002, 5003, 5004, 5007, 5100, 5555]


def task_for(spec):
    t = Task(name=spec["name"], family=spec["family"], prompt=spec["prompt"], tools=spec["tools"],
             ground_truth=spec["ground_truth"], reward=lambda turns, ws: 0.0,
             meta={"seed": spec["seed"], "checks": []})
    t.reward = reward_for(t)
    return t


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.ws = Workspace(Path(self._td.name) / "ws", Runner("host"))

    def tearDown(self):
        self._td.cleanup()

    def fresh(self, spec):
        self.ws.reset()
        write_files(spec["files"], self.ws.root)

    def grade(self, spec):
        return D.grade_doc(self.ws.root, json.loads(spec["ground_truth"]))

    def put(self, spec, md):
        """Write md the way the agent would (md -> write_file, pdf -> make_pdf)."""
        t = json.loads(spec["ground_truth"])
        if t["format"] == "md":
            (self.ws.root / t["path"]).write_text(md, encoding="utf-8")
        else:
            doc_tools.make_pdf(self.ws, {"path": t["path"], "markdown": md})


class Generation(Base):
    def test_deterministic_and_dispatch(self):
        for s in (9401, 9405, 5000, 5123):
            self.assertEqual(D.specs(s), D.specs(s))
            self.assertEqual(make_specs(s, "doc_make"), D.specs(s))
        self.assertEqual([sp["tier"] for sp in D.specs(9402)], [1, 2, 3])
        sp = find_spec(9402, "doc_make", "docmake-t2-9402")
        self.assertIn("make_pdf", sp["tools"])
        self.assertIn("write_file", find_spec(9402, "doc_make", "docmake-t1-9402")["tools"])

    def test_reserved_seed_refused(self):
        for s in (9400, 9406, 9449):
            with self.assertRaises(ValueError):
                D.specs(s)

    def test_split_by_root(self):
        hold_text = " ".join(t for s in D.HOLDOUT_SEEDS for t in D.specs(s)[0]["files"].values())
        self.assertFalse(set(D.PAIRS_T) & set(D.PAIRS_H))
        for a, b in ((D.ORGS_T, D.ORGS_H), (D.LAST_T, D.LAST_H), (D.VENDORS_T, D.VENDORS_H)):
            self.assertFalse(set(a) & set(b))
        for s in TRAIN_SEEDS + list(range(5200, 5230)):
            self.assertEqual(D.world(s)["side"], "train")
            for sp in D.specs(s):
                body = " ".join(sp["files"].values())
                for pair in D.PAIRS_H:
                    self.assertNotIn(pair[0], body)
                for name in D.ORGS_H + D.VENDORS_H + D.LAST_H:
                    self.assertNotIn(name, body)
        self.assertTrue(any(h[0] in hold_text for h in D.PAIRS_H))

    def test_material_contains_facts_and_plants(self):
        """Every required fact is findable in the notes (tolerant), and the near-misses
        really are planted there (otherwise they test nothing)."""
        for s in list(D.HOLDOUT_SEEDS) + TRAIN_SEEDS:
            for sp in D.specs(s):
                t = json.loads(sp["ground_truth"])
                mat = "\n".join(v for k, v in sp["files"].items() if k.startswith("notes/"))
                norm, dn, nums = D._words(mat), D._date_text(mat), D.numbers_in(mat)
                for f in t["facts"]:
                    self.assertTrue(D.fact_present(f, mat, norm, nums, dn), (sp["name"], f["label"]))
                for it in t["forbidden"]:
                    self.assertTrue(D.forbidden_hit(it, norm, nums, dn), (sp["name"], it["label"]))
                if t.get("table"):
                    csv_nums = D.numbers_in(sp["files"]["data.csv"])
                    for r in t["table"]["rows"][:-1]:
                        self.assertLessEqual({float(x) for x in r[1:]}, csv_nums)

    def test_turn_budget(self):
        for sp in D.specs(9401):
            self.assertLessEqual(len(sp["files"]) + 3, 10, "list + reads + write + answer fits cap 10")

    def test_prompts_do_not_leak_values(self):
        for s in D.HOLDOUT_SEEDS:
            for sp in D.specs(s):
                t = json.loads(sp["ground_truth"])
                nums = D.numbers_in(sp["prompt"])
                for f in t["facts"]:
                    if f["kind"] == "num":
                        self.assertNotIn(float(f["value"]), nums)
                    if f["kind"] == "text":
                        self.assertNotIn(f["value"], sp["prompt"])


class Oracle(Base):
    def test_oracle_passes_holdouts_and_train(self):
        for s in list(D.HOLDOUT_SEEDS) + TRAIN_SEEDS:
            for sp in D.specs(s):
                self.fresh(sp)
                ans, calls = D.oracle(sp, self.ws)
                ok, why = self.grade(sp)
                self.assertTrue(ok, (sp["name"], why))
                self.assertLessEqual(calls + 1, 10, sp["name"])

    def test_missing_section_fails(self):
        for tier in (1, 2, 3):
            sp = D.specs(9403)[tier - 1]
            t = json.loads(sp["ground_truth"])
            for h in (t["headings"][0], t["headings"][-1]):
                self.fresh(sp)
                self.put(sp, D.ideal_markdown(t, drop_heading=h))
                ok, why = self.grade(sp)
                self.assertFalse(ok, (tier, h))

    def test_wrong_number_fails(self):
        for tier in (1, 2, 3):
            sp = D.specs(9404)[tier - 1]
            t = json.loads(sp["ground_truth"])
            f = next(f for f in t["facts"] if f["kind"] == "num")
            wrong = f"${int(f['value']) + 500:,}" if "USD" in f["label"] else str(int(f["value"]) + 1)
            self.fresh(sp)
            self.put(sp, D.ideal_markdown(t, corrupt={f["label"]: wrong}))
            ok, why = self.grade(sp)
            self.assertFalse(ok, tier)
            self.assertTrue(why.startswith("fact:"), why)

    def test_wrong_table_number_fails(self):
        sp = D.specs(9401)[2]
        t = json.loads(sp["ground_truth"])
        ideal = json.loads(sp["ground_truth"])
        t["table"]["rows"][1][2] += 1000              # the document says a different value
        self.fresh(sp)
        self.put(sp, D.ideal_markdown(t))
        ok, why = D.grade_doc(self.ws.root, ideal)
        self.assertFalse(ok)
        self.assertTrue(why.startswith("table-row"), why)

    def test_near_miss_and_paste_everything_fail(self):
        sp = D.specs(9402)[1]
        t = json.loads(sp["ground_truth"])
        self.fresh(sp)
        sib = next(i for i in t["forbidden"] if i["label"].startswith("sibling/draft"))
        self.put(sp, D.ideal_markdown(t) + f"\nSister project budget: ${sib['value']:,}\n")
        ok, why = self.grade(sp)
        self.assertFalse(ok)
        self.assertTrue(why.startswith("near-miss"), why)
        # dumping every note is blocked by the planted values even though all facts are present
        self.fresh(sp)
        dump = "\n".join(v for k, v in sp["files"].items() if k.startswith("notes/"))
        self.put(sp, "\n".join(f"## {h}" for h in t["headings"]) + "\n" + dump)
        self.assertFalse(self.grade(sp)[0])

    def test_order_validity_and_length(self):
        sp = D.specs(9405)[1]
        t = json.loads(sp["ground_truth"])
        self.fresh(sp)
        self.assertEqual(self.grade(sp), (False, "missing-file"))
        (self.ws.root / "report.pdf").write_bytes(b"not a pdf at all")
        self.assertTrue(self.grade(sp)[1].startswith("pdf-invalid"))
        (self.ws.root / "report.pdf").write_bytes(b"")
        self.assertEqual(self.grade(sp), (False, "empty-file"))
        swapped = dict(t, headings=list(reversed(t["headings"])))
        self.put(sp, D.ideal_markdown(t))
        self.assertTrue(self.grade(sp)[0])
        self.assertFalse(D.grade_doc(self.ws.root, swapped)[0])
        self.put(sp, D.ideal_markdown(t) + "\nfiller words. " * 600)
        self.assertTrue(self.grade(sp)[1].startswith("length"))
        sp1 = D.specs(9405)[0]
        self.fresh(sp1)
        self.put(sp1, "   \n")
        self.assertFalse(self.grade(sp1)[0])


class Tolerance(unittest.TestCase):
    def test_numbers(self):
        n = D.numbers_in
        self.assertIn(7000.0, n("budget $7,000."))
        self.assertIn(7000.0, n("budget 7000 USD"))
        self.assertIn(7000.0, n("USD 7,000.00"))
        self.assertNotIn(7000.0, n("budget 70,000"))
        self.assertNotIn(7000.0, n("budget 700"))

    def test_dates_lists_names(self):
        f = {"kind": "date", "value": "2027-03-14"}
        for txt in ("14 March 2027", "March 14, 2027", "march 14th 2027", "2027-03-14", "Mar 14, 2027"):
            self.assertTrue(D.fact_present(f, txt, D._words(txt), set(), D._date_text(txt)), txt)
        for txt in ("15 March 2027", "14 March 2026"):
            self.assertFalse(D.fact_present(f, txt, D._words(txt), set(), D._date_text(txt)), txt)
        lst = {"kind": "list", "items": ["A One", "B Two", "C Three"]}
        for txt in ("A One, B Two and C Three", "- a one\n- b two\n- c three", "C Three; B Two; A One"):
            self.assertTrue(D.fact_present(lst, txt, D._words(txt), set(), ""), txt)
        self.assertFalse(D.fact_present(lst, "A One and B Two", D._words("A One and B Two"), set(), ""))
        nm = {"kind": "text", "value": "Elena Rossi"}
        for txt in ("ELENA ROSSI", "Rossi, Elena", "lead: Elena  Rossi."):
            self.assertTrue(D.fact_present(nm, txt, D._words(txt), set(), ""), txt)
        self.assertFalse(D.fact_present(nm, "Elena Rossini", D._words("Elena Rossini"), set(), ""))

    def test_headings(self):
        hs = ["Team and Vendors", "Risks"]
        self.assertTrue(D.headings_in_order("# 1. team and vendors\ntext\n## RISKS\n", hs)[0])
        self.assertFalse(D.headings_in_order("Risks\nTeam and Vendors\n", hs)[0])
        self.assertFalse(D.headings_in_order("Team and Vendors\nTeam\n", hs)[0])


class Wiring(Base):
    def test_reward_for(self):
        sp = D.specs(9401)[0]
        t = task_for(sp)
        self.fresh(sp)
        D.oracle(sp, self.ws)
        good = [Turn(content="ANSWER: report.md")]
        self.assertEqual(t.reward(good, self.ws), 1.0)
        self.assertEqual(t.reward([Turn(content="ANSWER: report.md", truncated=True)], self.ws), 0.0)
        self.assertEqual(t.reward([Turn(content="done")], self.ws), 0.0)
        self.assertEqual(t.reward(good, None), 0.0)
        self.fresh(sp)                                   # nothing written
        self.assertEqual(t.reward(good, self.ws), 0.0)

    def test_make_pdf_tool(self):
        self.ws.reset()
        out = agent.execute_tool(self.ws, Action(name="make_pdf", arguments={
            "path": "out/r.pdf", "markdown": "# Title\n\n- a\n- b\n\n| X | Y |\n|---|---|\n| 1,200 | 3 |\n"}))
        self.assertTrue(out.startswith("wrote out/r.pdf"), out)
        text, err = D.extract_text(self.ws.root / "out" / "r.pdf", "pdf")
        self.assertIn("1,200", text)
        run = lambda a: agent.run_action(self.ws, Action(name="make_pdf", arguments=a), ["make_pdf"])  # noqa: E731
        self.assertTrue(run({"path": "x.txt", "markdown": "a"}).startswith("error: path must end in .pdf"))
        self.assertIn("missing required argument", run({"path": "x.pdf"}))
        self.assertTrue(run({"path": "../x.pdf", "markdown": "a"}).startswith("error:"))   # jail
        self.assertIn("make_pdf(path, markdown)", T.describe_tools(["make_pdf"]))
        names = [x["function"]["name"] for x in agent._openai_tools(
            [{"role": "system", "content": T.describe_tools(D.TOOLS_PDF)}])]
        self.assertIn("make_pdf", names)

    def test_sealed_holdouts_match_generator(self):
        for tier, path in SEALED.items():
            ok, m = seal.verify(path)
            self.assertTrue(ok, path)
            self.assertEqual(m["family"], "doc_make")
            self.assertEqual(m["seeds"], list(D.HOLDOUT_SEEDS))
            self.assertEqual(len(m["items"]), 5)
            for it in m["items"]:
                sp = find_spec(it["seed"], "doc_make", it["name"])
                self.assertEqual(sp["tier"], tier)
                self.assertEqual((sp["prompt"], sp["ground_truth"]), (it["prompt"], it["ground_truth"]))


if __name__ == "__main__":
    unittest.main()
