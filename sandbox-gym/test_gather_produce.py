"""Checks for family gather_produce (lookups over repo_q -> FINDINGS.md). No model, no GPU.
Needs git + the read-only repo at repo_q.REPO (skips otherwise).

  ..\\.venv\\Scripts\\python.exe -m unittest test_gather_produce -v
"""

import os
import re
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import gather_produce as G
import repo_q as R
import seal
from graders import ANSWER_RE
from runner import Runner
from schema import Task, Turn
from workspace import Workspace

HOLDOUT_DIR = Path(__file__).parent / "data" / "holdout"
SEALED = {1: HOLDOUT_DIR / "smoke_gather_t1_9501-9505.json",
          2: HOLDOUT_DIR / "smoke_gather_t2_9501-9505.json"}
TRAIN_SEEDS = list(range(1, 41)) + [123, 4567]
CALL_CAP = {1: 5, 2: 13}               # oracle tool calls incl. write_file (+1 ANSWER turn)


def plugins_in(files):
    return {PurePosixPath(p).parts[1] for p in files if p.startswith("plugins/")}


def task_for(sp):
    t = Task(name=sp["name"], family=sp["family"], prompt=sp["prompt"], tools=sp["tools"],
             ground_truth=sp["ground_truth"], reward=lambda turns, ws: 0.0,
             meta={"seed": sp["seed"], "checks": sp["checks"]})
    t.reward = G.reward(t)
    return t


class MatchTests(unittest.TestCase):
    def test_tolerant_match(self):
        m = G.match
        ok = [("7,000", "7000"), ("7000.0f", "7000"), ("7000 Hz", "7000"), ("**7000**", "7000"),
              ("`-12`", "-12"), ("0.5", "0.5"), (".5", "0.5"),
              ("Corona, Ember and Nadir", "Corona, Ember, Nadir"),
              ("nadir, corona & ember", "Corona, Ember, Nadir"),
              ("Corona; Ember; Nadir.", "Corona, Ember, Nadir"),
              ("Corona, Ember, and Nadir", "Corona, Ember, Nadir"),
              ("tight", "Tight"), ('"Tight"', "Tight"),
              ("ScaledLookAndFeel", "vstkit::ScaledLookAndFeel"),
              ("public juce::Button", "juce::Button"),
              ("./plugins/Corona/source/CoronaDSP.h", "plugins/Corona/source/CoronaDSP.h"),
              ("plugins\\Corona\\source\\CoronaDSP.h", "plugins/Corona/source/CoronaDSP.h"),
              ("`plugins/Corona/source/CoronaDSP.h:42`", "plugins/Corona/source/CoronaDSP.h"),
              ("plugins/Corona/source/CoronaDSP.h (line 42)", "plugins/Corona/source/CoronaDSP.h"),
              ("Formant", "Formant"), ("Bypass and Bias", "Bias, Bypass"),
              ("yes", "true"), ("False", "false")]
        bad = [("700", "7000"), ("7000, 8000", "7000"), ("-12 to 12", "12"),
               ("Corona, Ember", "Corona, Ember, Nadir"),
               ("Corona, Ember, Nadir, Polaris", "Corona, Ember, Nadir"),
               ("Loose", "Tight"), ("CoronaDSP.h", "plugins/Corona/source/CoronaDSP.h"),
               ("LookAndFeel_V4", "vstkit::ScaledLookAndFeel"), ("", "7000"), (None, "7000"),
               ("true", "false"), ("Bias", "Bias, Bypass")]
        for a, g in ok:
            self.assertTrue(m(a, g), (a, g))
        for a, g in bad:
            self.assertFalse(m(a, g), (a, g))

    def test_findings_never_trip_answer_line(self):
        text = G.render_findings(["7000", "a/b.h"], [["a/b.h"], ["c/d.cpp"]])
        self.assertIsNone(ANSWER_RE.search(text))


@unittest.skipUnless((R.REPO / ".git").exists(), f"repo {R.REPO} not present")
class GatherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        os.environ["GATHER_PARTIAL_LOG"] = str(Path(cls.td.name) / "partial.jsonl")
        cls.ws = Workspace(Path(cls.td.name) / "ws", Runner("host"))

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("GATHER_PARTIAL_LOG", None)
        cls.td.cleanup()

    def episode(self, sp, findings, answer="ANSWER: done", truncated=False):
        self.ws.reset()
        from repo_gen import write_files
        write_files(sp["files"], self.ws.root)
        if findings is not None:
            (self.ws.root / G.FINDINGS).write_text(findings, encoding="utf-8")
        return task_for(sp).reward([Turn(content=answer, truncated=truncated)], self.ws)

    def ideal(self, sp):
        return G.render_findings([it["gt"] for it in sp["items"]],
                                 [it["ev_paths"] for it in sp["items"]])

    def test_deterministic_tiers_and_reserved(self):
        for s in (9501, 9504, 3, 123):
            self.assertEqual(G.specs(s), G.specs(s))
        for s in G.HOLDOUT_SEEDS:
            sps = G.specs(s)
            self.assertEqual([sp["tier"] for sp in sps], [1, 2], s)
            for sp in sps:
                self.assertIn(len(sp["items"]), G.TIER_N[sp["tier"]])
                hops = [it["hops"] for it in sp["items"]]
                self.assertEqual(max(hops), sp["tier"], sp["name"])
                self.assertEqual(len({it["key"] for it in sp["items"]}), len(hops))
        for s in (9500, 9506, 9549, 9301, 9350):
            with self.assertRaises(ValueError):
                G.specs(s)

    def test_split_by_root(self):
        bad = R._holdout_entities() | G.SOURCES["repo_q"].holdout_entities()
        for s in G.HOLDOUT_SEEDS:
            for sp in G.specs(s):
                self.assertFalse(plugins_in(sp["files"]) & set(R.TRAIN_PLUGINS), sp["name"])
                self.assertTrue(all(it["plugin"] in R.HOLDOUT_PLUGINS for it in sp["items"]))
        for s in TRAIN_SEEDS:
            for sp in G.specs(s):
                self.assertFalse(plugins_in(sp["files"]) & set(R.HOLDOUT_PLUGINS), sp["name"])
                for it in sp["items"]:
                    self.assertIn(it["plugin"], R.TRAIN_PLUGINS)
                    self.assertFalse(set(it["entities"]) & bad, (sp["name"], it["key"]))

    def test_oracle_passes_via_tools(self):
        n = 0
        for s in list(G.HOLDOUT_SEEDS) + TRAIN_SEEDS:
            for sp in G.specs(s):
                self.ws.reset()
                from repo_gen import write_files
                write_files(sp["files"], self.ws.root)
                text, calls, _ = G.oracle(sp, self.ws)
                self.assertLessEqual(calls, CALL_CAP[sp["tier"]], sp["name"])
                rw = task_for(sp).reward([Turn(content="ANSWER: done")], self.ws)
                self.assertEqual(rw, 1.0, (sp["name"], text))
                n += 1
        self.assertGreater(n, 60)

    def test_negatives(self):
        for s in (9502, 9504, 7):
            for sp in G.specs(s):
                good = self.ideal(sp)
                self.assertEqual(self.episode(sp, good), 1.0, sp["name"])
                it0 = sp["items"][0]
                gt0, ev0 = it0["gt"], ", ".join(it0["ev_paths"])
                # wrong answer
                wrong = "plugins/Nope/x.h" if "/" in gt0 else ("Nope, Zilch" if "," in gt0 else "987654")
                self.assertEqual(self.episode(sp, good.replace(f"Result: {gt0}\n", f"Result: {wrong}\n", 1)), 0.0)
                # missing evidence path
                self.assertEqual(self.episode(sp, good.replace(f"Evidence: {ev0}\n", "Evidence: plugins/Nope/missing.cpp\n", 1)), 0.0)
                # evidence that exists but doesn't contain the answer
                other = next(p for p in sp["files"] if p.endswith("CMakeLists.txt"))
                self.assertEqual(self.episode(sp, good.replace(f"Evidence: {ev0}\n", f"Evidence: {other}\n", 1)), 0.0)
                # citing the findings file itself (not an original file)
                self.assertEqual(self.episode(sp, good.replace(f"Evidence: {ev0}\n", f"Evidence: {G.FINDINGS}\n", 1)), 0.0)
                # no evidence line, missing heading, extra wrong answer, extra section
                self.assertEqual(self.episode(sp, good.replace(f"Evidence: {ev0}\n", "", 1)), 0.0)
                self.assertEqual(self.episode(sp, good.replace("## Q1\n", "## Summary\n")), 0.0)
                self.assertEqual(self.episode(sp, good.replace(f"Result: {gt0}\n", f"Result: {gt0}\nResult: {wrong}\n", 1)), 0.0)
                self.assertEqual(self.episode(sp, good + f"\n## Q{len(sp['items']) + 1}\nResult: 5\nEvidence: {ev0}\n"), 0.0)
                # no file / no ANSWER line / truncated final turn
                self.assertEqual(self.episode(sp, None), 0.0)
                self.assertEqual(self.episode(sp, good, answer="done, I think"), 0.0)
                self.assertEqual(self.episode(sp, good, truncated=True), 0.0)

    def test_tolerant_findings_format(self):
        sp = G.specs(9501)[1]
        secs = []
        for i, it in enumerate(sp["items"], 1):
            gt = it["gt"].replace(", ", " and ") if "," in it["gt"] else f"**{it['gt']}**"
            ev = "\n".join(f"- `./{p}:3`" for p in it["ev_paths"])
            secs.append(f"### Q{i}: lookup {i}\n- **Answer:** {gt}\n- **Evidence:**\n{ev}")
        self.assertEqual(self.episode(sp, "# Corona findings\n\n" + "\n\n".join(secs) + "\n"), 1.0)

    def test_partial_logged(self):
        sp = G.specs(9503)[1]
        good = self.ideal(sp)
        gt0 = sp["items"][0]["gt"]
        self.episode(sp, good.replace(f"Result: {gt0}\n", "Result: Nope, Zilch\n", 1))
        last = Path(os.environ["GATHER_PARTIAL_LOG"]).read_text(encoding="utf-8").splitlines()[-1]
        self.assertIn('"correct": %d' % (len(sp["items"]) - 1), last)
        self.assertIn("wrong-answer", last)

    def test_dispatch(self):
        from graders import reward_for
        from repo_gen import find_spec, make_specs
        self.assertEqual(make_specs(9505, G.FAMILY), G.specs(9505))
        sp = find_spec(9505, G.FAMILY, G.specs(9505)[0]["name"])
        t = task_for(sp)
        t.reward = reward_for(t)
        self.ws.reset()
        from repo_gen import write_files
        write_files(sp["files"], self.ws.root)
        (self.ws.root / G.FINDINGS).write_text(self.ideal(sp), encoding="utf-8")
        self.assertEqual(t.reward([Turn(content="ANSWER: done")], self.ws), 1.0)

    def test_sealed_holdouts_match_generator(self):
        from repo_gen import find_spec
        for tier, path in SEALED.items():
            ok, m = seal.verify(path)
            self.assertTrue(ok, path)
            self.assertEqual((m["family"], len(m["items"])), (G.FAMILY, 5))
            for it in m["items"]:
                sp = find_spec(it["seed"], G.FAMILY, it["name"])
                self.assertEqual(sp["tier"], tier)
                self.assertEqual((sp["prompt"], sp["ground_truth"], sp["checks"]),
                                 (it["prompt"], it["ground_truth"], it["checks"]))


if __name__ == "__main__":
    unittest.main()
