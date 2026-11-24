"""Checks for family repo_q (questions over the real plugin repo). No model, no GPU.
Needs git + the read-only repo at repo_q.REPO (skips otherwise).

  ..\\.venv\\Scripts\\python.exe -m unittest test_repo_q -v
"""

import re
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import repo_q as R
import seal
from graders import reward_for
from repo_gen import find_spec, make_specs, write_files
from runner import Runner
from schema import Task, Turn
from workspace import Workspace

HOLDOUT_DIR = Path(__file__).parent / "data" / "holdout"
SEALED = {1: HOLDOUT_DIR / "smoke_repoq_t1_9301-9305.json",
          2: HOLDOUT_DIR / "smoke_repoq_t2_9301-9305.json"}
TRAIN_SEEDS = [1, 2, 3, 4, 5, 6, 17, 18, 55, 123, 4567]
CALL_CAP = {1: 2, 2: 4}                       # oracle tool calls per tier (turn cap = calls + 1)
SECRET_TEXT = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}"
                         r"|ghp_[A-Za-z0-9]{30,}|xox[bap]-[A-Za-z0-9-]{10,}")


def task_for(spec):
    t = Task(name=spec["name"], family=spec["family"], prompt=spec["prompt"],
             tools=spec["tools"], ground_truth=spec["ground_truth"],
             reward=lambda turns, ws: 0.0, meta={"seed": spec["seed"], "checks": []})
    t.reward = reward_for(t)
    return t


def plugins_in(files):
    return {PurePosixPath(p).parts[1] for p in files if p.startswith("plugins/")}


@unittest.skipUnless((R.REPO / ".git").exists(), f"repo {R.REPO} not present")
class RepoQTests(unittest.TestCase):
    def test_deterministic_and_dispatch(self):
        for s in (9301, 9304, 3, 123):
            self.assertEqual(R.specs(s), R.specs(s))
            self.assertEqual(make_specs(s, "repo_q"), R.specs(s))
        self.assertEqual([x[1] for x in R.lookup_items(9302)],
                         [sp["ground_truth"] for sp in R.specs(9302)])
        q, gt, files = R.lookup_items(9301)[0]
        self.assertIn("ANSWER:", q)
        self.assertTrue(gt and isinstance(files, dict) and files)

    def test_reserved_seed_refused(self):
        with self.assertRaises(ValueError):
            R.specs(9306)

    def test_split_by_root(self):
        hold_ents = R._holdout_entities()
        for s in R.HOLDOUT_SEEDS:
            sps = R.specs(s)
            self.assertEqual(sorted(sp["tier"] for sp in sps), [1, 2], s)
            for sp in sps:
                self.assertEqual(sp["commit"], R.PIN)
                self.assertFalse(plugins_in(sp["files"]) & set(R.TRAIN_PLUGINS), sp["name"])
        for s in TRAIN_SEEDS:
            for sp in R.specs(s):
                self.assertIn(sp["plugin"], R.TRAIN_PLUGINS)
                self.assertFalse(plugins_in(sp["files"]) & set(R.HOLDOUT_PLUGINS), sp["name"])
                self.assertFalse(set(sp["entities"]) & hold_ents, sp["name"])

    def test_oracle_solves_holdouts_and_train(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d) / "ws", Runner("host"))
            for s in list(R.HOLDOUT_SEEDS) + TRAIN_SEEDS:
                for sp in R.specs(s):
                    ws.reset()
                    write_files(sp["files"], ws.root)
                    ans, calls, _ = R.oracle(sp, ws)
                    self.assertTrue(R.grade(ans, sp["ground_truth"])[0], (sp["name"], ans))
                    self.assertLessEqual(calls, CALL_CAP[sp["tier"]], sp["name"])

    def test_no_secrets_or_binaries(self):
        for s in list(R.HOLDOUT_SEEDS) + TRAIN_SEEDS[:4]:
            for sp in R.specs(s):
                total = 0
                for path, text in sp["files"].items():
                    self.assertFalse(R.SECRET_RE.search(path), path)
                    self.assertNotIn("third_party", path)
                    self.assertTrue(path.endswith(tuple(R.CODE_SUFFIX)) or path.endswith("CMakeLists.txt"), path)
                    self.assertNotIn("\x00", text, path)
                    self.assertIsNone(SECRET_TEXT.search(text), path)
                    total += len(text.encode("utf-8"))
                self.assertLessEqual(total, R.MAX_TOTAL + 200_000)

    def test_grader(self):
        g = R.grade
        self.assertTrue(g("7000", "7000")[0])
        self.assertTrue(g("7000.0", "7000")[0])
        self.assertTrue(g("7000f", "7000")[0])
        self.assertFalse(g("700", "7000")[0])
        self.assertTrue(g("polaris, Ember,nadir", "Ember, Nadir, Polaris")[0])
        self.assertFalse(g("Ember, Polaris", "Ember, Nadir, Polaris")[0])
        self.assertFalse(g("Ember, Nadir, Polaris, Corona", "Ember, Nadir, Polaris")[0])
        self.assertTrue(g("`./plugins/Corona/source/CoronaDSP.h`", "plugins/Corona/source/CoronaDSP.h")[0])
        self.assertTrue(g("plugins\\Corona\\source\\CoronaDSP.h", "plugins/Corona/source/CoronaDSP.h")[0])
        self.assertFalse(g("CoronaDSP.h", "plugins/Corona/source/CoronaDSP.h")[0])
        self.assertTrue(g("ScaledLookAndFeel", "vstkit::ScaledLookAndFeel")[0])
        self.assertFalse(g("LookAndFeel_V4", "vstkit::ScaledLookAndFeel")[0])
        self.assertTrue(g("tight", "Tight")[0])
        self.assertFalse(g("Loose", "Tight")[0])
        self.assertFalse(g(None, "Tight")[0])

    def test_reward_for_wiring(self):
        sp = find_spec(9304, "repo_q", "repoq-t2-history-9304")
        t = task_for(sp)
        good = [Turn(content="ANSWER: " + sp["ground_truth"])]
        bad = [Turn(content="ANSWER: Bypass")]
        cut = [Turn(content="ANSWER: " + sp["ground_truth"], truncated=True)]
        self.assertEqual(t.reward(good, None), 1.0)
        self.assertEqual(t.reward(bad, None), 0.0)
        self.assertEqual(t.reward(cut, None), 0.0)
        self.assertEqual(t.reward([Turn(content="no answer line")], None), 0.0)

    def test_sealed_holdouts_match_generator(self):
        for tier, path in SEALED.items():
            ok, m = seal.verify(path)
            self.assertTrue(ok, path)
            self.assertEqual(m["family"], "repo_q")
            self.assertEqual(len(m["items"]), 5)
            for it in m["items"]:
                sp = find_spec(it["seed"], "repo_q", it["name"])
                self.assertEqual(sp["tier"], tier)
                self.assertEqual((sp["prompt"], sp["ground_truth"]), (it["prompt"], it["ground_truth"]))


if __name__ == "__main__":
    unittest.main()
