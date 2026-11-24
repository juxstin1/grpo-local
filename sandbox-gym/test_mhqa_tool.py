"""Checks for family mhqa_t (tool-fetched multi-hop KB). No model, no GPU.

  ..\\.venv\\Scripts\\python.exe -m unittest test_mhqa_tool -v
"""

import re
import tempfile
import unittest
from pathlib import Path

import mhqa_tool as M
from graders import reward_for
from repo_gen import find_spec, make_specs, write_files
from runner import Runner
from schema import Task, Turn
from workspace import Workspace

SEEDS = list(range(9201, 9206)) + list(range(9211, 9216)) + list(range(9221, 9226)) + [123, 4567]


def task_for(spec):
    t = Task(name=spec["name"], family=spec["family"], prompt=spec["prompt"],
             tools=spec["tools"], ground_truth=spec["ground_truth"],
             reward=lambda turns, ws: 0.0, meta={"seed": spec["seed"], "checks": []})
    t.reward = reward_for(t)
    return t


class MhqaToolTests(unittest.TestCase):
    def test_deterministic(self):
        for s in SEEDS[:4]:
            self.assertEqual(M.specs(s), M.specs(s))
            self.assertEqual(make_specs(s, "mhqa_t"), M.specs(s))
        self.assertNotEqual(M.kb_files(9201), M.kb_files(9202))

    def test_eval_world_refused(self):
        with self.assertRaises(ValueError):
            M.specs(M.mh.SEED)

    def test_oracle_solves_every_item(self):
        for s in SEEDS:
            for sp in M.specs(s):
                chain, reads = M.oracle(sp)
                self.assertEqual(chain, sp["ground_truth"].split(M.SEP), sp["name"])
                self.assertEqual(reads, sp["hops"], sp["name"])
                self.assertEqual(len(chain), sp["hops"] + 1)

    def test_facts_not_in_prompt_and_no_single_file_shortcut(self):
        for s in SEEDS:
            for sp in M.specs(s):
                chain = sp["ground_truth"].split(M.SEP)
                for hop in chain[1:]:
                    self.assertNotIn(hop, sp["prompt"], sp["name"])
                for path, text in sp["files"].items():
                    self.assertFalse(chain[0] in text and chain[-1] in text,
                                     f"{sp['name']}: {path} names subject and answer")
                    self.assertLess(sum(c in text for c in chain), len(chain), path)

    def test_distractors_present_and_off_chain(self):
        for s in SEEDS:
            _, _, dis = M.world(s)
            names = [n for d in dis.values() for n in d]
            self.assertGreaterEqual(len(names), 10)
            for sp in M.specs(s):
                chain = sp["ground_truth"].split(M.SEP)
                self.assertFalse(set(names) & set(chain), sp["name"])
            for d in ("people", "companies", "cities"):   # list_dir shows all (cap 60)
                n = sum(p.startswith(f"kb/{d}/") for p in M.kb_files(s))
                self.assertLess(n, 60)

    def test_grader(self):
        sp = M.specs(9221)[2]
        chain = sp["ground_truth"].split(M.SEP)
        self.assertTrue(M.grade(M.SEP.join(chain), sp["ground_truth"])[0])
        self.assertTrue(M.grade(M.SEP.join(chain[1:]), sp["ground_truth"])[0])   # subject optional
        self.assertTrue(M.grade(M.SEP.join(chain).upper(), sp["ground_truth"])[0])
        self.assertFalse(M.grade(chain[-1], sp["ground_truth"])[0])              # endpoint only
        self.assertFalse(M.grade(M.SEP.join(chain[:-1]), sp["ground_truth"])[0]) # one hop short
        wrong = chain[:]
        wrong[2] = chain[3]
        self.assertFalse(M.grade(M.SEP.join(wrong), sp["ground_truth"])[0])      # bad middle hop
        self.assertFalse(M.grade(None, sp["ground_truth"])[0])

    def test_reward_for_and_workspace_walk(self):
        """Real Workspace tools reach the chain; reward_for grades the transcript."""
        sp = find_spec(9211, "mhqa_t", "mhqat-h5-9211")
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(Path(d) / "ws", Runner("host"))
            ws.reset()
            write_files(sp["files"], ws.root)
            subject = sp["ground_truth"].split(M.SEP)[0]
            hit = ws.search(subject.split()[0], "kb/people")
            self.assertIn(M.slug(subject), hit)
            self.assertIn("Mentor:", ws.read_file(f"kb/people/{M.slug(subject)}.md"))
            self.assertIn("file", ws.list_dir("kb/cities"))
            t = task_for(sp)
            good = [Turn(content="ANSWER: " + sp["ground_truth"])]
            bad = [Turn(content="ANSWER: " + sp["ground_truth"].split(M.SEP)[-1])]
            cut = [Turn(content="ANSWER: " + sp["ground_truth"], truncated=True)]
            self.assertEqual(t.reward(good, ws), 1.0)
            self.assertEqual(t.reward(bad, ws), 0.0)
            self.assertEqual(t.reward(cut, ws), 0.0)


if __name__ == "__main__":
    unittest.main()
