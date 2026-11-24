"""One budget: default 8192 untouched; GYM_BUDGET=32768 moves every derived threshold."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PROBES = {
    "gym": r"""
import json, sys
sys.path[:0] = [%(h)r, %(r)r]
import notes
from config import Config
b = Config().episode.budget
print(json.dumps({"b": b, "trig": notes.compact_trigger_tokens(b), "ctx": notes.compact_context_max(b)}))
""",
    "handoff": r"""
import json, sys
sys.path[:0] = [%(r)r + "/handoff-gym"]
import common as C
print(json.dumps({"ho": C.BUDGET}))
""",
    "sft": r"""
import json, sys
sys.path[:0] = [%(r)r + "/sft"]
import sft_schema as S
print(json.dumps({"sft": S.MAX_TOKENS_MULTITURN}))
""",
    "mixed": r"""
import json, sys
sys.path[:0] = [%(h)r]
import grpo_mixed as M
print(json.dumps({"endpoint": M.MAX_TOKENS}))
""",
}


def probe(env_budget=None, argv=()):
    env = {k: v for k, v in os.environ.items() if k != "GYM_BUDGET"}
    if env_budget:
        env["GYM_BUDGET"] = str(env_budget)
    out = {}
    for name, src in PROBES.items():
        r = subprocess.run([sys.executable, "-c", src % {"h": str(HERE), "r": str(ROOT)}, *argv],
                           capture_output=True, text=True, env=env, cwd=str(HERE))
        assert r.returncode == 0, name + r.stderr[-800:]
        out.update(json.loads(r.stdout.strip().splitlines()[-1]))
    return out


class BudgetTest(unittest.TestCase):
    def test_default_unchanged(self):
        d = probe()
        self.assertEqual(d, {"b": 8192, "ho": 8192, "sft": 8192,
                             "trig": 6144, "ctx": 3277, "endpoint": 4096})

    def test_env_override_moves_everything(self):
        d = probe(32768)
        self.assertEqual(d, {"b": 32768, "ho": 32768, "sft": 32768,
                             "trig": 24576, "ctx": 13107, "endpoint": 32768})

    def test_cli_override(self):
        self.assertEqual(probe(argv=["--budget", "16384"])["trig"], 12288)

    def test_require_same_and_floor(self):
        sys.path.append(str(ROOT))
        import gym_budget as GB
        self.assertEqual(GB.require_same([None, 8192]), 8192)    # legacy record = default
        with self.assertRaises(SystemExit):
            GB.require_same([8192, 32768], "x")
        with self.assertRaises(SystemExit):
            GB.resolve([], {"GYM_BUDGET": "100"})

    def test_check_server(self):
        sys.path.append(str(ROOT))
        import gym_budget as GB
        orig = GB.server_max_model_len
        try:
            GB.server_max_model_len = lambda *a, **k: 8192
            GB.check_server("http://x", 8192)
            with self.assertRaises(SystemExit) as e:
                GB.check_server("http://x", 32768)
            self.assertIn("max_model_len 8192", str(e.exception))
            GB.server_max_model_len = lambda *a, **k: None
            GB.check_server("http://x", 32768)   # unknown: warn only
        finally:
            GB.server_max_model_len = orig


if __name__ == "__main__":
    unittest.main()
