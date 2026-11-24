"""Offline checks for leg-3 pool isolation and token-exact MHQA rewards."""

import json
import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import grpo_mixed as mixed
import repo_gen
from config import Config
from schema import Task, Trajectory


class FakeTokenizer:
    completion = "</think>\n\nANSWER: wrong"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] and kwargs["add_generation_prompt"]
        assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
        return [11, 12, 13]

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 99

    def decode(self, ids, **kwargs):
        return self.completion + ("<|im_end|>" if ids and ids[-1] == 99 else "")


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def mhqa_task():
    record = mixed.synth_multihop.records(7300, {3: 2, 4: 2})[0]
    return Task(name=record["uid"], family="mhqa", prompt=record["prompt"],
                tools=[], ground_truth="", reward=lambda turns, ws: 0,
                meta={"seed": 7300, "root_id": record["root_id"], "record": record})


def response(task, content, ids=None, prompt=None, finish="stop"):
    return FakeResponse({"prompt_token_ids": [11, 12, 13] if prompt is None else prompt,
                         "choices": [{"token_ids": [7, 99] if ids is None else ids,
                                      "finish_reason": finish,
                                      "message": {"content": content}}]})


class MhqaRolloutTests(unittest.TestCase):
    def setUp(self):
        self.task = mhqa_task()
        self.cfg = Config()
        self.cfg.serve.served_model = "step20"

    def test_full_chain_pass_uses_exact_ids_and_endpoint_prompt(self):
        answer = "ANSWER: " + " -> ".join(self.task.meta["record"]["answer"]["chain"])
        tokenizer = FakeTokenizer()
        tokenizer.completion = "</think>\n\n" + answer
        with patch.object(mixed.requests, "post", return_value=response(self.task, answer)) as post:
            out = mixed.mhqa_rollout(self.cfg, tokenizer, self.task)
        self.assertIsInstance(out, Trajectory)
        self.assertEqual((out.reward, out.stop, out.truncated), (1.0, "answer", False))
        self.assertEqual(out.tok_ids, [11, 12, 13, 7, 99])
        self.assertEqual(out.trainable, [False, False, False, True, True])
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["messages"], mixed.endpoint_graders.messages(self.task.meta["record"]))
        self.assertEqual(sent["max_tokens"], mixed.MAX_TOKENS - 3)
        self.assertEqual(sent["model"], "step20")

    def test_wrong_chain_is_zero_and_truncation_is_zero(self):
        with patch.object(mixed.requests, "post", return_value=response(self.task, "ANSWER: wrong")):
            wrong = mixed.mhqa_rollout(self.cfg, FakeTokenizer(), self.task)
        self.assertEqual(wrong.reward, 0.0)
        with patch.object(mixed.requests, "post", return_value=response(
                self.task, "ANSWER: " + " -> ".join(self.task.meta["record"]["answer"]["chain"]),
                ids=[7], finish="length")):
            truncated = mixed.mhqa_rollout(self.cfg, FakeTokenizer(), self.task)
        self.assertEqual((truncated.reward, truncated.stop, truncated.truncated),
                         (0.0, "truncated", True))

    def test_served_content_mismatch_and_overflow_are_errors(self):
        answer = "ANSWER: " + " -> ".join(self.task.meta["record"]["answer"]["chain"])
        with patch.object(mixed.requests, "post", return_value=response(self.task, answer)):
            out = mixed.mhqa_rollout(self.cfg, FakeTokenizer(), self.task)
        self.assertIsInstance(out, str)
        self.assertIn("sampled IDs differ", out)
        with patch.object(mixed.requests, "post", return_value=response(
                self.task, "ANSWER: wrong", ids=[7] * mixed.MAX_TOKENS + [99])):
            out = mixed.mhqa_rollout(self.cfg, FakeTokenizer(), self.task)
        self.assertIsInstance(out, str)
        self.assertIn("token budget exceeded", out)

    def test_prompt_mismatch_and_request_timeout_are_errors(self):
        with patch.object(mixed.requests, "post", return_value=response(
                self.task, "ANSWER: wrong", prompt=[55])):
            out = mixed.mhqa_rollout(self.cfg, FakeTokenizer(), self.task)
        self.assertIsInstance(out, str)
        self.assertIn("fidelity", out)
        with patch.object(mixed.requests, "post", side_effect=mixed.requests.Timeout("timeout")) as post, \
                patch.object(mixed.time, "sleep"):
            out = mixed.mhqa_rollout(self.cfg, FakeTokenizer(), self.task)
        self.assertIsInstance(out, str)
        self.assertIn("request", out)
        self.assertEqual(post.call_count, 3)


class MixedPoolTests(unittest.TestCase):
    def fixture(self, directory):
        folder = Path(directory)
        config = json.loads((mixed.ROOT / ".planning/leg3-config.json").read_text())
        manifest = {"adapter_step": 20, "attempts": 8, "keep_passes": [1, 6], "families": {}}
        config["prescreen"]["pool_manifest"] = str(folder / "pool-manifest.json")
        config["prescreen"]["attestation"] = str(folder / "pool-attestation.json")
        config["prescreen"]["diagnostics_script"] = str(folder / "diagnostics.sh")
        config["prescreen"]["source_sha256"] = {}
        (folder / "diagnostics.sh").write_text("screen script fixture\n")
        config["prescreen"]["diagnostics_script_sha256"] = mixed.file_sha256(folder / "diagnostics.sh")
        for family in ("find_h", "edit", "mhqa"):
            seeds = list(range(2200, 2206)) if family != "mhqa" else list(range(7300, 7306))
            generated = ([sp["name"] for seed in seeds for sp in repo_gen.make_specs(seed, family)]
                         if family != "mhqa" else
                         [r["uid"] for seed in seeds for r in mixed.synth_multihop.records(seed, {3: 2, 4: 2})])
            kept = sorted(generated)[:16]
            per_task = {uid: {"pass": 1 if uid in kept else 8,
                              "fail": 7 if uid in kept else 0, "truncated": 0, "error": 0}
                        for uid in generated}
            stats = folder / f"{family}.stats.json"
            stats.write_text(json.dumps({"family": family, "attempts": 8,
                                         "args": {"served_model": config["prescreen"]["model"],
                                                  "mhqa_seeds": "7300-7305", "hops": "3,4",
                                                  "per_hop": 2, "temperature": 0.8},
                                         "refused_seeds": [], "refused_leak_gate": 0,
                                         "skipped_prompt_too_long": 0,
                                         "per_task": per_task}))
            (folder / f"pool-{family}.txt").write_text("\n".join(kept) + "\n")
            source = config["prescreen"]["families"][family]
            source["seeds"] = "2200-2205" if family != "mhqa" else "7300-7305"
            source["expected_tasks"] = len(generated)
            source["stats"] = str(stats)
            manifest["families"][family] = {"source_stats": str(stats),
                                             "screened_tasks": len(generated),
                                             "mixed_tasks": len(kept)}
        path = folder / "config.json"
        path.write_text(json.dumps(config))
        (folder / "pool-manifest.json").write_text(json.dumps(manifest))
        att = {"adapter_sha256": config["init_adapter"]["adapter_model_sha256"],
               "served_model": config["prescreen"]["model"],
               "temperature": config["training"]["temperature"],
               "attempts": 8,
               "manifest_sha256": mixed.file_sha256(folder / "pool-manifest.json"),
               "diagnostics_script_sha256": config["prescreen"]["diagnostics_script_sha256"],
               "source_sha256": {},
               "stats_sha256": {f: mixed.file_sha256(folder / f"{f}.stats.json")
                                for f in ("find_h", "edit", "mhqa")},
               "pool_sha256": {f: mixed.file_sha256(folder / f"pool-{f}.txt")
                               for f in ("find_h", "edit", "mhqa")}}
        (folder / "pool-attestation.json").write_text(json.dumps(att))
        return path, folder

    def test_regenerates_exact_three_family_pool_and_quotas(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(mixed, "prior_training_roots", return_value=(set(), set())), \
                patch.object(mixed, "forbidden_gym_seeds", return_value=set()), \
                patch.object(mixed, "forbidden_endpoint_roots", return_value=set()):
            config_path, _ = self.fixture(tmp)
            _, tasks, quotas = mixed.load_mixed_pool(config_path)
        self.assertEqual(quotas, {"find_h": 4, "edit": 2, "mhqa": 2})
        self.assertEqual(Counter(t.family for t in tasks), {"find_h": 16, "edit": 16, "mhqa": 16})
        self.assertTrue(all(t.meta.get("record", {}).get("answer", {}).get("chain")
                            for t in tasks if t.family == "mhqa"))

    def test_refuses_wrong_pool_id_and_training_root_overlap(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(mixed, "prior_training_roots", return_value=(set(), set())), \
                patch.object(mixed, "forbidden_gym_seeds", return_value=set()), \
                patch.object(mixed, "forbidden_endpoint_roots", return_value=set()):
            config_path, folder = self.fixture(tmp)
            path = folder / "pool-edit.txt"
            ids = path.read_text().splitlines()
            path.write_text("\n".join(ids[:-1] + [ids[0]]) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate|pool"):
                mixed.load_mixed_pool(config_path)
            path.write_text("\n".join(ids) + "\n")
            with patch.object(mixed, "prior_training_roots", return_value=({2200}, set())):
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    mixed.load_mixed_pool(config_path)

    def test_attestation_rejects_changed_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, folder = self.fixture(tmp)
            with (folder / "edit.stats.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "stats/pool hash mismatch"):
                mixed.validate_attestation(json.loads(config_path.read_text()),
                                           folder / "pool-manifest.json")

    def test_resume_source_and_ledger_provenance(self):
        config = json.loads((mixed.ROOT / ".planning/leg3-config.json").read_text())
        correct = config["init_adapter"]["adapter_model_sha256"]
        mixed.validate_resume_provenance(config, 21, correct, {}, "config", "attestation")
        with self.assertRaisesRegex(ValueError, "loaded step-20 adapter"):
            mixed.validate_resume_provenance(config, 21, "wrong", {}, "config", "attestation")
        mixed.validate_resume_provenance(config, 22, "step21",
                                         {"leg3_config_sha256": "config",
                                          "leg3_attestation_sha256": "attestation"},
                                         "config", "attestation")
        with self.assertRaisesRegex(ValueError, "different config or pool attestation"):
            mixed.validate_resume_provenance(config, 22, "step21",
                                             {"leg3_config_sha256": "old",
                                              "leg3_attestation_sha256": "attestation"},
                                             "config", "attestation")


class TrainerDispatchTests(unittest.TestCase):
    def test_workspace_task_gets_its_own_family_config(self):
        import train_grpo

        cfg = Config()
        cfg.episode.family = "find_h"
        task = Task(name="edit-example", family="edit", prompt="", tools=[],
                    ground_truth="", reward=lambda turns, ws: 0, meta={"seed": 1})
        with patch.object(train_grpo, "_rollout", return_value="sentinel") as run:
            self.assertEqual(train_grpo.rollout(cfg, None, task, 0, "s21"), "sentinel")
        self.assertEqual(run.call_args.args[0].episode.family, "edit")
        self.assertEqual(cfg.episode.family, "find_h")

    def test_mixed_load_rejects_stale_adapter_name(self):
        import train_grpo

        served = SimpleNamespace(status_code=200, json=lambda: {"data": [{"id": "leg3-s20"}]},
                                 raise_for_status=lambda: None)
        with patch.object(train_grpo.requests, "get", return_value=served), \
                patch.object(train_grpo.requests, "post") as post:
            with self.assertRaisesRegex(RuntimeError, "already served"):
                train_grpo.vllm_load("http://127.0.0.1:8000", "leg3-s20", Path("adapter"), strict=True)
        post.assert_not_called()


class AttestationBuilderTests(unittest.TestCase):
    def test_completed_screen_is_attested_and_source_drift_rejected(self):
        script = mixed.ROOT / "ops/build-leg3-attestation.py"
        spec = importlib.util.spec_from_file_location("build_leg3_attestation", script)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "obj/leg3-diagnostics"
            output.mkdir(parents=True)
            (output / "screen.done").write_text("")
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"step20")
            source = root / "source.py"
            source.write_text("original\n")
            diagnostics = root / "diagnostics.sh"
            diagnostics.write_text("script\n")
            families, manifest_families, report_lines = {}, {}, []
            for family, short in (("find_h", "findh"), ("edit", "edit"), ("mhqa", "mhqa")):
                stats_path = root / f"{family}.stats.json"
                ids = [f"{family}-{n}" for n in range(16)]
                stats_path.write_text(json.dumps({"attempts": 8,
                    "per_task": {uid: {"pass": 1, "fail": 7, "truncated": 0, "error": 0}
                                 for uid in ids},
                    "args": {"served_model": "step20", "mhqa_seeds": "7300-7305",
                             "hops": "3,4", "per_hop": 2, "temperature": 0.8}}))
                (output / f"pool-{family}.txt").write_text("\n".join(ids) + "\n")
                families[family] = {"stats": str(stats_path), "expected_tasks": 16,
                                    "seeds": "7300-7305", "hops": [3, 4], "per_hop": 2}
                manifest_families[family] = {"source_stats": str(stats_path)}
                report_lines.append(f"STATS_OK leg3-screen-step20-{short}.stats.json "
                                    "tasks=16 attempts=8 errors=0")
            (root / "REPORT-leg3-diagnostics.md").write_text("\n".join(report_lines))
            (output / "pool-manifest.json").write_text(json.dumps({
                "adapter_step": 20, "attempts": 8, "keep_passes": [1, 6],
                "families": manifest_families}))
            config = {"init_adapter": {"pod_path": str(adapter),
                                       "adapter_model_sha256": builder.digest(adapter / "adapter_model.safetensors")},
                      "training": {"temperature": 0.8},
                      "prescreen": {"attempts_per_task": 8, "keep_pass_counts_inclusive": [1, 6],
                                    "model": "step20", "families": families,
                                    "diagnostics_script": str(diagnostics),
                                    "diagnostics_script_sha256": builder.digest(diagnostics),
                                    "source_sha256": {"source.py": builder.digest(source)}}}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            att = builder.build(config_path, root, output)
            self.assertEqual(att["adapter_sha256"], config["init_adapter"]["adapter_model_sha256"])
            self.assertEqual(set(att["pool_sha256"]), {"find_h", "edit", "mhqa"})
            source.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "source drift"):
                builder.build(config_path, root, output)


if __name__ == "__main__":
    unittest.main()
