"""Audited leg-3 pool loading and token-exact single-turn MHQA rollouts.

The pool files contain task IDs, not prompts. Regenerate the two task sources and
refuse a mismatched screen or an eval/holdout root before any training begins.
"""

from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path

import requests

import encoding
import repo_gen
import seal as gym_seal
from eval import make_task
from schema import Task, Trajectory, Turn

# The trainer is normally launched from sandbox-gym/. Import endpoint modules by
# their package path: bare `graders` would resolve to the workspace gym grader.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evals import graders as endpoint_graders  # noqa: E402
from evals.datasets import synth_multihop  # noqa: E402
from evals.holdout import seal as endpoint_seal  # noqa: E402
from gym_budget import ENDPOINT_MAX_TOKENS as MAX_TOKENS  # noqa: E402  (4096 unless budget overridden)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_endpoint_seal(directory: Path) -> tuple[dict, list[dict], set[str]]:
    """Check the same hashes and ledger binding as evals/holdout/seal.py verify."""
    manifest, body, roots_bytes = endpoint_seal.load_seal(directory)
    items = [json.loads(line) for line in body.splitlines() if line.strip()]
    root_hashes = endpoint_seal.root_hashes(items, manifest["root_field"])
    ledger, chain_ok = endpoint_seal.ledger_entries()
    manifest_sha = endpoint_seal.sha(endpoint_seal.canon(manifest))
    checks = (
        endpoint_seal.sha(body) == manifest["holdout_sha256"],
        endpoint_seal.sha(roots_bytes) == manifest["roots_sha256"],
        root_hashes == manifest["root_hashes"],
        endpoint_seal.merkle(root_hashes.values()) == manifest["merkle_root"],
        chain_ok,
        any(row.get("event") == "seal" and row.get("manifest_sha256") == manifest_sha
            for row in ledger),
    )
    roots = set(json.loads(roots_bytes)["root_ids"])
    if not all(checks) or roots != set(root_hashes):
        raise ValueError(f"invalid endpoint holdout seal: {directory}")
    return manifest, items, roots


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        lo, sep, hi = part.strip().partition("-")
        seeds.extend(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seed in {spec}")
    return seeds


def forbidden_endpoint_roots() -> set[str]:
    """Use the same root sources and canonicalizer as sft.collect's leak gate."""
    eval_set = json.loads((ROOT / "evals/data/sets/post_cutoff_v1_5pb.json").read_text())
    roots = {endpoint_seal.root_of(item, "root_id")
             for items in eval_set["items"].values() for item in items}
    sealed = ROOT / "evals/holdout/sealed"
    for directory in sealed.iterdir() if sealed.exists() else []:
        _, _, sealed_roots = verify_endpoint_seal(directory)
        roots.update(sealed_roots)
    roots.discard(None)
    return roots


def forbidden_gym_seeds() -> set[int]:
    holdout = Path(__file__).resolve().parent / "data/holdout/gym_v1.json"
    if not holdout.exists():
        raise ValueError(f"missing gym holdout: {holdout}")
    ok, manifest = gym_seal.verify(holdout)
    if not ok:
        raise ValueError(f"invalid gym holdout seal: {holdout}")
    return set(manifest["seeds"])


def prior_training_roots() -> tuple[set[int], set[str]]:
    """Reject a prescreen task previously used as a kept SFT training example."""
    gym, endpoint = set(), set()
    for directory in (ROOT / "sandbox-gym/data/collect", ROOT / "sft/data"):
        for path in directory.glob("*.jsonl") if directory.exists() else []:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != "train":
                    continue
                root = row.get("root_id")
                if root is None:
                    raise ValueError(f"training row without root_id: {path}")
                if isinstance(root, int) or (isinstance(root, str) and root.isdigit()):
                    gym.add(int(root))
                else:
                    endpoint.add(endpoint_seal.root_of(row, "root_id"))
    return gym, endpoint


def _screen(path: Path, family: str, expected: int, attempts: int) -> dict:
    stats = json.loads(path.read_text(encoding="utf-8"))
    per_task = stats["per_task"]
    if stats["attempts"] != attempts or len(per_task) != expected:
        raise ValueError(f"{family}: screen size/attempt count mismatch")
    if stats.get("refused_seeds") or stats.get("refused_leak_gate") or stats.get("skipped_prompt_too_long"):
        raise ValueError(f"{family}: screen refused or skipped tasks")
    if family != "mhqa" and stats["family"] != family:
        raise ValueError(f"{family}: screen family mismatch")
    for uid, row in per_task.items():
        if row["error"] or sum(row[k] for k in ("pass", "fail", "truncated")) != attempts:
            raise ValueError(f"{family}: incomplete or errored screen group {uid}")
    return per_task


def validate_attestation(config: dict, manifest_path: Path) -> str:
    """Bind pool bytes, screening code, and model/sampling identity."""
    spec = config["prescreen"]
    attestation_path = Path(spec["attestation"])
    att = json.loads(attestation_path.read_text(encoding="utf-8"))
    if (att["adapter_sha256"] != config["init_adapter"]["adapter_model_sha256"] or
            att["served_model"] != spec["model"] or
            att["temperature"] != config["training"]["temperature"] or
            att["attempts"] != spec["attempts_per_task"] or
            att["manifest_sha256"] != file_sha256(manifest_path)):
        raise ValueError("screen attestation model/sampling/manifest mismatch")
    script = Path(spec["diagnostics_script"])
    if (att["diagnostics_script_sha256"] != spec["diagnostics_script_sha256"] or
            file_sha256(script) != att["diagnostics_script_sha256"]):
        raise ValueError("screen diagnostic script hash mismatch")
    for relative, expected in spec["source_sha256"].items():
        if att["source_sha256"].get(relative) != expected or file_sha256(ROOT / relative) != expected:
            raise ValueError(f"screen source drift: {relative}")
    if set(att["source_sha256"]) != set(spec["source_sha256"]):
        raise ValueError("screen source list mismatch")
    for family, source in spec["families"].items():
        stats = Path(source["stats"])
        pool = manifest_path.parent / f"pool-{family}.txt"
        if (att["stats_sha256"].get(family) != file_sha256(stats) or
                att["pool_sha256"].get(family) != file_sha256(pool)):
            raise ValueError(f"{family}: screen stats/pool hash mismatch")
    return file_sha256(attestation_path)


def validate_resume_provenance(config: dict, step: int, source_sha: str,
                               previous_row: dict, config_sha: str, attestation_sha: str) -> None:
    """Keep the first and resumed leg-3 steps tied to the approved sources."""
    first = config["training"]["checkpoint_steps"][0]
    if step == first:
        if source_sha != config["init_adapter"]["adapter_model_sha256"]:
            raise ValueError("loaded step-20 adapter differs from pre-registered hash")
    elif (previous_row.get("leg3_config_sha256") != config_sha or
          previous_row.get("leg3_attestation_sha256") != attestation_sha):
        raise ValueError("resumed leg-3 checkpoint has different config or pool attestation")


def validate_future_gate(config: dict) -> set[str]:
    """Require the pre-registered 240-item MHQA fixture to be sealed first."""
    spec = config["future_mhqa_gate_preregistered_not_run"]
    directory = Path(spec["sealed_dir"])
    expected_hash = spec["fixture_manifest_sha256"]
    if not expected_hash or not (directory / "manifest.json").is_file() or \
            file_sha256(directory / "manifest.json") != expected_hash:
        raise ValueError("future MHQA gate fixture is missing or has a different manifest hash")
    manifest, items, roots = verify_endpoint_seal(directory)
    if manifest["n_items"] != spec["distinct_items"] or len(items) != spec["distinct_items"]:
        raise ValueError("future MHQA gate fixture item count differs from plan")
    expected = {}
    plan = {int(h): spec["per_hop"] for h in spec["hops"]}
    for seed in parse_seeds(spec["seeds"]):
        expected.update((record["uid"], record) for record in synth_multihop.records(seed, plan))
    actual = {record["uid"]: record for record in items}
    if len(actual) != len(items) or set(actual) != set(expected):
        raise ValueError("future MHQA gate fixture IDs differ from pre-registered seeds")
    for uid, record in actual.items():
        if any(record.get(key) != expected[uid][key]
               for key in ("root_id", "prompt", "answer", "hops")):
            raise ValueError(f"future MHQA gate task drift: {uid}")
    return roots


def load_mixed_pool(config_path: Path, manifest_path: Path | None = None) -> tuple[dict, list[Task], dict[str, int]]:
    """Return config, regenerated tasks, and quotas after strict pool validation."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quotas = config["training"]["family_live_quotas"]
    if set(quotas) != {"find_h", "edit", "mhqa"} or any(n < 1 for n in quotas.values()):
        raise ValueError("leg-3 quotas must cover find_h, edit, and mhqa")
    if sum(quotas.values()) != config["training"]["live_groups_per_step"]:
        raise ValueError("family quotas do not add to live group count")
    manifest_path = manifest_path or Path(config["prescreen"]["pool_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_attestation(config, manifest_path)
    attempts = config["prescreen"]["attempts_per_task"]
    if manifest["adapter_step"] != 20 or manifest["attempts"] != attempts:
        raise ValueError("pool manifest adapter/attempt mismatch")
    if manifest["keep_passes"] != config["prescreen"]["keep_pass_counts_inclusive"]:
        raise ValueError("pool manifest pass filter mismatch")
    low, high = manifest["keep_passes"]
    if not (0 < low <= high < attempts):
        raise ValueError("pool filter must exclude 0/N and N/N")

    prior_gym, prior_endpoint = prior_training_roots()
    gym_bad = forbidden_gym_seeds() | prior_gym
    endpoint_bad = forbidden_endpoint_roots() | prior_endpoint
    diagnostic_seeds = set(parse_seeds(config["diagnostics_unsealed"]["mhqa_seeds"]))
    gate_seeds = set(parse_seeds(config["future_mhqa_gate_preregistered_not_run"]["seeds"]))
    tasks: list[Task] = []
    for family, quota in quotas.items():
        source = config["prescreen"]["families"][family]
        stat_path = Path(source["stats"])
        recorded = manifest["families"][family]
        if recorded["source_stats"] != str(stat_path):
            raise ValueError(f"{family}: manifest source differs from config")
        if family == "mhqa":
            stats_args = json.loads(stat_path.read_text(encoding="utf-8")).get("args", {})
            if (stats_args.get("served_model") != config["prescreen"]["model"] or
                    stats_args.get("mhqa_seeds") != source["seeds"] or
                    stats_args.get("hops") != ",".join(map(str, source["hops"])) or
                    stats_args.get("per_hop") != source["per_hop"] or
                    stats_args.get("temperature") != config["training"]["temperature"]):
                raise ValueError("MHQA screen collector arguments differ from config")
        per_task = _screen(stat_path, family, source["expected_tasks"], attempts)
        pool_file = manifest_path.parent / f"pool-{family}.txt"
        ids = pool_file.read_text(encoding="utf-8").splitlines()
        if len(ids) != len(set(ids)) or len(ids) != recorded["mixed_tasks"] or len(ids) < max(16, quota):
            raise ValueError(f"{family}: missing, duplicate, or undersized pool")
        eligible = {uid for uid, row in per_task.items() if low <= row["pass"] <= high}
        if set(ids) != eligible:
            raise ValueError(f"{family}: pool IDs differ from screened 1-6/N groups")
        if recorded["screened_tasks"] != len(per_task):
            raise ValueError(f"{family}: manifest task count mismatch")
        if "histogram" in recorded:
            histogram = {str(n): sum(row["pass"] == n for row in per_task.values())
                         for n in range(attempts + 1)}
            if recorded["histogram"] != histogram:
                raise ValueError(f"{family}: manifest histogram differs from screen")
        generated = []
        seeds = parse_seeds(source["seeds"])
        if family == "mhqa":
            if set(seeds) & (diagnostic_seeds | gate_seeds | {synth_multihop.SEED}):
                raise ValueError("MHQA training seeds overlap diagnostic, gate, or eval")
            plan = {int(h): source["per_hop"] for h in source["hops"]}
            for seed in seeds:
                for record in synth_multihop.records(seed, plan):
                    root = endpoint_seal.root_of(record, "root_id")
                    if not root or root in endpoint_bad:
                        raise ValueError(f"sealed/eval MHQA root: {record['uid']}")
                    generated.append(Task(name=record["uid"], family="mhqa",
                                          prompt=record["prompt"], tools=[], ground_truth="",
                                          reward=lambda turns, ws: 0.0,
                                          meta={"seed": seed, "root_id": record["root_id"],
                                                "record": record}))
        else:
            diagnostic_find = set(parse_seeds(config["diagnostics_unsealed"]["find_h_seeds"]))
            if set(seeds) & (gym_bad | diagnostic_find):
                raise ValueError(f"{family}: training seed overlaps holdout, SFT, or diagnostics")
            for seed in seeds:
                generated.extend(make_task(spec) for spec in repo_gen.make_specs(seed, family))
        lookup = {task.name: task for task in generated}
        if len(lookup) != source["expected_tasks"] or set(lookup) != set(per_task):
            raise ValueError(f"{family}: regenerated tasks differ from screened tasks")
        tasks.extend(lookup[uid] for uid in ids)
    if len({(t.family, t.name) for t in tasks}) != len(tasks):
        raise ValueError("duplicate family/task in mixed pool")
    if config["status"] == "APPROVED_TO_RUN":
        gate_roots = validate_future_gate(config)
        if any(endpoint_seal.root_of(t.meta, "root_id") in gate_roots
               for t in tasks if t.family == "mhqa"):
            raise ValueError("training MHQA root overlaps future gate")
    return config, tasks, quotas


def mhqa_rollout(cfg, tokenizer, task: Task) -> Trajectory | str:
    """One endpoint answer, graded exactly as sft.collect; errors stay errors."""
    record = task.meta["record"]
    messages = endpoint_graders.messages(record)
    prompt_ids = encoding.render_chat(tokenizer, messages, True)
    room = MAX_TOKENS - len(prompt_ids)
    if room < 256:
        return f"prompt-too-long: {task.name} room={room}"
    body = {"model": cfg.serve.served_model or cfg.serve.model_name,
            "messages": messages, "max_tokens": room,
            "temperature": cfg.serve.temperature, "top_p": cfg.serve.top_p,
            "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": True}}
    headers = {"Authorization": f"Bearer {cfg.serve.api_key}"} if cfg.serve.api_key else None
    for attempt in range(3):
        try:
            response = requests.post(f"{cfg.serve.base.rstrip('/')}/v1/chat/completions",
                                     json=body, headers=headers, timeout=3600)
            response.raise_for_status()
            data = response.json()
            break
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                return f"request: {type(exc).__name__}: {str(exc)[:120]}"
            time.sleep(10)
    if not isinstance(data, dict):
        return f"response: {task.name} vLLM body is not an object"
    if data.get("prompt_token_ids") != prompt_ids:
        return f"fidelity: {task.name} vLLM prompt IDs differ from local template"
    try:
        choice = data["choices"][0]
        completion_ids = choice["token_ids"]
        if not completion_ids or not all(isinstance(i, int) for i in completion_ids):
            return f"fidelity: {task.name} missing sampled token IDs"
        if len(prompt_ids) + len(completion_ids) > MAX_TOKENS:
            return f"fidelity: {task.name} token budget exceeded"
        eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
        stopped = choice["finish_reason"] == "stop" and completion_ids[-1] == eos
        decoded = tokenizer.decode(completion_ids, skip_special_tokens=False)
        content = encoding.response_section(
            tokenizer.decode(completion_ids[:-1], skip_special_tokens=False), True)
        served_content = choice["message"].get("content") or ""
        if stopped and content.strip() != served_content.strip():
            return f"fidelity: {task.name} sampled IDs differ from served content"
        passed, _ = endpoint_graders.grade(record, content) if stopped else (False, "truncated")
        turn = Turn(content=decoded, resp_text=content,
                    reasoning=choice["message"].get("reasoning_content") or "",
                    gen_ids=completion_ids, truncated=not stopped)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return f"response: {task.name}: {type(exc).__name__}: {str(exc)[:120]}"
    return Trajectory(task=task, turns=[turn], tok_ids=prompt_ids + completion_ids,
                      trainable=[False] * len(prompt_ids) + [True] * len(completion_ids),
                      reward=float(passed), done=stopped, truncated=not stopped,
                      stop="answer" if stopped else "truncated")
