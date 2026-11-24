"""GRPO with vLLM rollouts, checkpoints and resume (the walk-away trainer).

Differences from train.py (the HF-generate smoke loop):
- Rollouts go through the served vLLM (fast, batched). The policy adapter is
  hot-loaded into vLLM after every optimizer step (/v1/load_lora_adapter), so
  rollouts are always sampled from the current weights (on-policy).
- The policy IS the SFT adapter (r3), trained in place, so every checkpoint is a
  normal LoRA vLLM can serve and eval. kl_beta defaults to 0 (the reference
  would be the raw base, which is not what we want to pull toward).
- Token stream per trajectory = the same segments SFT records use (served
  prompt ids + exact sampled ids + locally encoded observations); every episode
  must pass collect.fidelity_error or it is dropped as an error.
- A failed request is never a result: errored / unfaithful episodes are excluded
  from their group, and a group with < 2 valid rollouts is skipped.
- Reward = task.reward, forced to 0 on cap hit or truncation (gym rule).
- Every step: ckpt/step-K/{adapter, optim.pt, state.json, sha256}; one JSON line
  per step in ckpt/steps.jsonl. --resume continues from the latest step-K.
- Every rollout (errors included) -> one line in ckpt/rollouts.jsonl: step, seed,
  task, family, full transcript, reward, stop reason, rollout adapter.
- Early stop without a kill: after step K is fully written, if the flag file
  (--stop-file, default <ckpt>/../STOP_AFTER) exists and holds N <= K, or is empty,
  log "[stop] STOP_AFTER hit at step K", rename it to STOP_AFTER.hit-stepK, exit 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import requests
import torch
from peft import PeftModel
from transformers import AutoTokenizer

import agent
import compact
import encoding
import repo_gen
import vram
from collect import _segments, fidelity_error
from config import Config
import gym_budget as GB
from eval import make_task
from grpo_sampling import DynamicGroupSampler
from grpo_mixed import file_sha256, load_mixed_pool, mhqa_rollout, validate_resume_provenance
from rl import grpo_backward
from runner import BridgeError, Runner
from schema import Trajectory
from train import _model_class
from workspace import Workspace


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def vllm_load(base: str, name: str, path: Path, strict: bool = False) -> None:
    if strict:
        models = requests.get(f"{base}/v1/models", timeout=30)
        models.raise_for_status()
        if any(model["id"] == name for model in models.json()["data"]):
            raise RuntimeError(f"adapter name {name} is already served; cannot prove its weights")
    r = requests.post(f"{base}/v1/load_lora_adapter",
                      json={"lora_name": name, "lora_path": str(path)}, timeout=300)
    if r.status_code != 200 and (strict or "already" not in r.text):
        raise RuntimeError(f"load_lora_adapter {name}: {r.status_code} {r.text[:300]}")
    if strict:
        models = requests.get(f"{base}/v1/models", timeout=30)
        models.raise_for_status()
        if not any(model["id"] == name for model in models.json()["data"]):
            raise RuntimeError(f"adapter {name} absent from vLLM model list after load")


def vllm_unload(base: str, name: str) -> None:
    requests.post(f"{base}/v1/unload_lora_adapter", json={"lora_name": name}, timeout=60)


def rollout(cfg: Config, tok, task, k: int, tag: str) -> Trajectory | None | str:
    """One episode on vLLM -> Trajectory, or an error string (excluded). A dead reward
    bridge (Runner 'remote') is an error, never a 0 reward."""
    if task.family == "mhqa":
        return mhqa_rollout(cfg, tok, task)
    task_cfg = replace(cfg, episode=replace(cfg.episode, family=task.family))
    try:
        return _rollout(task_cfg, tok, task, k, tag)
    except BridgeError as e:
        return f"bridge: {e}"


def _rollout(cfg: Config, tok, task, k: int, tag: str) -> Trajectory | None | str:
    ws = Workspace(cfg.episode.data_root / "workspaces" / f"grpo-{tag}-{task.name}-a{k}",
                   Runner(cfg.episode.runner))
    ws.reset()
    ws.builds_left = cfg.episode.max_builds
    repo_gen.write_files(repo_gen.find_spec(task.meta["seed"], task.family,
                                            task.name)["files"], ws.root)
    ep = agent.run_episode(
        cfg, task, ws, return_ids=True,
        prompt_len_fn=lambda m: len(encoding.render_chat(tok, m, cfg.episode.thinking)),
        tokenizer=tok)
    if ep.error:
        return f"request: {ep.error}"
    if not ep.turns:
        return "no turns"
    if cfg.episode.compact:
        return _compact_trajectory(cfg, tok, task, ep, ws)
    f = fidelity_error(ep, tok, cfg.episode.thinking, agent.counter_turns(cfg), cfg.episode.hint)
    if f:
        return f"fidelity: {f}"
    ids, trainable = [], []
    for s in _segments(ep, tok, cfg.episode.thinking, agent.counter_turns(cfg)):
        ids += s["ids"]
        trainable += [s["train"]] * len(s["ids"])
    reward = float(task.reward(ep.turns, ws))
    if ep.cap_hit or ep.truncated:
        reward = 0.0
    tr = Trajectory(task=task, turns=ep.turns, tok_ids=ids, trainable=trainable,
                    reward=reward, done=ep.stop == "answer", truncated=ep.truncated, stop=ep.stop)
    return tr


def _compact_trajectory(cfg: Config, tok, task, ep, ws) -> Trajectory | str:
    """Compacted episode -> Trajectory whose windows (context windows + harness note
    calls) are each a token-exact sequence; one reward for all of them."""
    f = compact.fidelity_error(ep, tok, cfg.episode.thinking, agent.counter_turns(cfg))
    if f:
        return f"fidelity: {f}"
    seqs = compact.sequences(ep, tok, cfg.episode.thinking, agent.counter_turns(cfg))
    windows = []
    for sq in seqs:
        ids, trainable = [], []
        for s in sq["segments"]:
            ids += s["ids"]
            trainable += [s["train"]] * len(s["ids"])
        windows.append((ids, trainable))
    reward = float(task.reward(ep.turns, ws))
    if ep.cap_hit or ep.truncated or ep.compact_fail:
        reward = 0.0
    tr = Trajectory(task=task, turns=ep.turns, tok_ids=windows[0][0],
                    trainable=windows[0][1], reward=reward, done=ep.stop == "answer",
                    truncated=ep.truncated, stop=ep.stop, windows=windows)
    tr.compaction = compact.summary(ep)       # every reset, for rollouts.jsonl
    return tr


def rollout_record(step: int, adapter: str, task, k: int, out,
                   sample_round: int | None = None) -> dict:
    """One rollouts.jsonl row. `out` = Trajectory or error string (reward None)."""
    rec = {"step": step, "adapter": adapter, "seed": task.meta.get("seed"),
           "root_id": task.meta.get("root_id", task.meta.get("seed")),
           "task": task.name, "family": task.family, "attempt": k}
    if sample_round is not None:
        rec["sample_round"] = sample_round
    if isinstance(out, str):
        return {**rec, "reward": None, "stop": "error", "error": out, "messages": []}
    if getattr(out, "compaction", None) is not None:
        rec["compaction"] = out.compaction
    return {**rec, "reward": out.reward, "stop": out.stop,
            "truncated": out.truncated, "prompt": task.prompt,
            "messages": [{"content": t.content, "reasoning": t.reasoning,
                          "action": ({"name": t.action.name, "arguments": t.action.arguments}
                                     if t.action else None),
                          "obs": t.obs, "truncated": t.truncated} for t in out.turns]}


def stop_requested(flag: Path, step: int) -> bool:
    """True if the STOP_AFTER flag says to stop now that `step` is fully written.
    Absent -> False. Empty -> True (stop after the current step). Integer N -> step >= N.
    Unparseable -> True (the intent to stop is clear; the number is not)."""
    if not flag.is_file():
        return False
    txt = flag.read_text(encoding="utf-8", errors="replace").strip()
    if not txt:
        return True
    try:
        return step >= int(txt)
    except ValueError:
        print(f"[stop] {flag} content {txt[:40]!r} is not an integer; stopping now", flush=True)
        return True


def consume_stop(flag: Path, step: int) -> None:
    """Rename (never delete) the flag so a later relaunch/resume does not stop at once."""
    try:
        flag.rename(flag.with_name(f"{flag.name}.hit-step{step}"))
    except OSError as e:
        print(f"[stop] could not rename {flag}: {e}", flush=True)


def latest_ckpt(root: Path) -> Path | None:
    ks = sorted((int(p.name.split("-")[1]), p) for p in root.glob("step-*")
                if (p / "state.json").exists())
    return ks[-1][1] if ks else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="find_h")
    ap.add_argument("--seeds", help="single-family seed range")
    ap.add_argument("--pool", default=None, help="task names to keep (prescreen output)")
    ap.add_argument("--mixed-config", help="audited three-family leg-3 config JSON")
    ap.add_argument("--init-adapter", required=True, help="SFT LoRA the policy starts from")
    ap.add_argument("--ckpt", required=True, help="checkpoint root (step-K dirs)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, required=True, help="total optimizer steps")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--task-batch", type=int, default=8)
    ap.add_argument("--dynamic-sampling", action="store_true",
                    help="oversample until task-batch full-variance groups are available")
    ap.add_argument("--oversample-factor", type=float, default=2.0,
                    help="candidate groups per still-needed live group (dynamic mode)")
    ap.add_argument("--max-candidates", type=int, default=64,
                    help="hard per-step task-group budget; fail without checkpoint if exhausted")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--kl-beta", type=float, default=0.0)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--runner", default="docker")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--tag", default="run", help="vLLM adapter name prefix")
    ap.add_argument("--compact", action="store_true",
                    help="in-episode compaction (75%% trigger + checkpoint tool); default off")
    ap.add_argument("--compact-note-mode", choices=["off", "force"], default="off")
    ap.add_argument("--max-resets", type=int, default=None)
    ap.add_argument("--stop-file", default=None,
                    help="flag file: empty = stop after current step, N = stop after step N "
                         "(default <ckpt>/../STOP_AFTER)")
    ap.add_argument("--budget", type=int, default=None,
                    help="episode token budget (default 8192 / env GYM_BUDGET); read by gym_budget at import")
    a = ap.parse_args()
    mixed, quotas = None, {a.family: a.task_batch}
    mixed_config_sha = mixed_attestation_sha = None
    if a.mixed_config:
        if a.seeds or a.pool or not a.resume:
            ap.error("mixed mode needs --resume and its config pool; omit --seeds and --pool")
        draft = json.loads(Path(a.mixed_config).read_text(encoding="utf-8"))
        if draft["status"] != "APPROVED_TO_RUN":
            ap.error("mixed config is a draft; wait for the user's leg-3 go")
        mixed, tasks, quotas = load_mixed_pool(Path(a.mixed_config))
        mixed_config_sha = file_sha256(Path(a.mixed_config))
        mixed_attestation_sha = file_sha256(Path(mixed["prescreen"]["attestation"]))
        spec = mixed["training"]
        if (a.steps != spec["checkpoint_steps"][1] or
                a.group_size != spec["group_size"] or
                a.task_batch != sum(quotas.values())):
            ap.error("mixed CLI step/group/batch values differ from leg-3 config")
        a.dynamic_sampling = True
        a.oversample_factor = spec["dynamic_sampling"]["oversample_factor"]
        a.max_candidates = spec["dynamic_sampling"]["max_candidate_groups_per_step"]
        a.lr, a.kl_beta = spec["learning_rate"], spec["kl_beta"]
    elif not a.seeds:
        ap.error("--seeds is required without --mixed-config")
    if a.dynamic_sampling and (a.group_size < 2 or a.oversample_factor < 1
                               or a.max_candidates < a.task_batch):
        ap.error("dynamic sampling needs group-size >=2, oversample-factor >=1, "
                 "and max-candidates >= task-batch")

    cfg = Config()
    cfg.episode.family, cfg.episode.runner = a.family, a.runner
    cfg.serve.base = a.base
    cfg.serve.temperature, cfg.serve.top_p = cfg.train.temperature, cfg.train.top_p
    cfg.train.kl_beta = a.kl_beta
    cfg.episode.compact = a.compact
    cfg.episode.compact_note_mode, cfg.episode.max_resets = a.compact_note_mode, a.max_resets
    cfg.train.group_size, cfg.train.task_batch = a.group_size, a.task_batch
    if mixed:
        cfg.serve.temperature = mixed["training"]["temperature"]
        cfg.episode.max_turns = mixed["training"]["workspace_turn_cap"]
        cfg.episode.budget = (GB.BUDGET if GB.OVERRIDDEN
                              else mixed["training"]["workspace_token_budget"])
    if a.lr:
        cfg.train.lr = a.lr
    GB.check_server(cfg.serve.base, cfg.episode.budget, cfg.serve.api_key)
    root = Path(a.ckpt)
    if mixed and not root.is_dir():
        ap.error("mixed mode needs the existing step-20 checkpoint directory")
    root.mkdir(parents=True, exist_ok=True)
    stop_file = Path(a.stop_file) if a.stop_file else root.resolve().parent / "STOP_AFTER"

    if mixed:
        if sha256(Path(a.init_adapter) / "adapter_model.safetensors") != mixed["init_adapter"]["adapter_model_sha256"]:
            ap.error("mixed init adapter does not match the pre-registered step-20 hash")
    else:
        seeds = []
        for part in a.seeds.split(","):
            lo, _, hi = part.partition("-")
            seeds += list(range(int(lo), int(hi or lo) + 1))
        tasks = [make_task(sp) for s in seeds for sp in repo_gen.make_specs(s, a.family)]
        if a.pool:
            keep = {ln.strip() for ln in open(a.pool, encoding="utf-8") if ln.strip()}
            tasks = [t for t in tasks if t.name in keep]
    if len(tasks) < a.task_batch:
        sys.exit(f"[grpo] only {len(tasks)} tasks < task_batch {a.task_batch}")

    start, prev = 1, None
    src = Path(a.init_adapter)
    if a.resume and (prev := latest_ckpt(root)):
        src = prev / "adapter"
        start = json.loads((prev / "state.json").read_text())["step"] + 1
    if mixed and (prev is None or not mixed["training"]["checkpoint_steps"][0] <= start <= a.steps):
        ap.error("mixed mode must resume from a completed step 20-29 checkpoint")
    if mixed:
        rows = {row["step"]: row for row in
                map(json.loads, (root / "steps.jsonl").read_text(encoding="utf-8").splitlines())}
        source_sha = sha256(src / "adapter_model.safetensors")
        if rows.get(start - 1, {}).get("sha256") != source_sha:
            ap.error("latest checkpoint adapter hash differs from steps.jsonl")
        try:
            validate_resume_provenance(mixed, start, source_sha, rows[start - 1],
                                       mixed_config_sha, mixed_attestation_sha)
        except ValueError as exc:
            ap.error(str(exc))
    tok = AutoTokenizer.from_pretrained(cfg.serve.model_name)
    base = _model_class(cfg.serve.model_name).from_pretrained(
        cfg.serve.model_name, torch_dtype=cfg.train.torch_dtype, low_cpu_mem_usage=True)
    policy = PeftModel.from_pretrained(base, str(src), is_trainable=True)
    policy = policy.to(cfg.train.resolved_device)
    if cfg.train.grad_checkpointing:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optim = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=cfg.train.lr)
    if prev is not None:
        optim.load_state_dict(torch.load(prev / "optim.pt", map_location="cpu"))
        rng = json.loads((prev / "state.json").read_text())["rng"]
        random.setstate((rng[0], tuple(rng[1]), rng[2]))
    else:
        random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed + start)
    served = f"{a.tag}-s{start - 1}" if prev else f"{a.tag}-init"
    vllm_load(a.base, served, src, strict=bool(mixed))
    cfg.serve.served_model = served
    print(f"[grpo] {len(tasks)} tasks | start step {start}/{a.steps} | from {src} "
          f"sha256 {sha256(src / 'adapter_model.safetensors')} | group {a.group_size} x "
          f"batch {a.task_batch} | lr {cfg.train.lr} kl {cfg.train.kl_beta} "
          f"T {cfg.serve.temperature} | dynamic {a.dynamic_sampling} "
          f"oversample {a.oversample_factor} max-candidates {a.max_candidates} "
          f"| stop-file {stop_file}", flush=True)

    vram.install()
    for step in range(start, a.steps + 1):
        t0 = time.time()
        def run_candidates(batch, sample_round=None):
            jobs = [(t, k) for t in batch for k in range(a.group_size)]
            started = time.time()
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                outs = list(ex.map(lambda j: rollout(cfg, tok, j[0], j[1], f"s{step}"), jobs))
            elapsed = time.time() - started
            with (root / "rollouts.jsonl").open("a", encoding="utf-8") as f:
                for (t, k), o in zip(jobs, outs):
                    f.write(json.dumps(rollout_record(step, served, t, k, o, sample_round),
                                       ensure_ascii=False) + "\n")
            return outs, elapsed

        errors, rewards = [], []
        n_dead = n_none = 0
        extra = {}
        if a.dynamic_sampling:
            sampler = DynamicGroupSampler(tasks, quotas, a.group_size,
                                          a.oversample_factor, a.max_candidates, random)
            t_roll = 0.0
            while not sampler.ready:
                batch = sampler.next_candidates()
                if not batch:
                    raise RuntimeError(
                        f"[dynamic] step {step}: only {len(sampler.live)}/{a.task_batch} "
                        f"live groups after {sampler.candidate_groups} candidates; "
                        f"dead={sampler.dead_groups} error_groups={sampler.error_groups}; "
                        "no optimizer step or checkpoint written")
                outs, elapsed = run_candidates(batch, sampler.rounds)
                t_roll += elapsed
                for i, t in enumerate(batch):
                    group_outs = outs[i * a.group_size:(i + 1) * a.group_size]
                    errors.extend(f"{t.family}:{t.name}: {o}" for o in group_outs
                                  if isinstance(o, str))
                    status = sampler.observe(t, group_outs)
                    valid = [o for o in group_outs if not isinstance(o, str)]
                    passes = sum(o.reward > 0 for o in valid)
                    print(f"[group] step {step} round {sampler.rounds} "
                          f"{t.family}:{t.name}: pass {passes}/{len(valid)} {status}", flush=True)
            group_items = [(f"{t.family}:{t.name}", group)
                           for t, group in sampler.selected_groups()]
            rewards = sampler.rewards
            n_dead, n_none = sampler.dead_groups, sampler.zero_pass_groups
            candidate_groups = sampler.candidate_groups
            extra = {"dynamic_sampling": True, "used_groups": len(group_items),
                     "used_by_family": {family: sum(t.family == family for t, _ in sampler.selected_groups())
                                        for family in quotas},
                     "resampled_groups": sampler.resampled_groups,
                     "resample_rounds": sampler.rounds - 1,
                     "error_groups": sampler.error_groups,
                     "unused_live_groups": sampler.unused_live_groups,
                     "zero_pass_groups": sampler.zero_pass_groups,
                     "all_pass_groups": sampler.all_pass_groups}
        else:
            batch = random.sample(tasks, a.task_batch)
            outs, t_roll = run_candidates(batch)
            groups = {}
            for (t, _), o in zip(((t, k) for t in batch for k in range(a.group_size)), outs):
                if isinstance(o, str):
                    errors.append(f"{t.name}: {o}")
                else:
                    groups.setdefault(t.name, []).append(o)
            group_items = list(groups.items())
            candidate_groups = len(groups)

        policy.train()
        optim.zero_grad()
        loss, n_used = 0.0, 0
        for name, g in group_items:
            if not a.dynamic_sampling:
                r = [x.reward for x in g]
                rewards += r
                passes = sum(x > 0 for x in r)
                n_none += passes == 0
                print(f"[group] step {step} {name}: pass {passes}/{len(g)}", flush=True)
                if len(g) < 2 or len(set(r)) == 1:
                    n_dead += 1
                    continue
            n_used += 1
            loss += grpo_backward(policy, None, g, cfg, scale=1.0 / a.task_batch)
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0) if n_used else torch.tensor(0.0)
        if n_used:
            optim.step()
        ck = root / f"step-{step}"
        policy.save_pretrained(str(ck / "adapter"))
        torch.save(optim.state_dict(), ck / "optim.pt")
        rs = random.getstate()
        digest = sha256(ck / "adapter" / "adapter_model.safetensors")
        row = {"step": step, "rollout_adapter": served, "temperature": cfg.serve.temperature,
               "time": round(time.time() - t0), "rollout_s": round(t_roll),
               "groups": candidate_groups, "dead": n_dead, "solve_none": n_none,
               "dead_pct": round(n_dead / max(1, candidate_groups), 3),
               "solve_none_pct": round(n_none / max(1, candidate_groups), 3),
               "mean_reward": round(sum(rewards) / max(1, len(rewards)), 4),
               "errors": len(errors), "loss": round(loss, 5), "grad_norm": round(float(grad), 4),
               "sha256": digest, "vram": vram.step_report(), **extra}
        if mixed:
            row.update(leg3_config_sha256=mixed_config_sha,
                       leg3_attestation_sha256=mixed_attestation_sha)
        (ck / "state.json").write_text(json.dumps(
            {"step": step, "rng": [rs[0], list(rs[1]), rs[2]], **row}))
        with (root / "steps.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if errors:
            (ck / "errors.txt").write_text("\n".join(errors))
        print(f"[step] {json.dumps(row)}", flush=True)
        if stop_requested(stop_file, step):   # ckpt + steps.jsonl + rollouts.jsonl all written
            print(f"[stop] STOP_AFTER hit at step {step}", flush=True)
            consume_stop(stop_file, step)
            sys.exit(0)
        new = f"{a.tag}-s{step}"
        vllm_load(a.base, new, ck / "adapter", strict=bool(mixed))
        if served != new:
            vllm_unload(a.base, served)
        served = cfg.serve.served_model = new
    print("[grpo] DONE", flush=True)


if __name__ == "__main__":
    main()
