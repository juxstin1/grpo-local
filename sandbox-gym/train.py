"""GRPO training loop for the sandbox gym (ported from grpo-local train.py).

Each rollout samples a FRESH workspace per trajectory (edits can't leak across
rolls), is rewarded by the family's mechanical grader, and its token-exact
stream is scored with the GRPO loss. sampler == trainer: no optimizer step
until a whole group is sampled and scored, so recomputed logprobs are
bit-identical to sample time.

Run:
  .\\cli.py train --family find --seeds 1000-1009 --steps 50 --group-size 6

The VRAM guard refuses to start while vLLM or a stray process holds the GPU.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import cast

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)

import encoding
import repo_gen
import tools as T
import vram
from config import Config
from gyms import spec_to_task, system_prompt
from rl import grpo_backward
from rollout import sample_trajectory
from schema import Task

try:  # Qwen3.5 needs transformers >= 5.x
    from transformers import Qwen3_5ForConditionalGeneration  # type: ignore[attr-defined]
except ImportError:
    Qwen3_5ForConditionalGeneration = AutoModelForCausalLM  # type: ignore[misc]


def _model_class(model_name: str):
    try:
        mt = AutoConfig.from_pretrained(model_name).model_type
    except Exception:
        return AutoModelForCausalLM
    if mt == "qwen3_5":
        return Qwen3_5ForConditionalGeneration  # type: ignore[return-value]
    return AutoModelForCausalLM


def build_policy_and_ref(cfg: Config):
    """(policy, ref, tokenizer). ref is None under LoRA: reference logprobs come
    from the same model with the adapter disabled, so no second base in VRAM."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.serve.model_name)
    base = cast(PreTrainedModel,
                _model_class(cfg.serve.model_name).from_pretrained(
                    cfg.serve.model_name, torch_dtype=cfg.train.torch_dtype,
                    low_cpu_mem_usage=True,
                ))
    base = base.to(device=cfg.train.resolved_device)  # type: ignore[call-arg]
    base.eval()
    if cfg.train.grad_checkpointing:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if not cfg.train.use_lora:
        return base, base, tokenizer

    lora = LoraConfig(
        r=cfg.train.lora_r, lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=cfg.serve.model_name.split("/")[-1].startswith("Qwen") and
        r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj"
        r"|gate_proj|up_proj|down_proj)" or r".*\.(q_proj|k_proj|v_proj|o_proj"
        r"|gate_proj|up_proj|down_proj)",
        bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(base, lora), None, tokenizer


def make_task_pool(cfg: Config, seeds: list[int], family: str) -> list[Task]:
    out = []
    for s in seeds:
        for spec in repo_gen.make_specs(s, family):
            out.append(spec_to_task(spec))
    return out


def run_step(policy, ref, tokenizer, tasks, cfg) -> tuple[float, dict]:
    batch = random.sample(tasks, min(cfg.train.task_batch, len(tasks)))
    total_loss = 0.0
    rewards, tool_use, parsed, truncated = [], 0, 0, 0
    n_traj, dead_groups = 0, 0
    for task in batch:
        policy.eval()  # sampling: KV cache on, checkpointing off
        group = [sample_trajectory(policy, tokenizer, task, cfg, i)
                 for i in range(cfg.train.group_size)]
        policy.train()
        total_loss += grpo_backward(policy, ref, group, cfg,
                                    scale=1.0 / len(batch))
        r = [t.reward for t in group]
        rewards.extend(r)
        if len(set(r)) == 1:
            dead_groups += 1
        for t in group:
            n_traj += 1
            tool_use += int(any(turn.action is not None for turn in t.turns))
            parsed += int(any(T.parse_action(turn.resp_text) is not None
                              for turn in t.turns))
            truncated += int(t.truncated)
    stats = {
        "mean_reward": sum(rewards) / len(rewards),
        "dead_groups": dead_groups / len(batch),
        "tool_use": tool_use / n_traj,
        "parse_rate": parsed / n_traj,
        "truncation_rate": truncated / n_traj,
    }
    return total_loss, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", required=True, help="e.g. 1000-1009 or 1000,1002")
    ap.add_argument("--family", default="find")
    ap.add_argument("--runner", default="docker")
    a = ap.parse_args()

    cfg = Config()
    if a.steps:
        cfg.train.total_steps = a.steps
    if a.group_size:
        cfg.train.group_size = a.group_size
    if a.model:
        cfg.serve.model_name = a.model
    cfg.episode.family = a.family
    cfg.episode.runner = a.runner

    seeds = []
    for part in a.seeds.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))

    torch.manual_seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    print(f"device={cfg.train.resolved_device} dtype={cfg.train.dtype} "
          f"steps={cfg.train.total_steps} group={cfg.train.group_size} "
          f"family={a.family} seeds={seeds}")

    vram.install()
    policy = ref = optim = None
    try:
        policy, ref, tokenizer = build_policy_and_ref(cfg)
        optim = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=cfg.train.lr)
        tasks = make_task_pool(cfg, seeds, a.family)
        print(f"[train] {len(tasks)} tasks")
        if not tasks:
            raise SystemExit("no tasks; pick seeds/family that generate tasks")

        worst_prompt = max(
            len(encoding.encode_prompt(tokenizer, system_prompt(t.tools),
                                       t.prompt, cfg.episode.thinking))
            for t in tasks)
        if worst_prompt >= cfg.train.max_seq_tokens:
            raise SystemExit(f"prompt of {worst_prompt} tokens >= max_seq_tokens")
        vram.check(policy, cfg.train.max_seq_tokens,
                   cfg.train.max_seq_tokens - worst_prompt,
                   cfg.train.grad_checkpointing,
                   where="preflight (worst-case trajectory)")

        for step in range(1, cfg.train.total_steps + 1):
            optim.zero_grad()
            total_loss, stats = run_step(policy, ref, tokenizer, tasks, cfg)
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            print(
                f"step {step}/{cfg.train.total_steps}  loss={total_loss:.4f}  "
                f"grad_norm={grad.item():.3f}  "
                f"reward={stats['mean_reward']:.3f}  "
                f"dead={stats['dead_groups']:.0%}  "
                f"trunc={stats['truncation_rate']:.0%}  "
                f"tool_use={stats['tool_use']:.0%}  "
                f"parse={stats['parse_rate']:.0%}  "
                f"{vram.step_report()}"
            )

        out = f"lora-{cfg.serve.model_name.split('/')[-1]}-gym-{a.family}-grpo"
        if cfg.train.use_lora:
            policy.save_pretrained(out)
            print(f"saved LoRA to ./{out}")
    except vram.VramCeiling as e:
        print(e, file=sys.stderr)
        raise SystemExit(2)
    finally:
        del policy, ref, optim
        vram.release()


if __name__ == "__main__":
    main()