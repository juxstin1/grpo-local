"""Growable GRPO training loop for an agentic coder.

Run:  python train.py            (smoke test: 2 steps, small tasks)
      python train.py --steps 50 --group-size 6

Growth path (one seam to swap at a time):
  - tasks.py:  add the real buckets. Multi-hop questions generated from your
               repos; C++-edit tasks with per-episode perturbations; JUCE
               processBlock alloc/lock verifiers over MCP. Reward fns stay
               mechanical, keep them sparse.
  - rollout.py: swap MockEnv for the opencode executor (agent loop = opencode,
               verifier = MCP servers). If sampling moves off-process, capture
               per-token logprobs at the sampler and store them on Trajectory.
  - train.py:  small edits only when the above force them.
"""

from __future__ import annotations

import argparse
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from rl import grpo_loss
from rollout import sample_trajectory
from tasks import MockEnv, make_tasks, parse_action, used_tool


def run_step(policy, ref, tokenizer, env, tasks, cfg) -> tuple[torch.Tensor, list, dict]:
    batch = random.sample(tasks, min(cfg.task_batch, len(tasks)))
    group_losses = []
    rewards: list[float] = []
    dead_groups = 0
    tool_use = 0
    n_traj = 0

    for task in batch:
        group = [sample_trajectory(policy, tokenizer, env, task, cfg)
                 for _ in range(cfg.group_size)]
        loss = grpo_loss(policy, ref, group, cfg)
        group_losses.append(loss)

        r = [t.reward for t in group]
        rewards.extend(r)
        if len(set(r)) == 1:
            dead_groups += 1  # all-equal group: zero advantage, dead step
        for t in group:
            n_traj += 1
            tool_use += int(used_tool(t.turns))

    total_loss = torch.stack(group_losses).mean()
    total_loss.backward()

    stats = {
        "mean_reward": sum(rewards) / len(rewards),
        "dead_groups": dead_groups / len(batch),
        "tool_use": tool_use / n_traj,
    }
    return total_loss, batch, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--lora", type=lambda s: s.lower() == "true", default=None)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.steps:
        cfg.total_steps = args.steps
    if args.group_size:
        cfg.group_size = args.group_size
    if args.lora is not None:
        cfg.use_lora = args.lora
    if args.model:
        cfg.model_name = args.model

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    print(f"device={cfg.resolved_device} dtype={cfg.dtype} "
          f"steps={cfg.total_steps} group={cfg.group_size}")

    policy, ref, tokenizer = build_policy_and_ref(cfg)
    optim = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    env = MockEnv()
    tasks = make_tasks()

    for step in range(1, cfg.total_steps + 1):
        optim.zero_grad()
        total_loss, batch, stats = run_step(policy, ref, tokenizer, env, tasks, cfg)
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim.step()
        print(
            f"step {step}/{cfg.total_steps}  loss={total_loss.item():.4f}  "
            f"grad_norm={grad.item():.3f}  "
            f"reward={stats['mean_reward']:.3f}  "
            f"dead={stats['dead_groups']:.0%}  "
            f"tool_use={stats['tool_use']:.0%}"
        )

    out = f"lora-{cfg.model_name.split('/')[-1]}-grpo"
    if cfg.use_lora:
        policy.save_pretrained(out)
        print(f"saved LoRA to ./{out}")


if __name__ == "__main__":
    main()