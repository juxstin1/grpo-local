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
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:  # Qwen3.5 needs transformers >= 5.x (run via .venv); absent on 4.x
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None

import vram
from config import Config
from context import encode_prompt
from rl import grpo_backward
from rollout import sample_trajectory
from tasks import SYSTEM, MockEnv, make_tasks, parse_action, turns_parsed, used_tool


def _model_class(model_name: str):
    """Qwen3.5 is a conditional-generation (multimodal) wrapper, not a causal
    LM; its logits still follow the standard causal alignment, and LoRA lands
    only on language_model.layers attention. Everything else is a plain causal
    LM and uses the auto path."""
    try:
        mt = AutoConfig.from_pretrained(model_name).model_type
    except Exception:
        return AutoModelForCausalLM
    if mt == "qwen3_5":
        return Qwen3_5ForConditionalGeneration or AutoModelForCausalLM
    return AutoModelForCausalLM


def build_policy_and_ref(cfg: Config):
    """Load base on device, wrap in LoRA. Returns (policy, ref, tokenizer).

    ref is None when training LoRA: the reference logprobs come from the same
    model with the adapter disabled (rl.py), so no second copy of the base
    lives in VRAM - that is the 12B case this matters for.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    base = _model_class(cfg.model_name).from_pretrained(
        cfg.model_name,
        torch_dtype=cfg.torch_dtype,
        low_cpu_mem_usage=True,
    )
    base.to(cfg.resolved_device)
    base.eval()
    if cfg.grad_checkpointing:
        # Non-reentrant: works with a frozen base (no enable_input_require_grads).
        # HF only checkpoints in train() mode; run_step toggles train/eval.
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if not cfg.use_lora:
        return base, base, tokenizer  # no KL (ref == policy); full-FT corner case

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(base, lora)
    return policy, None, tokenizer


def run_step(policy, ref, tokenizer, env, tasks, cfg) -> tuple[float, list, dict]:
    """Sample + accumulate grads for one optimizer step. Each trajectory is
    backwarded on its own inside grpo_backward, so no two graphs are ever
    alive at once; the caller does clip + optim.step()."""
    batch = random.sample(tasks, min(cfg.task_batch, len(tasks)))
    total_loss = 0.0
    rewards: list[float] = []
    dead_groups = 0
    tool_use = 0
    parsed = 0
    truncated = 0
    n_traj = 0

    for task in batch:
        policy.eval()  # sampling: KV cache on, checkpointing off
        group = [sample_trajectory(policy, tokenizer, env, task, cfg)
                 for _ in range(cfg.group_size)]
        policy.train()  # loss: checkpointing active (lora_dropout=0, no noise)
        total_loss += grpo_backward(policy, ref, group, cfg, scale=1.0 / len(batch))

        r = [t.reward for t in group]
        rewards.extend(r)
        if len(set(r)) == 1:
            dead_groups += 1  # all-equal group: zero advantage, dead step
        for t in group:
            n_traj += 1
            tool_use += int(used_tool(t.turns))
            parsed += int(turns_parsed(t.turns))
            truncated += int(t.truncated)

    stats = {
        "mean_reward": sum(rewards) / len(rewards),
        "dead_groups": dead_groups / len(batch),
        "tool_use": tool_use / n_traj,
        "parse_rate": parsed / n_traj,
        "truncation_rate": truncated / n_traj,
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
          f"steps={cfg.total_steps} group={cfg.group_size} "
          f"ckpt={cfg.grad_checkpointing}")

    vram.install()  # PID lock + release() on every exit path we control
    policy = ref = optim = None
    try:
        policy, ref, tokenizer = build_policy_and_ref(cfg)
        optim = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=cfg.lr
        )
        env = MockEnv()
        tasks = make_tasks()

        # Preflight: refuse up front if the longest possible trajectory can't
        # fit, instead of discovering it after minutes of sampling.
        # rollout.py hard-stops every trajectory at cfg.max_seq_tokens, so that
        # is the worst case; the prompt only has to leave room for one turn.
        worst_prompt = max(
            len(encode_prompt(tokenizer, SYSTEM, t.prompt, cfg.raw_text, cfg.thinking))
            for t in tasks
        )
        if worst_prompt >= cfg.max_seq_tokens:
            raise SystemExit(f"prompt of {worst_prompt} tokens >= max_seq_tokens={cfg.max_seq_tokens}")
        worst_T = cfg.max_seq_tokens
        est = vram.check(policy, worst_T, worst_T - worst_prompt,
                         cfg.grad_checkpointing, where="preflight (worst-case trajectory)")
        if est:
            usable, total = vram.available_bytes()
            print(f"[vram] preflight ok: worst T={worst_T} est={est / vram.GiB:.2f} GiB, "
                  f"usable {usable / vram.GiB:.2f} of {total / vram.GiB:.1f} GiB")

        for step in range(1, cfg.total_steps + 1):
            optim.zero_grad()
            total_loss, batch, stats = run_step(policy, ref, tokenizer, env, tasks, cfg)
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            print(
                f"step {step}/{cfg.total_steps}  loss={total_loss:.4f}  "
                f"grad_norm={grad.item():.3f}  "
                f"reward={stats['mean_reward']:.3f}  "
                f"dead={stats['dead_groups']:.0%}  "
                f"trunc={stats['truncation_rate']:.0%}  "
                f"tool_use={stats['tool_use']:.0%}  "
                f"parse={stats['parse_rate']:.0%}  "
                f"{vram.step_report()}"
            )

        out = f"lora-{cfg.model_name.split('/')[-1]}-grpo"
        if cfg.use_lora:
            policy.save_pretrained(out)
            print(f"saved LoRA to ./{out}")
    except vram.VramCeiling as e:
        print(e, file=sys.stderr)
        raise SystemExit(2)
    except torch.OutOfMemoryError as e:  # estimate was too optimistic
        a, t = (vram.available_bytes() if torch.cuda.is_available() else (0, 0))
        print(f"[vram] OOM despite guard (estimate too low - raise vram.SAFETY): "
              f"usable {a / vram.GiB:.2f} of {t / vram.GiB:.1f} GiB\n  {e}", file=sys.stderr)
        raise SystemExit(3)
    finally:
        del policy, ref, optim  # drop the last refs, or empty_cache frees nothing
        vram.release()


if __name__ == "__main__":
    main()