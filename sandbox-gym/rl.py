"""GRPO loss over a group of trajectories (ported from grpo-local rl.py).

Each trajectory is scored with ONE forward pass over its full token stream.
The trainable mask selects exactly the policy-sampled positions -- observations,
prompt and template tokens are never loss targets, so multi-turn agent GRPO
cannot reward the model for "writing" tool output it never produced.

Per-token loss, policy vs frozen reference:

    -a * log_pi + beta * k3(log_pi, log_ref)

where k3 is the unbiased reference-pull KL estimator from the GRPO paper:

    log_ratio = log_ref - log_pi
    k3 = exp(log_ratio) - log_ratio - 1      # >= 0, gradient pulls pi -> ref

The reference is the LoRA base with the adapter disabled -- never a second
model in memory.

INVARIANTS (each fails silently; check before a real run):
- on-policy ONLY for exactly one optimizer step per rollout batch
- log_softmax in float32 (bf16 mantissa is too small for the KL term)
- EOS stays in the stream and stays trainable
- lm_head never builds the full (T, V) logits; runs LOGP_CHUNK positions at a
  time under checkpoint
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.utils.checkpoint import checkpoint

import vram
from config import Config
from schema import Trajectory

LOGP_CHUNK = vram.LOGP_CHUNK


def _chunk_logp(lm_head, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = lm_head(h).float()  # (C, V) fp32, lives only inside this chunk
    tgt = logits.gather(-1, targets[:, None]).squeeze(-1)
    return tgt - torch.logsumexp(logits, dim=-1)


def _masked_logprobs(model, traj: Trajectory, device: str) -> torch.Tensor:
    trainable = torch.tensor(traj.trainable, dtype=torch.bool, device=device)
    idx = trainable[1:].nonzero(as_tuple=True)[0] + 1  # token positions (j)
    if idx.numel() == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)

    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    seq = torch.tensor([traj.tok_ids], device=device)
    hidden = hf.model(input_ids=seq, use_cache=False).last_hidden_state[0]  # (T, H)
    h = hidden[idx - 1]  # hidden[j-1] predicts j
    targets = torch.tensor(traj.tok_ids, device=device)[idx]

    lm_head = hf.get_output_embeddings()
    out = []
    for s in range(0, idx.numel(), LOGP_CHUNK):
        hc, tc = h[s : s + LOGP_CHUNK], targets[s : s + LOGP_CHUNK]
        if torch.is_grad_enabled():
            out.append(checkpoint(_chunk_logp, lm_head, hc, tc, use_reentrant=False))
        else:
            out.append(_chunk_logp(lm_head, hc, tc))
    return torch.cat(out)


def group_advantages(trajectories: List[Trajectory], eps: float = 1e-4) -> torch.Tensor:
    rewards = torch.tensor([t.reward for t in trajectories], dtype=torch.float32)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def _n_policy_tokens(traj: Trajectory) -> int:
    return sum(traj.trainable[1:])


def grpo_backward(
    policy,
    ref: Optional[torch.nn.Module],
    group: List[Trajectory],
    cfg: Config,
    scale: float = 1.0,
) -> float:
    """Accumulate grads for one task's group; returns the loss value.

    Loss is sum over trajectories / total policy tokens in the group, times
    `scale` (1 / n_groups). Each trajectory runs forward+backward on its own and
    its graph is freed before the next, so peak activation memory is ONE
    trajectory, not the whole batch. ref is None when the policy is a LoRA
    adapter: reference logprobs come from the same model with the adapter
    disabled.
    """
    device = cfg.train.resolved_device
    adv = group_advantages(group)
    total_tokens = sum(_n_policy_tokens(t) for t in group)
    if total_tokens == 0:
        return 0.0

    loss_val = 0.0
    for traj, a in zip(group, adv):
        n = _n_policy_tokens(traj)
        if n == 0:
            continue
        vram.check(policy, len(traj.tok_ids), n, cfg.train.grad_checkpointing,
                   where=f"backward ({traj.task.name})")
        if ref is not None:
            with torch.no_grad():
                refv = _masked_logprobs(ref, traj, device)
        else:
            with torch.no_grad(), policy.disable_adapter():
                refv = _masked_logprobs(policy, traj, device)
        pol = _masked_logprobs(policy, traj, device)

        log_ratio = refv - pol                     # log(ref / pi)
        kl = torch.exp(log_ratio) - log_ratio - 1  # k3: >= 0, correct gradient
        loss = ((-a * pol).sum() + cfg.train.kl_beta * kl.sum()) * (scale / total_tokens)
        loss.backward()
        loss_val += loss.item()

    return loss_val