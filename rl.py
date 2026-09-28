"""GRPO loss over a group of trajectories for one task.

Each trajectory is scored with ONE forward pass over its full token stream
(the running concatenated list from rollout.py). The `trainable` mask
selects exactly the policy-sampled token positions - observations, prompt
and template tokens are never loss targets, so multi-turn agent GRPO cannot
silently reward the model for "writing" tool output it never produced.

Per-token loss, policy vs frozen reference:

    -a * log_pi + beta * k3(log_pi, log_ref)

where k3 is the unbiased reference-pull KL estimator from the GRPO paper:

    log_ratio = log_ref - log_pi
    k3 = exp(log_ratio) - log_ratio - 1      # >= 0, gradient pulls pi -> ref

The naive log_pi - log_ref has gradient just -1 * beta * grad(log_pi): a
coefficient blend on the policy gradient, not a KL. Use k3.

The reference is the LoRA base model with the adapter disabled - never a
second model in memory (the 12B case this matters for).

INVARIANT: this loss is on-policy ONLY for exactly one optimizer step per
rollout batch. sampler == trainer holds only until the first parameter
update touches those logprobs. If you add a second optim.step() on the same
rollouts, you must introduce the importance ratio + clipping. Deliberate
per-token normalization: longer trajectories carry more gradient mass
(DAPO-style weighting). Known choice, not an accident.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from config import Config
from schema import Trajectory


def _masked_logprobs(model, traj: Trajectory, device: str) -> torch.Tensor:
    """(n_policy_tokens,) per-token logprob of the policy-sampled ids.

    One forward over traj.tok_ids; position i predicts token i+1, so the
    logprob of a token at position j is taken from logits[j-1] - and only
    at positions where traj.trainable[j] is True.

    logits are cast to float32 before the softmax: in bf16 the mantissa is
    too small for the tiny logprob differences between policy and ref that
    the KL term reads, so the term would be mostly rounding noise.
    """
    seq = torch.tensor([traj.tok_ids], device=device)
    logits = model(seq).logits[0]  # (T, V)

    trainable = torch.tensor(traj.trainable, dtype=torch.bool, device=device)
    idx = trainable[1:].nonzero(as_tuple=True)[0] + 1  # token positions (j)
    if idx.numel() == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)

    targets = torch.tensor([traj.tok_ids[int(j)] for j in idx], device=device)
    logp = F.log_softmax(logits[idx - 1].float(), dim=-1)  # logits[j-1] predicts j
    return logp[torch.arange(idx.numel(), device=device), targets]


def group_advantages(trajectories: List[Trajectory], eps: float = 1e-4) -> torch.Tensor:
    """Standardized rewards within the group (the 'relative' part of GRPO)."""
    rewards = torch.tensor([t.reward for t in trajectories], dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


def grpo_loss(
    policy,
    ref: Optional[torch.nn.Module],
    group: List[Trajectory],
    cfg: Config,
) -> torch.Tensor:
    """Scalar loss for one task's group, normalized by total policy tokens.

    ref is None when the policy is a LoRA adapter over the base model: the
    reference logprobs come from the same model with the adapter disabled.
    """
    device = cfg.resolved_device

    adv = group_advantages(group)

    total_tokens = 0
    loss = torch.tensor(0.0, device=device, requires_grad=True)

    for traj, a in zip(group, adv):
        pol = _masked_logprobs(policy, traj, device)
        if ref is not None:
            with torch.no_grad():
                refv = _masked_logprobs(ref, traj, device)
        else:
            with torch.no_grad(), policy.disable_adapter():
                refv = _masked_logprobs(policy, traj, device)

        n = pol.numel()
        if n == 0:
            continue
        total_tokens += n

        log_ratio = refv - pol                     # log(ref / pi)
        kl = torch.exp(log_ratio) - log_ratio - 1  # k3: >= 0, correct gradient
        loss = loss + (-a * pol).sum() + cfg.kl_beta * kl.sum()

    if total_tokens == 0:
        return loss * 0.0

    return loss / total_tokens