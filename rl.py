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
from torch.utils.checkpoint import checkpoint

import vram
from config import Config
from schema import Trajectory


LOGP_CHUNK = vram.LOGP_CHUNK  # positions per lm_head chunk (vram.py budgets it)


def _chunk_logp(lm_head, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = lm_head(h).float()  # (C, V) fp32, lives only inside this chunk
    tgt = logits.gather(-1, targets[:, None]).squeeze(-1)
    return tgt - torch.logsumexp(logits, dim=-1)


def _masked_logprobs(model, traj: Trajectory, device: str) -> torch.Tensor:
    """(n_policy_tokens,) per-token logprob of the policy-sampled ids.

    One decoder forward over traj.tok_ids; position i predicts token i+1, so
    the logprob of a token at position j comes from hidden[j-1] - and only at
    positions where traj.trainable[j] is True.

    The full (T, V) logits are never built: V is ~248k, so that tensor plus
    its grad and fp32 copy was most of the VRAM. Instead lm_head runs on the
    selected hidden states LOGP_CHUNK at a time, each chunk checkpointed so
    backward recomputes its logits instead of keeping them.

    logits are cast to float32 before logsumexp: in bf16 the mantissa is too
    small for the tiny logprob differences between policy and ref that the
    KL term reads, so the term would be mostly rounding noise.
    """
    trainable = torch.tensor(traj.trainable, dtype=torch.bool, device=device)
    idx = trainable[1:].nonzero(as_tuple=True)[0] + 1  # token positions (j)
    if idx.numel() == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)

    # LoRA is injected in place, so calling the inner modules still runs the
    # adapter (and disable_adapter() still turns it off).
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    seq = torch.tensor([traj.tok_ids], device=device)
    hidden = hf.model(input_ids=seq, use_cache=False).last_hidden_state[0]  # (T, H), normed
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
    """Standardized rewards within the group (the 'relative' part of GRPO)."""
    rewards = torch.tensor([t.reward for t in trajectories], dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)


def _n_policy_tokens(traj: Trajectory) -> int:
    """Count of positions _masked_logprobs will score (trainable, j >= 1)."""
    return sum(traj.trainable[1:])


def grpo_backward(
    policy,
    ref: Optional[torch.nn.Module],
    group: List[Trajectory],
    cfg: Config,
    scale: float = 1.0,
) -> float:
    """Accumulate grads for one task's group; returns the loss value.

    The loss is sum over trajectories / total policy tokens in the group,
    times `scale` (1 / n_groups in the step). Because that normalizer is known
    before any forward (it's just the trainable mask), each trajectory runs
    forward + backward on its own and its graph is freed before the next one.
    Peak activation memory is ONE trajectory, not the whole batch; the grads
    summed in .grad are identical to one backward over the full loss.

    ref is None when the policy is a LoRA adapter over the base model: the
    reference logprobs come from the same model with the adapter disabled.
    """
    device = cfg.resolved_device

    adv = group_advantages(group)
    total_tokens = sum(_n_policy_tokens(t) for t in group)
    if total_tokens == 0:
        return 0.0

    loss_val = 0.0
    for traj, a in zip(group, adv):
        n = _n_policy_tokens(traj)
        if n == 0:
            continue
        vram.check(policy, len(traj.tok_ids), n, cfg.grad_checkpointing,
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
        loss = ((-a * pol).sum() + cfg.kl_beta * kl.sum()) * (scale / total_tokens)
        loss.backward()
        loss_val += loss.item()

    return loss_val