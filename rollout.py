"""Multi-turn trajectory sampling.

Two invariants, both enforced structurally:

1. Token stream is the concatenation of per-segment encodes, never the encode
   of a joined string. Prompt, each generated segment, and each observation
   are encoded on their own and id lists appended to a running list. BPE
   merges across segment boundaries break when you join text first, so we
   never do that - this is the verl/SkyRL convention.

2. Every token id carries a trainable flag: True for policy-sampled tokens,
   False for observations/template/prompt. The loss only ever targets
   trainable positions, so a future single-forward-with-mask rewrite cannot
   accidentally reward the model for "writing" context it never produced.

Thinking mode (cfg.thinking=True): the model emits reasoning, then
`</think>`, then its actual output. Only the post-marker text is
stored as Turn.resp_text and parsed for actions/answers - reasoning is not
action. A turn that exhausts max_new_tokens without sampling EOS is marked
truncated; a trajectory whose FINAL turn is truncated is incomplete and earns
no reward (see tasks.incomplete), and truncation rate is logged per step so a
wrong budget is visible, not silent.

The correctness core from before is unchanged: because sampler == trainer
(same weights, no optimizer step until a whole group is sampled and scored),
the logprobs recomputed at update time are bit-identical to sample-time
logprobs - there is nothing to log on the rollout. If you move sampling to
an external engine (Llamacpp/ExLlama), THIS seam is where per-token logprobs
must be captured and stored instead.
"""

from __future__ import annotations

import torch

from config import Config
from context import encode_observation, encode_prompt
from tasks import MockEnv, SYSTEM, parse_action, response_section
from schema import Task, Trajectory, Turn


def sample_trajectory(
    model: torch.nn.Module,
    tokenizer,
    env: MockEnv,
    task: Task,
    cfg: Config,
) -> Trajectory:
    tok_ids: list[int] = []
    trainable: list[bool] = []

    def append(ids: list[int], learn: bool) -> None:
        if not ids and tokenizer.bos_token_id is not None and not tok_ids:
            ids = [tokenizer.bos_token_id]
        tok_ids.extend(ids)
        trainable.extend([learn] * len(ids))

    append(encode_prompt(tokenizer, SYSTEM, task.prompt, cfg.raw_text, cfg.thinking), learn=False)
    turns: list[Turn] = []
    for _ in range(cfg.max_turns):
        # Trajectory budget: the turn gets whatever is left, so the stream never
        # exceeds cfg.max_seq_tokens (the length the VRAM preflight signed off on).
        budget = min(cfg.max_new_tokens, cfg.max_seq_tokens - len(tok_ids))
        if budget <= 0:
            break
        prefix = list(tok_ids)  # running stream up to this turn, never strings
        gen_ids, gen_text, truncated = _generate(model, tokenizer, prefix, cfg, budget)

        turn = Turn(
            gen_text=gen_text,
            resp_text=response_section(gen_text, cfg.thinking),
            gen_ids=gen_ids,
            truncated=truncated,
        )
        turns.append(turn)
        # Append the EXACT sampled ids - never a re-encode of the decoded
        # text. decode(gen_ids) may not re-encode identically (BPE merges,
        # unicode), and the loss must see the real sampled stream.
        tok_ids.extend(gen_ids)
        trainable.extend([True] * len(gen_ids))

        # Actions are parsed from the response section only; reasoning is not
        # action, so a tool call described mid-thought is not a real call.
        action = parse_action(turn.resp_text)
        if action is None:
            break  # plain answer (or truncated mid-reasoning) -> terminal

        if truncated:
            # Budget ran out mid-stream. The parsed action may still be real,
            # but the turn did not finish; stop so the episode ends with a
            # visibly-incomplete last turn instead of fabricating context.
            break

        tool, arg = action
        obs = env.step(tool, arg)
        turn.obs_text = obs
        obs_ids = encode_observation(tokenizer, obs, cfg.raw_text, cfg.thinking)
        if len(tok_ids) + len(obs_ids) >= cfg.max_seq_tokens:
            break  # out of budget with a pending action: episode ends not-done
        append(obs_ids, learn=False)

    if not turns:  # train.py's preflight refuses this config; never score an empty episode
        raise ValueError(f"prompt ({len(tok_ids)} tokens) leaves no room under "
                         f"max_seq_tokens={cfg.max_seq_tokens}")
    done = parse_action(turns[-1].resp_text) is None
    reward = task.reward_fn(turns, done)
    return Trajectory(
        task=task,
        turns=turns,
        tok_ids=tok_ids,
        trainable=trainable,
        reward=reward,
        done=done,
        truncated=any(t.truncated for t in turns),
    )


def _generate(model, tokenizer, prefix_ids: list[int], cfg: Config, max_new: int):
    inp = torch.tensor([prefix_ids], device=model.device)

    with torch.no_grad():
        out = model.generate(
            inp,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    gen_ids = out.sequences[0, inp.shape[-1] :].tolist()

    # Honest truncation: budget exhausted without sampling EOS -> the turn is
    # incomplete. Any ANSWER/tool text in it is partial, so it must never earn
    # credit and the episode should stop (handled by the caller).
    truncated = (
        len(gen_ids) >= max_new
        and (not gen_ids or gen_ids[-1] != tokenizer.eos_token_id)
    )

    # EOS stays in the stream: its logprob gets the same advantage as every
    # other token, so winners reinforce ending and losers suppress it. Stripping
    # it here would teach the model never to emit a turn boundary.
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_ids, gen_text, truncated