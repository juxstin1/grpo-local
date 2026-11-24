"""Multi-turn trajectory sampling for GRPO, in the sandbox.

Two structural invariants (from the runbook):

1. The token stream is the concatenation of per-segment encodes, never the
   encode of a joined string. Prompt, each generation, each observation are
   encoded separately and appended. BPE merges across segment boundaries break
   when you join text first.
2. Every token id carries a trainable flag: True for policy-sampled tokens,
   False for observations/prompt/template. The loss only ever targets
   trainable positions.

The environment is the real sandbox (workspace tools), not a mock. Each
trajectory gets a FRESH workspace so edits in one roll can't leak into the
reward of another. Rewards are the family's mechanical graders (train reward ==
eval metric, per the runbook).

Thinking mode: the model emits reasoning, then a ' response' marker, then its
actual output. Only the post-marker text is parsed for actions/answers. A final
turn that exhausts the budget without EOS is truncated and earns no reward.
"""

from __future__ import annotations

import torch

import encoding
import tools as T
from agent import counter_turns, run_action
from config import Config
from graders import ANSWER_RE
from gyms import system_prompt
from repo_gen import find_spec, write_files
from runner import Runner
from schema import Task, Trajectory, Turn
from workspace import Workspace


def sample_trajectory(
    model: torch.nn.Module,
    tokenizer,
    task: Task,
    cfg: Config,
    roll_idx: int = 0,
) -> Trajectory:
    if cfg.episode.compact:
        raise NotImplementedError("compaction runs through agent.run_episode "
                                  "(eval / collect / train_grpo.py), not legacy rollout.py")
    ws = _fresh_workspace(cfg, task, roll_idx)
    tok_ids: list[int] = []
    trainable: list[bool] = []

    def append(ids: list[int], learn: bool) -> None:
        if not ids and tokenizer.bos_token_id is not None and not tok_ids:
            ids = [tokenizer.bos_token_id]
        tok_ids.extend(ids)
        trainable.extend([learn] * len(ids))

    N = counter_turns(cfg)                    # turn counter (None when off)
    append(encoding.encode_prompt(tokenizer, system_prompt(task.tools, N),
                                  task.prompt, cfg.episode.thinking), learn=False)
    turns: list[Turn] = []
    for _ in range(cfg.episode.max_turns):
        budget = min(cfg.train.max_new_tokens,
                     cfg.train.max_seq_tokens - len(tok_ids))
        if budget <= 0:
            break
        gen_ids, gen_text, truncated = _generate(
            model, tokenizer, list(tok_ids), cfg, budget)

        turn = Turn(content=gen_text,
                    resp_text=encoding.response_section(gen_text, cfg.episode.thinking),
                    gen_ids=gen_ids, truncated=truncated)
        turns.append(turn)
        tok_ids.extend(gen_ids)  # EXACT sampled ids, never a re-encode
        trainable.extend([True] * len(gen_ids))

        action = T.parse_action(turn.resp_text)
        if ANSWER_RE.search(turn.resp_text):
            break  # plain answer: episode over
        if action is None or truncated:
            break  # no tool call, or budget ran out mid-stream
        result = run_action(ws, action, task.tools)
        turn.action = action
        turn.obs = result
        obs_ids = encoding.encode_observation(
            tokenizer, encoding.observation_body(action.name, action.arguments, result,
                                                 len(turns), N),
            cfg.episode.thinking)
        if len(tok_ids) + len(obs_ids) >= cfg.train.max_seq_tokens:
            break
        append(obs_ids, learn=False)

    if not turns:
        raise ValueError(f"prompt ({len(tok_ids)} tokens) leaves no room under "
                         f"max_seq_tokens={cfg.train.max_seq_tokens}")
    reward = task.reward(turns, ws)
    return Trajectory(
        task=task, turns=turns, tok_ids=tok_ids, trainable=trainable,
        reward=reward, done=ANSWER_RE.search(turns[-1].resp_text) is not None,
        truncated=any(t.truncated for t in turns),
    )


def _fresh_workspace(cfg: Config, task: Task, roll_idx: int) -> Workspace:
    seed = task.meta["seed"]
    spec = find_spec(seed, task.family, task.name)
    root = cfg.episode.data_root / "rollouts" / f"{task.name}-r{roll_idx}"
    ws = Workspace(root, Runner(cfg.episode.runner))
    ws.reset()
    ws.builds_left = cfg.episode.max_builds
    write_files(spec["files"], root)
    return ws


def _generate(model, tokenizer, prefix_ids: list[int], cfg: Config, max_new: int):
    inp = torch.tensor([prefix_ids], device=model.device)
    with torch.no_grad():
        out = model.generate(
            inp, max_new_tokens=max_new, do_sample=True,
            temperature=cfg.train.temperature, top_p=cfg.train.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    gen_ids = out.sequences[0, inp.shape[-1]:].tolist()
    truncated = (len(gen_ids) >= max_new
                 and (not gen_ids or gen_ids[-1] != tokenizer.eos_token_id))
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen_ids, gen_text, truncated