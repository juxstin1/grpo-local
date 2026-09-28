"""Segment encoding for prompts and observations.

The seam for chat-template support (review item #6). The default is raw text
(raw_text=True): the prompt and each observation are encoded as plain text
segments - fine for plumbing and for the MockEnv.

For real runs with instruct models (Qwen/Gemma), flip raw_text=False and move
the template-specific wrapping here:
  - encode_prompt: use apply_chat_template(..., tokenize=True) for the user
    turn, plus add_generation_prompt=True so the model knows it is its turn.
  - encode_observation: wrap the tool result in the template's tool-role
    tokens as its OWN segment, flagged non-trainable by the caller. Each
    model family differs (Qwen <|im_start|>tool..., Gemma uses its own turn
    tokens) - this is exactly where the fidelity assertion from the design
    applies: verify the wrapped segment re-tokenizes identically to what the
    deploy-time harness will feed the model.
"""

from __future__ import annotations


def encode_prompt(tokenizer, prompt: str, raw_text: bool) -> list[int]:
    if raw_text:
        return tokenizer.encode(prompt, add_special_tokens=False)
    # Chat-template path: model/template-specific, verify before real runs.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )


def encode_observation(tokenizer, obs: str, raw_text: bool) -> list[int]:
    if raw_text:
        return tokenizer.encode("\n" + obs, add_special_tokens=False)
    # Tool-role wrap goes here for real runs. Must be the exact byte sequence
    # the deploy harness renders, or training/dev observations diverge.
    return tokenizer.encode(f"\n<|im_start|>tool\n{obs}\n<|im_end|>", add_special_tokens=False)