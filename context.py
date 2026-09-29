"""Segment encoding for prompts and observations.

Two modes:

- raw_text=True: prompt and each observation are encoded as plain text
  segments. Plumbing-only; no chat semantics.
- raw_text=False (default): the Qwen chat-template path. The system + user
  turns are rendered once with apply_chat_template, and each tool observation
  is rendered as the template would: a `tool`-role block wrapped as
  `<|im_start|>user\n<tool_response>...<|im_end|>`, followed by the next
  assistant generation prompt. All of this is non-trainable; the caller flags
  it by appending with learn=False.

Thinking mode (cfg.thinking=True): the assistant generation prompt ends with
`<think>\n` and the model emits its reasoning, then `</think>`, then its actual
output. rollout.py keeps only the text after the last `</think>` for
actions/answers. Non-thinking mode inserts an empty `<think>\n\n</think>\n\n`
directly and everything the model writes is output. (Fixed 2026-09-29: this
used to emit literal " thinking"/" response" text, which is not Qwen3.5's format.)

Fidelity rule (from the runbook): these segments must re-tokenize identically
to what the deploy harness (vLLM with the same tokenizer) will feed the model.
Every special token is emitted via convert_tokens_to_ids and every plain-text
segment is encoded on its own with add_special_tokens=False - never a joined
string that could BPE-merge across a boundary.
"""

from __future__ import annotations


def _ids(enc) -> list[int]:
    """transformers 5.x returns BatchEncoding (or Encoding) from tokenize=True;
    4.x returns a plain list. Normalize all to a flat list of ints."""
    try:
        return list(enc["input_ids"])
    except (KeyError, TypeError, IndexError):
        pass
    try:
        return list(enc.ids)
    except (KeyError, TypeError, AttributeError):
        pass
    return list(enc)


def encode_prompt(tokenizer, system: str, user: str, raw_text: bool, thinking: bool) -> list[int]:
    if raw_text:
        return tokenizer.encode(system + "\n" + user, add_special_tokens=False)
    enc = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    return _ids(enc)


def encode_observation(tokenizer, obs: str, raw_text: bool, thinking: bool) -> list[int]:
    if raw_text:
        return tokenizer.encode("\n" + obs, add_special_tokens=False)

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    body = tokenizer.encode(
        f"user\n<tool_response>\n{obs}\n</tool_response>", add_special_tokens=False
    )
    newline = tokenizer.encode("\n", add_special_tokens=False)
    # Leading newline: the template puts "\n" after the assistant turn's <|im_end|>
    # (the sampled ids end at <|im_end|>), before the next <|im_start|>.
    return newline + [im_start] + body + [im_end] + newline + generation_header(tokenizer, thinking)


def generation_header(tokenizer, thinking: bool) -> list[int]:
    """Qwen3.5's assistant generation prompt, exactly as its chat template renders it
    (verified 2026-09-29 against Qwen/Qwen3.5-4B):
      thinking:     <|im_start|>assistant\\n<think>\\n
      non-thinking: <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    think = tokenizer.convert_tokens_to_ids("<think>")
    end_think = tokenizer.convert_tokens_to_ids("</think>")
    hdr = [im_start] + tokenizer.encode("assistant\n", add_special_tokens=False) + [think]
    if thinking:
        return hdr + tokenizer.encode("\n", add_special_tokens=False)
    nn = tokenizer.encode("\n\n", add_special_tokens=False)
    return hdr + nn + [end_think] + nn