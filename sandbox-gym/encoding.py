"""Token-exact segment encoding for the local (torch) GRPO path.

Invariants (from the runbook; each one fails silently, check before any run):

- Never re-encode generated text. Append sampled gen_ids straight to the
  running stream. Decoded text is for parsing/logs only.
- Never tokenize a joined string. Prompt, each generation, each observation:
  encode separately, concatenate id lists.
- Every id has a trainable flag. Model-sampled = True; prompt/observations/
  template/role wrappers = False.
- Observations are rendered as the Qwen template would render a tool result:
  a user turn wrapping <tool_response>...</tool_response>, then the assistant
  generation prompt (thinking re-enabled). This must re-tokenize identically
  to what the served loop (vLLM, same tokenizer + template) feeds the model --
  collect.py aborts on any mismatch.

`render_chat` (OpenAI-style message list -> ids) is the served-side twin; the
incremental appenders below must equal rendering the same messages.
"""

from __future__ import annotations


def _ids(enc) -> list[int]:
    """transformers 5.x returns BatchEncoding (or Encoding); 4.x a list."""
    try:
        return list(enc["input_ids"])
    except (KeyError, TypeError, IndexError):
        pass
    try:
        return list(enc.ids)
    except (KeyError, TypeError, AttributeError):
        pass
    return list(enc)


def render_chat(tokenizer, messages: list[dict], thinking: bool) -> list[int]:
    """ids of an OpenAI-style message list, generation prompt appended."""
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=thinking,
    )
    return _ids(enc)


def encode_prompt(tokenizer, system: str, user: str, thinking: bool) -> list[int]:
    return render_chat(
        tokenizer,
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        thinking,
    )


def encode_observation(tokenizer, obs: str, thinking: bool) -> list[int]:
    """Non-trainable segment appended after an assistant tool call: the tool
    result as a user turn, then the assistant generation prompt again."""
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


def response_section(text: str, thinking: bool) -> str:
    """In thinking mode the prompt ends with <think>, so a generation is
    reasoning, then </think>, then the actual output. Only the text after the
    last </think> is action/answer -- reasoning is not action. No </think> yet
    means the answer never came (still thinking / truncated)."""
    if not thinking:
        return text
    idx = text.rfind("</think>")
    if idx == -1:
        return ""
    return text[idx + len("</think>"):].lstrip("\n")