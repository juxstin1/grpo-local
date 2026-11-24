"""Configuration for the sandbox gym.

Three sections:
  serve    how we talk to the model (vLLM OpenAI endpoint or local torch)
  episode  the sandbox loop (budget, turns, which tools, where work happens)
  train    GRPO hyperparameters

Budget and turn cap are ONE number each -- 8192 total prompt+completion tokens
and 10 turns per episode (episode.budget, episode.max_turns),
decided 2026-09-29 -- and is read by eval, collection and GRPO alike. The budget
lives in grpo-local/gym_budget.py (default 8192; override env GYM_BUDGET or --budget).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))  # grpo-local (appended: never shadows)
from gym_budget import BUDGET  # noqa: E402  (THE episode budget; GYM_BUDGET / --budget)


@dataclass
class Serve:
    model_name: str = "Qwen/Qwen3.5-4B"      # HF id: tokenizer + chat template
    served_model: str = ""                   # LoRA adapter name served by vLLM ("" = base)
    base: str = "http://127.0.0.1:8000"      # always http://127.0.0.1, never localhost
    api_key: str = ""                        # bearer token for a pod vLLM
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass
class Episode:
    budget: int = BUDGET                     # total prompt+completion tokens (one budget)
    max_turns: int = 10                      # THE turn cap: eval, collect, GRPO, CLI.
                                             # One number, like budget -- never a second
                                             # copy (Train had its own 6 until 2026-09-29)
    thinking: bool = True                    # reasoning is never scored
    family: str = "find"                     # find | edit | fix
    runner: str = "docker"                   # docker (production) | host (selftest only)
    max_builds: int = 4                      # build/run_tests cap per episode
    hint: str = ""                            # COLLECTION ONLY (context distillation):
                                             # extra system line; stripped before SFT
                                             # (sft/from_gym.py --strip-hint). Never eval.
    turn_counter: bool = True                # tell her the turn budget: system line +
                                             # "[turn t of N]" after every tool result
    compact: bool = False                    # in-episode compaction (compact.py): note at
                                             # 75% of budget + checkpoint tool; OFF = today
    compact_note_mode: str = "off"           # harness note call: off (thinking off) | force
    max_resets: int | None = None            # None = bounded only by turn cap + thrash guard
    data_root: Path = field(default_factory=lambda: Path("data"))
    toolset: dict = field(default_factory=lambda: {  # gym ring -> enabled tools
        "find": ["list_dir", "read_file", "search"],
        "edit": ["list_dir", "read_file", "search", "edit_file", "write_file"],
        "fix": ["list_dir", "read_file", "search", "edit_file", "write_file",
                "build", "run_tests"],
        "cpp": ["list_dir", "read_file", "search", "edit_file", "write_file",
                "build", "run_tests", "definition", "references", "diagnostics"],
    })


@dataclass
class Train:
    # --- model -----------------------------------------------------------
    dtype: str = "bfloat16"                  # bfloat16 | float32 (float32 for CPU)
    device: str = "auto"                     # auto | cuda | cpu
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    grad_checkpointing: bool = True
    lr: float = 5e-5
    kl_beta: float = 0.04
    # --- rollout -----------------------------------------------------------
    group_size: int = 4                      # responses per task (the GRPO group)
    task_batch: int = 2                      # tasks per optimizer step
    total_steps: int = 2                     # keep small to smoke-test
    seed: int = 0
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 4096               # per turn
    max_seq_tokens: int = 12288              # whole trajectory; bounds VRAM (see vram.py)

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float32

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    serve: Serve = field(default_factory=Serve)
    episode: Episode = field(default_factory=Episode)
    train: Train = field(default_factory=Train)

    @property
    def tools(self) -> list[str]:
        return self.episode.toolset[self.episode.family]