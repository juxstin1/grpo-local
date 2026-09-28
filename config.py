from dataclasses import dataclass

import torch


@dataclass
class Config:
    # --- model -----------------------------------------------------------
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"          # bfloat16 | float32 (float32 for CPU)
    device: str = "auto"             # auto | cuda | cpu

    # --- policy adapter ---------------------------------------------------
    use_lora: bool = True            # train LoRA on a frozen base, not full FT
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # --- optimizer ----------------------------------------------------------
    lr: float = 5e-5
    kl_beta: float = 0.04            # simple-KL weight vs the frozen reference

    # --- rollout ------------------------------------------------------------
    group_size: int = 4              # responses sampled per task (the GRPO group)
    task_batch: int = 2              # tasks per optimizer step
    total_steps: int = 2             # keep small to smoke-test
    seed: int = 0

    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 96
    max_turns: int = 6               # tool rounds before we force-stop

    raw_text: bool = True            # False = chat-template segments (verify first)

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float32

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def lora_targets(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]