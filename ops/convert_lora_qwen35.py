"""convert_lora_to_gguf.py for Qwen3.5 LoRAs that touch the linear-attention layers.

Upstream fails on these: Qwen3.5's converter reorders V heads (grouped -> tiled) with reshape/permute, and
LoraTorchTensor can't reshape a LoRA's input dimension (out_proj) or torch.cat factors (in_proj_qkv).
Both reorders are pure permutations, which map exactly onto the factors of W = B @ A:
  permute output rows    -> permute rows of B
  permute input columns  -> permute columns of A
and cat of row slices that share one A -> cat their B's. Only this process is patched.

  python ops/convert_lora_qwen35.py --base <hf snapshot> --outtype f16 --outfile out.gguf <peft adapter dir>
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

LLAMA_CPP = Path.home() / "llama.cpp-source"
sys.path.insert(0, str(LLAMA_CPP))
sys.path.insert(1, str(LLAMA_CPP / "gguf-py"))

from conversion import qwen  # noqa: E402

_orig = qwen._LinearAttentionVReorderBase._reorder_v_heads


def _is_lora(t) -> bool:
    return type(t).__name__ == "LoraTorchTensor"


def _reorder(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    if not _is_lora(tensor):
        return _orig(tensor, dim, num_k_heads, num_v_per_k, head_dim)
    A, B = tensor.get_lora_A_B()                      # A (r, in), B (out, r)
    if A.ndim != 2:
        raise NotImplementedError("only 2-D LoRA factors")
    dim = dim % 2
    n = B.shape[0] if dim == 0 else A.shape[1]
    # the exact index permutation the dense path applies, computed on an index vector
    perm = _orig(torch.arange(n).unsqueeze(-1), 0, num_k_heads, num_v_per_k, head_dim).squeeze(-1)
    if dim == 0:
        return type(tensor)(A, B[perm])
    return type(tensor)(A[:, perm], B)


qwen._LinearAttentionVReorderBase._reorder_v_heads = staticmethod(_reorder)

_cat = torch.cat


def _lora_cat(tensors, dim=0, **kw):
    ts = list(tensors)
    if ts and all(_is_lora(t) for t in ts):
        As = [t.get_lora_A_B()[0] for t in ts]
        if dim not in (0, -2) or any(not torch.equal(As[0], a) for a in As[1:]):
            raise NotImplementedError("LoRA cat only for row slices sharing A")
        return type(ts[0])(As[0], _cat([t.get_lora_A_B()[1] for t in ts], dim=0))
    return _cat(tensors, dim=dim, **kw)


torch.cat = _lora_cat

sys.argv = [str(LLAMA_CPP / "convert_lora_to_gguf.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
