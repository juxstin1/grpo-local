"""VRAM ceiling guard + exit cleanup (ported from grpo-local, same approach).

On Windows ROCm, overflowing VRAM does not raise a clean OOM -- it spills into
shared memory until the driver times out (TDR) and the GPU resets. So we
estimate each trajectory's forward+backward footprint BEFORE running it,
compare to what is actually free, and refuse with numbers instead of letting
the driver hang.

A PID lock in runs/ catches a zombie train.py still holding VRAM, and
release() frees the caching allocator on any exit path we control.
"""

from __future__ import annotations

import atexit
import gc
import os
import signal
import sys
from pathlib import Path

import psutil
import torch

GiB = 1024**3
SAFETY = 1.4              # multiplier on the estimate
RESERVE = int(0.75 * GiB)  # fragmentation + kernel workspace slack
LOGP_CHUNK = 256           # rl.py scores this many positions per lm_head call

STEP = {"est_max": 0}


class VramCeiling(RuntimeError):
    pass


def _text_cfg(model):
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def estimate_bytes(model, seq_len: int, n_policy: int, checkpointing: bool) -> dict:
    tc = _text_cfg(model)
    H, L = tc.hidden_size, tc.num_hidden_layers
    I = getattr(tc, "intermediate_size", 4 * H)
    V = tc.vocab_size
    heads = getattr(tc, "num_attention_heads", 16)
    b = 2 if next(model.parameters()).dtype in (torch.bfloat16, torch.float16) else 4
    T, n = seq_len, n_policy
    layer_full = T * (12 * H + 4 * I) * b
    graph = (L * T * H * b + layer_full) if checkpointing else L * layer_full
    attn = heads * T * T * b
    logits = min(n, LOGP_CHUNK) * V * 12
    total = int((graph + attn + logits) * SAFETY)
    return {"graph": graph, "attn": attn, "logits": logits, "total": total}


def available_bytes() -> tuple[int, int]:
    free, total = torch.cuda.mem_get_info()
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return free + cached, total


def check(model, seq_len: int, n_policy: int, checkpointing: bool, where: str) -> int:
    if not torch.cuda.is_available() or next(model.parameters()).device.type != "cuda":
        return 0
    est = estimate_bytes(model, seq_len, n_policy, checkpointing)
    STEP["est_max"] = max(STEP["est_max"], est["total"])
    avail, total = available_bytes()
    if est["total"] + RESERVE > avail:
        raise VramCeiling(
            f"[vram] REFUSED {where}: T={seq_len} policy_tokens={n_policy} "
            f"ckpt={checkpointing}\n"
            f"  need  {est['total'] / GiB:6.2f} GiB (graph {est['graph'] / GiB:.2f} + "
            f"attn {est['attn'] / GiB:.2f} + logits {est['logits'] / GiB:.2f}, x{SAFETY})"
            f" + reserve {RESERVE / GiB:.2f}\n"
            f"  have  {avail / GiB:6.2f} GiB usable of {total / GiB:.2f} "
            f"(torch allocated {torch.cuda.memory_allocated() / GiB:.2f})\n"
            f"  fix: shorter max_new_tokens/max_turns, grad_checkpointing=True, "
            f"or a smaller model"
        )
    return est["total"]


def step_report() -> str:
    if not torch.cuda.is_available():
        return ""
    peak = torch.cuda.max_memory_allocated()
    _, total = torch.cuda.mem_get_info()
    s = (f"vram est={STEP['est_max'] / GiB:.2f} peak={peak / GiB:.2f} "
         f"of {total / GiB:.1f} GiB")
    STEP["est_max"] = 0
    torch.cuda.reset_peak_memory_stats()
    return s


# --- cleanup ----------------------------------------------------------------

_LOCK = Path(__file__).resolve().parent / "runs" / ".train.lock"
_released = False


def acquire_lock() -> None:
    if _LOCK.exists():
        try:
            pid = int(_LOCK.read_text().strip())
        except ValueError:
            pid = -1
        if pid != os.getpid() and psutil.pid_exists(pid):
            try:
                cmd = " ".join(psutil.Process(pid).cmdline())
            except psutil.Error:
                cmd = ""
            if "train.py" in cmd:
                raise SystemExit(
                    f"[vram] previous run PID {pid} is still alive and may hold VRAM:\n"
                    f"  {cmd}\n  kill it first:  Stop-Process -Id {pid}"
                )
        print(f"[vram] removed stale lock from dead PID {pid}")
    _LOCK.parent.mkdir(parents=True, exist_ok=True)
    _LOCK.write_text(str(os.getpid()))


def release() -> None:
    global _released
    if _released:
        return
    _released = True
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[vram] cleanup: {type(e).__name__}: {e}", file=sys.stderr)
    try:
        if _LOCK.exists() and _LOCK.read_text().strip() == str(os.getpid()):
            _LOCK.unlink()
    except OSError:
        pass


def install() -> None:
    acquire_lock()
    atexit.register(release)

    def _die(signum, _frame):
        raise SystemExit(128 + signum)

    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _die)