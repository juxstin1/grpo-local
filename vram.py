"""VRAM ceiling guard and exit cleanup.

Why this exists: on Windows ROCm, overflowing VRAM does not raise a clean
OOM - it spills into shared memory until the driver times out (TDR) and the
GPU resets. So we estimate each trajectory's forward+backward footprint
BEFORE running it, compare it to what is actually free, and refuse with the
numbers instead of letting the driver hang.

The estimate is deliberately pessimistic (SAFETY multiplier + fixed RESERVE).
Each step prints est-vs-actual peak so the model can be checked against
reality and tightened later.

Cleanup: a PID lock in runs/ catches a zombie train.py still holding VRAM
(the "one crash wedges the next run" case), and release() frees the caching
allocator on any exit path we get control of. A hard kill (Stop-Process,
TDR) skips Python entirely - the OS frees VRAM when the process dies, and the
lock check catches it if it didn't.
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
SAFETY = 1.4             # multiplier on the estimate; r1 SFT measured ~10% over the 1.25x estimate
RESERVE = int(0.75 * GiB)  # fragmentation + kernel workspace slack
LOGP_CHUNK = 256         # rl.py scores this many positions per lm_head call

# Per optimizer step: largest estimate seen, for est-vs-actual logging.
STEP = {"est_max": 0}


class VramCeiling(RuntimeError):
    pass


def _text_cfg(model):
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def estimate_bytes(model, seq_len: int, n_policy: int, checkpointing: bool) -> dict:
    """Peak bytes for one trajectory's policy forward + backward.

    - graph: saved activations. With checkpointing only each layer's input
      (T*H) is kept, plus one layer recomputed in full during backward.
      Without it, every layer's full set is kept.
    - attn: one layer's T^2 score matrix, in case the kernel materializes it.
    - logits: rl.py never builds (T,V). lm_head runs on LOGP_CHUNK positions
      at a time under checkpoint, so the peak is one chunk being recomputed
      in backward: bf16 logits + fp32 copy + fp32 softmax grad + bf16 grad
      = 12 bytes per (position, vocab) entry.
    The no-grad ref pass runs first and is freed, so it never overlaps.
    """
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
    """(usable, total). Usable = driver-free + memory torch cached but isn't using."""
    free, total = torch.cuda.mem_get_info()
    cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    return free + cached, total


def check(model, seq_len: int, n_policy: int, checkpointing: bool, where: str) -> int:
    """Raise VramCeiling if this forward+backward won't fit. Returns the estimate."""
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
    """'est X / peak Y / total Z' for the step just finished; resets counters."""
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
    """Refuse to start if a previous train.py is still alive (and likely
    still holding VRAM). A lock left by a dead process is just removed."""
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
    """Idempotent: free the caching allocator and drop the lock. Callers must
    drop their own model/optimizer references first, or nothing is freed."""
    global _released
    if _released:
        return
    _released = True
    gc.collect()
    if torch.cuda.is_available():
        try:  # after a driver reset these can throw; never mask the real error
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
    """Lock + route every catchable exit (normal, exception, Ctrl+C,
    Ctrl+Break, SIGTERM) through release()."""
    acquire_lock()
    atexit.register(release)

    def _die(signum, _frame):
        raise SystemExit(128 + signum)  # unwinds through main's finally

    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _die)
