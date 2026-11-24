"""LoRA SFT on collected wins. Loss is ONLY on completion_ids; prompt ids are context.

  # stop vLLM first (it holds 85% of VRAM); the VRAM guard refuses otherwise
  .venv\\Scripts\\python.exe sft\\train_sft.py --data sft\\data\\mhqa-s7.jsonl --name mhqa-s7

Before any GPU work: every record is validated (schema.validate), its
tokenizer_template_sha must equal the current tokenizer's, and its root_id must
not be in the eval set or a sealed holdout (same gate as collect.py). Then:
per-example forward+backward (one graph alive at a time), grads accumulated over
--batch examples, loss = -sum(logp(completion)) / completion tokens in the batch.

Output: lora-<model>-sft-<name>/ (PEFT adapter, hot-loadable by vLLM) plus
sft_manifest.json with the data files' sha256, so a checkpoint names its data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "evals" / "datasets")]
import registry as DR  # noqa: E402
from sft_schema import load, template_sha, validate, training_tokens_and_mask  # noqa: E402
import admission as admission  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--registry", default=os.environ.get(DR.ENV), help="Explicit public dataset registry")
    ap.add_argument("--allow-model-run", action="store_true", help="Required even for tokenizer-loading dry runs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8, help="examples per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-root", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="validate data only, no GPU")
    a = ap.parse_args()
    if not a.allow_model_run:
        raise SystemExit("Model execution disabled; tokenizer-loading dry runs also require --allow-model-run")
    registry = DR.load_registry(a.registry)
    recs, shas = [], {}
    for p in a.data:
        shas[str(Path(p).resolve())] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        recs.extend(load(p))
    DR.assert_unique_ids(recs, "id")
    registry.validate_training_rows(recs)
    for record in recs:
        if record.get("dataset_registry_sha256") != registry.sha256 or record.get("grade", {}).get("status") != "valid":
            raise DR.RegistryError("Collection lacks this registry and a valid grading receipt; legacy wins require requalification")
    if not recs:
        raise DR.RegistryError("No training records")
    task_index = admission.check_dataset(recs, registry)
    import torch
    import vram
    from config import Config
    from rl import _masked_logprobs
    from train import build_policy_and_ref
    cfg = Config()
    if a.model:
        cfg.model_name = a.model
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    tsha = template_sha(tok)
    admission_replays = {}
    for r in recs:
        validate(r, eos)
        if r["tokenizer_template_sha"] != tsha:
            sys.exit(f"{r['id']}: tokenizer/template changed since collection - ids are stale")
        admission_replays[r['id']] = admission.verify_tokens_and_replay(r, tok, registry, task_index)
    # Max wins per root task across ALL files (collections can overlap), so easy
    # tasks don't dominate. Dedupe identical completions first.
    seen, per_root, kept = set(), {}, []
    for r in recs:
        key = (r["root_id"], tuple(r["completion_ids"]))
        if key in seen or per_root.get(r["root_id"], 0) >= a.max_per_root:
            continue
        seen.add(key)
        per_root[r["root_id"]] = per_root.get(r["root_id"], 0) + 1
        kept.append(r)
    if len(kept) < len(recs):
        print(f"[sft] per-root cap {a.max_per_root}: {len(recs)} -> {len(kept)} records")
    recs = kept
    by_bucket: dict[str, int] = {}
    for r in recs:
        by_bucket[r["bucket"]] = by_bucket.get(r["bucket"], 0) + 1
    ntok = sum(len(r["completion_ids"]) for r in recs)
    print(f"[sft] {len(recs)} records {by_bucket}, {ntok} completion tokens, "
          f"max n_tokens {max(r['n_tokens'] for r in recs)}")
    if a.dry_run:
        return

    vram.install()
    policy = optim = None
    try:
        policy, _, _ = build_policy_and_ref(cfg)
        optim = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=a.lr)
        longest = max(recs, key=lambda r: r["n_tokens"])
        vram.check(policy, longest["n_tokens"], len(longest["completion_ids"]),
                   cfg.grad_checkpointing, where="preflight (longest record)")
        policy.train()

        step = 0
        for ep in range(1, a.epochs + 1):
            order = recs[:]
            random.shuffle(order)
            for i in range(0, len(order), a.batch):
                batch = order[i : i + a.batch]
                denom = sum(len(r["completion_ids"]) for r in batch)
                optim.zero_grad()
                loss_val = 0.0
                for r in batch:
                    ids, mask = training_tokens_and_mask(r, eos)
                    ex = SimpleNamespace(tok_ids=ids, trainable=mask)
                    vram.check(policy, len(ids), len(r["completion_ids"]), cfg.grad_checkpointing,
                               where=f"sft {r['id']}")
                    logp = _masked_logprobs(policy, ex, cfg.resolved_device)
                    loss = -logp.sum() / denom
                    loss.backward()
                    loss_val += loss.item()
                grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()
                step += 1
                print(f"ep {ep} step {step}  nll/tok={loss_val:.4f}  grad_norm={grad.item():.3f}  "
                      f"tokens={denom}  {vram.step_report()}", flush=True)

        out = ROOT / f"lora-{cfg.model_name.split('/')[-1]}-sft-{a.name}"
        policy.save_pretrained(str(out))
        (out / "sft_manifest.json").write_text(json.dumps({
            "data": shas, "records": len(recs), "by_bucket": by_bucket, "completion_tokens": ntok,
            "dataset_registry_sha256": registry.sha256,
            "admission_replays": {r['id']: admission_replays[r['id']] for r in recs},
            "epochs": a.epochs, "batch": a.batch, "lr": a.lr, "model": cfg.model_name,
            "tokenizer_template_sha": tsha, "lora_r": cfg.lora_r, "lora_targets": cfg.lora_targets,
        }, indent=1), encoding="utf-8")
        print(f"saved {out}")
    except vram.VramCeiling as e:
        print(e, file=sys.stderr)
        raise SystemExit(2)
    finally:
        del policy, optim
        vram.release()


if __name__ == "__main__":
    main()
