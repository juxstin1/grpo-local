"""Build the frozen bake-off set (baseline_v1.json).

Deterministic: fixed seeds, fixed selection. The output file is the source of
truth for eval_runner.py and must NEVER be regenerated with different seeds or

editors. If you need a different set, write baseline_v2 from a different seed
and keep v1 around for comparison.

Item IDs are stable across models: lcb-001..030, bfcl-001..050, w2c-001..030,
holdout-001..008 (holdout content lives in evals/holdout/holdout.py, sealed).
"""

from __future__ import annotations

import base64
import json
import pickle
import random
import zlib
from pathlib import Path

SEED = 20260928
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "sets" / "baseline_v1.json"

BUCKETS = {"lcb": 30, "bfcl": 50, "when2call": 30, "holdout": 8}


def b64(s: str) -> bytes:
    s = s.strip().replace("-", "+").replace("_", "/")
    return base64.b64decode(s + "=" * (-len(s) % 4))


def decode_lcb_tests(s: str) -> list:
    if s.startswith("["):
        return json.loads(s)
    b = b64(s)
    if b[:1] == b"\x78":
        b = zlib.decompress(b)
    try:
        v = pickle.loads(b)
    except Exception:
        v = b.decode("utf-8")
    if isinstance(v, str):
        v = json.loads(v)
    return v


def load_lcb():
    items = [json.loads(l) for l in (RAW / "lcb" / "test6.jsonl").open(encoding="utf-8")]
    out = {}
    for it in items:
        out[it["question_id"]] = {
            "title": it["question_title"],
            "question": it["question_content"],
            "starter_code": it.get("starter_code") or "",
            "difficulty": it.get("difficulty", ""),
            "tests": decode_lcb_tests(it["private_test_cases"]),
        }
    return out


def load_bfcl():
    """Only sources with downloaded ground truth are gradeable. The two
    irrelevance files' possible_answer came back as 404; When2Call already
    covers the must-not-call direction, so they are dropped from BFCL."""
    sources = [
        ("BFCL_v4_simple_python.json", 20),
        ("BFCL_v4_live_simple.json", 15),
        ("BFCL_v4_multiple.json", 15),
    ]
    items = []
    for f, take in sources:
        data = [json.loads(l) for l in (RAW / "bfcl" / f).open(encoding="utf-8")]
        pa_raw = (RAW / "bfcl" / "possible_answer" / f).read_text(encoding="utf-8")
        try:
            pa = json.loads(pa_raw)
        except json.JSONDecodeError:
            pa = [json.loads(l) for l in pa_raw.splitlines() if l.strip()]
        if isinstance(pa, list) and pa and isinstance(pa[0], dict) and "id" in pa[0]:
            pa = {x["id"]: x for x in pa}
        by_id = {x.get("id"): x for x in data}
        items.append(
            {
                "source": f,
                "take": take,
                "items": [
                    by_id[k]
                    for k in by_id
                    if k in pa and "question" in by_id[k]
                ],
                "ground_truth": pa,
            }
        )
    return items


def load_w2c():
    items = [
        json.loads(l)
        for l in (RAW / "when2call" / "when2call_test_mcq.jsonl").open(encoding="utf-8")
    ]
    return [
        {
            "uuid": x["uuid"],
            "source_id": x["source_id"],
            "question": x["question"],
            "tools": json.loads(x["tools"]) if isinstance(x["tools"], str) else x["tools"],
            "correct_answer": x["correct_answer"],
        }
        for x in items
        if x.get("tools")
    ]


def main() -> None:
    rng = random.Random(SEED)

    lcb = load_lcb()
    ids = sorted(lcb)
    sel_lcb = rng.sample(ids, BUCKETS["lcb"])
    lcb_set = [
        {
            "id": f"lcb-{i:03d}",
            "question_id": qid,
            "title": lcb[qid]["title"],
            "question": lcb[qid]["question"],
            "starter_code": lcb[qid]["starter_code"],
            "difficulty": lcb[qid]["difficulty"],
            "tests": lcb[qid]["tests"],
        }
        for i, qid in enumerate(sel_lcb, 1)
    ]

    bfcl_pool = load_bfcl()
    bfcl_set = []
    n = 0
    for src in bfcl_pool:
        pool = src["items"]
        chosen = rng.sample(pool, min(src["take"], len(pool)))
        for item in chosen:
            n += 1
            bfcl_set.append(
                {
                    "id": f"bfcl-{n:03d}",
                    "source": src["source"],
                    "item_id": item["id"],
                    "question": item["question"],
                    "functions": item["function"],
                    "ground_truth": src["ground_truth"].get(item["id"]),
                }
            )

    w2c_pool = load_w2c()
    sel_w2c = rng.sample(w2c_pool, BUCKETS["when2call"])
    w2c_set = [
        {
            "id": f"w2c-{i:03d}",
            "uuid": item["uuid"],
            "question": item["question"],
            "tools": item["tools"],
            "correct_answer": item["correct_answer"],
        }
        for i, item in enumerate(sel_w2c, 1)
    ]

    holdout_set = [{"id": f"holdout-{i:03d}"} for i in range(1, BUCKETS["holdout"] + 1)]

    frozen = {
        "schema": "baseline_v1",
        "seed": SEED,
        "created": "2026-09-28",
        "buckets": {
            "lcb": {"n": len(lcb_set), "max_tokens": 3072},
            "bfcl": {"n": len(bfcl_set), "max_tokens": 512},
            "when2call": {"n": len(w2c_set), "max_tokens": 64},
            "holdout": {"n": len(holdout_set), "max_tokens": 256, "max_turns": 6},
        },
        "items": {
            "lcb": lcb_set,
            "bfcl": bfcl_set,
            "when2call": w2c_set,
            "holdout": holdout_set,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(frozen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()