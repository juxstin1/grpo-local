"""Seal a holdout set by ROOT TASK before any training touches it, and prove it stayed clean.

  python seal.py seal   HOLDOUT.jsonl --name math-v1 --min-date 2026-03-02 [--root-field root_id] [--role eval]
  python seal.py verify sealed/math-v1
  python seal.py check  TRAIN.jsonl [...] --against sealed/math-v1 [--root-field root_id]

Leakage is defined by root task, not by text. Every record carries a root id naming the original task it
descends from (e.g. aime:2025:i:1, cf:2034:e, lc:two-sum) -- rewrites, variants and cross-dataset copies
share it. Root ids are canonicalized (lowercase, separators collapsed to ':') so both sides compare equal.

--role : eval (default) requires every item first_published >= min-date (post-cutoff).
         train-dist requires every item < min-date: an in-distribution overfit/leak detector for datasets
         with no post-cutoff items. It is never an eval and is reported separately. Role is in the manifest.
seal   : groups items by root, stamps a canary GUID into each item, writes holdout.jsonl + manifest.json
         (file sha256, per-root hashes, merkle root over roots) + roots.json, marks files read-only,
         appends a hash-chained entry to SEALS.log.
verify : recomputes every hash; nonzero exit if the holdout, manifest, or ledger chain changed.
check  : leak gate for training files. Fails if any record's root is sealed, any record has NO root
         (fail closed), or the canary appears. Result + train file sha256 appended to SEALS.log.
Stdlib only.
"""
import argparse, datetime as dt, hashlib, json, os, re, stat, sys, uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "SEALS.log"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def root_of(r, field):
    v = r.get(field)
    if v is None or str(v).strip() == "":
        return None
    return re.sub(r"[^a-z0-9]+", ":", str(v).strip().lower()).strip(":")


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def merkle(hashes):
    layer = sorted(hashes)
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [sha((layer[i] + layer[i + 1]).encode()) for i in range(0, len(layer), 2)]
    return layer[0] if layer else sha(b"")


def root_hashes(items, field):
    groups = {}
    for r in items:
        groups.setdefault(root_of(r, field), []).append(sha(canon(r)))
    return {k: sha(canon([k, sorted(v)])) for k, v in groups.items()}


def ledger_append(entry):
    prev = "0" * 64
    if LEDGER.exists():
        lines = [l for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            prev = json.loads(lines[-1])["entry_hash"]
    entry = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "prev": prev, **entry}
    entry["entry_hash"] = sha(canon(entry))
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def ledger_entries():
    if not LEDGER.exists():
        return [], True
    out, prev = [], "0" * 64
    for n, line in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        e = json.loads(line)
        h = e.pop("entry_hash")
        if e["prev"] != prev or sha(canon(e)) != h:
            print(f"LEDGER BROKEN at line {n}")
            return out, False
        e["entry_hash"] = prev = h
        out.append(e)
    return out, True


def cmd_seal(a):
    out = HERE / "sealed" / a.name
    if out.exists():
        sys.exit(f"{out} already sealed; seals are immutable -- pick a new --name")
    canary = f"HOLDOUT-CANARY {uuid.uuid4()} DO NOT TRAIN"
    items = []
    for i, r in enumerate(read_jsonl(a.holdout)):
        if root_of(r, a.root_field) is None:
            sys.exit(f"record {i} has no '{a.root_field}' -- every holdout item needs a root task id")
        pub = str(r.get("first_published", ""))[:10]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", pub):
            sys.exit(f"record {i} first_published={pub or 'missing'} -- every holdout item needs a date")
        if a.role == "eval" and pub < a.min_date:
            sys.exit(f"record {i} first_published={pub} -- eval holdout needs >= {a.min_date} (post-cutoff only)")
        if a.role == "train-dist" and pub >= a.min_date:
            sys.exit(f"record {i} first_published={pub} -- train-dist holdout must be < {a.min_date}")
        items.append({**r, "_canary": canary})
    rh = root_hashes(items, a.root_field)
    out.mkdir(parents=True)
    body = b"".join(canon(r) + b"\n" for r in items)
    (out / "holdout.jsonl").write_bytes(body)
    roots = {"root_ids": sorted(rh), "canary": canary}
    (out / "roots.json").write_bytes(canon(roots))
    manifest = {
        "name": a.name, "source": str(Path(a.holdout).resolve()), "root_field": a.root_field,
        "min_date": a.min_date, "role": a.role, "n_items": len(items), "n_root_tasks": len(rh), "holdout_sha256": sha(body),
        "roots_sha256": sha(canon(roots)), "root_hashes": rh, "merkle_root": merkle(rh.values()), "canary": canary,
    }
    (out / "manifest.json").write_bytes(canon(manifest))
    ms = sha(canon(manifest))
    for f in out.iterdir():
        os.chmod(f, stat.S_IREAD)
    e = ledger_append({"event": "seal", "name": a.name, "role": a.role, "manifest_sha256": ms, "merkle_root": manifest["merkle_root"],
                       "n_items": len(items), "n_root_tasks": len(rh)})
    print(f"SEALED {a.name} ({a.role}): {len(rh)} root tasks / {len(items)} items")
    print(f"  manifest_sha256 {ms}\n  merkle_root     {manifest['merkle_root']}\n  ledger_entry    {e['entry_hash']}")


def load_seal(d):
    d = Path(d)
    return json.loads((d / "manifest.json").read_bytes()), (d / "holdout.jsonl").read_bytes(), (d / "roots.json").read_bytes()


def cmd_verify(a):
    m, body, rb = load_seal(a.sealdir)
    items = [json.loads(l) for l in body.splitlines() if l.strip()]
    rh = root_hashes(items, m["root_field"])
    entries, chain_ok = ledger_entries()
    ms = sha(canon(m))
    checks = {
        "holdout_sha256": sha(body) == m["holdout_sha256"],
        "roots_sha256": sha(rb) == m["roots_sha256"],
        "root_hashes": rh == m["root_hashes"],
        "merkle_root": merkle(rh.values()) == m["merkle_root"],
        "ledger_chain": chain_ok,
        "manifest_in_ledger": any(e.get("event") == "seal" and e.get("manifest_sha256") == ms for e in entries),
    }
    for k, v in checks.items():
        print(f"  {'OK  ' if v else 'FAIL'} {k}")
    for k in sorted(set(rh) ^ set(m["root_hashes"]) | {k for k in rh if rh[k] != m["root_hashes"].get(k)}):
        print(f"  changed root: {k}")
    print(f"manifest_sha256 {ms}")
    sys.exit(0 if all(checks.values()) else 1)


def cmd_check(a):
    m, _, rb = load_seal(a.against)
    sealed = json.loads(rb)
    roots, canary = set(sealed["root_ids"]), sealed["canary"]
    rf = a.root_field or m["root_field"]
    hits, scanned, file_hashes = [], 0, {}
    for path in a.train:
        raw = Path(path).read_bytes()
        file_hashes[str(Path(path).resolve())] = sha(raw)
        if canary.encode() in raw or b"HOLDOUT-CANARY" in raw:
            hits.append((path, None, "canary present (holdout rows copied in)"))
        for ln, line in enumerate(raw.decode("utf-8").split("\n"), 1):  # not splitlines(): JSON strings may hold U+2028/0x1c
            if not line.strip():
                continue
            scanned += 1
            root = root_of(json.loads(line), rf)
            if root is None:
                hits.append((path, ln, f"no '{rf}' -- unrooted items can't be proven clean"))
            elif root in roots:
                hits.append((path, ln, f"sealed root task {root}"))
    ok = not hits
    ledger_append({"event": "check", "seal": m["name"], "manifest_sha256": sha(canon(m)),
                   "train_files": file_hashes, "records_scanned": scanned, "leaks": len(hits), "clean": ok})
    for f, ln, why in hits[:50]:
        print(f"  LEAK {f}:{ln}  {why}")
    if len(hits) > 50:
        print(f"  ... {len(hits) - 50} more")
    print(f"{'CLEAN' if ok else 'LEAKED'}: {scanned} train records vs {m['name']} ({len(hits)} hits) -- logged to SEALS.log")
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    a = s.add_parser("seal"); a.add_argument("holdout"); a.add_argument("--name", required=True)
    a.add_argument("--root-field", default="root_id")
    a.add_argument("--min-date", required=True, help="model cutoff, YYYY-MM-DD (Qwen3.5: 2026-03-02)")
    a.add_argument("--role", choices=["eval", "train-dist"], default="eval")
    v = s.add_parser("verify"); v.add_argument("sealdir")
    c = s.add_parser("check"); c.add_argument("train", nargs="+"); c.add_argument("--against", required=True)
    c.add_argument("--root-field")
    a = p.parse_args()
    {"seal": cmd_seal, "verify": cmd_verify, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
