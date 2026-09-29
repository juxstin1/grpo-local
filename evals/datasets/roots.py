"""Root-task clustering: a problem and every variant of it get one root_id.

Two kinds of edges, merged with union-find:
  1. lineage keys  - explicit identity carried by the data (contest+problem#,
                     arXiv id, LeetCode slug minus I/II suffix, MuSiQue
                     single-hop ids). Records sharing any key are one task.
  2. skeleton dups - text with numbers, single-letter variables and declared
                     identifiers masked, compared on word n-gram shingles.
                     Catches renamed-variable / reworded copies that carry no
                     shared id. MinHash-LSH proposes pairs, exact Jaccard
                     confirms.

Over-merging is the safe failure: a bigger cluster just moves more items to
one side of the split. root_id = the smallest lineage key in the cluster, so
it is stable across re-runs on the same data.
"""

from __future__ import annotations

import html
import re
import zlib
from collections import defaultdict

import numpy as np

NUM_PERM = 64
BANDS, ROWS = 16, 4           # LSH candidate threshold ~ (1/16)^(1/4) = 0.5
MAX_BUCKET = 400              # a bucket this big is template noise, not a variant family
_P = np.uint64(4294967291)    # prime > 2^32
_rng = np.random.default_rng(20260302)
_A = _rng.integers(1, 2**31 - 1, NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, 2**31 - 1, NUM_PERM, dtype=np.uint64)


class UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if rb < ra:
            ra, rb = rb, ra
        self.p[rb] = ra
        return True


def skeleton(text: str, idents: set[str] = frozenset()) -> list[str]:
    t = html.unescape(re.sub(r"<[^>]+>", " ", text or "")).lower()
    t = re.sub(r"\\[a-z]+", " ", t)                       # latex commands
    out = []
    for tok in re.findall(r"[a-z_][a-z0-9_]*|\d+(?:\.\d+)?", t):
        if tok[0].isdigit():
            out.append("#")
        elif len(tok) == 1 or "_" in tok or any(c.isdigit() for c in tok) or tok in idents:
            out.append("$v")
        else:
            out.append(tok)
    return out


def shingles(tokens: list[str], k: int) -> set[int]:
    if len(tokens) < k:
        return {zlib.crc32(" ".join(tokens).encode())} if tokens else set()
    return {zlib.crc32(" ".join(tokens[i : i + k]).encode()) for i in range(len(tokens) - k + 1)}


def minhash(sh: set[int]) -> np.ndarray:
    h = np.fromiter(sh, dtype=np.uint64, count=len(sh))
    return ((np.outer(_A, h) + _B[:, None]) % _P).min(axis=1)


def cluster(records: list[dict], k: int, tau: float, log) -> dict[str, str]:
    """records need: uid, lineage (list[str]), skel_text, idents (optional).
    Returns uid -> root_id."""
    uf = UnionFind()
    key_owner: dict[str, str] = {}
    for r in records:
        uf.find(r["uid"])
        for key in r["lineage"]:
            if key in key_owner:
                uf.union(r["uid"], key_owner[key])
            else:
                key_owner[key] = r["uid"]
    lineage_merges = len(records) - len({uf.find(r["uid"]) for r in records})

    sh = {r["uid"]: shingles(skeleton(r["skel_text"], set(r.get("idents", ()))), k) for r in records}
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for uid, s in sh.items():
        if not s:
            continue
        sig = minhash(s)
        for b in range(BANDS):
            buckets[(b, sig[b * ROWS : (b + 1) * ROWS].tobytes())].append(uid)
    tested: set[tuple[str, str]] = set()
    skel_edges = skipped = 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        if len(members) > MAX_BUCKET:
            skipped += 1
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = sorted((members[i], members[j]))
                if (a, b) in tested:
                    continue
                tested.add((a, b))
                sa, sb = sh[a], sh[b]
                if len(sa & sb) / len(sa | sb) >= tau:
                    skel_edges += 1
                    uf.union(a, b)

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[uf.find(r["uid"])].append(r)
    root_of = {}
    for members in groups.values():
        rid = min(key for m in members for key in (m["lineage"] or [m["uid"]]))
        for m in members:
            root_of[m["uid"]] = rid
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    log.info("  clustering: %d records -> %d root tasks | lineage merges %d | skeleton pairs tested %d, "
             "merged %d (tau %.2f, k %d) | oversized LSH buckets skipped %d | largest roots %s",
             len(records), len(groups), lineage_merges, len(tested), skel_edges, tau, k, skipped, sizes[:5])
    return root_of
