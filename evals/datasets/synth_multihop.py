"""Synthetic multi-hop (mhqa) items from a seeded fictional world.

Vendored from llamacpp-prebuilt/tuning/fleet/multihop.py (v1 world + chains,
unchanged). The world is invented per seed, so no item can be in any model's
training data: contamination-free by construction, no dating needed.

  .venv\\Scripts\\python.exe evals\\datasets\\synth_multihop.py            # 5 eval items -> 5pb set
  .venv\\Scripts\\python.exe evals\\datasets\\synth_multihop.py --seed N --per-hop K --hops 2,3,4 --out pool.jsonl
                                                                          # GRPO tasks: any other seed

Difficulty knob = hop count (2..5). Eval uses SEED; training must use other seeds.
Graded by evals/graders.py grade_mhqa: the final "ANSWER:" line must give the FULL chain
(person -> company -> city ...), every hop normalized-exact. Headline metric is per-attempt
accuracy, reported next to the guess floor (see guess_floor()).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SET = HERE.parent / "data" / "sets" / "post_cutoff_v1_5pb.json"
SEED = 20260302          # the eval world; never train on this seed
EVAL_PLAN = {2: 3, 3: 2}  # 5 items: three 2-hop, two 3-hop
CREATED = "2026-09-28"    # world invented on this date

_SYL = ["ka", "ro", "vel", "mi", "tan", "sor", "ul", "de", "bri", "an", "qo", "zel",
        "fa", "nor", "ith", "pa", "gre", "lu", "mar", "os", "ve", "tra", "yen", "cal"]


def _name(rng, used, parts=2, cap=True):
    while True:
        n = "".join(rng.choice(_SYL) for _ in range(parts))
        n = n.capitalize() if cap else n
        if n not in used and len(n) >= 4:
            used.add(n)
            return n


WORLD = "w3"  # w3 = 12/12/16/16 world, ~1.4k-token prompt, answers graded as the FULL chain.
              # w1 (36/16/12/12, ~2.85k prompt) starved the answer; w2 (12/6/6/6) let a
              # guesser pass ~52-68% at pass@4. Both retired.


def build_world(seed, n_people=12, n_companies=12, n_cities=16, n_rivers=16):
    rng = random.Random(seed)
    used = set()
    people = [f"{_name(rng, used)} {_name(rng, used, 3)}" for _ in range(n_people)]
    companies = [f"{_name(rng, used, 3)} {rng.choice(['Works', 'Labs', 'Foundry', 'Group', 'Holdings'])}"
                 for _ in range(n_companies)]
    cities = [_name(rng, used, 3) for _ in range(n_cities)]
    rivers = [f"the {_name(rng, used)} River" for _ in range(n_rivers)]
    w = {
        "employer": {p: rng.choice(companies) for p in people},
        "mentor": {p: rng.choice([q for q in people if q != p]) for p in people},
        "born": {p: rng.choice(cities) for p in people},
        "hq": {c: rng.choice(cities) for c in companies},
        "founder": {c: rng.choice(people) for c in companies},
        "river": dict(zip(cities, rng.sample(rivers, n_cities))),
    }
    tmpl = {
        "employer": ["{a} works for {b}.", "{a} is employed at {b}.", "{b} employs {a}."],
        "mentor":   ["{a} was mentored by {b}.", "{b} is the mentor of {a}."],
        "born":     ["{a} was born in {b}.", "{a}'s birthplace is {b}."],
        "hq":       ["{a} is headquartered in {b}.", "The head office of {a} is in {b}."],
        "founder":  ["{a} was founded by {b}.", "{b} founded {a}."],
        "river":    ["{b} flows through {a}.", "The city of {a} sits on {b}."],
    }
    facts = [rng.choice(tmpl[rel]).format(a=a, b=b) for rel, m in w.items() for a, b in m.items()]
    for p in rng.sample(people, n_people // 2):          # decoys: true-looking, never on a chain
        facts.append(f"{p} once visited {rng.choice(cities)} on business.")
        facts.append(f"{p} admires the work of {rng.choice(people)}.")
    rng.shuffle(facts)
    return w, people, facts


def make_questions(w, people, per_hop, seed, hops):
    rng = random.Random(seed + 1)
    E, HQ, F, B, R, M = w["employer"], w["hq"], w["founder"], w["born"], w["river"], w["mentor"]
    # Each chain returns the full path [person, ..., answer]; the answer line must name every hop.
    chains = {
        2: (lambda p: [p, E[p], HQ[E[p]]], ["person", "company", "city"],
            "In which city is the company that {p} works for headquartered?"),
        3: (lambda p: [p, E[p], HQ[E[p]], R[HQ[E[p]]]], ["person", "company", "city", "river"],
            "Which river flows through the city where {p}'s employer is headquartered?"),
        4: (lambda p: [p, E[p], F[E[p]], B[F[E[p]]], R[B[F[E[p]]]]],
            ["person", "company", "founder", "city", "river"],
            "Which river flows through the birth city of the founder of the company {p} works for?"),
        5: (lambda p: [p, M[p], E[M[p]], F[E[M[p]]], B[F[E[M[p]]]], R[B[F[E[M[p]]]]]],
            ["person", "mentor", "company", "founder", "city", "river"],
            "Which river flows through the birth city of the founder of the company that "
            "{p}'s mentor works for?"),
    }
    qs = []
    for h in hops:
        fn, kinds, t = chains[h]
        seen, picked = set(), []                         # distinct answers: no "guess the common river"
        for p in rng.sample(people, len(people)):
            if fn(p)[-1] not in seen:
                seen.add(fn(p)[-1]); picked.append(p)
        picked += [p for p in people if p not in picked]
        fmt = " -> ".join(f"<{k}>" for k in kinds)
        for p in picked[:per_hop]:
            q = t.format(p=p) + f"\nGive the full chain on the last line: ANSWER: {fmt}"
            qs.append({"hops": h, "q": q, "chain": fn(p), "expected": fn(p)[-1], "person": p})
    return qs


def guess_floor(w: dict, hops: int) -> dict:
    """Per-attempt probability a non-reasoner passes full-chain grading.
    blind:  names every hop independently at random from its entity type.
    follow: picks a random first hop, then follows the facts (the chain after
            hop 1 is determined) - the best cheap strategy, so it's the floor."""
    n = {"person": len(w["mentor"]), "company": len(w["hq"]), "city": len(w["river"]),
         "river": len(set(w["river"].values()))}
    kinds = {2: ["company", "city"], 3: ["company", "city", "river"],
             4: ["company", "person", "city", "river"], 5: ["person", "company", "person", "city", "river"]}[hops]
    blind = 1.0
    for k in kinds:
        blind /= n[k]
    return {"blind": blind, "follow": 1 / n[kinds[0]]}


def records(seed: int, plan: dict[int, int]) -> list[dict]:
    w, people, facts = build_world(seed)
    out, used = [], set()
    for h, n in plan.items():
        # one person per item: items sharing a person share a chain prefix (same root task)
        qs = [q for q in make_questions(w, people, len(people), seed, [h]) if q["person"] not in used][:n]
        used |= {q["person"] for q in qs}
        for q in qs:
            ans = q["expected"]
            aliases = [ans[4:]] if ans.startswith("the ") else []   # "the X River" -> "X River"
            out.append({
                "uid": f"mhqa:synth:{WORLD}:{seed}:h{h}:{q['person'].replace(' ', '_')}",
                "dataset": "mhqa", "source": f"synth_multihop {WORLD} world seed={seed}",
                "root_id": f"mhqa:synth:{WORLD}:{seed}:{q['person'].replace(' ', '_')}",
                "first_published": CREATED, "date_basis": "invented world (contamination-free)",
                "hops": h,
                "prompt": "Facts:\n" + "\n".join(facts) + f"\n\nQuestion: {q['q']}",
                "answer": {"answer": ans, "aliases": aliases, "chain": q["chain"]},
                "guess_floor": guess_floor(w, h),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--per-hop", type=int)
    ap.add_argument("--hops", default="2,3")
    ap.add_argument("--out", help="write a JSONL pool instead of the eval set")
    a = ap.parse_args()
    if a.out:
        if a.seed == SEED:
            sys.exit(f"seed {SEED} is the eval world; use another seed for training pools")
        recs = records(a.seed, {int(h): a.per_hop or 5 for h in a.hops.split(",")})
        Path(a.out).write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode())
        print(f"wrote {len(recs)} items -> {a.out}")
        return
    recs = records(SEED, EVAL_PLAN)
    s = json.loads(SET.read_bytes())
    s["items"]["mhqa"] = [{**r, "id": r["uid"]} for r in recs]
    s["buckets"]["mhqa"] = {"n": len(recs), "max_tokens": 2048}
    SET.write_bytes(json.dumps(s, ensure_ascii=False, indent=1).encode())
    for r in recs:
        print(f"h{r['hops']}  {r['prompt'].rsplit('Question: ', 1)[1]}  ->  {r['answer']['answer']}")
    print({k: v["n"] for k, v in s["buckets"].items()})


if __name__ == "__main__":
    main()
