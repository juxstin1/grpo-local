"""Family mhqa_t: multi-hop QA over a knowledge base she must look up with tools.

Same invented world and chains as mhqa (evals/datasets/synth_multihop.py, imported,
never run bare), but the facts are NOT in the prompt: each seed writes the world as
one file per entity under kb/ (people/, companies/, cities/), and every file holds
only that entity's own facts, so each hop costs at least one tool call. Near-name
distractor entities (same first name, same company stem, extended city name) sit
next to the real files, so a sloppy read or a guessed path gives a wrong chain.

Grading is grade_mhqa from evals/graders.py (full chain, every hop normalized-exact),
on her final ANSWER line. ground_truth = the chain joined by " -> ".

Hops (difficulty knob), one item per hop per seed:
  3  person -> company -> city -> river
  5  person -> mentor -> company -> founder -> city -> river
  7  person -> mentor x3 -> company -> founder -> city -> river
A competent episode needs one read per hop plus the answer (h7 >= 8 turns).
"""

from __future__ import annotations

import importlib.util
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # grpo-local
_DS = str(ROOT / "evals" / "datasets")
if _DS not in sys.path:
    sys.path.append(_DS)                               # append: never shadow gym modules
import synth_multihop as mh  # noqa: E402

# evals/graders.py shares a module name with sandbox-gym/graders.py: load by path.
_spec = importlib.util.spec_from_file_location("evals_graders", ROOT / "evals" / "graders.py")
evals_graders = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evals_graders)

FAMILY = "mhqa_t"
HOPS = (3, 5, 7)
WORLD = dict(n_people=24, n_companies=16, n_cities=20, n_rivers=20)
TOOLS = ["list_dir", "read_file", "search"]
SEP = " -> "


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _deep(w, people, seed, hops: int) -> list[dict]:
    """hops >= 6: p -> mentor x (hops-4) -> company -> founder -> city -> river.
    Same rule as handoff-gym/common._deep_hop (distinct answers, no repeated
    person on the mentor run); not imported because that module rewrites sys.path."""
    E, F, B, R, M = w["employer"], w["founder"], w["born"], w["river"], w["mentor"]
    d = hops - 4

    def fn(p):
        c = [p]
        for _ in range(d):
            c.append(M[c[-1]])
        co = E[c[-1]]
        return c + [co, F[co], B[F[co]], R[B[F[co]]]]

    rng = random.Random(seed + hops)
    seen, picked = set(), []
    for p in rng.sample(people, len(people)):
        ch = fn(p)
        if len(set(ch[: d + 1])) == d + 1 and ch[-1] not in seen:
            seen.add(ch[-1])
            picked.append(p)
    whose = "{p}'s mentor"
    for _ in range(d - 1):
        whose = "the mentor of " + whose
    q = ("Which river flows through the birth city of the founder of the company that "
         f"{whose} works for?")
    kinds = ["person"] + ["mentor"] * d + ["company", "founder", "city", "river"]
    fmt = " -> ".join(f"<{k}>" for k in kinds)
    return [{"hops": hops, "person": p, "chain": fn(p),
             "q": q.format(p=p) + f"\nGive the full chain on the last line: ANSWER: {fmt}"}
            for p in picked]


def world(seed: int):
    """(w, people, distractors). Distractors: name -> facts, never on any chain."""
    if seed == mh.SEED:
        raise ValueError("that seed is the mhqa eval world")
    w, people, _ = mh.build_world(seed, **WORLD)
    rng = random.Random(seed * 7919 + 1)
    companies, cities = sorted(w["hq"]), sorted(w["river"])
    rivers = sorted(set(w["river"].values()))
    taken = set(people) | set(companies) | set(cities)
    syl = mh._SYL

    def fresh(make):
        while True:
            n = make()
            if n not in taken:
                taken.add(n)
                return n

    dp = {}
    for p in rng.sample(people, len(people) // 3):     # same first name, other surname
        n = fresh(lambda: f"{p.split()[0]} {''.join(rng.choice(syl) for _ in range(3)).capitalize()}")
        dp[n] = {"Employer": rng.choice(companies), "Mentor": rng.choice(people),
                 "Born in": rng.choice(cities)}
    dc = {}
    for c in rng.sample(companies, len(companies) // 3):   # same stem, other suffix
        stem, suf = c.rsplit(" ", 1)
        n = fresh(lambda: f"{stem} {rng.choice([s for s in ('Works', 'Labs', 'Foundry', 'Group', 'Holdings') if s != suf])}")
        dc[n] = {"Headquarters": rng.choice(cities), "Founded by": rng.choice(people)}
    dt = {}
    for c in rng.sample(cities, len(cities) // 4):         # extended city name
        n = fresh(lambda: c + rng.choice(syl))
        dt[n] = {"River": rng.choice(rivers)}
    return w, people, {"person": dp, "company": dc, "city": dt}


def kb_files(seed: int) -> dict[str, str]:
    w, people, dis = world(seed)
    rng = random.Random(seed * 104729 + 3)
    files: dict[str, str] = {}

    def put(kind: str, name: str, facts: list[tuple[str, str]]):
        rng.shuffle(facts)
        body = f"# {kind.capitalize()}: {name}\n\n" + "".join(f"- {k}: {v}\n" for k, v in facts)
        path = f"kb/{_KIND_DIR[kind]}/{slug(name)}.md"
        if path in files:
            raise AssertionError(f"seed {seed}: two entities share {path}")
        files[path] = body

    for p in people:
        f = [("Employer", w["employer"][p]), ("Mentor", w["mentor"][p]), ("Born in", w["born"][p])]
        if rng.random() < 0.5:                              # decoys, as in mhqa
            f.append(("Visited on business", rng.choice(sorted(w["river"]))))
            f.append(("Admires", rng.choice(people)))
        put("person", p, f)
    for c in w["hq"]:
        put("company", c, [("Headquarters", w["hq"][c]), ("Founded by", w["founder"][c])])
    for c in w["river"]:
        put("city", c, [("River", w["river"][c])])
    for kind, d in dis.items():
        for n, facts in d.items():
            put(kind, n, list(facts.items()))
    files["kb/README.md"] = ("# Knowledge base\n\nOne file per entity: kb/people/, kb/companies/, "
                             "kb/cities/. File names are the entity name in lower case with "
                             "underscores. Each file lists only that entity's own facts.\n")
    return files


def questions(seed: int) -> list[dict]:
    w, people, _ = world(seed)
    out = []
    for h in HOPS:
        qs = (mh.make_questions(w, people, 1, seed, [h]) if h <= 5 else _deep(w, people, seed, h))
        out.append(qs[0])
    return out


PREAMBLE = ("Answer the question from the knowledge base in kb/ (one file per person, "
            "company and city). The facts are only in those files: look each one up "
            "with the tools.\n\nQuestion: ")


def specs(seed: int) -> list[dict]:
    files = kb_files(seed)
    return [{"family": FAMILY, "seed": seed, "files": files, "tools": list(TOOLS),
             "name": f"mhqat-h{q['hops']}-{seed}",
             "hops": q["hops"],
             "ground_truth": SEP.join(q["chain"]),
             "prompt": PREAMBLE + q["q"]}
            for q in questions(seed)]


def grade(answer: str | None, ground_truth: str) -> tuple[bool, str]:
    """evals/graders.grade_mhqa on her final ANSWER line (full chain)."""
    if answer is None:
        return False, "no-answer-line"
    chain = ground_truth.split(SEP)
    record = {"answer": {"answer": chain[-1], "aliases": [], "chain": chain}}
    return evals_graders.grade_mhqa(record, f"ANSWER: {answer}")


# --- oracle (tests / sealing): walk the files exactly as a careful agent would ---

_KIND_DIR = {"person": "people", "mentor": "people", "founder": "people",
             "company": "companies", "city": "cities"}
_FIELD = {("person", "mentor"): "Mentor", ("mentor", "mentor"): "Mentor",
          ("person", "company"): "Employer", ("mentor", "company"): "Employer",
          ("company", "city"): "Headquarters", ("company", "founder"): "Founded by",
          ("founder", "city"): "Born in", ("city", "river"): "River"}


def _kinds(hops: int) -> list[str]:
    return {3: ["person", "company", "city", "river"],
            5: ["person", "mentor", "company", "founder", "city", "river"]}.get(
        hops, ["person"] + ["mentor"] * (hops - 4) + ["company", "founder", "city", "river"])


def oracle(spec: dict) -> tuple[list[str], int]:
    """(chain, files read). Reads one file per hop, by slug, field by name."""
    files, kinds = spec["files"], _kinds(spec["hops"])
    cur = spec["ground_truth"].split(SEP)[0]            # the subject named in the question
    if cur not in spec["prompt"]:
        raise AssertionError("subject not in prompt")
    chain, reads = [cur], 0
    for a, b in zip(kinds, kinds[1:]):
        text = files[f"kb/{_KIND_DIR[a]}/{slug(cur)}.md"]
        reads += 1
        m = re.search(rf"^- {re.escape(_FIELD[(a, b)])}: (.+)$", text, re.M)
        cur = m.group(1).strip()
        chain.append(cur)
    return chain, reads
