"""Family doc_make: write a document (Markdown or PDF) from notes + a CSV in the workspace.

Each seed invents one project world and writes it as notes/{overview,finance,team,schedule}.md
plus data.csv. The prompt asks for a report with exact section headings and named facts
that she has to look up. Facts sit in different files and in mixed formats (7,000 vs 7000,
"14 March 2027" vs 2027-03-14, "A, B and C" vs bullets). Near-miss plants in the notes
(a sibling project with a similar name and different numbers, a superseded draft figure)
must NOT appear in the output.

Tiers (one item per tier per seed):
  t1  report.md   3 sections, 5 facts                      ~8 turns (list, 4-5 reads, write)
  t2  report.pdf  4 sections, 8 facts, via make_pdf         ~8 turns
  t3  report.pdf  6 sections, 12 facts + a table of numbers from data.csv with a Total row

Mechanical grader (no LLM), on the END STATE of the workspace plus an ANSWER line:
  1. file exists, valid (PDF parses with pypdf and has text; markdown non-empty UTF-8)
  2. every required heading is present as its own line, in order (case/numbering tolerant)
  3. every required fact is present, tolerant: numbers by value (7,000 = 7000 = $7,000),
     dates in common formats, names case-insensitive, lists as sets of items
  4. table (t3): every required row has its label and all its numbers on one line
  5. no planted near-miss value appears (sibling project's and superseded figures)
  6. extracted text length inside [min_chars, max_chars] (blocks "paste everything")
ground_truth is a JSON string holding all of the above, so the sealed holdout is
self-contained.

Split rule (root = world): holdout seeds 9401-9405 draw project/org/people/vendor names
from the *_H pools, train seeds from the *_T pools; pools are disjoint. Seeds 9400-9449
are reserved: never train seeds. Suggested train seeds: 5000-5999.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

FAMILY = "doc_make"
HOLDOUT_SEEDS = tuple(range(9401, 9406))
RESERVED = range(9400, 9450)
TOOLS_T1 = ["list_dir", "read_file", "search", "write_file"]
TOOLS_PDF = ["list_dir", "read_file", "search", "make_pdf"]

PAIRS_T = [("Alder", "Alden"), ("Cinder", "Cinder Ridge"), ("Fenwick", "Fenwood"),
           ("Granite", "Granita"), ("Juniper", "Junipero"), ("Lantern", "Lanternfish"),
           ("Meridian", "Meridiem"), ("Quartz", "Quarto"), ("Sable", "Sabine"),
           ("Thistle", "Thistledown"), ("Vesper", "Vesperal"), ("Willow", "Willowbank")]
PAIRS_H = [("Marlow", "Marlowe"), ("Tessel", "Tessen"), ("Corvid", "Corvina"),
           ("Harrow", "Harrod"), ("Ostrel", "Ostrava"), ("Pylon", "Pyland")]
ORGS_T = ["Northgate Freight", "Bluepeak Labs", "Copperline Foods", "Driftwood Media",
          "Evergreen Rail", "Foxhall Energy", "Greywater Textiles", "Ironbridge Health"]
ORGS_H = ["Kestrel Logistics", "Lowmoor Analytics", "Mistral Grain Co", "Oakhurst Devices",
          "Pinecrest Marine", "Quillon Press"]
FIRST = ["Elena", "Marcus", "Priya", "Tomas", "Naomi", "Dmitri", "Sofia", "Kwame", "Ingrid",
         "Rafael", "Mei", "Callum", "Amara", "Jonas", "Leila", "Owen"]
LAST_T = ["Rossi", "Okafor", "Lindqvist", "Hartmann", "Castillo", "Nakamura", "Abara",
          "Whitcombe", "Duarte", "Sorensen", "Petrov", "Beaumont", "Achterberg", "Volkov"]
LAST_H = ["Brightwell", "Eskola", "Fontaine", "Gallagher", "Hollis", "Imanov", "Jovanovic",
          "Kowalczyk", "Ledesma", "Mbeki", "Novak", "Quirke"]
VENDORS_T = ["Tallis Cloud", "Brevik Systems", "Oakmere Print", "Quanta Supply", "Halden Audit",
             "Corvale Security", "Ridgeline Staffing", "Sunmark Courier", "Pellam Design",
             "Ardent Testing", "Wexford Hosting", "Lumen Surveys"]
VENDORS_H = ["Zephyr Cloudworks", "Marrow Analytics", "Nordhaven Freight", "Ostwick Labs",
             "Pallisade Legal", "Rexford Print", "Sable Courier", "Tindall Testing",
             "Umber Hosting", "Vantor Audit"]
ROLES = ["engineering lead", "finance controller", "operations manager", "design lead",
         "quality manager", "procurement officer", "data analyst", "field coordinator"]
RISKS = ["supplier delay", "data migration slip", "key staff turnover", "regulatory review",
         "cloud cost overrun", "integration defects", "vendor lock-in", "training shortfall",
         "permit backlog", "scope creep"]
MILESTONES = ["design freeze", "pilot go-live", "security audit", "staff training complete",
              "vendor contract signed", "data cutover", "public beta", "final acceptance"]
REGIONS = ["North", "South", "East", "West", "Central", "Coastal", "Highland", "Metro"]

SECTIONS = {
    1: ["Overview", "Budget", "Schedule"],
    2: ["Summary", "Team and Vendors", "Budget and Schedule", "Risks"],
    3: ["Summary", "Team", "Vendors", "Budget", "Schedule and Risks", "Regional Results"],
}
BOUNDS = {1: (150, 3500), 2: (350, 4500), 3: (600, 6000)}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


# --- normalization / matching ----------------------------------------------------

def _words(s: str) -> str:
    """casefold, everything non-alphanumeric -> single space (for names/phrases/headings)."""
    return " " + re.sub(r"[^a-z0-9]+", " ", s.casefold()).strip() + " "


def numbers_in(text: str) -> set[float]:
    """Every numeric token by value; thousands commas folded (7,000 -> 7000)."""
    out: set[float] = set()
    for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        tok = tok.rstrip(",")
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", tok):
            out.add(float(tok.replace(",", "")))
        else:
            for part in tok.split(","):
                if part:
                    out.add(float(part))
    return out


def date_forms(iso: str) -> list[str]:
    d = date.fromisoformat(iso)
    mon, ab = MONTHS[d.month - 1], MONTHS[d.month - 1][:3]
    forms = [f"{d.day} {mon} {d.year}", f"{mon} {d.day} {d.year}", f"{d.day} {ab} {d.year}",
             f"{ab} {d.day} {d.year}", f"{d.year} {d.month:02d} {d.day:02d}",
             f"{d.year} {d.month} {d.day}"]
    return [_words(f) for f in forms]


def _date_text(text: str) -> str:
    """normalize ordinals ('14th') then words-normalize; ISO dashes become spaces."""
    return _words(re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text, flags=re.I))


def _has_phrase(norm: str, phrase: str) -> bool:
    return _words(phrase) in norm


def fact_present(fact: dict, text: str, norm: str, nums: set[float], dnorm: str) -> bool:
    k = fact["kind"]
    if k == "num":
        return float(fact["value"]) in nums
    if k == "date":
        return any(f in dnorm for f in date_forms(fact["value"]))
    if k == "text":
        v = fact["value"]
        parts = v.split()
        alts = [v] + ([f"{parts[-1]} {' '.join(parts[:-1])}"] if len(parts) > 1 else [])
        return any(_has_phrase(norm, a) for a in alts)
    if k == "list":
        return all(_has_phrase(norm, it) for it in fact["items"])
    raise ValueError(k)


def forbidden_hit(item: dict, norm: str, nums: set[float], dnorm: str) -> bool:
    k = item["kind"]
    if k == "num":
        return float(item["value"]) in nums
    if k == "date":
        return any(f in dnorm for f in date_forms(item["value"]))
    return _has_phrase(norm, item["value"])


# --- file extraction + grading ------------------------------------------------------

def extract_text(path: Path, fmt: str) -> tuple[str | None, str]:
    """(text, error). PDF must parse with pypdf and yield text; md must be non-empty UTF-8."""
    if not path.is_file():
        return None, "missing-file"
    data = path.read_bytes()
    if not data.strip():
        return None, "empty-file"
    if fmt == "md":
        try:
            return data.decode("utf-8"), ""
        except UnicodeDecodeError:
            return None, "not-utf8"
    try:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        if len(r.pages) < 1:
            return None, "pdf-no-pages"
        text = "\n".join((p.extract_text() or "") for p in r.pages)
    except Exception as e:  # noqa: BLE001
        return None, f"pdf-invalid:{type(e).__name__}"
    return (text, "") if text.strip() else (None, "pdf-no-text")


def _heading_line(line: str) -> str:
    return _words(re.sub(r"^[\s#*>\-]*(?:\d+[.)]\s+)?", "", line))


def headings_in_order(text: str, headings: list[str]) -> tuple[bool, str]:
    want = [_words(h) for h in headings]
    i = 0
    for ln in text.split("\n"):
        if i < len(want) and _heading_line(ln) == want[i]:
            i += 1
    return (i == len(want), "" if i == len(want) else f"heading:{headings[i]}")


def grade_doc(root: Path, truth: dict) -> tuple[bool, str]:
    text, err = extract_text(Path(root) / truth["path"], truth["format"])
    if text is None:
        return False, err
    ok, why = headings_in_order(text, truth["headings"])
    if not ok:
        return False, why
    norm, dnorm, nums = _words(text), _date_text(text), numbers_in(text)
    for f in truth["facts"]:
        if not fact_present(f, text, norm, nums, dnorm):
            return False, f"fact:{f['label']}"
    tb = truth.get("table")
    if tb:
        lines = [(_words(ln), numbers_in(ln)) for ln in text.split(chr(10))]
        labs = [_words(str(r[0])) for r in tb["rows"]]
        ncol = len(tb["header"]) - 1
        for row, lab in zip(tb["rows"], labs):
            need = {float(x) for x in row[1:]}
            ok = False
            for i, (w, n) in enumerate(lines):
                if lab not in w:
                    continue
                got = set(n)                     # row on one line, or one cell per line (pypdf)
                for w2, n2 in lines[i + 1:i + 1 + ncol]:
                    if any(o in w2 for o in labs if o != lab):
                        break
                    got |= n2
                if need <= got:
                    ok = True
                    break
            if not ok:
                return False, f"table-row:{row[0]}"
    for item in truth["forbidden"]:
        if forbidden_hit(item, norm, nums, dnorm):
            return False, f"near-miss:{item['label']}"
    lo, hi = truth["min_chars"], truth["max_chars"]
    n = len(re.sub(r"\s+", " ", text).strip())
    if not lo <= n <= hi:
        return False, f"length:{n} not in [{lo},{hi}]"
    return True, "ok"


def reward_fn(task):
    """Task.reward for doc_make: end-state file check + a final ANSWER line."""
    from graders import last_answer, truncated_final
    truth = json.loads(task.ground_truth)

    def rw(turns, ws):
        if truncated_final(turns) or last_answer(turns) is None or ws is None:
            return 0.0
        return float(grade_doc(ws.root, truth)[0])
    return rw


# --- world generation -----------------------------------------------------------------

def _money(n: int) -> str:
    return f"${n:,}"


def _fmt_date(d: date, rng: random.Random) -> str:
    mon = MONTHS[d.month - 1]
    return rng.choice([f"{d.day} {mon} {d.year}", f"{mon} {d.day}, {d.year}", d.isoformat(),
                       f"{mon[:3]} {d.day}, {d.year}"])


def _fmt_money(n: int, rng: random.Random) -> str:
    return rng.choice([f"${n:,}", f"{n} USD", f"USD {n:,}", f"${n}"])


def _list_text(items: list[str], rng: random.Random) -> str:
    style = rng.choice(["and", "semi", "bullets"])
    if style == "and":
        return ", ".join(items[:-1]) + " and " + items[-1]
    if style == "semi":
        return "; ".join(items)
    return "\n" + "\n".join(f"- {i}" for i in items)


def _side(seed: int) -> str:
    if seed in HOLDOUT_SEEDS:
        return "holdout"
    if seed in RESERVED:
        raise ValueError(f"seed {seed} is reserved for doc_make holdouts")
    return "train"


def world(seed: int) -> dict:
    side = _side(seed)
    P = (PAIRS_H, ORGS_H, LAST_H, VENDORS_H) if side == "holdout" else (PAIRS_T, ORGS_T, LAST_T, VENDORS_T)
    pairs, orgs, lasts, vendors = P
    rng = random.Random(seed * 7919 + 13)
    for _ in range(200):
        w = _world(rng, pairs, orgs, lasts, vendors)
        if w:
            w["seed"], w["side"] = seed, side
            return w
    raise AssertionError(f"seed {seed}: no collision-free world")


def _world(rng, pairs, orgs, lasts, vendors) -> dict | None:
    project, sibling = rng.choice(pairs)
    org = rng.choice(orgs)
    names = [f"{f} {l}" for f, l in zip(rng.sample(FIRST, 6), rng.sample(lasts, 6))]
    lead, sponsor, sib_lead = names[0], names[1], names[2]
    team = list(zip(names[3:6], rng.sample(ROLES, 3)))
    vend = rng.sample(vendors, 3)
    base = date(2027, 1, 1) + timedelta(days=rng.randrange(0, 300))
    launch, sib_launch = base + timedelta(days=130), base + timedelta(days=rng.choice([-21, 14, 35]))
    ms_names = rng.sample(MILESTONES, 3)
    ms = [(n, base + timedelta(days=20 + 35 * i + rng.randrange(0, 6))) for i, n in enumerate(ms_names)]
    budget = rng.randrange(80, 900) * 500
    reserve = round(budget * rng.choice([0.08, 0.1, 0.12]) / 100) * 100
    draft = budget - rng.choice([-1, 1]) * rng.randrange(20, 90) * 500
    sib_budget = budget + rng.choice([-1, 1]) * rng.randrange(10, 60) * 500
    headcount = rng.randrange(120, 900)
    risks = rng.sample(RISKS, 3)
    risk_owner = rng.choice(names[3:6])
    regions = rng.sample(REGIONS, 6)
    table = [(r, rng.randrange(8, 90) * 100, rng.randrange(20, 900) * 1000, rng.randrange(10, 600) * 1000)
             for r in regions]
    # collision guard: near-miss numbers must not equal anything she legitimately writes
    legit = {budget, reserve, headcount, launch.year, *(m[1].year for m in ms)}
    for r in table:
        legit |= set(r[1:])
    for j in range(1, 4):
        legit |= {sum(r[j] for r in table[:4])}
    if len({budget, draft, sib_budget}) < 3 or {draft, sib_budget} & legit:
        return None
    if len({lead, sponsor, sib_lead, risk_owner}) < 4 or {sib_launch} & {launch, *(m[1] for m in ms)}:
        return None
    return dict(project=project, sibling=sibling, org=org, lead=lead, sponsor=sponsor,
                sib_lead=sib_lead, team=team, vendors=vend, launch=launch, sib_launch=sib_launch,
                milestones=ms, budget=budget, reserve=reserve, draft=draft, sib_budget=sib_budget,
                headcount=headcount, risks=risks, risk_owner=risk_owner, table=table,
                fmt_seed=rng.randrange(1 << 30))


def files_for(w: dict) -> dict[str, str]:
    r = random.Random(w["fmt_seed"])
    P, S = w["project"], w["sibling"]
    ov = (f"# Project {P} -- overview\n\n"
          f"{w['org']} runs Project {P}. The project lead is {w['lead']}; the executive sponsor is "
          f"{w['sponsor']}.\n\nThe team numbers {w['headcount']} people across all sites.\n\n"
          f"Public launch is planned for {_fmt_date(w['launch'], r)}.\n\n"
          f"Not to be confused with Project {S}, the sister initiative led by {w['sib_lead']}, "
          f"which launches {_fmt_date(w['sib_launch'], r)}. Project {S} has its own plan and "
          f"is not covered here.\n")
    fin = (f"# Finance notes\n\nThe approved budget for Project {P} is "
           f"{_fmt_money(w['budget'], r)}. A contingency reserve of {_fmt_money(w['reserve'], r)} "
           f"sits inside that figure.\n\n"
           f"History: the first draft planned {_fmt_money(w['draft'], r)}; that draft was "
           f"superseded by the approved figure above and must not be quoted.\n\n"
           f"For comparison only: Project {S} was approved at {_fmt_money(w['sib_budget'], r)}.\n\n"
           "Regional sales and costs are in data.csv (units, revenue and cost in USD).\n")
    people = "\n".join(f"- {n}: {role}" for n, role in w["team"])
    owner = w["risk_owner"]
    team = (f"# Team and vendors\n\nCore team for Project {P}:\n{people}\n\n"
            f"Contracted vendors: {_list_text(w['vendors'], r)}.\n")
    ms = "\n".join(f"- {n}: {_fmt_date(d, r)}" for n, d in w["milestones"])
    sch = (f"# Schedule and risks\n\nMilestones:\n{ms}\n\n"
           f"Top risks: {_list_text(w['risks'], r)}.\n\n"
           f"{owner} owns the risk register and chairs the weekly risk review.\n")
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(["region", "units", "revenue_usd", "cost_usd"])
    wr.writerows(w["table"])
    return {"notes/overview.md": ov, "notes/finance.md": fin, "notes/team.md": team,
            "notes/schedule.md": sch, "data.csv": buf.getvalue()}


# --- tiers -----------------------------------------------------------------------------

def _fact(section, label, kind, value, display, **kw):
    return {"section": section, "label": label, "kind": kind, "value": value,
            "display": display, **kw}


def _truth(w: dict, tier: int) -> dict:
    S = SECTIONS[tier]
    iso = lambda d: d.isoformat()  # noqa: E731
    m0 = w["milestones"][0]
    f_lead = _fact(S[0], "the project lead", "text", w["lead"], w["lead"])
    f_spon = _fact(S[0], "the executive sponsor", "text", w["sponsor"], w["sponsor"])
    f_budget = lambda s: _fact(s, "the approved budget (USD)", "num", w["budget"], _money(w["budget"]))  # noqa: E731
    f_launch = lambda s: _fact(s, "the public launch date", "date", iso(w["launch"]),  # noqa: E731
                               f"{w['launch'].day} {MONTHS[w['launch'].month - 1]} {w['launch'].year}")
    f_ms = lambda s: _fact(s, f"the date of the '{m0[0]}' milestone", "date", iso(m0[1]),  # noqa: E731
                           f"{m0[1].day} {MONTHS[m0[1].month - 1]} {m0[1].year}")
    f_res = lambda s: _fact(s, "the contingency reserve (USD)", "num", w["reserve"], _money(w["reserve"]))  # noqa: E731
    f_head = lambda s: _fact(s, "the total headcount", "num", w["headcount"], str(w["headcount"]))  # noqa: E731
    f_vend = lambda s: _fact(s, "the three contracted vendors", "list", None,  # noqa: E731
                             ", ".join(w["vendors"][:-1]) + " and " + w["vendors"][-1], items=w["vendors"])
    f_risks = lambda s: _fact(s, "the three top risks, named as in the notes", "list", None,  # noqa: E731
                              ", ".join(w["risks"][:-1]) + " and " + w["risks"][-1], items=w["risks"])
    f_owner = lambda s: _fact(s, "who owns the risk register", "text", w["risk_owner"], w["risk_owner"])
    f_team = lambda s: _fact(s, "the names of the three core team members", "list", None,  # noqa: E731
                             ", ".join(n for n, _ in w["team"]), items=[n for n, _ in w["team"]])
    if tier == 1:
        facts = [f_lead, f_spon] + [g for g in (f_budget(S[1]), f_launch(S[2]), f_ms(S[2]))]
    elif tier == 2:
        facts = [_fact(S[0], "the project lead", "text", w["lead"], w["lead"]), f_head(S[0]),
                 f_team(S[1]), f_vend(S[1]), f_budget(S[2]), f_launch(S[2]), f_risks(S[3]),
                 f_owner(S[3])]
    else:
        facts = [_fact(S[0], "the project lead", "text", w["lead"], w["lead"]),
                 _fact(S[0], "the executive sponsor", "text", w["sponsor"], w["sponsor"]),
                 f_head(S[0]), f_team(S[1]), f_vend(S[2]), f_budget(S[3]), f_res(S[3]),
                 f_launch(S[4]), f_ms(S[4]), f_risks(S[4]), f_owner(S[4])]
        facts.append(_fact(S[0], "the organisation running the project", "text", w["org"], w["org"]))
    forbidden = [{"label": "sibling/draft budget", "kind": "num", "value": w["sib_budget"]},
                 {"label": "superseded draft budget", "kind": "num", "value": w["draft"]},
                 {"label": "sibling launch date", "kind": "date", "value": iso(w["sib_launch"])},
                 {"label": "sibling project lead", "kind": "text", "value": w["sib_lead"]}]
    truth = {"path": "report.md" if tier == 1 else "report.pdf",
             "format": "md" if tier == 1 else "pdf", "headings": S, "facts": facts,
             "forbidden": forbidden, "min_chars": BOUNDS[tier][0], "max_chars": BOUNDS[tier][1]}
    if tier == 3:
        rows = w["table"][:4]
        trows = [[r[0], *r[1:]] for r in rows]
        trows.append(["Total", *(sum(r[j] for r in rows) for j in (1, 2, 3))])
        truth["table"] = {"section": S[5], "header": ["Region", "Units", "Revenue (USD)", "Cost (USD)"],
                          "rows": trows}
    return truth


def _prompt(w: dict, truth: dict, tier: int) -> str:
    P = w["project"]
    kind = "Markdown file" if tier == 1 else "PDF (use the make_pdf tool; write Markdown, it is typeset for you)"
    lines = [f"The workspace has notes about Project {P} in notes/ and a spreadsheet data.csv. "
             f"Write a status report on Project {P} as {truth['path']} -- a {kind}. "
             "Use these exact section headings, in this order, each as its own heading line:"]
    lines += [f"  {i}. {h}" for i, h in enumerate(truth["headings"], 1)]
    lines.append(f"The report must state these facts about Project {P} (look them up in the notes; "
                 "use the figures for this project only, not for any other project, and not "
                 "superseded drafts; write each fact where it fits):")
    by = {}
    for f in truth["facts"]:
        by.setdefault(f["section"], []).append(f["label"])
    for h in truth["headings"]:
        if h in by:
            lines.append(f"  - under \"{h}\": " + "; ".join(by[h]))
    if truth.get("table"):
        t = truth["table"]
        regs = ", ".join(r[0] for r in t["rows"][:-1])
        lines.append(f"  - under \"{t['section']}\": a table with columns "
                     f"{', '.join(t['header'])} for the regions {regs}, copied from data.csv, "
                     "plus a final Total row that sums each numeric column")
    lines.append("Keep it short and factual (a few sentences or bullets per section). "
                 "When the file is written, finish with a line `ANSWER: <the file path>`.")
    return "\n".join(lines)


def specs(seed: int) -> list[dict]:
    w = world(seed)
    files = files_for(w)
    out = []
    for tier in (1, 2, 3):
        truth = _truth(w, tier)
        out.append({"family": FAMILY, "seed": seed, "files": dict(files),
                    "tools": list(TOOLS_T1 if tier == 1 else TOOLS_PDF),
                    "name": f"docmake-t{tier}-{seed}", "tier": tier, "project": w["project"],
                    "ground_truth": json.dumps(truth, sort_keys=True, ensure_ascii=False),
                    "prompt": _prompt(w, truth, tier)})
    return out


# --- oracle (scripted ideal answer) -------------------------------------------------------

def ideal_markdown(truth: dict, drop_heading: str | None = None, corrupt: dict | None = None) -> str:
    """The ideal document. drop_heading / corrupt exist for the negative checks."""
    parts = []
    for h in truth["headings"]:
        if h == drop_heading:
            continue
        parts.append(f"## {h}\n")
        for f in truth["facts"]:
            if f["section"] == h:
                disp = (corrupt or {}).get(f["label"], f["display"])
                parts.append(f"- {f['label'][0].upper() + f['label'][1:]}: {disp}.")
        tb = truth.get("table")
        if tb and tb["section"] == h:
            parts.append("\n| " + " | ".join(tb["header"]) + " |")
            parts.append("|" + "---|" * len(tb["header"]))
            for r in tb["rows"]:
                parts.append("| " + " | ".join([str(r[0])] + [f"{int(x):,}" for x in r[1:]]) + " |")
        parts.append("")
    return "\n".join(parts)


def oracle(spec: dict, ws, markdown: str | None = None) -> tuple[str, int]:
    """Reads every file through ws like the agent would, then writes the ideal document.
    Returns (answer, tool calls). `markdown` overrides the document (negative checks)."""
    import doc_tools
    truth = json.loads(spec["ground_truth"])
    calls = 1
    listing = ws.list_dir(".")
    assert "notes" in listing and "data.csv" in listing
    for p in sorted(spec["files"]):
        ws.read_file(p)
        calls += 1
    md = markdown if markdown is not None else ideal_markdown(truth)
    if truth["format"] == "md":
        ws.write_file(truth["path"], md)
    else:
        out = doc_tools.make_pdf(ws, {"path": truth["path"], "markdown": md})
        assert out.startswith("wrote"), out
    return truth["path"], calls + 1
