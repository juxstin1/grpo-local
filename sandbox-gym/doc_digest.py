"""Family doc_digest: read a long generated report with the tools, write SUMMARY.md.

Each seed (the root) builds an invented world -- a regional works consortium, its
projects, towns, people, firms -- and, per tier, a programme review under docs/ of
1.5x / 3x / 5x the episode budget in Qwen3.5 tokens (TIERS; at 32768 that is ~48k / ~96k /
~160k). Train docs follow config episode.budget; sealed holdouts are pinned to
HOLDOUT_BUDGET = 32768. The doc is always bigger than the episode budget, so it can
only be read in pieces (read_file offset/limit); without compaction (`eval.py --compact`) a full read does not fit. That is expected.

Planted in seeded filler prose:
  KEY facts   N projects, one headline figure each (budget, headcount, opening date,
              main contractor, delivery partners, capacity, length, loss reduction,
              freight throughput). Spread one per 1/N slice of the doc, so skipping
              a chunk drops a fact.
  DECOYS      M wrong values: an early draft, a hypothetical, a withdrawn press figure,
              a scheme outside the Consortium (cited for comparison), and a plain early
              statement that a late erratum corrects.
Filler mentions projects only qualitatively and never names a firm; its numbers
(years, small counts, round sums) never equal a key or decoy value (asserted).

Grade (mechanical, binary; partial credit logged, never rewarded):
  SUMMARY.md exists, <= cap words, every key fact present and attached to its project
  (same line as the project name, or under the nearest preceding project mention),
  and NO decoy value anywhere. Matching is tolerant: case, articles, '&'/'and',
  number spellings (4.2 million / 4,200,000 / 4.2M / 4.2m), percent / per cent,
  dates (14 March 2026 / March 14 / 2026-03-14 / 14/03/2026), list formatting
  (items matched one by one: "A, B and C" == "A; B; C" == bullets).
  An ANSWER line is required (done = she said so), like edit/fix.

Split by root: holdout seeds HOLDOUT_SEEDS (t1 9601-9605, t2 9611-9615, t3 9621-9625;
range 9600-9699 reserved, never train), one item per holdout seed. Train seeds give
one item per tier, and their worlds never reuse a holdout project/scheme name.
Tools: list_dir, read_file, search, write_file.
"""

from __future__ import annotations

import json
import os
import random
import re
import textwrap
import time
from functools import lru_cache
from pathlib import Path

FAMILY = "doc_digest"
TOOLS = ["list_dir", "read_file", "search", "write_file"]
SUMMARY = "SUMMARY.md"
# Doc size is RELATIVE to the episode budget (x = tokens / budget), so a doc never fits
# one context window at any budget: compaction (eval.py --compact) is required.
TIERS = {1: dict(x=1.5, n_key=5, n_decoy=5, cap=150, parts=1),
         2: dict(x=3.0, n_key=8, n_decoy=7, cap=220, parts=2),
         3: dict(x=5.0, n_key=12, n_decoy=10, cap=300, parts=3)}
HOLDOUT_BUDGET = 32768          # sealed holdouts are sized for this budget, whatever config says
READ_LINES = 150                # one read_file of this many lines fits the 14k tool-output cap
HOLDOUT_SEEDS = {1: tuple(range(9601, 9606)), 2: tuple(range(9611, 9616)),
                 3: tuple(range(9621, 9626))}
RESERVED = range(9600, 9700)
CHARS_PER_TOKEN = 4.4           # measured on this filler with the Qwen3.5-4B tokenizer (4.48-4.52)
WRAP = 90
DECOY_TYPES = ("draft", "hypo", "retract", "outside", "late")


def train_budget() -> int:
    """The episode budget train docs are sized for: config (its env override included)."""
    from config import Config
    return int(Config().episode.budget)


def turn_cap(files: dict[str, str]) -> int:
    """Turns for one full read (READ_LINES per call) + a checkpoint per 3 reads + list,
    write, answer. Turns bind, not tokens: eval with --max-turns >= this."""
    import math
    reads = sum(math.ceil((t.count("\n") + 1) / READ_LINES) for t in files.values())
    return reads + math.ceil(reads / 3) + 3


def holdout_tier(seed: int) -> int | None:
    for t, ss in HOLDOUT_SEEDS.items():
        if seed in ss:
            return t
    return None


# --- invented world ------------------------------------------------------------

ONSET = ("Vel", "Or", "Tess", "Hal", "Bren", "Cal", "Dun", "Ess", "Fen", "Gor", "Ith", "Jor",
         "Kel", "Lun", "Mor", "Nev", "Ost", "Pell", "Quen", "Ros", "Sar", "Thal", "Ul", "Var",
         "Wen", "Yor", "Zan", "Ash", "Bel", "Cor", "Dar", "Eld", "Fal", "Gal", "Hart", "Ir",
         "Kast", "Lor", "Mar", "Nor", "Pry", "Rhen", "Sel", "Tor", "Ves", "Wil")
CODA = ("mar", "len", "aly", "dor", "wick", "ford", "by", "ton", "holm", "stead", "mere",
        "gate", "wyn", "ric", "oth", "ane", "ira", "ell", "ard", "usk", "avon", "ess", "ley")
SUFFIX = {"budget": ("Works", "Exchange", "Quarter"), "headcount": ("Yard", "Depot", "Mill"),
          "date": ("Bridge", "Market Hall", "Library"), "contractor": ("Viaduct", "Tunnel", "Lock"),
          "partners": ("Food Hub", "Training Centre", "Business Park"),
          "capacity": ("Power Station", "Solar Farm", "Hydro Scheme"),
          "length": ("Cycleway", "Tramline", "Flood Wall"),
          "reduction": ("Reservoir", "Water Works", "Pumping Station"),
          "throughput": ("Wharf", "Freight Terminal", "Rail Sidings")}
OUTSIDE_SUFFIX = ("Basin scheme", "Valley programme", "Coast partnership", "Moor project")
FIRST = ("Anna", "Bertil", "Clara", "Daan", "Edda", "Felix", "Greta", "Hugo", "Ilse", "Jonas",
         "Karin", "Lars", "Mira", "Niels", "Olga", "Pieter", "Runa", "Stig", "Tilde", "Ulla",
         "Viktor", "Wanda", "Yrsa", "Emil")
PSURN = ("Aldane", "Brisk", "Corwen", "Dalby", "Ensor", "Fairlie", "Grantham", "Holloway",
         "Ingram", "Jessop", "Kettle", "Lambeth", "Marlow", "Nesbitt", "Orme", "Pargeter",
         "Quayle", "Rudd", "Sowerby", "Tamsin", "Upshaw", "Vane", "Whitlow", "Yeardley")
FSURN = ("Abrams", "Beckwith", "Carrow", "Dimmock", "Eastlake", "Furness", "Gildart", "Hepworth",
         "Iveson", "Jarrow", "Kinsella", "Lockyer", "Mabbott", "Northey", "Ottley", "Prowse",
         "Rendall", "Scarth", "Thwaite", "Ulley", "Vosper", "Wardle", "Yelland", "Brenner",
         "Calloway", "Dunmore", "Fennick", "Gorst", "Harrowby", "Lowther")
FIRM_FMT = ("{a} & {b}", "{a} Brothers", "{a} Holdings", "{a} and {b}", "{a} Civil", "{a} Group")
ROLES = ("chair of the board", "programme director", "finance officer", "community liaison lead",
         "head of estates", "planning adviser", "environment officer", "secretary to the board")
COMMITTEES = ("the audit committee", "the estates committee", "the community panel",
              "the transport working group", "the environment forum")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")
SEASONS = ("spring", "summer", "autumn", "winter")
NUMW = ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


@lru_cache(maxsize=1)
def _vocab() -> frozenset:
    """Every word the templates and pools already use: invented names must differ."""
    text = " ".join(FILLER + SECTIONS + FIRST + PSURN + FSURN + ROLES + COMMITTEES + MONTHS + NUMW
                    + OUTSIDE_SUFFIX + tuple(ATTR.values()) + sum(SUFFIX.values(), ()))
    return frozenset(re.findall(r"[a-z]+", text.lower())) | {"galley", "valley", "darby", "selby"}


def _names(rng: random.Random, n: int, banned: set[str]) -> list[str]:
    out: list[str] = []
    while len(out) < n:
        w = re.sub(r"(.)\1\1", r"\1\1", rng.choice(ONSET) + rng.choice(CODA))   # Pell+ley -> Pelley
        if w not in out and w.lower() not in banned and w.lower() not in _vocab():
            out.append(w)
    return out


@lru_cache(maxsize=256)
def world(seed: int, banned: frozenset = frozenset()) -> dict:
    rng = random.Random(seed * 7919 + 13)
    heads = _names(rng, 40, set(banned))
    proj_heads, out_heads, places = heads[:12], heads[12:16], heads[16:40]
    people = [f"{f} {s}" for f, s in zip(rng.sample(FIRST, 16), rng.sample(PSURN, 16))]
    fs = rng.sample(FSURN, len(FSURN))
    firms = []
    for i in range(0, len(fs) - 1, 2):
        firms.append(rng.choice(FIRM_FMT).format(a=fs[i], b=fs[i + 1]))
    return {"seed": seed, "consortium": f"{places[0]} Valley Works Consortium",
            "region": places[0], "proj_heads": proj_heads, "out_heads": out_heads,
            "towns": places[1:13], "rivers": places[13:17], "hills": places[17:20],
            "people": people, "firms": firms,
            "roles": {p: rng.choice(ROLES) for p in people}}


# --- key / decoy values ----------------------------------------------------------

ATTR = {"budget": "approved budget", "headcount": "headcount", "date": "opening date",
        "contractor": "main contractor", "partners": "delivery partners",
        "capacity": "installed capacity", "length": "total length",
        "reduction": "reduction in water losses", "throughput": "annual freight throughput"}
KINDS = tuple(ATTR)


def _value(kind: str, rng: random.Random, firms: list[str]) -> dict:
    """{'kind', 'text' (as written in the doc), 'atoms': [...]} -- atoms are what the
    grader looks for: ('num', float) | ('date', m, d, y) | ('name', str)."""
    if kind == "budget":
        v = rng.randint(11, 99)
        v += (v % 10 == 0)
        return {"text": f"{v / 10} million crowns", "atoms": [("num", v * 100_000.0)]}
    if kind == "headcount":
        v = rng.randint(121, 979)
        return {"text": f"{v} staff", "atoms": [("num", float(v))]}
    if kind == "date":
        m, d = rng.randint(1, 12), rng.randint(2, 28)
        return {"text": f"{d} {MONTHS[m - 1]} 2026", "atoms": [("date", m, d, 2026)]}
    if kind == "contractor":
        f = firms.pop()
        return {"text": f, "atoms": [("name", f)]}
    if kind == "partners":
        a, b, c = firms.pop(), firms.pop(), firms.pop()
        return {"text": f"{a}, {b} and {c}", "atoms": [("name", a), ("name", b), ("name", c)]}
    if kind == "capacity":
        v = rng.randint(12, 95) + 0.5
        return {"text": f"{v} megawatts", "atoms": [("num", v)]}
    if kind == "length":
        v = rng.randint(21, 489)
        v += (v % 5 == 0)
        return {"text": f"{v / 10} kilometres", "atoms": [("num", v / 10)]}
    if kind == "reduction":
        v = rng.randint(11, 48) + 0.5
        return {"text": f"{v} percent", "atoms": [("num", v)]}
    if kind == "throughput":
        v = rng.randint(101, 989) * 100
        v += 100 * (v % 1000 == 0)
        return {"text": f"{v:,} tonnes", "atoms": [("num", float(v))]}
    raise ValueError(kind)


def _atom_key(a) -> tuple:
    return (a[0], round(a[1], 6)) if a[0] == "num" else (a[0], *a[1:]) if a[0] == "date" \
        else ("name", " ".join(name_tokens(a[1])))


# --- tolerant matching (also used to police the filler) ---------------------------

SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "mio": 1e6, "m": 1e6,
         "billion": 1e9, "bn": 1e9, "b": 1e9}
NUM_RE = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:[,   ]\d{3})+|\d+)(?:\.(\d+))?"
    r"(?:\s*(thousand|million|billion|mn|mio|bn|k|m|b)(?![a-z]))?", re.I)
MONTH_RE = ("jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
            "sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
DATE_DM = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({MONTH_RE})\.?(?:,?\s+(\d{{4}}))?\b", re.I)
DATE_MD = re.compile(rf"\b({MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}}))?", re.I)
DATE_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
DATE_SL = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")
STOP = {"the", "a", "an", "and", "ltd", "limited", "co", "company", "plc", "inc", "llp"}


def _month(s: str) -> int:
    return [m[:3].lower() for m in MONTHS].index(s[:3].lower()) + 1


def clean(text: str) -> str:
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"[*_`#>|]", " ", text)


def numbers(text: str) -> list[float]:
    out = []
    for m in NUM_RE.finditer(clean(text)):
        whole = re.sub(r"[^\d]", "", m.group(1))
        v = float(f"{whole}.{m.group(2)}" if m.group(2) else whole)
        if m.group(3):
            v *= SCALE[m.group(3).lower()]
        out.append(round(v, 6))
    return out


def dates(text: str) -> list[tuple[int, int, int | None]]:
    """(month, day, year-or-None). d/m/y slashes are read both ways."""
    t, out = clean(text), []
    for m in DATE_DM.finditer(t):
        out.append((_month(m.group(2)), int(m.group(1)), int(m.group(3)) if m.group(3) else None))
    for m in DATE_MD.finditer(t):
        out.append((_month(m.group(1)), int(m.group(2)), int(m.group(3)) if m.group(3) else None))
    for m in DATE_ISO.finditer(t):
        out.append((int(m.group(2)), int(m.group(3)), int(m.group(1))))
    for m in DATE_SL.finditer(t):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out += [(b, a, y), (a, b, y)]
    return [d for d in out if 1 <= d[0] <= 12 and 1 <= d[1] <= 31]


def name_tokens(s: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", clean(s).lower().replace("&", " and "))
    return [t for t in toks if t not in STOP]


def has_atom(text: str, atom) -> bool:
    if atom[0] == "num":
        return any(abs(v - atom[1]) < 1e-6 * max(1.0, abs(atom[1])) for v in numbers(text))
    if atom[0] == "date":
        return any((m, d) == (atom[1], atom[2]) and y in (None, atom[3]) for m, d, y in dates(text))
    want, toks = name_tokens(atom[1]), name_tokens(text)
    return any(toks[i:i + len(want)] == want for i in range(len(toks) - len(want) + 1))


def words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _mentions(line: str, heads: list[str]) -> list[str]:
    low = line.lower()
    return [h for h in heads if re.search(rf"\b{re.escape(h.lower())}\b", low)]


def key_found(text: str, key: dict, heads: list[str]) -> bool:
    """Every atom of the key, on a line that names the project or under it (the
    nearest preceding project/scheme mention, when the line names none)."""
    for atom in key["atoms"]:
        ctx, ok = None, False
        for line in text.split("\n"):
            m = _mentions(line, heads)
            if has_atom(line, atom) and (key["head"] in m or (not m and ctx == key["head"])):
                ok = True
                break
            if m:
                ctx = m[-1]
        if not ok:
            return False
    return True


def grade_summary(text: str | None, gt: dict) -> tuple[bool, dict]:
    """(passed, info). info = found/n, decoys hit, words/cap -- partial credit for logs."""
    n = len(gt["keys"])
    if text is None:
        return False, {"found": 0, "n": n, "decoys_hit": [], "words": 0, "cap": gt["cap"],
                       "reason": f"no {SUMMARY}"}
    heads = gt["heads"]
    missing = [k["project"] for k in gt["keys"] if not key_found(text, k, heads)]
    hit = [d["text"] for d in gt["decoys"] if any(has_atom(text, a) for a in d["atoms"])]
    w = words(text)
    ok = not missing and not hit and w <= gt["cap"]
    reason = "ok" if ok else "; ".join(
        x for x in (f"missing {len(missing)}" if missing else "",
                    f"decoys {len(hit)}" if hit else "",
                    f"words {w}>{gt['cap']}" if w > gt["cap"] else "") if x)
    return ok, {"found": n - len(missing), "n": n, "missing": missing, "decoys_hit": hit,
                "words": w, "cap": gt["cap"], "reason": reason}


# --- filler ------------------------------------------------------------------------

SECTIONS = ("Background", "Governance", "Regional Context", "Methodology", "Stakeholder Engagement",
            "Risk Register", "Environmental Review", "Finance Overview", "Transport and Access",
            "Community Liaison", "Heritage and Archives", "Skills and Training", "Procurement",
            "Health and Safety", "Weather and Site Conditions", "Lessons Learned", "Outlook",
            "Programme Notes", "Monitoring", "Communications", "Land and Estates",
            "Partnership Working", "Appendix: Meeting Record", "Appendix: Site Visits")
FILLER = (
    "The {committee_t} met in {month} {year} to review progress across the valley, and {person} opened the session with a short account of the previous quarter.",
    "Much of the discussion turned on access roads around {town}, where residents have long complained about heavy vehicles on the old bridge over the {river}.",
    "{person}, the {role}, reminded members that the board's first duty is to the communities that host the works, not to the timetable.",
    "Records held in the {town} archive show that the river was first embanked in {year}, after a winter flood that reached the market square.",
    "The town of {town} has a population of about {pop}, and its high street still follows the line of the medieval road to {town2}.",
    "Members asked for clearer reporting on site safety, and {person2} agreed to bring a revised template to the next meeting.",
    "Fieldwork on the slopes of {hill} was delayed for {nw} weeks by snow, a pattern that the {season} records suggest is becoming more common.",
    "Work at {P} continued through the {season}, with {person} chairing the site liaison group that meets on the first {weekday} of each month.",
    "Local schools in {town} took part in a drawing competition about the {river}, and the winning entries now hang in the council chamber.",
    "The consortium's approach rests on three ideas: build what the valley needs, explain every decision in public, and leave each site better than it was found.",
    "In {year} the council spent {n9} million crowns on flood walls along the {river}, a sum that older residents still mention at public meetings.",
    "A barge on the {river} carries about {n} tonnes of gravel, and most of the quarry traffic once moved that way before the railway reached {town}.",
    "The old {n9} megawatt mill turbine at {town2} was decommissioned in {year} and is now a museum piece.",
    "The {n} kilometre ridge walk above {town} remains popular with visitors, and the board has promised to keep its stiles and gates in repair.",
    "Around {pct} percent of households in {town2} responded to the {year2} survey, a rate the community panel considers healthy for a rural district.",
    "A team of {n} staff from the regional office attended the open day, answering questions about noise, dust and working hours.",
    "{person} noted that procurement rules had changed twice since {year2}, and that suppliers needed plain guidance rather than another policy document.",
    "The {committee_t} asked for a short history of the valley's industries, which this review now includes as a separate section.",
    "Heavy rain in {month} washed out part of the towpath near {town}, and volunteers rebuilt it over {nw} weekends.",
    "Birdwatchers have recorded otters on the lower {river} for the first time since {year}, which the environment forum welcomed.",
    "Questions about parking at {P} were raised again, and the board agreed that the matter belongs with the transport working group.",
    "{person2} presented a map of footpaths, bridleways and lanes around {hill}, with the gaps that the estates team hopes to close.",
    "The meeting closed with thanks to the staff of the {town} library, who have hosted the public exhibition for {nw} years.",
    "Several speakers stressed that small contracts matter as much as large ones, because they keep local trades in work through the slack months.",
    "The archive holds a photograph of the {town2} quay taken in {year}, showing sailing barges moored three deep along the wall.",
    "Weather stations on {hill} and at {town} have run since {year}, giving the consortium one of the longest local records in the region.",
    "Residents of {town2} asked whether the evening bus could run later, and the transport working group promised to raise it with the operator.",
    "{person} explained that every risk on the register has a named owner, a review date and a plain description of what would trigger action.",
    "The heritage officer catalogued {n} boundary stones along the old parish line between {town} and {town2}.",
    "The board visited {P} in the {season} and spoke with apprentices about their training and their plans after the course.",
    "Minutes of each meeting are published within {nw} weeks, and paper copies are left at the libraries in {town} and {town2}.",
    "Dust monitoring near the {river} crossing found no breach of the agreed limits, though readings rose briefly during a dry spell in {month}.",
    "The community panel asked that public notices use plain language, avoid acronyms and give a named contact for every query.",
    "Older maps show a ferry crossing at {town2} until {year}, when the first iron bridge was opened by the county.",
    "{person2}, the {role2}, reported that the volunteer warden scheme now covers every public site in the valley.",
    "The environment forum asked for a hedgerow survey before any new fencing, and the estates team agreed to commission one.",
    "Members discussed the board's own carbon footprint, including travel to meetings, and agreed to hold more sessions online.",
    "A short film about the valley's working history was shown at the {town} festival and later placed in the public archive.",
    "The board's code of conduct was reviewed in {month}; no changes were proposed beyond clearer wording on gifts and hospitality.",
    "{person} reminded the room that the review is written for residents as much as for funders, and should be read that way.",
    "Snow closed the pass above {hill} for {nw} days, and supplies for the upland sites were moved by the valley road instead.",
    "The transport working group compared traffic counts from {year2} with the latest figures and found the pattern broadly unchanged.",
    "Local historians in {town2} have asked to see the consortium's site photographs, and the board has agreed in principle.",
    "Concerns about night lighting at {P} were passed to the environment officer, who will report on bat activity in the {season}.",
    "Apprentices from {town} spent a week with the survey team, learning to read levels and keep a clean field book.",
    "The {committee_t} noted that the valley's population has grown slowly since {year}, mostly in {town} and the villages around it.",
    "The audit trail for small grants was simplified, so that a parish council can now apply on a single page.",
    "{person2} described the clean-up of the old dye works at {town2}, where the soil was tested in {nw} separate places.",
    "Visitors to the {river} towpath are counted by a sensor near {town}, and the numbers rise sharply every {season}.",
    "The board thanked {person} for {nw} years of service on the community panel and wished them well in retirement.",
    "A water vole survey along the {river} found healthy colonies above {town2} and fewer below the weir.",
    "Local businesses in {town} asked for earlier notice of road closures, and the communications team now posts them two weeks ahead.",
    "The question of a permanent visitor centre was deferred again, pending a wider review of the valley's tourism plan.",
    "Members agreed that photographs in public reports should show the people who do the work, not only the finished structures.",
    "{person} summarised the risk register: most items are steady, a few have eased, and none has been closed without a review.",
    "The old quarry at {hill} is now a nature reserve, and the consortium keeps a small budget line for its paths and signs.",
    "Church records from {town2} mention the great frost of {year}, when the {river} froze hard enough for a fair on the ice.",
    "The estates team walked the boundary of {P} with neighbouring landowners in {month} and agreed the line of a new hedge.",
    "Feedback forms from the open day were mostly positive, though several asked for more seating and clearer signs to the toilets.",
    "The {committee_t} asked that future reviews keep the same section order, so that readers can compare one year with the next.",
    "The board confirmed the dates of next year's meetings and agreed to keep the {month} session in {town}.",
    "{person} confirmed that the final version of the minutes would be circulated before the {season} recess.",
    "No final decision was taken on the footbridge at {town2}; the figure most often quoted by residents has never been costed.",
    "The approved budget for the volunteer warden scheme has always been modest, and the board intends to keep it that way.",
    "An erratum slip was added to an earlier review after a misspelt place name in the {town} section was spotted by a reader.",
    "Site returns from the upland works confirm that the headcount rises every {season} as seasonal crews arrive.",
    "The board confirmed that each project keeps its own delivery partners and reports them separately in its annual return.",
    "Installed capacity, network length and freight throughput are all reported in the project returns, which this review draws on.",
    "{person2} confirmed that the correction notes in last year's review had been read by the audit committee and closed.",
    "The final accounts for the small grants fund were signed in {month}, and the {committee_t} saw no reason to reopen them.",
    "Members asked how the main contractor on each site is chosen, and {person} described the tender process step by step.",
    "Water losses across the valley's old mains are reported each year, and the environment forum follows the trend closely.",
)


def _fill(t: str, w: dict, rng: random.Random, projects: list[str]) -> str:
    towns = rng.sample(w["towns"], 2)
    p1, p2 = rng.sample(w["people"], 2)
    slots = {"town": towns[0], "town2": towns[1], "river": rng.choice(w["rivers"]),
             "hill": rng.choice(w["hills"]), "person": p1, "person2": p2,
             "role": w["roles"][p1], "role2": w["roles"][p2],
             "committee_t": rng.choice(COMMITTEES)[4:], "month": rng.choice(MONTHS),
             "season": rng.choice(SEASONS), "year": rng.randint(1890, 1990),
             "year2": rng.randint(1995, 2024), "n": rng.randint(4, 60), "n9": rng.randint(2, 9),
             "pct": rng.randint(20, 80), "pop": f"{rng.randint(12, 95)},000",
             "nw": rng.choice(NUMW), "P": rng.choice(projects),
             "weekday": rng.choice(("Monday", "Tuesday", "Wednesday", "Thursday"))}
    s = t.format(**slots)
    return s[0].upper() + s[1:]


def _para(w: dict, rng: random.Random, projects: list[str]) -> list[str]:
    return [_fill(t, w, rng, projects) for t in rng.sample(FILLER, rng.randint(4, 7))]


# --- planted statements --------------------------------------------------------------

def _is(kind: str) -> str:
    return "are" if kind == "partners" else "is"


def _final(k: dict, rng: random.Random) -> str:
    P, a, v, i = k["project"], ATTR[k["kind"]], k["text"], _is(k["kind"])
    return rng.choice((
        f"The {a} of {P} {i} {v}, the figure the board confirmed at its {rng.choice(MONTHS)} meeting.",
        f"Final accounts and site returns confirm that the {a} of {P} {i} {v}.",
        f"For {P}, the confirmed {a} {i} {v}; no further change is expected.",
    ))


def _decoy(d: dict, k: dict, w: dict, rng: random.Random) -> str:
    P, a, D, i = k["project"], ATTR[k["kind"]], d["text"], _is(k["kind"])
    t = d["type"]
    if k["kind"] == "partners" and t != "outside":     # one firm that is not a partner
        return {"draft": f"An early draft of the plan listed {D} among the delivery partners of {P}; "
                         "that firm withdrew before contracts were signed.",
                "hypo": f"Had the second phase gone ahead, {D} would also have joined the delivery "
                        f"partners of {P}.",
                "retract": f"A press notice in {rng.choice(MONTHS)} named {D} as a delivery partner of "
                           f"{P}; the notice was withdrawn a week later as an error.",
                "late": f"{D} is also listed among the delivery partners of {P}."}[t]
    if t == "draft":
        what = "choice" if k["kind"] == "contractor" else "figure"
        return f"An early draft of the business case gave the {a} of {P} as {D}; that {what} was revised before approval."
    if t == "hypo":
        return f"Had the northern extension gone ahead, the {a} of {P} would have been {D}."
    if t == "retract":
        return (f"A press notice in {rng.choice(MONTHS)} gave the {a} of {P} as {D}; the notice was "
                "withdrawn a week later as an error.")
    if t == "outside":
        O = d["entity"]
        if k["kind"] == "partners":
            return (f"Outside the Consortium, the {O}, run by {rng.choice(w['towns'])} council, counts {D} "
                    "among its delivery partners; it is cited here for comparison only.")
        return (f"Outside the Consortium, the {O}, run by {rng.choice(w['towns'])} council, reports "
                f"{'an' if a[0] in 'aeiou' else 'a'} {a} "
                f"of {D}; it is cited here for comparison only.")
    if t == "late":                                   # plain early statement, corrected later
        return f"The {a} of {P} {i} {D}."
    raise ValueError(t)


def _erratum(d: dict, k: dict) -> str:
    P, a, i = k["project"], ATTR[k["kind"]], _is(k["kind"])
    if k["kind"] == "partners":
        return (f"Erratum: an earlier section of this review lists {d['text']} among the delivery partners "
                f"of {P}. That is wrong; the delivery partners of {P} {i} {k['text']}.")
    return (f"Erratum: the {a} of {P} given earlier in this review ({d['text']}) was a transcription "
            f"error; the correct and final figure {i} {k['text']}.")


# --- doc generation ---------------------------------------------------------------------

def _plan(seed: int, tier: int, w: dict) -> dict:
    """Keys and decoys for one doc (no prose yet)."""
    cfg = TIERS[tier]
    rng = random.Random(seed * 1009 + tier * 17)
    N, M = cfg["n_key"], cfg["n_decoy"]
    kinds = list(KINDS) + list(rng.sample(KINDS, len(KINDS)))
    rng.shuffle(kinds)
    kinds = kinds[:N]
    firms = list(w["firms"])
    rng.shuffle(firms)
    used: set = set()

    def fresh(kind: str) -> dict:
        while True:
            v = _value(kind, rng, firms)
            ks = {_atom_key(a) for a in v["atoms"]}
            if not ks & used:
                used.update(ks)
                return {**v, "kind": kind}

    keys = []
    heads = list(w["proj_heads"])
    for i, kind in enumerate(kinds):
        head = heads[i]
        suf = rng.choice(SUFFIX[kind])
        pos = (i + rng.uniform(0.15, 0.85)) / N               # one key per 1/N slice
        keys.append({**fresh(kind), "head": head, "project": f"{head} {suf}", "pos": pos})
    order = list(range(N))
    rng.shuffle(order)
    types = [DECOY_TYPES[j % len(DECOY_TYPES)] for j in range(M)]
    rng.shuffle(types)
    decoys, outside = [], list(w["out_heads"])
    late_ok = [i for i in order if keys[i]["pos"] >= 0.45]
    late_used: set[int] = set()
    for j, t in enumerate(types):
        if t == "late":
            cands = [i for i in late_ok if i not in late_used]
            if not cands:
                t = "draft"
            else:
                ki = cands[0]
                late_used.add(ki)
        if t != "late":
            ki = order[j % N]
        k = keys[ki]
        dkind = "contractor" if k["kind"] == "partners" else k["kind"]   # partner decoy = one firm
        d = {**fresh(dkind), "type": t, "key": ki}
        if t == "outside":
            oh = outside.pop()
            d["entity"] = f"{oh} {rng.choice(OUTSIDE_SUFFIX)}"
            d["head"] = oh
        if t == "late":
            d["pos"] = rng.uniform(0.03, max(0.05, k["pos"] - 0.3))
            k["erratum"] = len(decoys)
        else:
            d["pos"] = rng.uniform(0.02, 0.98)
        decoys.append(d)
    return {"keys": keys, "decoys": decoys, "rng_state": rng.getstate()}


def _wrap(p: str) -> str:
    return textwrap.fill(p, WRAP, break_long_words=False, break_on_hyphens=False)


@lru_cache(maxsize=64)
def build_doc(seed: int, tier: int, budget: int, banned: frozenset = frozenset()) -> dict:
    w = world(seed, banned)
    cfg = TIERS[tier]
    plan = _plan(seed, tier, w)
    keys, decoys = plan["keys"], plan["decoys"]
    rng = random.Random(seed * 2003 + tier * 31)
    projects = [k["project"] for k in keys]
    bad = {_atom_key(a) for x in keys + decoys for a in x["atoms"]}
    target = int(cfg["x"] * budget * CHARS_PER_TOKEN)

    def safe(p: str) -> bool:                       # filler never states a key/decoy value
        return not ({("num", round(v, 6)) for v in numbers(p)} & bad) and not dates(p)

    paras: list[list[str]] = []                     # paragraphs as sentence lists
    size = 0
    while size < target:
        p = _para(w, rng, projects)
        if safe(" ".join(p)):
            paras.append(p)
            size += len(" ".join(p)) + 2
    # planted sentences go inside ordinary filler paragraphs, at their slice position
    plants = []
    for k in keys:
        plants.append((k["pos"], _erratum(decoys[k["erratum"]], k) if "erratum" in k
                       else _final(k, rng)))
    for d in decoys:
        plants.append((d["pos"], _decoy(d, keys[d["key"]], w, rng)))
    for pos, text in plants:
        para = paras[min(len(paras) - 1, max(1, int(pos * len(paras))))]
        para.insert(rng.randint(1, len(para)), text)
    paras = [" ".join(p) for p in paras]
    # sections every 6-11 paragraphs; split into parts at section boundaries
    title = f"# {w['consortium']}: Annual Programme Review"
    blocks, sec_n, i = [], 0, 0
    secs = list(SECTIONS)
    rng.shuffle(secs)
    while i < len(paras):
        n = rng.randint(6, 11)
        sec_n += 1
        name = secs[(sec_n - 1) % len(secs)]
        rnd = (sec_n - 1) // len(secs)
        blocks.append([f"## {sec_n}. {name}" + (f" ({rnd + 1})" if rnd else "")]
                      + [_wrap(p) for p in paras[i:i + n]])
        i += n
    parts, per = [], len(blocks) / cfg["parts"]
    for j in range(cfg["parts"]):
        parts.append(blocks[round(j * per):round((j + 1) * per)])
    files = {}
    for j, bl in enumerate(parts, 1):
        head = title + (f" (part {j} of {cfg['parts']})" if cfg["parts"] > 1 else "")
        body = head + "\n\n" + "\n\n".join("\n\n".join(b) for b in bl) + "\n"
        path = "docs/report.md" if cfg["parts"] == 1 else f"docs/report-part{j}.md"
        files[path] = body
    return {"files": files, "keys": keys, "decoys": decoys, "world": w}


# --- specs -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _holdout_heads() -> frozenset:
    out = set()
    for t, ss in HOLDOUT_SEEDS.items():
        for s in ss:
            w = world(s)
            out |= {h.lower() for h in w["proj_heads"] + w["out_heads"]}
    return frozenset(out)


def ideal_summary(keys: list[dict]) -> str:
    lines = ["# Summary", ""]
    lines += [f"- {k['project']}: {ATTR[k['kind']]} {k['text']}" for k in keys]
    return "\n".join(lines) + "\n"


def _gt(doc: dict, tier: int) -> dict:
    keys = [{"project": k["project"], "head": k["head"], "kind": k["kind"], "text": k["text"],
             "atoms": [list(a) for a in k["atoms"]]} for k in doc["keys"]]
    decoys = [{"type": d["type"], "text": d["text"], "atoms": [list(a) for a in d["atoms"]]}
              for d in doc["decoys"]]
    heads = [k["head"] for k in doc["keys"]] + [d["head"] for d in doc["decoys"] if "head" in d]
    return {"keys": keys, "decoys": decoys, "heads": heads, "cap": TIERS[tier]["cap"]}


def _prompt(doc: dict, tier: int) -> str:
    cfg, files = TIERS[tier], doc["files"]
    n_lines = sum(t.count("\n") for t in files.values())
    where = ("docs/report.md" if len(files) == 1
             else f"docs/ ({len(files)} files: " + ", ".join(sorted(files)) + ")")
    N = cfg["n_key"]
    return (f"{where} holds the {doc['world']['consortium']}'s annual programme review, about "
            f"{n_lines} lines in all -- far more than fits in your context at once, so read it "
            "in pieces with read_file offset/limit. The review tracks "
            f"{N} Consortium projects and reports one headline figure for each (a budget, a "
            "headcount, a date, a firm, a list of firms, a capacity, a length, a percentage or a "
            "tonnage).\n\n"
            f"Write {SUMMARY} in the workspace root giving, for each of the {N} projects, the "
            "project name and its final headline figure: the value the review finally confirms, "
            "after any corrections.\nRules:\n"
            "- Final values only. Do not write draft, superseded, withdrawn or hypothetical values, "
            "or figures of schemes outside the Consortium: a summary containing any of them fails.\n"
            f"- At most {cfg['cap']} words.\n"
            f"- write_file cannot overwrite, so write {SUMMARY} once, when you have all {N} "
            "figures.\n"
            f"When {SUMMARY} is written, end with: ANSWER: done")


@lru_cache(maxsize=64)
def _spec(seed: int, tier: int, budget: int) -> dict:
    banned = frozenset() if holdout_tier(seed) else _holdout_heads()
    doc = build_doc(seed, tier, budget, banned)
    gt = _gt(doc, tier)
    return {"family": FAMILY, "seed": seed, "tier": tier, "files": dict(doc["files"]),
            "tools": list(TOOLS), "name": f"docdigest-t{tier}-{seed}",
            "budget": budget, "turn_cap": turn_cap(doc["files"]), "gt": gt,
            "ground_truth": json.dumps(gt, sort_keys=True), "prompt": _prompt(doc, tier),
            "oracle": ideal_summary(doc["keys"])}


def specs(seed: int) -> list[dict]:
    t = holdout_tier(seed)
    if t is None and seed in RESERVED:
        raise ValueError(f"seed {seed} is reserved for doc_digest holdouts")
    if t:
        return [_spec(seed, t, HOLDOUT_BUDGET)]
    b = train_budget()
    return [_spec(seed, tt, b) for tt in TIERS]


# --- grader on the workspace --------------------------------------------------------------

PARTIAL_LOG = os.environ.get(
    "DOCDIGEST_PARTIAL_LOG", str(Path(__file__).parent / "data" / "results" / "doc_digest_partial.jsonl"))


def grade_workspace(root: Path, ground_truth: str, log_as: str | None = None) -> tuple[bool, dict]:
    """Binary pass + partial-credit info; appends info to PARTIAL_LOG (unless empty)."""
    gt = json.loads(ground_truth)
    p = Path(root) / SUMMARY
    try:
        text = p.read_text(encoding="utf-8") if p.is_file() else None
    except (OSError, UnicodeDecodeError):
        text = None
    ok, info = grade_summary(text, gt)
    if log_as and PARTIAL_LOG:
        try:
            Path(PARTIAL_LOG).parent.mkdir(parents=True, exist_ok=True)
            with open(PARTIAL_LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "ws": log_as,
                                     "passed": ok, **info}) + "\n")
        except OSError:
            pass
    return ok, info


def partial_report(path: str = PARTIAL_LOG) -> None:
    """Per-workspace-prefix aggregate of the partial-credit log (analysis only)."""
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["ws"].split("-a")[0], []).append(r)
    for name, rs in sorted(by.items()):
        f = sum(r["found"] for r in rs) / max(1, sum(r["n"] for r in rs))
        print(f"{name}: attempts {len(rs)} pass {sum(r['passed'] for r in rs)} "
              f"found {f:.0%} decoy-hit {sum(bool(r['decoys_hit']) for r in rs)} "
              f"over-cap {sum(r['words'] > r['cap'] for r in rs)} "
              f"no-file {sum(r.get('reason', '').startswith('no ') for r in rs)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "partial":
        partial_report(*sys.argv[2:3])
    else:
        for sp in specs(int(sys.argv[1]) if len(sys.argv) > 1 else 1):
            print(sp["name"], {k: len(v) for k, v in sp["files"].items()})
            print(sp["oracle"])
