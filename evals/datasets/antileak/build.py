#!/usr/bin/env python
"""Post-cutoff multi-hop QA builder following the AntiLeakBench method (ACL 2025).

Port of AntiLeakBench building/{find_knowledge_updates,fetch_wiki_articles,
build_singlehop,build_multihop}.py with the Wikidata dump replaced by the
Wikidata SPARQL endpoint + wbgetentities API, and Wikipedia pages fetched as the
*current* revision via the MediaWiki API.

Pipeline (mirrors AntiLeakBench):
  1. find_knowledge_updates: for each PID in metadata_pid.yaml, find subjects with
     a claim whose P580 start time is in (cutoff, current]; pull that subject's full
     claim history for the PID and apply get_old_new_items() verbatim (old = latest
     start <= cutoff, new = latest start <= current, objects differ, old has an end
     time and new.start >= old.end).
  2. fetch_wiki_articles: English Wikipedia article of the subject or object
     (per the PID's `wiki` option), plaintext via mwparserfromhell.strip_code().
  3. build_singlehop.check_valid_wikipedia: lead line must contain subject
     label/alias AND answer label/alias; question subject label = longest
     label/alias found in the lead.
  4. build_multihop: follow the PID's multihop templates (hop_PIDs) over current
     Wikidata (get_object = latest-start claim <= current time), each hop gated
     by check_valid_wikipedia on that hop's article; context doc appended only if
     the hop's wiki entity is new.
  5. build_multidoc (Noise5): 5 distractor docs sampled from the single-hop pool,
     excluding docs mentioning any hop entity label/alias; shuffled with seed.

Deliberate deviations (documented in the log header too):
  - regex labels are re.escape()d (upstream crashes/mis-matches on '(' '.' etc.)
  - deprecated-rank statements are ignored
  - hop objects whose chosen claim already ended (P582 <= current) are dropped,
    and undated fallbacks with >1 distinct value are dropped as ambiguous
  - the new (post-cutoff) claim must have day-precision start and not be ended
  - distractor exclusion uses word-boundary matching instead of raw substring
  - File:/Image:/Category: links are removed before strip_code()
  - docs are trimmed to ~1500 chars around the lead / answer mention

Outputs:
  evals/datasets/antileak/items.jsonl  (hard cap --max-items, default 200)
  evals/datasets/antileak/logs/build-<UTC>.log
  HTTP cache under evals/data/raw/antileak/cache (reruns are deterministic)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mwparserfromhell
import requests
import yaml

HERE = Path(__file__).resolve().parent
EVALS = HERE.parent.parent
CACHE_DIR = EVALS / "data" / "raw" / "antileak" / "cache"
LOG_DIR = HERE / "logs"
OUT_PATH = HERE / "items.jsonl"
METADATA_PATH = HERE / "metadata_pid.yaml"  # verbatim copy of AntiLeakBench metadata/metadata_pid.yaml

SPARQL_URL = "https://query.wikidata.org/sparql"
WD_API = "https://www.wikidata.org/w/api.php"
WP_API = "https://en.wikipedia.org/w/api.php"
UA = "grpo-local-antileak-builder/0.1 (AntiLeakBench-method eval dataset builder; python-requests)"

WD = "http://www.wikidata.org/entity/"
log = logging.getLogger("antileak")


# --------------------------------------------------------------------------- HTTP + cache
class Http:
    def __init__(self, cache_dir: Path, offline: bool = False):
        self.cache_dir = cache_dir
        self.offline = offline
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.last: dict[str, float] = {}
        self.n_requests = 0
        self.n_cache_hits = 0

    def _path(self, ns: str, key: str) -> Path:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / ns / h[:2] / f"{h}.json"

    def cache_get(self, ns: str, key: str):
        p = self._path(ns, key)
        if p.exists():
            self.n_cache_hits += 1
            with open(p, encoding="utf-8") as f:
                return json.load(f)["value"]
        return None

    def cache_put(self, ns: str, key: str, value):
        p = self._path(ns, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"key": key, "value": value}, f, ensure_ascii=False)
        tmp.replace(p)

    def request(self, url: str, params: dict, post: bool = False, min_interval: float = 1.0,
                timeout: int = 90, max_tries: int = 8):
        if self.offline:
            raise RuntimeError(f"offline mode: cache miss for {url} {str(params)[:200]}")
        host = url.split("/")[2]
        for attempt in range(max_tries):
            wait = self.last.get(host, 0) + min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last[host] = time.time()
            self.n_requests += 1
            try:
                if post:
                    r = self.s.post(url, data=params, timeout=timeout)
                else:
                    r = self.s.get(url, params=params, timeout=timeout)
            except requests.RequestException as e:
                backoff = min(60, 2 ** attempt)
                log.warning("request error %s (attempt %d), sleeping %ss", e, attempt + 1, backoff)
                time.sleep(backoff)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                ra = r.headers.get("Retry-After")
                backoff = int(ra) if ra and ra.isdigit() else min(120, 5 * 2 ** attempt)
                log.warning("HTTP %d from %s (attempt %d), sleeping %ss", r.status_code, host, attempt + 1, backoff)
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"giving up on {url} after {max_tries} tries")

    def sparql(self, query: str, ns: str = "sparql"):
        """Whole-query cache (used for the window queries)."""
        cached = self.cache_get(ns, query)
        if cached is not None:
            return cached
        rows = self.sparql_nocache(query)
        self.cache_put(ns, query, rows)
        return rows

    def sparql_nocache(self, query: str):
        data = self.request(SPARQL_URL, {"query": query, "format": "json"}, post=True, min_interval=1.0, timeout=120)
        return data["results"]["bindings"]


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def qid_of(uri: str) -> str | None:
    if uri.startswith(WD) and re.fullmatch(r"Q\d+", uri[len(WD):]):
        return uri[len(WD):]
    return None


def statement_id(uri: str) -> str:
    # http://www.wikidata.org/entity/statement/Q33-44fe... -> Q33$44fe...
    sid = uri.rsplit("/", 1)[-1]
    return sid.replace("-", "$", 1)


def time_lit(b: dict | None) -> str | None:
    """SPARQL dateTime literal -> 'YYYY-MM-DDTHH:MM:SSZ' (AntiLeakBench stores value[1:])."""
    if not b or b.get("type") != "literal":
        return None
    v = b["value"]
    if not re.match(r"^\d{4}-\d{2}-\d{2}T", v):
        return None
    return v[:19] + "Z"


# --------------------------------------------------------------------------- Wikidata
class Wikidata:
    def __init__(self, http: Http, cutoff: str, current: str):
        self.http = http
        self.cutoff = cutoff
        self.current = current

    def window_claims(self, pid: str, limit: int = 20000):
        """Claims of `pid` whose P580 start qualifier lies in [cutoff, current]."""
        q = f"""SELECT ?s ?st ?o ?t ?prec WHERE {{
  ?st pq:P580 ?t . hint:Prior hint:rangeSafe true .
  FILTER(?t >= "{self.cutoff}T00:00:00Z"^^xsd:dateTime && ?t <= "{self.current}T23:59:59Z"^^xsd:dateTime)
  ?s p:{pid} ?st .
  ?st ps:{pid} ?o .
  OPTIONAL {{ ?st pqv:P580 ?tv . ?tv wikibase:timePrecision ?prec . }}
}} LIMIT {limit}"""
        rows = self.http.sparql(q, ns="window")
        out = []
        for r in rows:
            s = qid_of(r["s"]["value"])
            o = qid_of(r["o"]["value"]) if r["o"]["type"] == "uri" else None
            if s is None or o is None:
                continue
            out.append({"qid": s, "claim_id": statement_id(r["st"]["value"]), "object_qid": o,
                        "start_time": time_lit(r.get("t"))})
        return out

    def get_claims(self, qids, pid: str) -> dict[str, list[dict]]:
        """All non-deprecated item-valued claims (qid, pid) with start/end/precision. Cached per (qid,pid)."""
        out, missing = {}, []
        for q in sorted(set(qids)):
            c = self.http.cache_get("claims", f"{q}|{pid}")
            if c is None:
                missing.append(q)
            else:
                out[q] = c
        for batch in chunks(missing, 60):
            values = " ".join(f"wd:{q}" for q in batch)
            query = f"""SELECT ?s ?st ?o ?rank ?start ?end ?prec WHERE {{
  VALUES ?s {{ {values} }}
  ?s p:{pid} ?st .
  ?st ps:{pid} ?o ; wikibase:rank ?rank .
  OPTIONAL {{ ?st pq:P580 ?start . }}
  OPTIONAL {{ ?st pq:P582 ?end . }}
  OPTIONAL {{ ?st pqv:P580 ?tv . ?tv wikibase:timePrecision ?prec . }}
}}"""
            rows = self.http.sparql_nocache(query)
            by_q: dict[str, dict[str, dict]] = {q: {} for q in batch}
            for r in rows:
                s = qid_of(r["s"]["value"])
                if s not in by_q:
                    continue
                if r["rank"]["value"].endswith("DeprecatedRank"):
                    continue
                o = qid_of(r["o"]["value"]) if r["o"]["type"] == "uri" else None
                if o is None:
                    continue
                cid = statement_id(r["st"]["value"])
                c = by_q[s].setdefault(cid, {"claim_id": cid, "object_qid": o, "start_time": None,
                                             "end_time": None, "start_precision": None})
                st, en = time_lit(r.get("start")), time_lit(r.get("end"))
                if st and (c["start_time"] is None or st < c["start_time"]):
                    c["start_time"] = st
                    if r.get("prec"):
                        c["start_precision"] = int(r["prec"]["value"])
                if en and (c["end_time"] is None or en > c["end_time"]):
                    c["end_time"] = en
            for q in batch:
                claims = sorted(by_q[q].values(), key=lambda c: c["claim_id"])
                self.http.cache_put("claims", f"{q}|{pid}", claims)
                out[q] = claims
        return out

    def get_entities(self, qids) -> dict[str, dict | None]:
        """label, aliases (label first, like AntiLeakBench), enwiki title. Cached per qid."""
        out, missing = {}, []
        for q in sorted(set(qids)):
            c = self.http.cache_get("entity", q)
            if c is None:
                missing.append(q)
            else:
                out[q] = c.get("data")
        for batch in chunks(missing, 50):
            data = self.http.request(WD_API, {
                "action": "wbgetentities", "ids": "|".join(batch), "props": "labels|aliases|sitelinks",
                "languages": "en", "sitefilter": "enwiki", "format": "json", "formatversion": 2,
            }, min_interval=0.5)
            ents = data.get("entities", {})
            for q in batch:
                e = ents.get(q)
                if e is None:  # redirected ids are keyed by target
                    e = next((v for v in ents.values() if v.get("redirects", {}).get("from") == q), None)
                rec = None
                if e and "missing" not in e:
                    label = e.get("labels", {}).get("en", {}).get("value")
                    if label:
                        aliases = [label] + [a["value"] for a in e.get("aliases", {}).get("en", [])]
                        wt = e.get("sitelinks", {}).get("enwiki", {}).get("title")
                        rec = {"qid": q, "label": label, "aliases": aliases, "wiki_title": wt}
                self.http.cache_put("entity", q, {"data": rec})
                out[q] = rec
        return out


# --------------------------------------------------------------------------- Wikipedia
def to_plaintext(wikitext: str) -> str:
    code = mwparserfromhell.parse(wikitext)
    for link in code.filter_wikilinks():
        t = str(link.title).strip().lower().lstrip(":")
        if t.startswith(("file:", "image:", "category:")):
            try:
                code.remove(link)
            except ValueError:
                pass
    return code.strip_code()


class Wikipedia:
    def __init__(self, http: Http):
        self.http = http
        self._pt: dict[str, dict | None] = {}

    def _fetch(self, titles: list[str]) -> dict[str, dict | None]:
        data = self.http.request(WP_API, {
            "action": "query", "prop": "revisions", "rvprop": "ids|timestamp|content", "rvslots": "main",
            "titles": "|".join(titles), "redirects": 1, "format": "json", "formatversion": 2,
        }, min_interval=0.5)
        q = data.get("query", {})
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
        pages = {p["title"]: p for p in q.get("pages", [])}
        out = {}
        for t in titles:
            final = norm.get(t, t)
            final = redir.get(final, final)
            p = pages.get(final)
            if p is None or "missing" in p or "invalid" in p:
                out[t] = None
            elif not p.get("revisions"):
                out[t] = "RETRY"
            else:
                rev = p["revisions"][0]
                out[t] = {"title": p["title"], "revision_id": rev["revid"], "revision_time": rev["timestamp"],
                          "wikitext": rev["slots"]["main"]["content"]}
        return out

    def prefetch(self, titles):
        missing = [t for t in sorted(set(t for t in titles if t)) if self.http.cache_get("wiki", t) is None]
        for batch in chunks(missing, 10):
            got = self._fetch(batch)
            for t, v in got.items():
                if v == "RETRY":
                    v = self._fetch([t])[t]
                    if v == "RETRY":
                        v = None
                self.http.cache_put("wiki", t, {"page": v})

    def get(self, title: str | None) -> dict | None:
        """{'title','plaintext','revision_id','revision_time'} for the current revision, or None."""
        if not title:
            return None
        if title in self._pt:
            return self._pt[title]
        c = self.http.cache_get("wiki", title)
        if c is None:
            self.prefetch([title])
            c = self.http.cache_get("wiki", title)
        page = c["page"]
        rnt = None
        if page:
            rnt = {"title": page["title"], "plaintext": to_plaintext(page["wikitext"]),
                   "revision_id": page["revision_id"], "revision_time": page["revision_time"]}
        self._pt[title] = rnt
        return rnt


# --------------------------------------------------------------------------- AntiLeakBench logic
def check_exist(text: str, label: str, label_aliases: list[str] | None = None) -> bool:
    # upstream: rf"\b{label}\b" unescaped; escaped here so labels with regex metachars work.
    labels = [label] + list(label_aliases or [])
    patterns = [rf"\b{re.escape(x)}\b" for x in labels if x]
    if not patterns:
        return False
    return bool(re.search("|".join(patterns), text))


def check_valid_wikipedia(subject_label, subject_aliases, answer, answer_aliases, wikipedia: dict):
    context_text, context_summary = [], []
    for value in wikipedia.values():
        context_text.append(value["plaintext"])
        context_summary.append(value["plaintext"].strip().split("\n", 1)[0])
    context_text = "\n".join(context_text)
    context_summary = "\n".join(context_summary)
    if not (check_exist(context_summary, subject_label, subject_aliases)
            and check_exist(context_summary, answer, answer_aliases)):
        return None, None
    question_subject_label = ""
    for label in [subject_label] + list(subject_aliases):
        if check_exist(context_summary, label) and len(label) > len(question_subject_label):
            question_subject_label = label
    return context_text, question_subject_label


def get_old_new_items(history: list[dict], cutoff: str, current: str):
    """Verbatim AntiLeakBench logic; returns (result, reason)."""
    objects = [c for c in history if c["start_time"] is not None]
    objects = sorted(objects, key=lambda x: x["start_time"], reverse=True)
    old_item = next((c for c in objects if c["start_time"] <= cutoff), None)
    new_item = next((c for c in objects if c["start_time"] <= current), None)
    if old_item is None or new_item is None:
        return None, "no_prior_value"
    if new_item["object_qid"] == old_item["object_qid"]:
        return None, "same_object"
    if old_item["end_time"] is None:
        return None, "old_value_no_end_time"
    if new_item["start_time"] < old_item["end_time"]:
        return None, "overlapping_values"
    return {"old_item": old_item, "new_item": new_item}, None


def get_object(claims: list[dict], current: str):
    """Upstream MultihopDatasetBuilder.get_object: object with latest start <= current, else last found."""
    obj = None
    last = None
    for c in claims:
        last = c
        st = c["start_time"]
        if st is not None and st <= current and (obj is None or obj["start_time"] < st):
            obj = c
    if obj is None and last is not None:
        distinct = {c["object_qid"] for c in claims}
        if len(distinct) > 1:
            return None, "ambiguous_undated_values"
        return {"qid": last["object_qid"], "start_time": None, "end_time": None}, None
    if obj is None:
        return None, "no_claim"
    if obj["end_time"] is not None and obj["end_time"] <= current:
        return None, "current_value_ended"
    return {"qid": obj["object_qid"], "start_time": obj["start_time"], "end_time": obj["end_time"]}, None


def trim_doc(text: str, groups: list[tuple[str, list[str]]], n: int) -> str:
    """Trim to ~n chars keeping the lead and the relevant mention(s)."""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= n:
        return text
    pos = 0
    for label, aliases in groups:
        pats = [rf"\b{re.escape(x)}\b" for x in [label] + aliases if x]
        m = re.search("|".join(pats), text) if pats else None
        if m:
            pos = max(pos, m.end())
    if pos + 150 <= n:
        start = 0
    else:
        start = max(0, pos - n // 2)
        nl = text.rfind("\n", 0, start + 1)
        dot = text.rfind(". ", 0, start + 1)
        cut = max(nl, dot)
        if cut > start - 300:
            start = cut + 1
    end = min(len(text), start + n)
    if end < len(text):
        seg = text[start:end]
        b = max(seg.rfind(". "), seg.rfind("\n"))
        if b > n * 0.6:
            end = start + b + 1
    return text[start:end].strip()


# --------------------------------------------------------------------------- builder
class Builder:
    def __init__(self, args):
        self.args = args
        self.http = Http(CACHE_DIR, offline=args.offline)
        self.wd = Wikidata(self.http, args.cutoff, args.current)
        self.wp = Wikipedia(self.http)
        with open(METADATA_PATH, encoding="utf-8") as f:
            self.meta = yaml.safe_load(f)
        self.cutoff_cmp = args.cutoff  # upstream compares 'YYYY-MM-DDT..' <= 'YYYY-MM-DD' (strict < next day)
        self.current_cmp = f"{args.current}T23:59:59Z"
        self.stats = defaultdict(Counter)

    # ---- step 1+2+3: updates -> single-hop samples
    def singlehop(self, pid: str, cap: int):
        st = self.stats[pid]
        win = self.wd.window_claims(pid)
        st["window_claims"] = len(win)
        subjects = sorted({w["qid"] for w in win})
        st["window_subjects"] = len(subjects)
        rng = random.Random(f"{self.args.seed}:{pid}")
        rng.shuffle(subjects)
        subjects = subjects[:cap]
        st["subjects_examined"] = len(subjects)
        hist = self.wd.get_claims(subjects, pid)

        updates = []
        for q in subjects:
            rnt, reason = get_old_new_items(hist.get(q, []), self.cutoff_cmp, self.current_cmp)
            if rnt is None:
                st["drop:" + reason] += 1
                continue
            new = rnt["new_item"]
            if new["start_time"][:10] < self.args.cutoff:
                st["drop:new_before_cutoff"] += 1
                continue
            if (new.get("start_precision") or 0) < 11:
                st["drop:start_not_day_precision"] += 1
                continue
            if new["end_time"] is not None and new["end_time"] <= self.current_cmp:
                st["drop:new_value_already_ended"] += 1
                continue
            updates.append((q, rnt))
        st["updates"] = len(updates)

        ents = self.wd.get_entities({x for q, r in updates for x in (q, r["old_item"]["object_qid"], r["new_item"]["object_qid"])})
        wiki_opt = self.meta[pid]["wiki"]
        samples = []
        for q, r in updates:
            s, o_old, o_new = ents.get(q), ents.get(r["old_item"]["object_qid"]), ents.get(r["new_item"]["object_qid"])
            if not (s and o_old and o_new):
                st["drop:no_en_label"] += 1
                continue
            if "subject" in wiki_opt and not s["wiki_title"]:
                st["drop:no_enwiki_subject"] += 1
                continue
            if "object" in wiki_opt and not o_new["wiki_title"]:
                st["drop:no_enwiki_object"] += 1
                continue
            samples.append({"subject": s, "pid": pid, "object_old": o_old,
                            "object": dict(o_new, claim_id=r["new_item"]["claim_id"],
                                           start_time=r["new_item"]["start_time"],
                                           end_time=r["new_item"]["end_time"])})
        titles = []
        for sm in samples:
            if "subject" in wiki_opt:
                titles.append(sm["subject"]["wiki_title"])
            if "object" in wiki_opt:
                titles.append(sm["object"]["wiki_title"])
        self.wp.prefetch(titles)

        out = []
        for sm in samples:
            wikipedia = {}
            ok = True
            for side in ("subject", "object"):
                if side in wiki_opt:
                    page = self.wp.get(sm[side]["wiki_title"])
                    if page is None:
                        ok = False
                        break
                    wikipedia[side] = page
            if not ok:
                st["drop:wiki_fetch_failed"] += 1
                continue
            answers = [sm["object"]["label"]] + [a for a in sm["object"]["aliases"] if a != sm["object"]["label"]]
            ctx, qlabel = check_valid_wikipedia(sm["subject"]["label"], sm["subject"]["aliases"],
                                                answers[0], answers[1:], wikipedia)
            if ctx is None:
                st["drop:wiki_lead_missing_subject_or_answer"] += 1
                continue
            sm["question_subject_label"] = qlabel
            sm["wikipedia"] = wikipedia
            sm["hop1_wiki_qids"] = ([sm["subject"]["qid"]] if "subject" in wiki_opt else []) + \
                                   ([sm["object"]["qid"]] if "object" in wiki_opt else [])
            out.append(sm)
        st["singlehop_kept"] = len(out)
        return out

    # ---- warm caches for multihop traversal (batch requests instead of 1 per hop)
    def warm_multihop(self, samples):
        frontier = []  # (object_qid, remaining hop pids)
        for sm in samples:
            for opt in self.meta[sm["pid"]].get("multihop", []):
                frontier.append((sm["object"]["qid"], tuple(opt["hop_PIDs"])))
        depth = 0
        while frontier:
            depth += 1
            by_pid = defaultdict(set)
            for q, hops in frontier:
                by_pid[hops[0]].add(q)
            claims = {pid: self.wd.get_claims(qs, pid) for pid, qs in sorted(by_pid.items())}
            nxt, need_ents = [], set()
            for q, hops in frontier:
                obj, _ = get_object(claims[hops[0]].get(q, []), self.current_cmp)
                need_ents.add(q)
                if obj:
                    need_ents.add(obj["qid"])
                    if len(hops) > 1:
                        nxt.append((obj["qid"], hops[1:]))
            ents = self.wd.get_entities(need_ents)
            titles = []
            for q, hops in frontier:
                obj, _ = get_object(claims[hops[0]].get(q, []), self.current_cmp)
                wopt = self.meta[hops[0]]["wiki"]
                if "subject" in wopt and ents.get(q):
                    titles.append(ents[q]["wiki_title"])
                if obj and "object" in wopt and ents.get(obj["qid"]):
                    titles.append(ents[obj["qid"]]["wiki_title"])
            self.wp.prefetch(titles)
            frontier = sorted(set(nxt))
            log.info("warm depth %d: %d (qid,pid) claim sets, %d entities, %d titles",
                     depth, sum(len(v) for v in by_pid.values()), len(need_ents), len(set(titles)))

    # ---- step 4: multihop (upstream build_multihop_sample)
    def build_multihop_sample(self, sm, template: str, hop_pids: list[str]):
        pre_wiki_qids = list(sm["hop1_wiki_qids"])
        # gold docs: title -> {"text", "groups"}; hop-1 doc(s) first
        docs: dict[str, dict] = {}
        doc_order = []
        hop1_answer = (sm["object"]["label"], [a for a in sm["object"]["aliases"] if a != sm["object"]["label"]])
        hop1_subj = (sm["subject"]["label"], sm["subject"]["aliases"])
        for side in ("subject", "object"):
            if side in sm["wikipedia"]:
                p = sm["wikipedia"][side]
                docs[p["title"]] = {"text": p["plaintext"], "groups": [hop1_answer, hop1_subj]}
                doc_order.append(p["title"])
        wiki_title_of_qid = {sm["subject"]["qid"]: sm["subject"].get("wiki_title"),
                             sm["object"]["qid"]: sm["object"].get("wiki_title")}
        chain = [{"subject": sm["subject"], "pid": sm["pid"], "object": sm["object"]}]
        answers = None
        for rel_pid in hop_pids:
            pre = chain[-1]
            qid = pre["object"]["qid"]
            ent_q = self.wd.get_entities([qid]).get(qid)
            if not ent_q or not ent_q["wiki_title"]:
                return None, f"hop_{rel_pid}:prev_object_no_enwiki"
            claims = self.wd.get_claims([qid], rel_pid).get(qid, [])
            obj, reason = get_object(claims, self.current_cmp)
            if obj is None:
                return None, f"hop_{rel_pid}:{reason}"
            ent_o = self.wd.get_entities([obj["qid"]]).get(obj["qid"])
            if not ent_o:
                return None, f"hop_{rel_pid}:object_no_en_label"
            wopt = self.meta[rel_pid]["wiki"]
            wikipedia = {}
            if "subject" in wopt:
                p = self.wp.get(ent_q["wiki_title"])
                if p is None:
                    return None, f"hop_{rel_pid}:wiki_fetch_failed"
                wikipedia["subject"] = p
            if "object" in wopt:
                if not ent_o["wiki_title"]:
                    return None, f"hop_{rel_pid}:object_no_enwiki"
                p = self.wp.get(ent_o["wiki_title"])
                if p is None:
                    return None, f"hop_{rel_pid}:wiki_fetch_failed"
                wikipedia["object"] = p
            answers = [ent_o["label"]] + [a for a in ent_o["aliases"] if a != ent_o["label"]]
            ctx, _ = check_valid_wikipedia(pre["object"]["label"], pre["object"]["aliases"],
                                           answers[0], answers[1:], wikipedia)
            if ctx is None:
                return None, f"hop_{rel_pid}:wiki_lead_missing_subject_or_answer"
            wiki_qids = ([qid] if "subject" in wopt else []) + ([ent_o["qid"]] if "object" in wopt else [])
            group_ans = (answers[0], answers[1:])
            group_subj = (pre["object"]["label"], pre["object"]["aliases"])
            if not set(wiki_qids).issubset(set(pre_wiki_qids)):
                for side, p in wikipedia.items():
                    if p["title"] not in docs:
                        docs[p["title"]] = {"text": p["plaintext"], "groups": []}
                        doc_order.append(p["title"])
                    docs[p["title"]]["groups"] += [group_ans, group_subj]
            else:
                # hop answered by an already-included doc: that doc must also keep this hop's mention
                for side, p in wikipedia.items():
                    if p["title"] in docs:
                        docs[p["title"]]["groups"] += [group_ans, group_subj]
            pre_wiki_qids.extend(wiki_qids)
            chain.append({"subject": pre["object"], "pid": rel_pid,
                          "object": dict(ent_o, start_time=obj["start_time"], end_time=obj["end_time"])})
        question = template.format(sm["question_subject_label"])
        return {"question": question, "answers": answers, "docs": docs, "doc_order": doc_order,
                "chain": chain}, None

    def finalize(self, raw, sm, hop_pids):
        """Trim gold docs, apply output rules. Returns item dict (without distractors) or (None, reason)."""
        answer, aliases = raw["answers"][0], raw["answers"][1:]
        if len(answer) > 60 or len(answer.split()) > 8:
            return None, "answer_not_short_label"
        q = raw["question"]
        if answer.lower() in q.lower() or check_exist(q, answer, [a for a in aliases if len(a) >= 4]):
            return None, "answer_in_question"
        gold = []
        for t in raw["doc_order"]:
            d = raw["docs"][t]
            text = trim_doc(d["text"], d["groups"], self.args.trim_chars)
            for lab, als in d["groups"]:
                if not check_exist(text, lab, als):
                    return None, "trim_lost_hop_mention"
            gold.append({"title": t, "text": text})
        if not check_exist("\n".join(g["text"] for g in gold), answer, aliases):
            return None, "gold_context_missing_answer"
        chain = raw["chain"]
        hops = [{"qid": h["subject"]["qid"], "pid": h["pid"], "value_qid": h["object"]["qid"],
                 "value_label": h["object"]["label"]} for h in chain]
        excluded = []
        for h in chain:
            excluded += [(h["subject"]["label"], h["subject"]["aliases"]), (h["object"]["label"], h["object"]["aliases"])]
        item = {
            "uid": f"antileak:{sm['subject']['qid']}:{sm['pid']}:{'-'.join(hop_pids)}",
            "question": q,
            "answer": answer,
            "aliases": aliases,
            "first_published": sm["object"]["start_time"][:10],
            "hops": hops,
            "lineage": [f"wd:{h['subject']['qid']}:{h['pid']}" for h in chain],
            "paragraphs": gold,
            "source": f"AntiLeakBench-method via Wikidata SPARQL {self.args.run_date}",
        }
        return (item, excluded), None

    # ---- step 5: Noise-k distractors (upstream sample_contexts, word-boundary exclusion)
    def add_distractors(self, item, excluded, pool, rng):
        gold_titles = {p["title"] for p in item["paragraphs"]}
        cands = [d for d in pool if d["title"] not in gold_titles
                 and not any(check_exist(d["text"], lab, als) for lab, als in excluded)]
        if len(cands) < self.args.noise:
            return False
        paras = rng.sample(cands, self.args.noise) + item["paragraphs"]
        rng.shuffle(paras)
        item["paragraphs"] = paras
        return True

    def run(self):
        a = self.args
        mh_pids = [p for p in self.meta if self.meta[p].get("multihop")]
        all_single = {}
        for pid in self.meta:
            cap = a.max_subjects_per_pid if pid in mh_pids else a.max_subjects_pool_pid
            log.info("== %s (%s): single-hop", pid, self.meta[pid]["str"].strip())
            try:
                all_single[pid] = self.singlehop(pid, cap)
            except Exception as e:  # keep going on one bad PID
                log.exception("PID %s failed: %s", pid, e)
                self.stats[pid]["error"] += 1
                all_single[pid] = []
            log.info("   %s", dict(self.stats[pid]))

        # distractor pool = every kept single-hop article (upstream: singledoc_dataset contexts)
        pool, seen = [], set()
        for pid in self.meta:
            for sm in all_single[pid]:
                for side, p in sm["wikipedia"].items():
                    if p["title"] in seen:
                        continue
                    seen.add(p["title"])
                    grp = [(sm["subject"]["label"], sm["subject"]["aliases"]),
                           (sm["object"]["label"], sm["object"]["aliases"])]
                    pool.append({"title": p["title"], "text": trim_doc(p["plaintext"], grp, a.trim_chars)})
        pool.sort(key=lambda d: d["title"])
        log.info("distractor pool: %d docs", len(pool))

        mh_samples = [sm for pid in mh_pids for sm in all_single[pid]]
        self.warm_multihop(mh_samples)

        # build all multihop candidates per first-hop PID
        cands = defaultdict(list)
        for pid in mh_pids:
            st = self.stats[pid]
            for sm in all_single[pid]:
                for ti, opt in enumerate(self.meta[pid]["multihop"]):
                    hop_pids = list(opt["hop_PIDs"])
                    st["mh_attempted"] += 1
                    raw, reason = self.build_multihop_sample(sm, opt["question"], hop_pids)
                    if raw is None:
                        st["mh_drop:" + reason] += 1
                        continue
                    fin, reason = self.finalize(raw, sm, hop_pids)
                    if fin is None:
                        st["mh_drop:" + reason] += 1
                        continue
                    st["mh_built"] += 1
                    cands[pid].append((ti, fin))

        # selection: round-robin across PIDs, interleave templates, <=per_pid, <=per_subject
        queues = {}
        for pid in mh_pids:
            by_t = defaultdict(list)
            for ti, fin in cands[pid]:
                by_t[ti].append(fin)
            for ti in by_t:
                by_t[ti].sort(key=lambda f: hashlib.sha1(f"{a.seed}:{f[0]['uid']}".encode()).hexdigest())
            inter = []
            for k in range(max((len(v) for v in by_t.values()), default=0)):
                for ti in sorted(by_t):
                    if k < len(by_t[ti]):
                        inter.append(by_t[ti][k])
            queues[pid] = inter
        rng = random.Random(a.seed)
        selected, per_pid, per_subj = [], Counter(), Counter()
        active = [p for p in mh_pids if queues[p]]
        while len(selected) < a.max_items and active:
            for pid in list(active):
                if len(selected) >= a.max_items:
                    break
                picked = False
                while queues[pid] and not picked:
                    item, excluded = queues[pid].pop(0)
                    subj = item["hops"][0]["qid"]
                    if per_subj[subj] >= a.per_subject:
                        self.stats[pid]["sel_skip:subject_cap"] += 1
                        continue
                    if not self.add_distractors(item, excluded, pool, rng):
                        self.stats[pid]["sel_skip:not_enough_distractors"] += 1
                        continue
                    selected.append(item)
                    per_pid[pid] += 1
                    per_subj[subj] += 1
                    picked = True
                if not queues[pid] or per_pid[pid] >= a.per_pid:
                    active.remove(pid)
        for pid in mh_pids:
            self.stats[pid]["selected"] = per_pid[pid]

        with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
            for it in selected:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="2026-03-02")
    ap.add_argument("--current", default="2026-09-28")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--per-pid", type=int, default=25)
    ap.add_argument("--per-subject", type=int, default=2)
    ap.add_argument("--noise", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trim-chars", type=int, default=1500)
    ap.add_argument("--max-subjects-per-pid", type=int, default=1500,
                    help="cap on candidate subjects examined for PIDs with multihop templates")
    ap.add_argument("--max-subjects-pool-pid", type=int, default=150,
                    help="cap for PIDs without multihop templates (distractor pool only)")
    ap.add_argument("--run-date", default=None, help="date stamp for the `source` field (default: --current)")
    ap.add_argument("--offline", action="store_true", help="fail on any cache miss")
    args = ap.parse_args()
    args.run_date = args.run_date or args.current

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"build-{stamp}.log"
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info("AntiLeakBench-method build: cutoff=%s current=%s args=%s", args.cutoff, args.current, vars(args))
    log.info("deviations: escaped regex labels; deprecated rank ignored; ended/ambiguous hop values dropped; "
             "new claim needs day precision & not ended; word-boundary distractor exclusion; trimmed docs")

    b = Builder(args)
    selected = b.run()

    log.info("==== per-PID summary ====")
    for pid in b.meta:
        st = b.stats[pid]
        drops = {k: v for k, v in sorted(st.items()) if k.startswith("drop:")}
        mh_drops = {k[8:]: v for k, v in sorted(st.items()) if k.startswith("mh_drop:")}
        sel_skips = {k[9:]: v for k, v in sorted(st.items()) if k.startswith("sel_skip:")}
        log.info("%-6s window_claims=%d subjects=%d examined=%d updates=%d singlehop_kept=%d mh_built=%d/%d selected=%d",
                 pid, st["window_claims"], st["window_subjects"], st["subjects_examined"], st["updates"],
                 st["singlehop_kept"], st["mh_built"], st["mh_attempted"], st["selected"])
        if drops:
            log.info("       update/singlehop drops: %s", drops)
        if mh_drops:
            log.info("       multihop drops: %s", mh_drops)
        if sel_skips:
            log.info("       selection skips: %s", sel_skips)
    log.info("total items: %d -> %s", len(selected), OUT_PATH)
    log.info("http requests: %d, cache hits: %d, log: %s", b.http.n_requests, b.http.n_cache_hits, log_path)


if __name__ == "__main__":
    main()
