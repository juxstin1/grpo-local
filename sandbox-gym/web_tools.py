"""web_search / fetch_url: the Bionic rebuild's websearch MCP tools over a frozen corpus.

Port of C:\\Users\\juxxs\\bionic\\llama-studio\\reference\\websearch-mcp-index.js (v2.2.0).
Everything she sees -- argument handling, result layout, hint, error strings, objective
trimming, fetch_url offset paging -- is byte-for-byte (web_tools_parity.py diffs it against
Bionic's own JS); only the backend changes: SearXNG + live fetch become a
corpus object (sandbox-gym/corpus.py) with

  search(query: str, k: int) -> list[{"docid", "url", "title", "snippet"}]   deterministic BM25
  get(url_or_docid: str)     -> {"url", "title", "text"} | None

Spec with line refs and golden strings: docs/bionic-web-tools-spec.md.
"""

from __future__ import annotations

import math
import re
from urllib.parse import parse_qsl, urlencode, urlsplit

# --- hint selection (W:23-38) --------------------------------------------------
STRONG_RE = re.compile(r"E2B|E4B|9B|12B|13B|14B|20B|24B|26B|27B|32B|70B|gpt-oss", re.I)
HINTS = {
    "h0": "",
    "h1": "Hint: You should not rely solely on these snippets. Use fetch_url to pull the "
          "relevant page content and verify the details.",
    "h2": "NEXT STEP: call fetch_url with the 1-3 most promising URLs above to read the full "
          "pages. Do not answer from snippets alone.",
}


def hint_for(client_model: str, mode: str = "auto") -> str:
    if mode != "auto":
        return HINTS.get(mode, HINTS["h2"])
    return HINTS["h0"] if client_model and STRONG_RE.search(client_model) else HINTS["h2"]


# --- JS string semantics --------------------------------------------------------
def js_slice(s: str, n: int) -> str:
    """String.prototype.slice(0, n): counts UTF-16 code units, not code points."""
    if len(s) <= n // 2 or not any(ord(c) > 0xFFFF for c in s):
        return s[:n]
    out, units = [], 0
    for c in s:
        w = 2 if ord(c) > 0xFFFF else 1
        if units + w > n:
            if w == 2 and units < n:    # JS keeps a lone high surrogate; UTF-8 on the wire
                out.append("�")    # turns it into U+FFFD, which is what the model sees
            break
        out.append(c)
        units += w
    return "".join(out)


def js_len(s: str) -> int:
    return sum(2 if ord(c) > 0xFFFF else 1 for c in s)


# --- objective-aware trimming (W:144-167) --------------------------------------
STOP = set("the a an my your is are was were what who when where how to of for and or in on "
           "it that this with from".split(" "))


def terms(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if len(w) > 2 and w not in STOP]


def trim_to_objective(text: str, objective: str, budget: int) -> str:
    if js_len(text) <= budget:
        return text
    paras = re.split(r"\n\n+", text)
    if len(paras) < 2 or not objective:
        return f"{js_slice(text, budget)} …[truncated]"
    q = terms(objective)
    scored = []
    for i, p in enumerate(paras):
        st = set(terms(p))
        scored.append((p, i, sum(1 for w in q if w in st)))
    keep = {0}
    used = js_len(paras[0])
    for p, i, _ in sorted(scored[1:], key=lambda s: (-s[2], s[1])):
        if used + js_len(p) + 2 > budget:
            continue
        keep.add(i)
        used += js_len(p) + 2
    out = "\n\n".join(p for p, i, _ in scored if i in keep)
    if js_len(out) > budget:
        return f"{js_slice(out, budget)} …[truncated]"
    return f"{out}\n\n…[trimmed to the parts relevant to the objective]"


# --- search fan-out (W:204-257) -------------------------------------------------
def norm_url(u: str) -> str:
    try:
        p = urlsplit(u)
        if not p.scheme or not p.netloc:
            return u
        q = p.query
        pairs = parse_qsl(q, keep_blank_values=True)
        if any(k.lower().startswith("utm_") for k, _ in pairs):
            kept = [(k, v) for k, v in pairs if not k.lower().startswith("utm_")]
            q = urlencode(kept)
        search = f"?{q}" if q else ""
        origin = f"{p.scheme}://{p.netloc}".lower()
        path = p.path or "/"
        return f"{origin}{re.sub(r'/$', '', path)}{search}"
    except ValueError:
        return u


def fan_out_search(corpus, queries: list[str]) -> tuple[list[dict], int]:
    lists = []
    for q in queries:
        try:
            lists.append(list(corpus.search(q, 10))[:10])
        except Exception:  # noqa: BLE001 -- Promise.allSettled: a failed query is an empty list
            lists.append([])
    by_url: dict[str, list] = {}                 # insertion-ordered, like JS Map
    raw = 0
    for lst in lists:
        for rank, r in enumerate(lst):
            raw += 1
            key = norm_url(r["url"])
            e = by_url.setdefault(key, [r, 0.0])
            e[1] += 1 / (60 + rank + 1)
    merged = [r for r, _ in sorted(by_url.values(), key=lambda e: -e[1])]   # stable
    return merged, raw - len(merged)


def format_results(objective: str, queries: list[str], merged: list[dict], dupes: int,
                   hint: str) -> str:
    top = merged[:8]
    lines = []
    for i, r in enumerate(top):
        snippet = js_slice(re.sub(r"\s+", " ", str(r.get("snippet") or "")).strip(), 200)
        date = f" ({js_slice(str(r['publishedDate']), 10)})" if r.get("publishedDate") else ""
        lines.append(f"[{i + 1}] {r.get('title', '')} — {r['url']}{date}\n    {snippet}")
    nq = len(queries)
    text = (f"Results for objective: {objective}\n\n" + "\n".join(lines)
            + f"\n({len(top)} results from {nq} {'query' if nq == 1 else 'queries'}; "
              f"{dupes} duplicates merged)")
    if hint:
        text += f"\n\n{hint}"
    return text


NO_RESULTS = ("The web search returned no results — the search backend may be rate-limited or "
              "offline. Do NOT retry the same search. Answer from your own knowledge if you can, "
              "and note that you could not verify it via search.")


def web_search(corpus, args: dict, client_model: str = "", hint_mode: str = "auto") -> str:
    a = args or {}
    queries = a.get("searchQueries") if isinstance(a.get("searchQueries"), list) else []
    queries = [js_slice(str(q).strip(), 200) for q in queries]
    queries = [q for q in queries if q][:5]
    if not queries and a.get("query"):
        queries = [q for q in [str(a["query"]).strip()] if q]
    if not queries:
        return 'web_search needs searchQueries: ["…"] — 1-3 keyword queries.'
    objective = str(a.get("objective") or "").strip() or queries[0]
    try:
        if corpus is None:
            raise RuntimeError("no corpus attached to this workspace")
        merged, dupes = fan_out_search(corpus, queries)
        if not merged:
            return NO_RESULTS
        return format_results(objective, queries, merged, dupes, hint_for(client_model, hint_mode))
    except Exception as e:  # noqa: BLE001
        return f"Search failed: {e}"


# --- pagination: fetch_url `offset` (W: "pagination" section, v2.2) --------------
# No `offset` argument -> the legacy objective-trimmed excerpt, byte-identical. With an
# `offset` (0 included) -> the RAW window text[offset : offset+budget] in document order,
# never objective-trimmed (trimming is non-contiguous, so offsets into it would be
# meaningless). Offsets count UTF-16 code units, like JS. A cut window ends with one line:
CONTINUATION = "[truncated — {left} chars left; call fetch_url again with offset={next}]"
OFFSET_ERROR = ("fetch_url offset must be a whole number ≥ 0 — the character position to "
                "resume reading from.")


def continuation_line(left: int, next_offset: int) -> str:
    return CONTINUATION.format(left=left, next=next_offset)


def parse_offset(v):
    """JS parseOffset: None -> None (legacy); int >= 0 or digit string -> int; else ValueError."""
    if v is None:
        return None
    if isinstance(v, bool):                       # JS typeof true === "boolean"
        raise ValueError(v)
    if isinstance(v, int) and v >= 0:
        return v
    if isinstance(v, float) and v.is_integer() and v >= 0:
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"[0-9]+", v):
        return int(v)
    raise ValueError(v)


def js_sub(s: str, start: int, end: int) -> str:
    """String.prototype.slice(start, end) in UTF-16 code units; a split surrogate pair becomes
    U+FFFD, which is what the model sees after the UTF-8 wire."""
    if not any(ord(c) > 0xFFFF for c in s):
        return s[start:end]
    return s.encode("utf-16-le")[2 * start:2 * end].decode("utf-16-le", "replace")


def window_text(text: str, offset: int, budget: int) -> str:
    n = js_len(text)
    if offset >= n:
        return f"(offset {offset} is past the end — the page has {n} chars)"
    end = min(n, offset + budget)
    win = js_sub(text, offset, end)
    return f"{win}\n{continuation_line(n - end, end)}" if end < n else win


def render_page(text, note, max_chars: int, objective: str, offset) -> str:
    if not text:
        return f"(couldn't fetch — {note or 'no readable text found'})"
    if offset is None:
        return trim_to_objective(text, objective, max_chars)
    return window_text(text, offset, max_chars)


# --- fetch_url (W: fetchText / fetchUrlTool) -------------------------------------
def _fetch_and_extract(corpus, url: str, max_chars: int, objective: str, offset=None) -> str:
    doc = None
    note = None
    try:
        if corpus is None:
            raise RuntimeError("no corpus attached to this workspace")
        doc = corpus.get(url)
        if doc is None and norm_url(url) != url:
            doc = corpus.get(norm_url(url))
        if doc is None:
            note = "HTTP 404"
    except Exception as e:  # noqa: BLE001
        note = str(e)
    return render_page((doc or {}).get("text") or None, note, max_chars, objective, offset)


def fetch_url(corpus, args: dict) -> str:
    a = args or {}
    urls = a.get("urls") if isinstance(a.get("urls"), list) else ([a["url"]] if a.get("url") else [])
    urls = [u for u in (str(u).strip() for u in urls) if re.match(r"^https?://", u, re.I)][:3]
    if not urls:
        return 'fetch_url needs urls: ["https://…"] — 1-3 full http(s) URLs.'
    try:
        offset = parse_offset(a.get("offset"))
    except ValueError:
        return OFFSET_ERROR
    objective = str(a.get("objective") or "").strip()
    per_url = max(1000, math.floor(8000 / len(urls)))
    texts = [_fetch_and_extract(corpus, u, per_url, objective, offset) for u in urls]
    return "\n\n".join(f"## {u}\n{t}" for u, t in zip(urls, texts))
