# Bionic web tools: spec for sandbox-gym `web_search` / `fetch_url`

Source of truth: the Bionic rebuild's websearch MCP server, vendored at
`C:\Users\juxxs\bionic\llama-studio\reference\websearch-mcp-index.js` (v2.1.0, spec 17 Stage A).
Below, `W:<n>` = that file, line n. Everything she sees (names, descriptions, arg names and
constraints, result layout, error strings) is copied byte-for-byte. Only the backend changes:
SearXNG + live fetch become the frozen corpus (`sandbox-gym/corpus.py`, deterministic BM25).

**Names:** the tools are `web_search` and **`fetch_url`** (not `fetch`) — W:273, W:297; the
llama-studio client routes on exactly these names (`src/lib/toolRegistry.ts:29`
`/^(web_search|fetch_url)$/`).

## web_search

| Item | Value | Where |
|---|---|---|
| description | `Search the PUBLIC INTERNET for external, real-world information: current events, news, documentation, public facts. Returns result snippets; follow up with fetch_url to read full pages. Do NOT use this for anything about the user or this conversation (their preferences, saved notes) — the memory tools own those.` | W:275 |
| arg `objective` | string, **required**. `Natural-language description of the web research goal, including source or freshness guidance and broader context from the task.` | W:281-284, W:293 |
| arg `searchQueries` | array of string, minItems 1, maxItems 5, **required**. `Concise keyword search queries, 3-6 words each. At least one query is required; provide 2-3 for best results. Maximum 5 queries, 200 characters per query.` | W:285-291, W:293 |
| query cleanup | each query `String(q).trim().slice(0, 200)`, drop empties, keep first 5 | W:342 |
| legacy salvage | no queries but `query` given → `[query.trim()]` (never rejected) | W:340, W:343 |
| objective default | `objective.trim() || queries[0]` | W:347 |
| no queries error | `web_search needs searchQueries: ["…"] — 1-3 keyword queries.` (isError) | W:345 |
| backend per query | top **10** results; a query that throws counts as an empty list (`Promise.allSettled`), it never fails the call | W:219, W:224-225 |
| merge | dedupe by normalized URL (drop `#hash`, drop `utm_*` params, lowercase origin, strip trailing `/` from path); Reciprocal Rank Fusion, score += `1/(60 + rank + 1)` (rank 0-based); sort by score desc (stable) | W:204-210, W:223-245 |
| top | first **8** merged | W:249 |
| snippet | `String(content).replace(/\s+/g, " ").trim().slice(0, 200)` | W:251 |
| date | ` (YYYY-MM-DD)` only if the result has `publishedDate` (corpus has none → omitted) | W:252 |
| line format | `[${i + 1}] ${title} — ${url}${date}\n    ${snippet}` (em dash U+2014, 4-space indent) | W:253 |
| body | `Results for objective: ${objective}\n\n${lines.join("\n")}\n(${n} results from ${q} ${q === 1 ? "query" : "queries"}; ${dupes} duplicates merged)`; dupes = raw results − merged | W:255 |
| hint | appended as `\n\n${hint}` when non-empty. `HINT_MODE=auto`: models matching `E2B\|E4B\|9B\|12B\|13B\|14B\|20B\|24B\|26B\|27B\|32B\|70B\|gpt-oss` get none (h0); everything else (incl. **Qwen3.5-4B**, unknown) gets **h2** | W:23-38, W:256 |
| h2 text | `NEXT STEP: call fetch_url with the 1-3 most promising URLs above to read the full pages. Do not answer from snippets alone.` | W:34 |
| no results | `The web search returned no results — the search backend may be rate-limited or offline. Do NOT retry the same search. Answer from your own knowledge if you can, and note that you could not verify it via search.` (not isError) | W:357 |
| backend exception | `Search failed: ${message}` (isError) | W:363 |
| unknown tool | `Unknown tool: ${name}` | W:337 |

## fetch_url

| Item | Value | Where |
|---|---|---|
| description | `Fetch 1-3 web pages and return their readable article text as markdown, trimmed to the parts relevant to your objective. Use this to READ pages — e.g. after web_search hands you links — so you answer from real content instead of snippets.` | W:299 |
| arg `urls` | array of string, minItems 1, maxItems 3, **required**. `1-3 full http(s) URLs to fetch` | W:303-308, W:312 |
| arg `objective` | string, optional. `Natural-language description of what to pull from the pages. Keep it specific so the excerpt stays focused.` | W:310 |
| legacy salvage | no `urls` array but `url` given → `[url]` | W:323-324 |
| url cleanup | `String(u).trim()`, keep only `/^https?:\/\//i`, first 3 | W:325 |
| no urls error | `fetch_url needs urls: ["https://…"] — 1-3 full http(s) URLs.` (isError) | W:327 |
| budget | per URL `max(1000, floor(8000 / urls.length))` chars | W:330 |
| body | `urls.map((u, i) => "## " + u + "\n" + text_i).join("\n\n")` (header is the URL as she wrote it, after trim) | W:332 |
| page text | live: defuddle/Readability → markdown, Jina fallback when < 800 chars (W:169-200). **Corpus:** the stored document text, no fallback | — |
| fetch failure | `(couldn't fetch — ${note})`, live notes `HTTP ${status}` / `non-HTML content: …` / error message. **Corpus:** URL not in corpus → `(couldn't fetch — HTTP 404)` | W:186-199 |
| trimming | `trimToObjective(text, objective, budget)` below | W:149-167 |

### trimToObjective (W:144-167), exact

```
STOP = "the a an my your is are was were what who when where how to of for and or in on it that this with from"
terms(s) = lowercase(s) matches of [a-z0-9]+, keep len > 2 and not in STOP
if len(text) <= budget: return text
paras = split(text, /\n\n+/)
if len(paras) < 2 or not objective: return text[:budget] + " …[truncated]"
q = terms(objective)                      # a LIST: duplicate objective terms count twice
score_i = number of q items present in set(terms(para_i))
keep = {0}; used = len(paras[0])
for s in paras[1:] sorted by (score desc, index asc):
    if used + len(s.p) + 2 > budget: continue
    keep s; used += len(s.p) + 2
out = "\n\n".join(kept paras in document order)
return out[:budget] + " …[truncated]"  if len(out) > budget
       else out + "\n\n…[trimmed to the parts relevant to the objective]"
```
(`…` is U+2026; the truncation marker has a leading space, the trimmed marker two newlines.)

## Golden strings (tests pin these)

`web_search({"objective":"capital rivers","searchQueries":["river city","capital"]})` against the
test corpus with 3 distinct hits, 1 duplicate across the two queries, model = 4B (h2):

```
Results for objective: capital rivers

[1] Alpha Town — https://example.org/alpha
    Alpha Town sits on the Blue River. It is the capital of the north.
[2] Beta City — https://example.org/beta
    Beta City is a river port.
[3] Gamma — https://example.org/gamma
    Gamma is the capital of the south.
(3 results from 2 queries; 1 duplicates merged)

NEXT STEP: call fetch_url with the 1-3 most promising URLs above to read the full pages. Do not answer from snippets alone.
```

`fetch_url({"urls":["https://example.org/missing"]})` →
```
## https://example.org/missing
(couldn't fetch — HTTP 404)
```

## Not mirrored (live-service only), with the deterministic equivalent

- SearXNG ranking/snippets → corpus BM25 top 10 per query; snippet = corpus `snippet` field
  (still passed through the W:251 whitespace-collapse + 200-char cut).
- Empty-results retry (W:226-233: wait 1.5 s, retry first query) → no-op; BM25 is deterministic.
- `_model` hint selection is client-injected (W:348-350); the gym passes the served model name
  and applies the same regex, so every 2B-4B run sees h2.
- JS `slice` counts UTF-16 units, Python counts code points; the port slices by UTF-16 units. A
  cut through an emoji leaves JS a lone surrogate, which UTF-8 on the wire turns into U+FFFD; the
  port emits U+FFFD there, i.e. the tokens she would actually receive.
- Parity is checked against the JS itself: `web_tools_parity.py` runs the vendored
  `trimToObjective` / `normUrl` / RRF + `formatResults` in node on 400+ random cases and diffs
  byte-for-byte (0 mismatches, 2026-10-02).
- Our harness lists tools in its own text block (same for every family). The tool **names,
  descriptions, argument names, required flags, limits and descriptions** are the Bionic ones;
  the wrapper that frames the tool list is the harness's.
