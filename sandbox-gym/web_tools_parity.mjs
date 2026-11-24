// Runs the Bionic websearch MCP's own JS (sections lifted from the source, never copied into
// grpo-local) over cases.json and writes the outputs. Driven by web_tools_parity.py.
// Bionic checkout: env BIONIC_DIR (default C:\Users\juxxs\bionic\llama-studio).
import fs from "fs";
import path from "path";
const BIONIC_DIR = process.env.BIONIC_DIR || "C:/Users/juxxs/bionic/llama-studio";
const src = fs.readFileSync(path.join(BIONIC_DIR, "reference", "websearch-mcp-index.js"), "utf8");
const grab = (start, endMarker) => {
  const i = src.indexOf(start); const j = src.indexOf(endMarker, i);
  if (i < 0 || j < 0) throw new Error(`section "${start}" … "${endMarker}" not found in ${BIONIC_DIR} (a Bionic checkout without fetch_url offset paging, i.e. before feat/fetch-url-offset, drifts from web_tools.py)`);
  return src.slice(i, j);
};
const code = [
  grab("const STOP", "// ── end fetch_url ──"),          // trimming + pagination + fetchUrlTool
  grab("function normUrl", "async function searxng"),
  grab("function formatResults", "// ── smoke test"),
].join("\n");
const mod = new Function(code + "\nreturn {trimToObjective, normUrl, formatResults, terms, fetchUrlTool};")();
const cases = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = [];
for (const c of cases) {
  if (c.kind === "trim") out.push(mod.trimToObjective(c.text, c.objective, c.budget));
  else if (c.kind === "norm") out.push(mod.normUrl(c.url));
  else if (c.kind === "fmt") {
    // RRF exactly as fanOutSearch, lists given
    const byUrl = new Map(); let raw = 0;
    c.lists.forEach((list) => list.forEach((r, rank) => { raw++; const key = mod.normUrl(r.url);
      const e = byUrl.get(key) || { r, score: 0 }; e.score += 1 / (60 + rank + 1); byUrl.set(key, e); }));
    const merged = [...byUrl.values()].sort((a, b) => b.score - a.score).map((e) => e.r);
    out.push(mod.formatResults(c.objective, c.queries, merged, raw - merged.length, c.hint));
  } else if (c.kind === "fetch") {
    // The whole fetch_url tool body; the network is the case's page map (missing = HTTP 404).
    const getText = async (u) => (Object.hasOwn(c.pages, u) ? { text: c.pages[u], note: null } : { text: null, note: "HTTP 404" });
    out.push((await mod.fetchUrlTool(c.args, getText)).text);
  } else throw new Error(`unknown case kind ${c.kind}`);
}
// The tool list the server advertises (ListTools), for the schema/description check.
const tools = new Function(`return {${grab("  tools: [", "\n}));")}};`)().tools;
fs.writeFileSync(process.argv[3], JSON.stringify({ cases: out, tools }));
