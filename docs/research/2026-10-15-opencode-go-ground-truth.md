# OpenCode Go ground truth probing

- **Date:** 2026-10-15 (probed live, same day)
- **Issue:** #34 (three questions: docs-page table contract, endpoint liveness, `/models` payload shape)
- **Raw artifacts saved:** `/tmp/opencode/go-docs.html` (the docs page HTML, 54,645 bytes), `/tmp/opencode/zen-models.json`, `/tmp/opencode/inference-models.json`
- **Method:** `curl` with `-sS`, default `curl` UA unless noted; raw HTML scraped exactly as a Python `HTMLParser` sees it (static HTML, no JS rendering needed — see §1.1).

---

## 1. https://opencode.ai/docs/go/ — page structure and every table

### 1.1 Redirect + rendering

- `curl https://opencode.ai/docs/go/` returns **HTTP 302** → `https://opencode.ai/v2/docs/console/go/` (final **HTTP 200** after `-L`). Source: observed via `curl -sSL -w "%{http_code} %{url_effective}"`.
- The page is **server-rendered static HTML** — no `__next_f`, no hardcoded Radix/Next "Tabs" component, zero `<script>`-Required content. All tables are in plain `<table>` elements in the HTML payload exactly as fetched.
- Nav links reveal a larger v2 docs site (`/v2/docs/console/api/`, `/v2/docs/console/api/inference/`, …) — see §2/§4 for why that matters.

### 1.2 The PlanTabs mechanism (HTML parser's view)

There are exactly **two PlanTabs groups**, each a pair of `<table>` elements (one per plan):

```html
<div class="docs-plan-tabs" data-plan-tabs data-sync-key="go-plan">
  <div class="docs-plan-tabs-list" role="tablist" aria-label="Go plan">
    <button role="tab" id="go-pricing-tab-go" aria-controls="go-pricing-panel-go"
            aria-selected="true" tabindex="0" data-plan-tab="go">Go</button>
    <button role="tab" id="go-pricing-tab-go-plus" aria-controls="go-pricing-panel-go-plus"
            aria-selected="false" tabindex="-1" data-plan-tab="go-plus">Go Plus</button>
  </div>
  <div role="tabpanel" id="go-pricing-panel-go" ... data-plan-panel="go">…table…</div>
  <div role="tabpanel" id="go-pricing-panel-go-plus" ... hidden data-plan-panel="go-plus">…table…</div>
</div>
```

Key facts for a scraper (all verbatim from the fetched HTML at offsets 15121 / 15277 / 29086 / 29244):

1. **Both tabs' tables are fully present in the DOM.** The inactive panel is merely marked `hidden` as an attribute — its `<table>` content is intact and any naive HTML parser (like the one in `llm_opencode.py`) will parse **both** plan tables. There is no client-side-only rendering.
2. Active vs inactive is distinguishable only by the `hidden` attribute on the `<div role="tabpanel">` wrapper, or `aria-selected` on the buttons. A table-level parser sees no difference between the two tables.
3. Both tab groups share `data-sync-key="go-plan"`, so selecting "Go Plus" on one group switches the other too (JS behavior; irrelevant to scraping but explains why both tables exist per group).
4. IDs are stable hooks: `go-pricing-panel-{go,go-plus}` and `go-requests-panel-{go,go-plus}`; `data-plan-panel` and `data-plan-tabs` attributes exist for targeting.

### 1.3 Every `<table>` on the page (9 total)

Table order and offsets are positions of `<table>` in the fetched HTML:

| # | Offset | Header row (verbatim `<th>`s) | Data rows | Content |
|---|--------|-------------------------------|-----------|---------|
| 0 | 7156 | `Plan, Price, Included usage` | 2 | Plans overview: **Go $10/month**, **Go Plus $40/month** |
| 1 | 11593 | `Client, Session support` | 7 | Validated clients (Hermes, Claude Code, Codex, ZCode, Pi, jcode, …) |
| 2 | 13428 | `Client, Status and tracking` | 4 | Known problematic clients (DeepSeek Harness, Copilot Chat, Kimi Code, MiMo Code) |
| 3 | 15880 | `Model, Input, Output, Cached Read, Cached Write, Monthly limit` | 40 | **Pricing — Go tab** |
| 4 | 21478 | `Model, Input, Output, Cached Read, Cached Write, Monthly limit` | 40 | **Pricing — Go Plus tab** (same models, different "Monthly limit" values) |
| 5 | 29721 | `Model, Requests per 5 hours, Requests per week, Requests per month` | 32 | **Request limits — Go tab** |
| 6 | 32753 | `Model, Requests per 5 hours, Requests per week, Requests per month` | 32 | **Request limits — Go Plus tab** |
| 7 | 39388 | `Model, Model ID, Endpoint, AI SDK Package` | 32 | Model-ID → endpoint/protocol mapping |
| 8 | 46993 | `Model, Model training, Data retention` | 32 | Privacy policy per model |

Row counts are 1:1 duplicates on the plan dimension — tables 3≈4 (identical prices, limits differ: e.g. GLM-5.3-Flash monthly limit `$60` Go vs `$180` Go Plus), tables 5≈6 (limit values differ, e.g. GLM-5.3-Flash `6,320`/month vs `18,960`/month).

### 1.4 Scraping landmines found in the data cells

1. **Tier-suffixed display names in pricing, plain names in the request-limit table.** Pricing table (3/4) names like:
   - `Qwen3.7 Plus (≤ 256K tokens)` / `Qwen3.7 Plus (> 256K tokens)`
   - `DeepSeek V4.1 Flash (Off-Peak)` / `(Peak)`, `DeepSeek V4 Pro (Off-Peak)` / `(Peak)`
   - `Grok 4.7 (≤ 200K tokens)` / `(> 200K tokens)`, same for Grok 4.6, GPT 6 Luna, GPT 5.6 Luna, Claude Haiku 5.5 `(≤ 100K tokens)`
   The request-limit table (5/6) uses **plain** names (`Grok 4.7`, `DeepSeek V4.1 Flash`) for these. Display-name matching between the two table families fails for these rows unless names are stripped of brackets/peaks before matching.
2. **Appended promo span in the request table:** one model row's first cell is `Gemini 4 Argon Contributor2x · Limited time` — text inside the first `<td>` from a nested element. Cell `strip()` alone won't fix name matching.
3. **The only Model-ID table (7) covers 32 models, but the API serves more.** API models with no first-party ID row: `glm-5.1`, `minimax-m2.5`, `omen-alpha`, `qwen3.6-plus`, `qwen3.7-max` (see §2.2). Every ID in table 7 is in the API; the reverse is not true.
4. Pricing cells: `-` for Cached Write on almost all rows; `Free` text for inputs/outputs of free-tier previews; monthly-limit cell contains `<strong>$60</strong><br/><small>limited time</small>` style markup — plain-text extraction yields `"$60"` plus `"limited time"` concatenated if the parser flattens inline tags (current `HTMLParser` approach returns them as one string: `"60limited time"`-adjacent; needs care).
5. Plans/clients tables (0, 1, 2) share the `<table>` shape but have no "Model ID"/pricing headers — a header-identified parser should skip them; a position-based parser cannot.

---

## 2. Endpoint liveness (probed 2026-10-15 with curl)

### 2.1 Routes on opencode.ai

| URL (method) | Status | Notes |
|---|---|---|
| `GET https://opencode.ai/zen/go/v1/models` | **200** | Serves the full Go model list **unauthenticated** (see §3). Still alive. |
| `GET https://opencode.ai/inference/go/v1/models` | **200** | Same route exists on the new host too. |
| `POST https://opencode.ai/zen/go/v1/chat/completions` (with valid body) | **400** | Route exists. Body: `{"type":"error","error":{"type":"MissingSessionID","message":"Request is missing x-opencode-session and cannot be routed efficiently. Please see https://opencode.ai/docs/go/#where-can-i-use-it"}}` |
| `POST https://opencode.ai/inference/go/v1/chat/completions` (with valid body) | **400** | Identical `MissingSessionID` error. |
| `POST https://opencode.ai/zen/go/v1/messages` (valid Anthropic body) | **400** | Route exists (note: `GET` on this path returns **404**; POST is the routed method). |
| `POST https://opencode.ai/inference/go/v1/messages` (valid Anthropic body) | **400** | Same error. |
| `POST` same 4 routes with `{}` body | **500** | Empty/invalid body → 500 (unhelpful, but confirms routers are mounted). |
| `GET https://opencode.ai/zen/go/v1/docs` | 404 | |
| `GET https://opencode.ai/inference/go/v1/docs` | **404** | The URL named in the issue text does **not** exist as an API docs route. |
| `GET https://opencode.ai/zen/go/v1/models/glm-5.3` | 404 | No per-model detail sub-route under this prefix. |

**Conclusion:** `/zen/go/v1/` was **not** replaced. Both `/zen/go/v1/...` and `/inference/go/v1/...` serve identical routes today; whereas the **first-party docs page** (§1, table 7) now constructs endpoint URLs on the `/inference/go/...` host only: per model, `https://opencode.ai/inference/go/v1/chat/completions`, `.../v1/responses`, `.../v1/messages`, and — for the Gemini-family model — `.../v1/models/gemini-4-argon-contributor`. The docs page itself (the place we scrape) is served from `/v2/docs/console/go/`.

### 2.2 First-party prose docs for the console Inference API

`https://opencode.ai/v2/docs/console/api/inference/` (HTTP 200) documents a broader, non-Go console API: base `https://opencode.ai/inference`, per-provider paths `/inference/openai/v1/chat/completions`, `/inference/openai/v1/responses`, `/inference/anthropic/v1/messages`, `/inference/google/v1beta/models/<model>:generateContent`, Bearer-token auth (free models may omit it), and — relevant here:

> "See the model endpoints table, or fetch the live list: `curl https://opencode.ai/inference/v1/models`"

That live list (§2.1's `/inference/v1/models` — a different path from `/inference/go/v1/models`) returns **85 models** including non-Go entries (`big-pickle`, `*…*` free variants like `kimi-k2.6-free`, `mimo-v2.6-flash-free` …). So the operator's catalog now spans a shared `/inference` gateway, with `/inference/go/...` as the Go-subscription family slice. Source: fetched page values above.

---

## 3. Shape of the `/models` payload

### 3.1 Live payload — no auth needed

`curl https://opencode.ai/zen/go/v1/models` → **HTTP 200, unauthenticated**, 3,090 bytes:

```json
{"object":"list","data":[
  {"id":"deepseek-v4.1-flash","object":"model","created":1792017571,"owned_by":"opencode"},
  {"id":"deepseek-v4-pro", ...},
  {"id":"glm-5.1", ...}, {"id":"glm-5.2"}, {"id":"glm-5.3"},
  {"id":"grok-4.6"}, {"id":"grok-4.7"},
  {"id":"muse-spark-1.2-contributor"}, {"id":"muse-spark-1.3-contributor"},
  {"id":"glm-5.3-flash"}, {"id":"omen-alpha"},
  {"id":"gpt-5.6-luna"}, {"id":"gpt-6-luna"},
  {"id":"claude-haiku-5-5"},
  {"id":"hy3"}, {"id":"hy4-preview"},
  {"id":"kimi-k2.6"}, {"id":"kimi-k2.7-code"}, {"id":"kimi-k3"},
  {"id":"mimo-v2.5"}, {"id":"mimo-v2.6-flash"}, {"id":"mimo-v2.5-pro"}, {"id":"mimo-v2.6-pro"},
  {"id":"minimax-m2.5"}, {"id":"minimax-m2.7"},
  {"id":"gemini-4-argon-contributor"}, {"id":"minimax-m3"},
  {"id":"space-bunny"},
  {"id":"longcat-2.0"}, {"id":"longcat-2.5-preview-free"},
  {"id":"step-5-preview-free"}, {"id":"cat-jam-free"},
  {"id":"qwen3.6-plus"}, {"id":"qwen3.7-max"}, {"id":"qwen3.8-max"},
  {"id":"qwen3.8-flash"}, {"id":"qwen3.7-plus"}]}
```

**ALL fields per model item are exactly: `id`, `object` (always `"model"`), `created` (unix ts), `owned_by` (always `"opencode"`).** No pricing, context length, protocol, or capability fields. No pagination (`object: "list"`, no `next_cursor`). `https://opencode.ai/inference/v1/models` returns the same four-field shape.

Implication: the plugin **must** keep scraping the docs page for anything beyond `id`/`created`/`owned_by` — there is no richer first-party machine-readable source found:
- `https://opencode.ai/inference/go/v1/docs` — 404.
- `https://opencode.ai/zen/go/v1/docs` — 404.
- No per-model endpoint (`/zen/go/v1/models/<id>` — 404).
- The page prose at https://opencode.ai/v2/docs/console/api/inference/ documents endpoint families in prose + the "model endpoints table" (rendered as an HTML table on the Go docs page, not an API).

### 3.2 What `llm_opencode.py` assumes about the payload (working dir)

`llm_opencode.py` reads `["data"]` off `https://opencode.ai/zen/go/v1/models` (line 78, 306–311, 557–563), then merges only `id`, `created`, `owned_by` into the unified entry (`_unified_entry`, line 258–270). That matches the live payload. **However** the docs-driven half of the merge is the fragile part — see §4.

---

## 4. Deltas vs current code (actions implied for #35–#38)

1. **[#35, #38] Duplicate plan tables hit the parser twice.** The fetched DOM contains both the Go and Go Plus tables (§1.2–1.3); `parse_docs_tables()` iterates every table unconditionally, so **Go Plus values silently overwrite Go values** for `pricing.usage` (Monthly limit) and `rate_limits` — the plugin currently reports Go Plus numbers for every model as though they were the plan's. Fix: either drop tables inside `[data-plan-panel="go-plus"]` wrappers (requires wrapper-aware parsing beyond the current flat `HTMLParser`), or key an attribute like `hidden`/panel ancestor, or treat tables 3–6 as plan-indexed and expose both.
2. **[#35] Display-name matching breaks on tier suffixes and promo text.** Pricing rows say `Qwen3.7 Plus (≤ 256K tokens)`, `DeepSeek V4.1 Flash (Off-Peak)`, `Gemini 4 Argon Contributor2x · Limited time` (§1.4). The current `_normalize_name()` lookup against the endpoint table's plain names fails for those, so those variants either lose their IDs or merge under fractional keys; the rate-limit table meanwhile uses the plain names. Name normalization needs to strip bracketed tier/peak qualifiers and trailing promo snippets before joining.
3. **[#36] Endpoint host is documented as `/inference/...` now, not `/zen/...`.** All first-party endpoint URLs on the model-endpoints table (table 7) are `https://opencode.ai/inference/go/v1/...` — `chat/completions`, `responses`, `messages`, and a **third protocol style**: `.../v1/models/gemini-4-argon-contributor` (@ai-sdk/google), which the plugin does not support — the Gemini-family model currently falls into the default "openai" bucket or the anthropic bucket unintentionally. Hardcoded `BASE_URL_OPENAI = ".../zen/go/v1"` still resolves correctly today (both hosts mounted), but it contradicts what the docs we scrape assert; the model-endpoints table's Endpoint column should be treated as the source of truth per model rather than the hardcoded openai/anthropic/responses sets.
4. **[#36] Hardcoded protocol sets are drifting from data.** Live API list contains 5 models (e.g. `qwen3.6-plus`, `qwen3.7-max`, `glm-5.1`, `minimax-m2.5`, `omen-alpha`) with **no** first-party endpoint row to check them against, and `minimax-m2.5`, `mimo-v2-omni`, `kimi-k2.5`, etc. in the hardcoded sets are not in the live API at all (stale). Also new API IDs not yet in any code set: `omen-alpha`, `minimax-m3` is handled anthropic (correct per docs). The endpoint table should drive registration, with hardcoded sets only for models the table doesn't document.
5. **[#37] `/models` payload is minimal and unauthenticated.** Only `id/object/created/owned_by` (§3.1) — nothing richer exists first-party, so the CLI detail surface is right to merge docs data; but the current CLI prints `owned_by`-sourced "provider" — on this API that is always `"opencode"`, so a `--provider` filter is effectively a no-op and should be repurposed or documented.
6. **[#37] Session-header contract.** POST without `x-opencode-session` on any of `chat/completions`/`responses`-family routes returns **400 `MissingSessionID`** with an Anthropic-style error envelope `{"type":"error","error":{...}}` even on OpenAI-style routes (§2.1). The plugin's chat classes do set this header in `execute()`; that per-HTTP-client dual path (`extra_headers` for openai SDK vs `default_headers` for Anthropic SDK vs `self.headers` override for Responses) is correct per this probe, and worth keeping as a CLI-visible detail.
7. **[#38] Docs URL is a moving target.** `https://opencode.ai/docs/go/` is a **302 redirect** to `https://opencode.ai/v2/docs/console/go/` (§1.1). `fetch_docs_page()` follows redirects today so it works, but the redirect target (a versioned `/v2/...` path) is the one whose HTML layout (including `data-plan-tabs`/`data-plan-panel` attributes) this research describes — re-verify these attributes against the v2 page structure if the scrape contract is rebuilt. Also note the error message in the CLI directs users to `https://opencode.ai/docs/go/`, which still resolves.
8. **[#38] The docs page is static HTML (server-rendered), not a client-side React island — good news for the scraper.** No hydration-only content was found; all 9 tables and the PlanTabs state are in the wire payload (§1.1–1.2). A resilience concern remains only for the redirect target's layout skews (v1 → v2 markdown structure could differ if the docs are re-authored) — a canary on table count (9) and header sets would catch that.
