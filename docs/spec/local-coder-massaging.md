# Local Coder Massaging

[< Spec Index](index.md)

## Purpose

This document catalogs every intervention the routing and tool layers apply specifically to help **small local coder models** succeed where they would fail on their own. "Local coder" here means the model wired into the `light_coder` role — currently models in the 4B–30B range at Q4 / Q6 quants running via Ollama (e.g. `devstral-small-2:q4_k_m`, `gemma4:e4b`).

These models are far less capable than frontier cloud models at:
- Tool-use discipline (picking the right tool, getting the arg shape right)
- Patch-format precision (producing valid unified diffs or Codex-native patches)
- Self-correction after an error (they tend to retry the exact same broken call)
- Staying on-task vs. announcing intent and stopping
- Grounding API calls in real documentation rather than guessing URL paths

Everything below exists because we observed one of those failure modes in practice and chose to fix it in the orchestration layer rather than wait for better local models.

---

## At a glance

Plain-English summary of every intervention. Numbers match the detailed sections below.

1. **Skip extras in local-only mode** — Normally a router, reasoner, and compactor help the main model. Locally we can't afford extras, so the Coder handles everything.
2. **Trim the tool menu** — Codex has ~120 tools. Small models get confused by big menus; we show them ~10.
3. **Plain-English tool cheat sheet** — We paste simple "here's how each tool works, with an example" text into the system prompt.
4. **Rewrite shell-ish tool names** — Models call `ls`, `cat`, `grep` like tools. We catch that and convert to a proper `shell` call.
5. **Browser user-agent for curl** — Sites block `curl/8.0`. We auto-add a real browser UA to any curl command.
6. **Web search + web fetch** — No built-in web search in local mode, so we added two tools for looking up real API docs.
7. **Fix broken patches** — Small models mangle patches: (a) write git-diff format, (b) leave `@@ -1,6 +1,6 @@` line numbers, (c) forget `+` prefixes and closing markers. We fix all three automatically.
8. **Better patch error messages** — Default errors are cryptic. We rewrote the common ones to explain what to try next.
9. **Better network errors** — Instead of "error sending request", we show the real cause (DNS, TLS, connection refused, etc.).
10. **Catch announce-without-act** — When the model says "Now I'll do X" and stops, a judge model spots it and we re-prompt "take the action"; the nudge **escalates** to a hard "emit one tool call, no prose" demand after it's ignored. A second, **deterministic** core-level guard catches the same pattern without the judge (and works for cloud routes too). Completion is **ground-truth-gated**: a coder turn only counts as done if it actually changed files or the judge confirms. Up to 3 retries.
11. **Stop repetition** — (a) same tool + same args 3× → STOP block. (b) same file failing 3× → forced-diagnosis. (c) same signature recurring interleaved → forced-diagnosis, but **only failing recurrences count**, so healthy re-runs (passing tests, `ls`/`git status`) don't trip it. Past the escalation threshold the loop is **excised from context** and the model is told it's stuck.
12. **Trim old conversation** — Keep the most recent turn intact, summarize older turns, drop stale file reads, pin errors so the model can't forget them.
13. **Log the model's thinking** — Reasoning text goes to debug logs so we can explain weird behavior later.
14. **Reroute wrong picks** — If the router picked "text-only model" but the conversation has tool calls, we upgrade to the tool-capable Coder.
15. **Diagnostic logs** — Extra logging for "which tools did we pass?", "did the STOP fire?", "what did the bail judge decide?"
16. **LM Studio / OpenAI-compat support** — Ollama and OpenAI-style servers disagree on URLs, payload shape, tool-call encoding, etc. A "flavor" switch lets one codebase talk to both.
17. **Token + time budget knobs** — New per-role `max_tokens` and `timeout_seconds` in `config.toml`. Set either to `0` for unlimited (reasoning models need this).
18. **Pin current file contents** — Models forget a file changed and generate patches based on the old version. We pin live on-disk contents at the top of the prompt.
19. **Catch thinking loops** — Some models spiral: "Actually, wait. Hmm. Let me reconsider." We watch the stream for 6+ self-doubt phrases after half the token budget is burned, abort mid-generation, and re-prompt "stop second-guessing."
20. **Stream the coder's output** — To make #19 work, we switched from "send, wait, parse" to "open stream, watch tokens, abort if needed." Also lets us log reasoning in real time.
21. **Recover leaked tool calls** — local models emit tool calls as TEXT in several dialects (Hermes `<tool_call>` JSON, the XML-function `<function=…>` form, or malformed JSON). One recovery pass promotes them to real, executed calls instead of letting the action vanish as prose. Single shared implementation on the one unified local path.
22. **Tolerate fenced JSON** — newer control models wrap their JSON verdict in a ```json fence, which silently broke the classifier and completion verifier. We extract the JSON object so the fence doesn't matter.
23. **Honest local-only signals** — a cloud role skipped by `local_only` no longer logs as "model not found (config error?)"; and the classifier honors its configured timeout instead of a hard-coded 10s, so a slow local server degrades gracefully instead of failing every turn.
24. **One path for coder and reasoner** — both roles run the same streaming/recovery/overflow path with the **same full tool set** (a reasoning model can code too). The only behavioral difference is one bit: whether a text-only turn is a valid completion (`role_text_is_product` — a coder must act; a reasoner's text can be the answer). Everything else that makes a "reasoner" — temperature, reasoning budget, `max_tokens` — is per-endpoint config. And when the prompt overflows the server's real context, the path re-trims to the server's reported numbers and retries instead of crashing.
25. **Tool-call constraint** (`tool_choice`) — the baseline (`"auto"`, unset) leaves the tool-call *format* unconstrained, the source of the leak/fence/prefill bugs. We enforce a valid call at the sampler (`tool_choice="required"`, which llama.cpp grammar-constrains) **only on the bail/rumination/quality retry of an actor role** — turning the prose "emit a single tool call" nudge into an enforced one, without blocking text-completion or termination on normal turns. Reasoners are never forced; a per-role config `tool_choice` is an operator override. The constraint can also force a **specific** tool, not just "required" — used to force `write_file` when a stuck patch needs a rewrite (see §30). The A/B (prose-nudge vs enforced) is a paper result.
26. **Loop detection beyond consecutive-identical** — the hard guard only catches the *same* call repeated back-to-back. Three detectors catch the loops it misses, all productivity-gated: (a) the same **assistant preamble** repeated across turns (normalized: stopwords stripped, light-stemmed), (b) a short **cyclic tool pattern** (patch→test→cat→patch…), (c) the **same file re-edited** with near-identical content. (a) re-prompts at turn end (bounded); (b)/(c) block the call at dispatch and redirect. Drawn directly from the Ada-handle thrash that ran 14 min undetected.
27. **Servability guarantees** — three fixes so a long/looping session can't dead-end: `apply_patch` `*** Add File` on an existing path now **errors** (was a silent overwrite that fed a rewrite loop); a `local_only` failover chain that strips cloud now **appends the other local role** as a backup instead of collapsing to one; and when protected assistant bulk alone exceeds the window, the prompt is reduced **semantic-first** — inline compaction summarizes the over-budget region (older turns, or the active turn's own middle), with a bounded last-resort drop (`drop_to_fit`, always keeping the user request) only as the floor — so the prompt always fits without crudely discarding the turn's history (see §12, §24).
28. **Real-token calibration** — Stop trusting the `chars/4` estimate (it runs 1.8–2.8× low on dense code/JSON — a prompt the estimator put at ~13k tokenized to **37k** on the server). Learn each model's real÷estimate ratio from the server's reported `prompt_tokens` (and the overflow error), and budget against it, so a dense prompt fits the real window on the **first** attempt instead of overflowing and being re-trimmed. Seeds at a safe default, self-corrects after one response.
29. **Active-turn compaction** — When a single long agentic turn (no new user message) outgrows the window, summarize its **middle** (keep the request + the last few steps verbatim) instead of crudely dropping it. Compacts whichever region holds the bulk — and since trim already collapses *older* turns to a small prelude, that's almost always the active turn. Fires only when the prompt genuinely won't fit (not preemptively), so it doesn't tax every turn. Ordering: *trim (mechanical) → compact (semantic) → drop (floor)*.
30. **`write_file` is the default editor; `apply_patch` retired locally** — `apply_patch` never once succeeded for the 9B in our logs (it can't reproduce matching context — 16/16 failures, mostly "Failed to find expected lines"). The local edit path is now **`write_file`** (whole file — nothing to match, nothing to fail) by default, `edit_file` for snippets, with a "keep files small and focused" nudge baked into the tool description. `apply_patch` is unadvertised; an emitted `*** Add File` converts to `write_file` (so it can't hit "Cannot add: exists"); a failed `Update` is steered to a rewrite (§31). Forcing a *specific* tool uses `tool_choice`'s object form (§25).
31. **Failed-patch → `write_file` rewrite** — When a patch fails (usually the model is editing code an earlier **failed write never landed**, so its `Update` context can't match), the harness pins the file's real on-disk contents, injects a `[PATCH DID NOT APPLY — REWRITE THE FILE]` directive, and for small files **forces** the `write_file` tool so the model rewrites the whole file instead of re-patching. A rewrite overwrites — there's nothing to match — which lands one clean write and re-syncs the model's mental model with disk. Clears the instant a rewrite succeeds (no infinite directive). The old inline hint that said "re-read and re-patch" — the loop-feeding advice — was changed to point at `write_file` too.
32. **Surfaced harness actions** — compaction, token-ratio calibration, last-resort drops, and patch-rewrite steering now show as TUI nudges (`push_nudge`), not just log lines, so the operator can watch the harness work in real time (e.g. *"Compacted the active turn (148 steps summarized)…"*, *"Calibrated context budget — this model packs ~2.8× the tokens our estimate assumed"*).
33. **MIME-aware output reduction + `web_fetch` navigation** — large tool outputs are reduced **lossless-first** (HTML→text, JSON minify, then a guarded prose-strip) instead of blind-truncated. `web_fetch` paginates with a copy-paste `cursor=` token, supports `find=` to jump straight to a section, and surfaces the **real HTTP status + body** (a 404's documentation body isn't thrown away) with a stop-guessing nudge only after 3 consecutive failures. The authoritative design lives in [content-reduce.md](content-reduce.md).

---

## 1. Local-only mode (no cloud)

**Problem:** With `local_only` set there must be no cloud inference — ever — but we still want *real* routing across the local roles (a classifier to pick coder vs reasoner, a reasoner for analysis, a compactor for summaries), not a degraded "coder does everything" mode.

**What we do:** `local_only` means "no cloud," enforced at two layers, and otherwise routes normally:
- Cloud roles are **stripped from every failover chain** (classification, coding, compaction, reasoning) up front, and `resolve_role` refuses any cloud role outright (`FailureType::RoleUnresolvable`) — so no cloud dispatch can happen regardless of what a chain contains.
- The classifier still runs (cloud-stripped) to pick the route; if it can't classify, the chain walks and defaults to the coder.
- **Coder and reasoner are both local roles on the one unified streaming path** (see §24) — both are full coders with the same tools; only their completion behavior and per-endpoint config differ. The reasoner isn't something that "falls through to the coder."
- Compaction routes through the `compaction` chain's first usable local role.

**Code:** [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — `resolve_role` (cloud-role guard), the per-chain cloud-strip, and `local_role_profile` (coder vs reasoner). See also [[project_local_only_mode]].

**Log signal:** `local_only: bypassing classifier — routing to LightCoder`

---

## 2. Tool catalog curation

**Problem:** Codex exposes ~120 tools (MCP connectors, multi-agent orchestration, dynamic tools, etc.). Small models lose attention when handed that many schemas, or hallucinate tool names that look plausible. Only ~10 of those tools matter for day-to-day coding work.

**What we do:** Filter the tool list down to a curated 10 before sending to the local Coder.

```rust
const LIGHT_CODER_TOOL_NAMES: &[&str] = &[
    "shell", "apply_patch", "list_dir", "view_image", "update_plan",
    "local_web_search", "web_fetch", "request_permissions",
    "exec_command", "write_stdin",
];
```

The subset is controllable per endpoint via `tool_subset: Focused` (default) vs `Full`. `Full` sends the entire catalog for capable local models that can handle it.

**Code:** [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — `LIGHT_CODER_TOOL_NAMES` and the filter in `try_local_model`.

**Log signal:** `Passing tool set to local coder tool_count=N available_in_prompt=M tools_passed=[...] tools_dropped=[...]`

---

## 3. Per-tool system-prompt hint

**Problem:** The formal tool schema (JSON Schema) that gets sent in the Ollama request is exhaustive but small models don't read it carefully. They call `ls` as a tool name, pass `command: "ls -la"` as a string instead of an array, forget prefixes on `apply_patch` bodies, etc.

**What we do:** Append a plain-English hint block to the system prompt that lists each available tool with its exact arg shape and a concrete example. The hint is rendered by `build_tool_hint` using the same tool names that were actually passed, so a tool that got filtered out never appears in the hint.

Sample hint entries include directives like:
- "If you find yourself wanting to call `ls`, `rg`, `cat`, `git`, or `pytest` directly, that is wrong — wrap it as `shell` with `command: [\"bash\", \"-lc\", \"<the command>\"]`."
- apply_patch: two accepted formats (unified diff + Codex native) with a prefix rule spelled out.
- local_web_search: suggests pairing with `web_fetch` to read a specific result.
- web_fetch: "use this BEFORE writing code against an unfamiliar API or library."

**Code:** [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — `build_tool_hint` function.

---

## 4. Shell-command alias translation

**Problem:** Models trained on shell sessions emit tool calls like `{"name": "ls", "arguments": {...}}`, `{"name": "cat", "arguments": {"path": "foo.py"}}`, or `{"name": "grep", "arguments": {"pattern": "x"}}` — none of which are real Codex tools. Rejecting these outright wastes a turn.

**What we do:** Detect common shell-command aliases in the tool name and rewrite them into `shell({command: ["bash", "-lc", "..."]})`. Also normalize `shell` calls where `command` was passed as a string instead of an array.

**Code:** [codex-rs/routing/src/tool_aliases.rs](../../codex-rs/routing/src/tool_aliases.rs) — `translate_native_tool_calls`, `translate_to_shell_call`.

**Log signal:** `Translated tool call (native) from=ls to=shell command_line=...`

---

## 5. `curl` User-Agent injection

**Problem:** Many sites behave differently when receiving a `curl/X.Y` User-Agent vs a browser UA — they might serve simplified HTML, return CAPTCHAs, or block the request outright. Training corpora rarely show models setting `-A`, so they just call `curl <url>` and get useless responses.

**What we do:** When the shell handler dispatches a command whose argv starts with `curl` (or a full path like `/usr/bin/curl`), inject `-A "<default browser UA>"` after the `curl` token — unless a UA is already set via `-A`, `--user-agent[=]`, `-H User-Agent:`, or `--header User-Agent:`. The same treatment applies to free-form shell strings via regex: `curl https://example.com | jq` becomes `curl --user-agent '...' https://example.com | jq`.

The default UA is a current Brave/Chrome Linux string (Brave intentionally identifies as Chrome). The same constant is the default for `local_web_search` and `web_fetch`.

**Code:** [codex-rs/routing/src/curl_ua.rs](../../codex-rs/routing/src/curl_ua.rs) + shell handler integration in [codex-rs/core/src/tools/handlers/shell.rs](../../codex-rs/core/src/tools/handlers/shell.rs).

---

## 6. New tools: `local_web_search` and `web_fetch`

**Problem:** In `local_only` mode, OpenAI's built-in `web_search` tool is unavailable. Local coders can't look things up, so they guess URL paths and API shapes from priors — frequently wrong.

**What we do:**
- **`local_web_search`** — Brave Search API with a configured key. Returns ranked titles, URLs, and snippets. Single HTTP GET, no retries.
- **`web_fetch`** — Single HTTP GET against an arbitrary URL, with the Brave/Chrome browser UA. Returns the response body as text for text-like content types, a placeholder for binary. Body capped at 512KB, 30s timeout, only http/https schemes.

Both tools are in `LIGHT_CODER_TOOL_NAMES` and are advertised in the tool hint. The hint explicitly tells the model to use `web_fetch` **before** writing code against an unfamiliar API rather than guessing.

**Code:**
- [codex-rs/routing/src/local_web_search.rs](../../codex-rs/routing/src/local_web_search.rs)
- [codex-rs/routing/src/web_fetch.rs](../../codex-rs/routing/src/web_fetch.rs)
- [codex-rs/tools/src/local_web_search_tool.rs](../../codex-rs/tools/src/local_web_search_tool.rs)
- [codex-rs/tools/src/web_fetch_tool.rs](../../codex-rs/tools/src/web_fetch_tool.rs)
- Handlers under [codex-rs/core/src/tools/handlers/](../../codex-rs/core/src/tools/handlers/).

---

## 7. `apply_patch` input normalization

Local models mangle `apply_patch` in three distinct ways. The normalizer pipeline (`normalize_apply_patch_call`) runs two passes in order:

### 7a. Unified-diff translation

**Problem:** Models emit `git diff` format (`--- a/path` / `+++ b/path` / `@@ -L,N +L,N @@`) because that's what their training corpus is full of. Codex's native format (`*** Begin Patch` / `*** Update File:` / context-anchored hunks) is rare in training data.

**What we do:** Detect unified-diff input and translate to Codex format. Handles:
- `--- a/path` + `+++ b/path` → `*** Update File: path`
- `--- /dev/null` + `+++ b/new.py` → `*** Add File: new.py`
- `--- a/old.py` + `+++ /dev/null` → `*** Delete File: old.py`
- `@@ -L,N +L,N @@ <anchor>` → `@@ <anchor>` (line numbers stripped — Codex matches by context)
- Git noise (`diff --git`, `index abc..def`, `rename from`, mode lines) is skipped
- `\ No newline at end of file` markers are dropped
- `a/`/`b/` path prefixes stripped; tab-delimited `diff -u` timestamps stripped; bare paths accepted

### 7b. Hybrid hunk-header normalization

**Problem:** A different failure mode: the model uses the Codex envelope (`*** Begin Patch` / `*** Update File:`) but puts unified-diff-style hunk headers *inside* it — `@@ -1,6 +1,6 @@`. The unified-diff translator returns `None` because there's no `---`/`+++` file header, and Codex's parser treats everything after `@@ ` as a literal anchor line.

**What we do:** Inside `fix_apply_patch_body`, when we see an `@@` line, strip any ` -L[,N] +L[,N] @@` segment and preserve an anchor text if present.
- `@@ -1,6 +1,6 @@` → `@@`
- `@@ -17,7 +17,7 @@ def foo():` → `@@ def foo():`
- `@@ def bar():` → unchanged
- `@@` → unchanged

### 7c. Prefix repair + end-of-patch terminator

**Problem:** Models emit patch bodies where lines lack any `+` / `-` / ` ` prefix — they just paste code, expecting the tool to figure it out. Also commonly forget the closing `*** End Patch` marker.

**What we do:** Inside a hunk, if a line doesn't start with `+`, `-`, a single space, or empty, prepend `+` (treat as addition). If the body has `*** Begin Patch` but no matching `*** End Patch`, auto-append.

**Code:** [codex-rs/routing/src/tool_aliases.rs](../../codex-rs/routing/src/tool_aliases.rs) — `translate_unified_diff_to_codex`, `normalize_codex_hunk_header`, `fix_apply_patch_body`, and `normalize_apply_patch_call` which wires them together.

**Log signal:** `Translated tool call (native) from=apply_patch to=apply_patch command_line=apply_patch (unified-diff translation + fixed prefixes, N bytes)`

---

## 8. `apply_patch` error-message improvements

Even with normalization, some patches genuinely can't apply — the context lines don't match, the model intended something the tool can't guess at, etc. Default errors like `Failed to find context '-17,7 +17,7 @@'` don't tell the model how to recover.

### 8a. Unified-diff hunk header detection

**Problem:** If a unified-diff-style hunk header slips past normalization (rare edge case), the error that Codex apply_patch produces is opaque.

**What we do:** When `Failed to find context` fires and the context looks like `-N,N +N,N`, swap in a directive error explaining that Codex doesn't use line numbers and instructing the model to omit the header or use a real anchor line.

**Code:** [codex-rs/apply-patch/src/lib.rs](../../codex-rs/apply-patch/src/lib.rs) — `looks_like_unified_diff_hunk_header`.

### 8b. Empty-args interception

**Problem:** After bailing on a hard turn, the model sometimes calls `apply_patch({})` — a syntactically valid but empty tool call. Codex's default error is the terse `missing field input at line 1 column 2`, which doesn't help the model recover.

**What we do:** In the apply_patch handler, if `arguments` is empty or `{}`, return a directive error that shows the expected shape with a concrete example and offers an escape hatch ("or use a different tool if you don't actually need to modify a file").

**Code:** [codex-rs/core/src/tools/handlers/apply_patch.rs](../../codex-rs/core/src/tools/handlers/apply_patch.rs) — the `ToolPayload::Function` branch.

---

## 9. `web_fetch` error enrichment

**Problem:** `reqwest::Error::to_string()` typically produces `error sending request for url (...)` — no visible root cause. DNS failure, TLS cert mismatch, and connection refused all look identical to the model.

**What we do:** Walk the error's `source()` chain up to 5 levels, deduplicate messages, join with ` → `, and prepend a category tag: `[connect]`, `[timeout]`, `[redirect]`, `[body]`, or `[decode]`. A TLS hostname mismatch now surfaces the actual "no alternative certificate subject name matches target host name 'X'" message instead of being buried.

**Code:** [codex-rs/routing/src/web_fetch.rs](../../codex-rs/routing/src/web_fetch.rs) — `describe_reqwest_error`.

---

## 10. Completion verifier (bail detector)

**Problem:** Local models often end a turn with text that announces intent but takes no action — "I will update the imports and then run the tests" with no tool call to actually do it. Codex interprets any text-only response as the end of a turn, emits `task_complete`, and the user is left with a broken task.

**What we do:** three cooperating layers.

### 10a. Completion verifier (judge) + escalating re-prompt

- After each Ollama call, if the response has non-empty text and zero tool calls, send it to a small judge model (the Coder itself in local-only mode) with a prompt defining BAIL vs COMPLETE patterns.
- If the verdict is BAIL, inject a `continuation_prompt` as a synthesized user message. The first nudge is gentle ("re-issue it as a tool call"); once the model **ignores it and keeps emitting prose**, the nudge **escalates** to a hard constraint: *"STOP EXPLAINING. Your ENTIRE next message must be a SINGLE tool call — no prose."* This breaks the explain-instead-of-edit loop that simple repetition of the same nudge can't.
- `MAX_BAIL_RETRIES = 3` — up to 3 nudges before Codex gives up.

The verifier prompt explicitly covers: "I will X" / "Let me X" / "Now I'll X" and stops; plans stated without a tool call; findings restated without being applied; and **code blocks are never actions** (a markdown fence is a suggestion unless the same content went to `apply_patch` / `shell`).

The verifier uses `light_coder` as its endpoint in `local_only` mode (the classifier is offline by design); cloud mode uses the fast classifier. Its JSON verdict is parsed **fence-tolerantly** (see #22) — a model that wraps `{"verdict":"bail"}` in a ```json fence was previously read as "unparseable" and defaulted to COMPLETE, silently letting a dangling turn finish.

### 10b. Ground-truth completion gate

The judge is a weak local model, so we don't trust it alone. A coder turn with no tool call only ends if **either** the active turn actually modified files (`files_modified_in_active_turn`) **or** the verifier returns COMPLETE. Text alone, with no file changes and no COMPLETE, never ends a coder turn — it re-prompts.

### 10c. Deterministic dangling-intent guard (route-agnostic)

The judge can still be fooled, and it doesn't run on cloud routes. So there is a second, **deterministic** guard at the core turn-completion gate (`run_turn`): if a turn is about to complete but the final message ends with an announced-but-unfulfilled action — a trailing colon plus a first-person action lead-in ("Let me fix the handler:") — re-prompt instead of completing. High-precision (needs both signals, so sign-offs and summaries don't trigger), bounded to 2 re-prompts/turn, and applies to **every** route including cloud.

**Code:**
- Verifier + escalating prompt: [codex-rs/routing/src/completion_verifier.rs](../../codex-rs/routing/src/completion_verifier.rs) — `verify_completion`, `continuation_prompt(prior, attempt)`.
- Ground-truth gate: [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — `did_real_work` / `files_modified_in_active_turn`.
- Deterministic guard: [codex-rs/core/src/codex.rs](../../codex-rs/core/src/codex.rs) — `ends_with_dangling_action` in `run_turn`.

**Log signal:** `Completion verifier judged a no-tool-call response verdict=Bail|Complete did_real_work=…` and `Model announced an action but stopped without a tool call; re-prompting`

---

## 11. Repetition alert

**Problem:** Local models frequently get stuck calling the same tool with the same arguments 5-10 times in a row, failing to learn from the identical outputs (or failures). Broader variant: repeatedly poking at the same **file** with subtly different commands (`cat foo.py`, then `wc -l foo.py`, then `head foo.py`) after every call fails — same failure mode but no exact-signature match.

**What we do:** Two detectors share the same STOP-block rendering:

### 11a. Exact-signature repetition

Walk the most recent `ToolCall` items and detect when 3+ consecutive calls share the same `(tool_name, signature)` where signature is a hash of the normalized args.

### 11b. Same-target-failure repetition

Walk the most recent `(ToolCall, ToolOutput)` pairs and detect when 3+ consecutive **failed** calls target the same file path, even with different argument shapes. Catches the "keep trying different ways to read a file that's broken" loop that 11a misses. Renders as a forced-diagnosis directive ("read the failure, state the root cause, then make ONE change").

### 11c. Unproductive recurrence (interleaved thrash)

The model works toward the same end with *varying* commands so 11a/11b miss it — e.g. running the same test 3× with edits between, never passing. Count tool-call signatures over a 24-call window; a signature recurring 3+ times is thrash. **Productivity-gated: only occurrences whose output FAILED count toward the threshold.** A signature that recurs but *succeeds* (a now-passing test, a routine `ls` / `grep` / `git status` re-run) is healthy and must not fire — and "diagnose the failure" is incoherent when there's no failure. This gate is what stopped the thrash nudge from firing constantly on normal repeated commands.

### Escalation ladder

Repetition rendering escalates with severity:
- `[STOP — REPETITION DETECTED]` for an exact byte-identical loop (11a).
- `[NO PROGRESS — DIAGNOSE …]` forced-diagnosis for the thrash variants (11b/11c).
- Past the escalation threshold (count ≥ 6), the loop's tool calls + outputs are **excised from the rendered context** so the model can't copy them out of its own history, and the prelude reframes to "you are stuck; the loop was removed."

**Code:**
- Detection: [codex-rs/routing/src/trim/state_extract.rs](../../codex-rs/routing/src/trim/state_extract.rs) — `detect_repetition`, `detect_same_target_failure_repetition`, `detect_unproductive_recurrence`
- Rendering: [codex-rs/routing/src/trim/render.rs](../../codex-rs/routing/src/trim/render.rs) — `render_repetition_alert`, `render_repetition_override`
- Signatures: [codex-rs/routing/src/trim/signatures.rs](../../codex-rs/routing/src/trim/signatures.rs)

**Log signal:** `Repetition alert fired — STOP block will be added to next prelude tool_name=X count=N`

---

## 12. Transcript trimming

**Problem:** Local models have small context windows (4K-32K tokens typical) and lose attention on long transcripts. Sending the raw Codex history would blow the budget and swamp the signal.

**What we do:** `trim_for_local` applies deterministic role-aware trimming:
- The **active turn** (everything from the most recent user message forward) is preserved verbatim.
- **Older turns** are replaced with a synthesized state prelude summarizing files seen, files modified, tests run, errors encountered.
- **Stale reads** (file reads followed by later writes that superseded them) are dropped.
- **Superseded outputs** (older tool outputs for files that have been re-read since) are dropped.
- **Errors are sticky** — any tool output containing an error is preserved regardless of age, so the model can't forget a failure and repeat it.
- The system prompt is never stubbed.

`trim_for_local` is **mechanical only** — it collapses older turns and truncates tool data, but it never drops whole messages. When even that leaves the prompt over budget (a long single active turn whose protected assistant bulk alone exceeds the window), the fix lives one layer up and is **semantic-first**: inline **compaction** summarizes the over-budget region — older turns if there are any, otherwise the **active turn's own middle** (keeping the user request + the most recent steps verbatim). Only if compaction is disabled or can't shrink it enough does a bounded **last-resort drop** (`drop_to_fit`, always keeping the user request) act as the floor that guarantees a servable prompt. This ordering — *trim → compact → drop* — is what stops a long agentic loop from either crudely losing its own history or dead-ending the turn. See §13 and `maybe_inline_compact`.

**Trigger calibration (the part that made compaction look broken).** The compactor must fire in **estimate space**, against the effective fit budget (`trim::effective_budget` = `trim_budget / SAFETY_FACTOR`), *not* a fraction of the raw `trim_budget`. The token estimate is `chars/4`; on JSON/code-dense content the real BPE count runs **1.8–2.8× higher** (observed live: a trimmed prompt the estimator put at ~13k tokens tokenized to **36.8k** on the server — a 2.84× undercount — and overflowed a 32k window). The old trigger (`0.85 × trim_budget`, real-token space ≈ 20.9k) sat far above any estimate the trimmer produces (~13k), so it **never fired**: the prompt overflowed while the estimate looked comfortably under budget, the server rejected it, and the overflow-retry loop scaled the budget down by the *real* overshoot and retried — repeatedly, per turn, never compacting. Triggering against `effective_budget` closes the dead band: an active turn whose estimate exceeds the fit budget now crosses the threshold and gets summarized **before** the first send.

**We don't have to estimate at all — the server tells us the truth.** Every response carries the real `prompt_tokens`, and an overflow carries `n_prompt_tokens`. So instead of a fixed safety factor we **learn the real ÷ estimate ratio per model** (`record_token_ratio`, EWMA, clamped `[1.8, 3.5]`) and budget against it: `calibrated_trim_budget` pre-scales the budget by `1.8 / observed_ratio`, so with an observed 2.84 the budget shrinks to ~63% and trim/compaction/`drop_to_fit` all target a prompt that actually fits on the **first** attempt. It seeds at the 1.8 default (so behaviour is unchanged until measured) and converges after the first response. Three details make it actually hold under load (the failure mode where it didn't: real prompts of 50–61k against a 49,664 window, *repeatedly*, while the estimate read ~13–19k):

- **Measure on the FULL prompt, not just messages.** The ratio is `real ÷ (messages + system + tool-schema estimate)`. If tools are left out of the denominator, the learned "ratio" silently absorbs the tool-overhead *fraction*, which swings as the message bulk grows and shrinks — so the number bounced 2.2→3.0 turn to turn and never settled. Including the schemas makes it pure tokenizer density, which is stable.
- **Reserve tool schemas in REAL tokens.** The schemas aren't in trim's chars/4 estimate at all. The old path subtracted their *estimate* inside the estimate-space budget (so it got ÷1.8'd along with everything else), under-reserving them by ~`ratio`×. We now subtract `tool_est × ratio` from the real window *before* calibrating the remainder, so `messages×ratio + schemas×ratio` lands exactly at the window.
- **Rise fast, fall slow.** The EWMA is asymmetric (`0.3/0.7` up, `0.8/0.2` down) because the costs are: under-estimating overflows the window (a wasted re-trim round-trip on a slow box); over-estimating just spends a little less context. A denser-than-seen turn pulls the guard up immediately; one light turn barely lowers it.

The active turn (where the bulk lives) is what gets compacted, since trim has already collapsed older turns to a small prelude. The chars/4 estimate now only sets the *starting* point — the server's real count corrects it within one turn. Code: `calibrated_trim_budget` / `record_token_ratio` / `observed_token_ratio`, the `real_tool_reserve` subtraction at the trim call site, and the `fit_budget` threaded through `maybe_inline_compact`. Tests: `token_ratio_rises_fast_and_falls_slow`, `budget_reserves_tools_so_real_prompt_fits_window`.

The same trimmer is also used as the first pass of compaction.

**Code:** [codex-rs/routing/src/trim/](../../codex-rs/routing/src/trim/) — entry point `trim_for_local` in `mod.rs`; `drop_to_fit` is the last-resort floor. Compaction split: `maybe_inline_compact` / `compact_active_turn` in [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs).

**Log signal:** `Trimmed transcript for local model trim_summary=kept N/M items; collapsed K older turns; dropped X stale reads, Y superseded outputs; elided Z chars; ~T input tokens`

---

## 13. Thinking / reasoning channel capture

**Problem:** Reasoning-heavy local models emit their chain-of-thought on a separate channel — `message.thinking` (Ollama) or `choices[0].delta.reasoning_content` / `message.reasoning_content` (OpenAI-compat). We don't feed it back to the model — it's private scratchpad, not user-facing — but losing it entirely makes debugging hard. When a local model makes a weird decision, the "why" often lives in the reasoning channel.

**What we do:** Accumulate reasoning deltas during streaming and, at turn end, log the full reasoning text at `debug!` level. Not part of the model's next-turn input; purely a diagnostic breadcrumb.

**Code:** [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — `StreamChunk::ReasoningDelta` branch of the coder's stream consumer.

**Log signal:** `Local coder reasoning channel reasoning_len=N reasoning_tokens=T reasoning=<content>` (debug level)

---

## 14. Conversation-state route override

**Problem:** A classifier or heuristic picks `LightReasoner` (a text-only route) but the transcript already has recent tool calls. Local reasoner models choke when handed an assistant message containing `tool_calls` without a corresponding tools array — they typically respond with empty output.

**What we do:** After classification but before dispatch, check if there are recent tool calls in the conversation. If so and the route is `LightReasoner`, upgrade to `LightCoder`. Deterministic override, layered on top of the classifier's output.

**Code:** [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — `conversation_has_recent_tool_calls` + the branch that upgrades the route.

**Log signal:** `Override: classifier picked LightReasoner but history has tool calls — upgrading to LightCoder`

(Moot in local-only mode since everything goes to `LightCoder` anyway, but preserved for cloud and mixed modes.)

---

## 15. Diagnostic logging

Beyond the per-feature log signals above, several diagnostics were added specifically because local-model problems are hard to reproduce outside the original session:

- **`tools_passed` / `tools_dropped` per turn** — reveals when a tool in `LIGHT_CODER_TOOL_NAMES` is missing from the session's `prompt.tools` (config drift or a feature flag).
- **`apply_patch (fixed prefixes, N bytes)` command line** — shows which normalization passes fired.
- **`Repetition alert fired`** — confirms whether the guard is actually triggering (separate from whether the model listened).
- **`Completion verifier judged`** — the final verdict and endpoint used, so we can tell "Bail was detected and we retried" from "Complete was returned too leniently".

All of these write to the standard tracing log (`~/.codex/log/codex-tui.log`).

---

## 16. OpenAI-compat wire adapter

**Problem:** Ollama and OpenAI-compat servers (LM Studio, llama.cpp's `server`, vLLM, LiteLLM, etc.) disagree on almost every surface: URL paths, payload shapes, tool-call JSON conventions, response-format hints, tool-result message roles. Writing Ollama-only code would lock out every OpenAI-compat server, which is most of the practical local-inference ecosystem.

**What we do:** A `ClientFlavor` enum on `OllamaEndpoint` (`Ollama` default, `OpenAICompat` selected via `provider = "openai-compat"` / `"lmstudio"` / `"openai"` in `config.toml`). Every wire operation branches on the flavor:

- **URL**: `/api/chat` (Ollama) vs `/v1/chat/completions` (OpenAI). Defensively strips trailing `/v1` so `http://host:1234` and `http://host:1234/v1` both resolve to the same endpoint.
- **Payload shape**:
    - Ollama — `options: { num_ctx, num_predict }`, `think: bool`, `format: "json"`.
    - OpenAI — top-level `max_tokens`, no `num_ctx` (server-set), no `think`.
- **Tool-call argument encoding**: Ollama accepts `arguments` as a JSON object; OpenAI requires `arguments` as a JSON-encoded STRING. Renderer branches so trimmed history matches what each server expects.
- **Tool-result messages**: Ollama expects `role: user` with the result wrapped in `<tool_result>` / `<tool_error>` tags; OpenAI expects `role: tool` with a `tool_call_id` field that matches the `id` on the assistant's `tool_calls[]` entry. The trimmer branches here too.
- **Streaming transport**: Ollama streams NDJSON (one JSON object per line); OpenAI streams SSE (`data: {...}` lines terminated by `data: [DONE]`). Two readers, one shared output enum (`StreamChunk`).
- **Usage decoding**: Ollama's `prompt_eval_count` / `eval_count` vs OpenAI's `usage.prompt_tokens` / `usage.completion_tokens` / `usage.completion_tokens_details.reasoning_tokens`.
- **Startup probe**: `/api/version` (Ollama) vs `/v1/models` (OpenAI).
- **response_format**: dropped for OpenAI-compat. The legacy `{"type": "json_object"}` shape that older OpenAI APIs accept is rejected by LM Studio (it demands `"text"` or `"json_schema"`, the latter requiring an actual schema we don't carry). Caller's system prompt enforces JSON instead — the same pattern the coder's tool-call flow already relies on.
- **Error surfaces**: non-2xx status bodies and HTTP-200 `{"error": ...}` bodies (some servers return 200 with an error field) are both decoded and logged so the caller gets a root cause instead of a silent `None`.

**Code:**
- Flavor enum, endpoint plumbing: [codex-rs/routing/src/config.rs](../../codex-rs/routing/src/config.rs)
- Wire branching: [codex-rs/routing/src/ollama.rs](../../codex-rs/routing/src/ollama.rs) — `build_chat_url`, `build_chat_payload`, `build_stream_payload`, `spawn_ollama_stream_reader`, `spawn_openai_sse_reader`, `translate_response_to_ollama_shape`
- Trimmer branching: [codex-rs/routing/src/trim/render.rs](../../codex-rs/routing/src/trim/render.rs) — flavor-aware tool-call and tool-result rendering

**Log signals:** `chat request returned non-success status url=... status=... body=...` / `chat response carried an error body — treating as failure`

---

## 17. Per-role `max_tokens` and `timeout_seconds`

**Problem:** Reasoning-capable local models (Qwopus 3.5, DeepSeek-R1, etc.) can legitimately take 5–30 minutes of wall clock for a single answer when they think heavily. The original 5-minute client timeout killed mid-flight inference; there was no way to set a per-role budget on either wall-clock time or output tokens.

**What we do:** Two new optional fields on the `[models.<role>]` config block:

- `max_tokens = N` — ceiling on generated tokens per response. `0` means unlimited (no cap). Maps to OpenAI `max_tokens` / Ollama `options.num_predict`. Normalized from `Some(0)` → `None` at config load.
- `timeout_seconds = N` — per-request wall-clock timeout. `0` means unlimited (no timeout). Applied to reqwest's `.timeout()` only when `> 0`, so unlimited = the `.timeout()` call is skipped entirely.

Both semantics mirror each other: `0` = the knob is off.

**Code:**
- Config shape: [codex-rs/routing/src/project_config.rs](../../codex-rs/routing/src/project_config.rs) + [codex-rs/routing/src/config.rs](../../codex-rs/routing/src/config.rs) — `endpoint_from_role`
- Wire plumbing: [codex-rs/routing/src/ollama.rs](../../codex-rs/routing/src/ollama.rs) — `build_chat_payload`, `build_stream_payload`, the `.timeout(...)` guards in `chat_with_tools` and `chat_stream`

---

## 18. Current-file-state prelude pin

**Problem:** A local model reads `foo.py`, edits it, reads it again, and then — several turns later — generates an `apply_patch` whose context lines are from the *original* read. The patch fails because the context doesn't match current disk state, and the model can't reliably reason about "which version of the file is authoritative" from scrolling back through the transcript.

**What we do:** The trimmer identifies files that were modified during the active turn (from tool outputs) and injects a dedicated `[Current file state — authoritative...]` block near the top of the prelude with the live on-disk contents. Capped at 10 KB per file; header includes a content hash, line count, and byte count so the model can cross-reference with its own mental model.

**Code:**
- Extraction: [codex-rs/routing/src/trim/state_extract.rs](../../codex-rs/routing/src/trim/state_extract.rs) — `files_modified_in_active_turn`
- Loading: [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — `load_active_turn_files`
- Rendering: [codex-rs/routing/src/trim/render.rs](../../codex-rs/routing/src/trim/render.rs) — `render_current_files`

---

## 19. Rumination detection (streaming)

**Problem:** Thinking-only local models (ones where the `<think>` channel can't be turned off — weights+template combo baked in) can spiral into self-interrupting loops: "Actually, wait. Let me reconsider. Hmm, on second thought..." until `max_tokens` runs out or the model finally stops. Symptom: after 2–10 minutes of wall clock, a response arrives with `content=""` and `tool_calls=[]`. The turn silently ends with no progress.

**What we do:** A streaming-time phrase-count detector watches the reasoning-channel deltas and aborts in-flight inference when the model shows signs of rumination.

- **Markers**: 23 case-insensitive word-boundary phrases characteristic of self-doubt — `actually`, `wait`, `but wait`, `hold on`, `hmm`, `let me reconsider`, `on second thought`, `let me think again`, `or maybe`, `or perhaps`, `rethinking`, `reconsider`, `going back`, `scratch that`, `nope`, `let me re-examine`, `let me revisit`, `i'm overthinking`, etc.
- **Budget gate**: Detector only fires once reasoning tokens exceed half of `max_tokens` (or half of a 4096 default when unset). Prevents false positives on a model that self-critiques once or twice during a normal chain.
- **Threshold**: ≥ 6 markers after the gate opens → flag as `Ruminating`.
- **Abort**: Dropping the SSE receiver closes the HTTP connection, signaling the server to stop generating and free its slot. No more tokens burned.
- **Re-prompt**: A `[RUMINATION GUARD]` continuation user-message is appended (hits count + approximate reasoning tokens) telling the model to pick the simplest next step and take it via a tool call, then the coder is re-invoked. Shares the same `MAX_BAIL_RETRIES = 3` cap as the completion-verifier loop.

**Code:**
- Detector (pure): [codex-rs/routing/src/rumination_detector.rs](../../codex-rs/routing/src/rumination_detector.rs) — `RuminationDetector`, `count_rumination_markers`, `continuation_prompt`
- Watcher wiring: [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — the coder's streaming loop

**Log signals:**
- `Rumination watch reasoning_chars=... reasoning_tokens=... budget_gate=... marker_count=... threshold=... gated=true|false` (every 500 bytes of new reasoning)
- `Rumination guard aborted local coder; re-prompting hits=... reasoning_tokens=... continuation_count=...` (on abort)

---

## 20. Streaming coder path with tool-call assembly

**Problem:** Rumination detection (section 19) requires watching reasoning as it streams — the non-streaming request-response pattern wouldn't let us see the loop until the full response returned, which is exactly what we wanted to avoid. But the existing tool-aware call (`chat_with_tools`) was non-streaming, and the existing streaming call (`chat_stream`) didn't carry tools.

**What we do:** New `chat_with_tools_stream` tool-aware streaming path. Two readers (Ollama NDJSON, OpenAI SSE) emit a unified `StreamChunk` enum covering four variants:

- `Delta(String)` — user-visible content delta
- `ReasoningDelta(String)` — private chain-of-thought delta (for the rumination watcher and for diagnostic logging)
- `ToolCallDelta { index, id, name, arguments_delta }` — incremental tool-call info. OpenAI streams tool-call `arguments` as multiple string fragments concatenated per `index`; Ollama typically emits whole tool calls atomically in the final chunk. The accumulator handles both.
- `Done { input_tokens, output_tokens, reasoning_tokens }` — stream terminator with usage.

Caller (`local_routing.rs`) consumes the stream, accumulates content / reasoning / tool-calls, runs the rumination check every 500 bytes of new reasoning, and on normal `Done` assembles a body in Ollama wire shape so the existing bail-verifier and tool-dispatch code works unchanged.

**Code:**
- Streaming pool method: [codex-rs/routing/src/ollama.rs](../../codex-rs/routing/src/ollama.rs) — `chat_with_tools_stream`, `spawn_ollama_stream_reader`, `spawn_openai_sse_reader`
- Caller-side assembly + watcher: [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — the streaming loop replacing the old `chat_with_tools` call, plus the inline `StreamToolCallAcc`

**Log signal (normal completion):** `Local coder response received content_len=... native_tool_calls=... reasoning_tokens=... continuation_count=...`

---

## 21. Leaked tool-call recovery

**Problem:** Local models frequently emit a tool call as **text** instead of as a structured `tool_calls` field — the server's chat template (even with `--jinja`) doesn't recognize the finetune's format, so the call arrives as prose. Without recovery the harness sees zero tool calls: the action silently vanishes and the turn can be mistaken for a completion. Observed in three dialects:

- **Hermes JSON** — `<tool_call>{"name":"exec_command","arguments":{…}}</tool_call>` (Qwen-family).
- **XML-function** — `<tool_call><function=exec_command><parameter=cmd>…</parameter></function></tool_call>` (Hermes-2-Pro / Qwen-Agent / Ornith). Tolerates `<function=NAME>` and `<function name="NAME">`, preserves multi-line shell commands verbatim, and coerces numerics (so `max_output_tokens` stays an int).
- **Malformed** — a `<tool_call>` block whose JSON doesn't parse (commonly a heredoc the model couldn't escape). Detected but not executed; instead the model is re-prompted to re-issue it cleanly and to prefer `write_file` over inline heredocs.

**What we do:** a **single** recovery pass promotes leaked calls to real, executed calls and strips them from the visible text. There is **one** entry point — `tool_recovery::recover_tool_calls` — called from the one unified local path (§24), so coder and reasoner recover identically. It internally uses `tool_aliases` for the `<tool_call>` dialects and also handles fenced JSON blobs and embedded `tool_use` blocks. (History: recovery used to be two parallel implementations plus a buried third copy across two separate code paths; a fix that landed in only the coder's copy let `light_reasoner` XML leaks slip through. Both the recovery *and* the paths are now unified.)

**Code:**
- Single entry: [codex-rs/routing/src/tool_recovery.rs](../../codex-rs/routing/src/tool_recovery.rs) — `recover_tool_calls`.
- `<tool_call>` dialect parser: [codex-rs/routing/src/tool_aliases.rs](../../codex-rs/routing/src/tool_aliases.rs) — `parse_leaked_tool_calls`, `parse_xml_function_call`, `has_leaked_tool_call`, `strip_leaked_tool_calls`.
- Call site: [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — the unified path (`tool_call_to_wire` → `translate_native_tool_calls`).

**Log signal:** `Recovered leaked tool call(s) from text — server didn't parse the model's tool-call blocks` / `Local model emitted an unparseable <tool_call> (malformed JSON); re-prompting`

---

## 22. Fenced-JSON tolerance for control models

**Problem:** The classifier and completion verifier ask a local model for a JSON verdict. Newer models (e.g. Ornith) wrap that JSON in a ```json markdown fence, so `serde_json::from_str` rejects it. The failure is silent and severe: every classification fails (`Classifier returned non-JSON` → chain exhausted → defaults every turn), and the verifier — which defaults to COMPLETE when it can't parse — lets dangling turns finish. Looks like a timeout or routing death but is purely a parse bug.

**What we do:** before parsing, extract the JSON object itself — slice from the first `{` to the last `}` — which tolerates code fences, surrounding prose, and leftover tags without a brittle fence-stripping ladder. Applied to both the classifier and the verifier.

**Code:** [codex-rs/routing/src/classifier.rs](../../codex-rs/routing/src/classifier.rs) — `extract_json_object`, used by the classifier parse path and by [completion_verifier.rs](../../codex-rs/routing/src/completion_verifier.rs).

**Log signal (the failure it fixes):** `Classifier returned non-JSON, falling back` disappearing; real `Request classified` lines returning.

---

## 23. Honest local-only failover signals

**Problem:** Two diagnostics actively misled debugging. (1) Under `local_only`, a failover chain that walks past a cloud role logged `model not found (config error?)` — implying a model-name typo when it's the intended cloud-skip. (2) The classifier hard-coded a 10s timeout, so a slow local server (e.g. a model split onto a slow second GPU) failed classification *every turn* while the coder, with a longer timeout, succeeded — read as a routing failure rather than a too-tight knob.

**What we do:**
- A role that won't resolve uses a distinct `FailureType::RoleUnresolvable` → logged as `role not resolvable (cloud disabled or unconfigured)`, not a model-name error. (Cloud roles are also stripped from chains up front under `local_only`, in classification, compaction, and dispatch alike.)
- The classifier honors its configured `timeout_seconds`, clamped to a sane 15–60s window, instead of forcing 10s.

**Code:**
- [codex-rs/routing/src/failover.rs](../../codex-rs/routing/src/failover.rs) — `FailureType::RoleUnresolvable`.
- [codex-rs/routing/src/classifier.rs](../../codex-rs/routing/src/classifier.rs) — `classify_timeout = timeout_seconds.clamp(15, 60)`.

**Log signal:** `Role not resolvable (cloud disabled by local_only, or unconfigured) — walking chain`

---

## 24. Unified local path: one streaming path for coder and reasoner

**Problem:** Coder (`light_coder`, tool-aware) and reasoner (`light_reasoner`, text) used to be two separate code paths — different pool methods, different response handlers, different recovery. They drifted, and every drift was a bug: a leaked-tool-call fix that landed only in the coder let `light_reasoner` leaks slip through; context-overflow handling existed in the coder and not the reasoner. The split also assumed "reasoner = no tools," which is wrong — a reasoner often needs to fetch docs to ground its reasoning.

**What we do:** there is now **one** local streaming path. Coder and reasoner run the same body — streaming, leak recovery (§21), rumination detection (§19), completion gating (§10), and overflow self-correction (below) — with the **same full tool set** (coding tools + `read_file` + edit tools). A reasoning model is a coding model with a different temperament, not a different capability, so every local role is a full coder. The **only behavioral difference** is one bit, `role_text_is_product(role)`:

| | tools | text is a valid completion? |
|---|---|---|
| **coder** | full set | **no** — must take an action |
| **reasoner** | full set (identical) | **yes** — text (analysis/answer/plan) can be the turn's product; finishes unless it *bailed* |

Everything else that makes a "reasoner" a reasoner — temperature, `reasoning` budget, `max_tokens` ("room to explore") — is **per-endpoint config** in its `[models.light_reasoner]` block, not behavior. A fix to the shared path now *cannot* land in only one of the two, and a coding task classified as reasoner is served by a full coder (which is why the old reasoner→coder route override was removed).

### Context-overflow self-correction

When the prompt's *real* tokenization exceeds the server's context window (the trimmer's estimate undercounts dense content like addresses/JSON/code by up to ~1.6×), llama.cpp returns a 400 `exceed_context_size_error` with the actual `n_prompt_tokens` and `n_ctx`. Rather than treat that as a dead stream and fail over to a tools-less role (which is how a truncated tool call once leaked as text), the path **re-trims to the server's real numbers and retries the same endpoint** — scaling the budget by how far it overshot. Context maxing out degrades gracefully instead of crashing the turn.

**Code:**
- One path + role profile: [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — `local_role_profile`, `build_local_tools`, `synthetic_local_read_tools`.
- Overflow signal + self-correction: [codex-rs/routing/src/ollama.rs](../../codex-rs/routing/src/ollama.rs) — `SendError::ContextOverflow`, `classify_send_error`; the re-trim/retry loop in `try_local_model`.
- Read tool: [codex-rs/routing/src/tool_aliases.rs](../../codex-rs/routing/src/tool_aliases.rs) — `normalize_read_file_call`.

**Log signal:** `Coder prompt overflowed server context — re-trimming smaller and retrying same endpoint` and `Passing tool set to local model … text_is_product=…`

---

## 25. Tool-call constraint (`tool_choice`)

**Problem:** With `tools` attached but no `tool_choice`, the OpenAI default is `"auto"` — and on llama.cpp that means **the tool-call format is unconstrained**. The model generates freely and the server parses a tool call out of the result *after the fact* (via `--jinja`). When the model's output doesn't match the template (the XML dialect, fenced JSON, a malformed heredoc, an assistant prefill), the parse fails and it leaks. Every format bug in §21–§23 happens under this default. It is the *baseline* most people get with a local model + tools, and it's the regime where small models are unusable for agentic work.

Confirmed on the test server (llama.cpp b8881): a request with `tool_choice: "required"` comes back with a clean structured `tool_calls` and empty content — no leak, no fence, no XML. The server genuinely grammar-constrains a valid call when told to. So the lever is real; the only question is *when* to pull it.

**The naive way is wrong.** Setting `"required"` as a blanket per-role default forces a tool call on *every* turn — the model can never answer in text or signal "done", so the agentic loop never terminates naturally. Same trap whether set statically or keyed off the classifier's `tools_potential` (which stays true for a whole coding task).

**What we do — enforce on the retry, not the turn.** The constraint is applied *exactly* where the model has already failed: when the completion verifier / rumination / quality escalation re-prompts an actor role that gave us a no-tool-call response we rejected (§7, §11, §12), the retry is sent with `tool_choice="required"`. This turns the existing "your entire next message must be a single tool call" *prose* nudge into a *sampler-enforced* one. It is scoped to the retry (a fresh turn starts clean), so normal text-completion and natural termination are never blocked. Reasoners (`text_is_product`) are never forced — their text is the product. Gate: `enforce_tool_call_on_retry(continuation_count > 0 && use_tools && !text_is_product && operator_tool_choice.is_none())`.

**Large writes (the other half).** The hardest format failure isn't *which* tool but the **JSON-string escaping of large/multiline file content** — a 9B dumps the file with raw newlines and bare double-quotes, so `serde_json` rejects the whole `write_file` object and the write is lost. The fix is to **tolerate the model's known limitation, not re-prompt it to "escape better"** (it can't):

1. `write_file`/`create_file` are **lowered to a `shell` base64 write and re-presented inbound as `write_file`** (§34), so the model still only ever sees its own tool — the real handler (`handlers/write_file.rs`) stays registered as a degradation fallback. (The earlier one-way `shell printf` translation, which the model *did* misread as mangled, is superseded: the inbound re-presentation is the half it lacked.)
2. When the args don't parse, `translate_one_native_call` calls `tool_aliases::recover_write_file_args` to pull `path` + `content` out of the raw text and rebuild a clean `{path, content}` object (content is taken verbatim between its opening quote and the last quote before `}`; recognized escapes are still decoded, so raw, escaped, and mixed all work). The handler then writes the file normally — no retry, no dead-end.
3. The `surviving_untranslated_synthetic` re-prompt now applies **only to the genuinely translated synthetics** (`edit_file`/`read_file`/`str_replace`/`cat_file`), which have no real handler to fall back to. **Regression fixed:** `write_file`/`create_file` were left in that list after gaining handlers, so *every* `write_file` call was flagged "malformed args" and re-prompted instead of dispatching — the source of a flood of `write_file arguments were malformed JSON` nudges.

When a re-prompt *does* fire (the translated synthetics), it routes through `enforce_tool_call_on_retry` so the retry is grammar-constrained.

**Operator override.** A per-endpoint `tool_choice` config field (`[models.*]`) still exists and always wins over the dynamic logic: set `"required"` to constrain unconditionally (for A/B experiments — expect the termination caveat), or a function name to force a specific tool. Unset = the dynamic enforce-on-retry behavior above.

**Why it matters (and the paper hook):** format discipline is where a 9B bleeds most; enforcing it at the source on the exact turns the model stalls collapses §21–§23 from load-bearing to fallback, without sacrificing the text-completion path. The clean A/B — **bail-recovery success rate, prose-nudge vs sampler-enforced** — is a quantifiable result. (A further refinement — a "free text **or** valid call" union GBNF applied to *every* turn — would constrain without the retry indirection, but the server's lazy grammar on `"auto"` isn't firing here, so that's future work.)

**Code:** [codex-rs/core/src/local_routing.rs](../../codex-rs/core/src/local_routing.rs) — `enforce_tool_call_on_retry` + the per-retry endpoint override in the coder loop; [config.rs](../../codex-rs/routing/src/config.rs) — `OllamaEndpoint::tool_choice` (operator override); [ollama.rs](../../codex-rs/routing/src/ollama.rs) — sent in `build_stream_payload` / `build_chat_payload` only when tools are present. Lives in `codex-routing` + the loop driver, so it travels into the [nudge service](nudge-service.md).

---

## 26. Loop detection beyond consecutive-identical

**Problem:** The hard repetition guard (`ToolRepetitionGuard`) only counts *consecutive byte-identical* tool calls and resets on any change. The Ada-handle session thrashed for 14 minutes — 19 `apply_patch`, 37 `exec`, the same "I see the issue… let me fix" preamble ~18× — and **nothing fired**, because the loop was a *cycle* (patch→test→cat→patch) and the edits/preambles varied just enough to keep resetting the counter. The user had to interrupt it.

**What we do:** a session-scoped [`LoopDetector`](../../codex-rs/routing/src/loop_detector.rs) with three productivity-gated detectors:
- **(a) repeated assistant text** — normalizes each turn-ending message to a content-token preamble (lowercase, apostrophes folded, stopwords dropped, light suffix-stem) and trips when the same preamble recurs ≥3×. Catches confident restating that the rumination detector (self-doubt markers only) misses. Re-prompts at turn end, bounded by `MAX_LOOP_TEXT_REPROMPTS` so a stuck model is nudged a couple of times then allowed to stop.
- **(b) cyclic tool pattern** — a ring of recent call signatures; trips when a period-2…4 cycle repeats ≥3×. Blocks the call at dispatch and tells the model to break the loop.
- **(c) same-target near-identical edit** — per-path fingerprints of recent edit content; trips when one file gets ≥3 essentially-identical edits. Blocks at dispatch. **Productivity gate:** genuinely different edits (content changes) never trip — that's progress.

**Why normalize for (a):** function words create a similarity floor (every sentence shares "the/a/is"), so stripping them + stemming makes "I see the issue… let me fix" and "Now I see an issue… I'll fix" collapse to the same fingerprint. See the design discussion on stopword/stemming trade-offs.

**Code:** `loop_detector.rs` (pure logic + tests); wired via `Session::note_loop_tool_call` (in `tools/router.rs`, beside the hard guard) and `Session::note_loop_assistant_text` (in `codex.rs`, at the turn-ending branch before the dangling-intent guard).

**Log signal:** `Blocking agentic loop (cycle/same-target guard)` / `Model is repeating itself without progress; re-prompting`.

---

## 27. Servability guarantees (no dead-end turns)

Three fixes ensure a long or struggling session degrades gracefully instead of dead-ending:

- **`apply_patch` Add-on-existing errors.** `*** Add File` on a path that already exists used to silently overwrite and report success — which let a weak model "re-create" a file forever, getting positive feedback each time (the Ada-handle loop). It now errors with *"Cannot add X: it already exists. Use `*** Update File`"*, preserving the file. [apply-patch/src/lib.rs](../../codex-rs/apply-patch/src/lib.rs).
- **`local_only` chains keep a local backup.** Stripping cloud roles could collapse a chain to a single local entry (`reasoning = [light_reasoner, cloud_*]` → `[light_reasoner]`), so one failure killed the turn. Since coder and reasoner now share the full-tool path, the chain build appends the other local role as a backup. [local_routing.rs](../../codex-rs/core/src/local_routing.rs).
- **Overflow-trim drops oldest turns.** `enforce_token_budget` only truncates tool data; a long single turn's protected assistant bulk can exceed the window on its own, leaving an oversized prompt the overflow-retry loop can't fix (the real root of the "reasoning chain exhausted" deaths). The trimmer now drops the oldest messages (front-first, recent context preserved) until the prompt fits. See §24. [trim/mod.rs](../../codex-rs/routing/src/trim/mod.rs).

---

## 28. System-prompt budget

**Problem:** The base system prompt is sent in the chat `system` field and was **never trimmed** (the "system is never stubbed" invariant). Codex's is ~9k real tokens; in a 32k local window that's a third of the budget gone before any conversation — a fixed, un-shrinkable floor that forces overflow on long sessions. And it's not Codex-specific: once this is a harness-agnostic service, *any* harness's system prompt lands in that field.

**What we do:** a configurable `[routing] system_budget_pct` (default **20**, `0` disables) caps the system prompt's share of `trim_budget`. Over budget, two tiers compress it:
- **(2) Cached LLM summary (primary).** `maybe_summarize_system` routes the oversized prompt through the **compaction track** (`compactor` role, falling back per the `compaction` chain), instructed to *preserve every rule / constraint / tool / output-format directive verbatim and drop only prose and examples*. Keyed by a content hash in a global cache, so the same prompt is summarized **once** and reused across requests/sessions — the right shape for a service. Falls back to ↓ if the compactor is unreachable or the summary didn't shrink.
- **(1) Deterministic head/tail elision (floor).** In `trim_for_local` (pure): keep the head (role/framing) + tail (output rules), elide the middle with a marker. Always available, offline-capable, and hard-enforces the budget even if the LLM summary overshoots.

The prompt's freshly-generated state prelude is always preserved on top, and the compaction *request* itself leaves its own system prompt uncompressed (full fidelity for the summarizer).

**Why generic, not a smaller Codex prompt:** hardcoding a leaner prompt fixes only this fork; bounding *whatever arrives* in the trim layer is what makes the harness reusable.

**Code:** [trim/mod.rs](../../codex-rs/routing/src/trim/mod.rs) — `compress_system_prompt` + the `system_budget_pct` plumbing; [local_routing.rs](../../codex-rs/core/src/local_routing.rs) — `maybe_summarize_system` / `summarize_system_via_compactor` + `SYSTEM_SUMMARY_CACHE`. **Config:** `[routing] system_budget_pct`.

**Log signal:** `Compressed oversized system prompt via compaction track (cached)`.

---

## 34. `write_file` → shell base64 (the bidirectional massage)

**The agnostic insight.** `write_file` is *not* the portable primitive — `shell` is. Every coding harness exposes command execution; `write_file`-as-a-named-tool is harness-specific convenience (Codex has none natively — we *added* the handler; Claude Code calls it `Write`; others differ). So the portable design is to let the **model** speak the ergonomic `write_file` while Shephard **lowers** it to the one substrate every harness has. This is the "Shephard owns no executors" principle made concrete: rich tools facing the model, irreducible primitives facing the harness.

**Outbound (`translate_one_native_call`).** A `write_file{path, content}` call becomes a `shell` call:
`mkdir -p '<dir>' && printf %s '<content_b64>' | base64 -d > '<path>' && printf 'write_file: wrote %s bytes…' …  # shephard-write:<path_b64>`.
**base64, not heredoc/printf-escaping:** the payload is pure `[A-Za-z0-9+/=]`, so it's immune to the entire shell escaping / quoting / heredoc-marker / trailing-newline bug-class — *any* content (quotes, newlines, `$`, backticks, even the literal sentinel text) round-trips byte-exact. The byte-count echo gives a small model positive confirmation the write landed (silent `>` redirects triggered "let me try write_file again" loops).

**Inbound (`represent_shell_writes`, runs before trim).** Codex records the *translated* `shell` call (not the model's `write_file` — confirmed at `ollama_tool_response_to_stream`, which emits the already-translated call). So next turn the transcript would show a base64 shell blob where the model called `write_file`. The inbound pass rewrites those recorded `shell` calls back to `write_file{path, content}` — recognized **statelessly** from the `# shephard-write:<path_b64>` sentinel (survives process restarts; no session map), call_id preserved so the output still matches. This is the half the **old one-way `shell printf` translation lacked** — without it the model saw a mangled shell command and panicked. Running before trim also means state-extraction / current-file pinning still see a `write_file` and recognize the write.

**Fallback.** The real `write_file` handler stays registered; the massage only declines (→ handler) if the args carry no usable `path`. base64 has no other failure mode, so in practice the massage always fires.

**Code:** [tool_aliases.rs](../../codex-rs/routing/src/tool_aliases.rs) — `write_file_to_base64_shell` / `parse_shephard_write`; [local_routing.rs](../../codex-rs/core/src/local_routing.rs) — the `write_file`/`create_file` match arm in `translate_one_native_call`, `represent_shell_writes` / `shell_command_str`. **Tests:** `write_file_base64_round_trips_hostile_content`, `parse_shephard_write_ignores_other_shell_commands`, `represent_shell_writes_restores_write_file`.

**Log signal:** `Translated tool call (native) from=write_file to=shell`.

---

## 35. Compaction: storm, hardening, and the persist endgame

**The storm.** Active-turn compaction (`compact_active_turn`) re-summarized the
**whole growing turn from scratch every overflow** — the summary lived in the
transient per-request `trimmed`, never persisted, so each turn trim re-derived the
full turn and re-ran the 4–5-chunk LLM pipeline. One stuck turn measured: **66 min,
~half of it inside compaction — 13 from-scratch runs, 60 chunk-extraction LLM
calls** ≈ the model's own 67 calls. That, not the model "thinking", was what pegged
the GPU (the model's own churn has only sub-second tool gaps but is at least *its*
work; compaction is pure overhead).

**Fixes (this layer):**
- **Incremental rolling summary.** The active turn is append-only, so cache the
  summary of the unchanged prefix (`ActiveCompactEntry`: prefix len + hash) and only
  LLM-compact the **new tail**, folding it into the prior summary. On the real loop
  content: re-compacted items dropped **2340 → 140 (16.7×)**. Falls back to a full
  compaction the moment the prefix changes (trim dropped/superseded an early item),
  so it's never wrong, only sometimes slower. Code: `plan_active_compaction`,
  `hash_prefix`, `active_compact_cache`.
- **Per-chunk timeout** (`EXTRACTION_TIMEOUT = 60s`) — a small compactor can wedge
  in a repetition loop (observed: one chunk generating 8+ min, freezing the turn);
  the deadline turns a wedge into a skipped chunk.
- **Fence-strip the extractor** — small compactors wrap the object in ```` ```json ````;
  `parse_extraction` runs `extract_json_object` first ([[project_local_model_fenced_json]]),
  so chunks stop dying on `expected value at line 1 column 1`.

**Why it's still a workaround.** The summary is transient because we're a stream
transform, not the conversation's owner. The endgame for a **local-only** harness is
to **persist** — drive the harness's NATIVE compaction (config-triggered, rewrites
history) at the real local window, so it's done once. Codex's native compaction
(`compact.rs::run_inline_auto_compact_task`) fires on
`total_usage >= model_auto_compact_token_limit` and is blind to the local window
(reads `model_context_window` from registry/config, never the server). Local turns
*do* feed its accounting (the local stream emits `Completed { token_usage }`), so the
tap-in is pure config: set `model_context_window` + `model_auto_compact_token_limit`
to the local values (we already detect the window via `/props`). See nudge-service.md
§ "Compaction: persist via the harness".

**Log signals:** `Active turn over budget — compacting its middle (full)` /
`Active-turn compaction reusing rolling summary (incremental)`;
`Compactor extraction timed out — skipping chunk`.

---

## 36. Context surgery without swiss cheese

When the context-reset guard **excises** a loop from the active turn, deleting the
calls into a **gap** is harmful: only the matching-signature calls go, so surviving
interleaved calls (the test runs) and dangling `<reasoning>` blocks remain with no
record of the edits between them — which a small model reads as *"I haven't acted
yet"* and **repeats**. Fix: collapse the excised loop into **one coherent inline
marker** at its position (`[loop collapsed — N repeated X attempts removed, result
unchanged, don't repeat]`), so the transcript still reads "I tried this N times".
Audit of the other surgery sites: older-turn compression renders as prose summaries
(safe, no protocol pairs); `drop_oldest_until_fit` now strips leading orphan
tool-results in **both** the OpenAI `role:"tool"` and Ollama `<tool_result>` forms.
Residual: loop `<reasoning>` blocks aren't keyed by call_id, so they survive — a
smaller follow-up. Code: `render_messages` (`loop_collapsed`), `drop_oldest_until_fit`.

**Observability fix (matters for diagnosis):** the loop guards surfaced *only* via
`push_nudge` → the TUI queue, which never reaches the tracing log — so from the logs
a guard that fired every turn looked like it "never fired". They now also
`warn!(guard=…, repetition_count=…)`. (This exact gap caused a multi-hour
misdiagnosis: context-reset *was* firing the whole time; the loop persisted because
excising can't stop a model that keeps re-attempting a bug it can't solve.)

---

## Keeping this document current

When you add a new intervention that targets local-model fragility, add a section here that covers:

1. **Problem** — a one-sentence description of the failure mode in the wild
2. **What we do** — the intervention
3. **Code** — file path(s) with clickable links
4. **Log signal** — the grep-able log line that proves the intervention fired

The goal is that a future engineer can read this document, understand why every weird knob exists, and confidently decide whether a given knob is still needed (maybe the next generation of local models doesn't need it anymore).
