# Heuristic Assists — the catalog

[< Spec Index](index.md) · explanatory overview: [shephard.md](shephard.md) · the why + code pointers: [local-coder-massaging.md](local-coder-massaging.md)

Every assist Shephard applies to keep a small local model (9B-class) + an agent harness doing real agentic coding, in five families. **Name first, one line each** — the "why", thresholds, and code pointers live in [local-coder-massaging.md](local-coder-massaging.md); the conceptual overview in [shephard.md](shephard.md).

- **Nudges** — in-context directives that steer the *model* (it sees them) to break loops and force progress.
- **Massages** — silent repairs of the model's *output* so the *harness* accepts it (the model never knows).
- **Context shaping** — managing what the model sees and how much, so it fits the window and stays grounded.
- **Probes** — Shephard makes its OWN read-only tool calls for ground truth; verify by *doing*, not reading.
- **Reasoned guidance** — the PRIMARY response to a stuck loop: hand the reasoner FRESH ground truth (the repeated failing action + its output, the actual files re-read from disk, the lint probe) and have it author the coder's next step. Fires whenever there's ANY real signal — a repeated action counts even when lint is CLEAN (the action-loop case the old dirty-only gate dropped to a bare nudge); a groundless call (clean probe, nothing else) is skipped, because reasoning over nothing hallucinates. Canned directives remain only as the reasoner-unavailable fallback.

## Nudges

- **Repetition guard** — same tool + same args 3× → a grounded reasoner redirect authored from the repeated call + its actual output (the `[STOP — REPETITION DETECTED]` directive is now the reasoner-unavailable fallback).
- **Thrash guard** — same goal via varying commands, still failing (24-call window, productivity-gated) → force a diagnosis. (A former same-file-failing-3× guard was removed — it never earned its keep; interleaved partial successes reset its streak.)
- **Context-reset guard (reasoned excise)** — a repeat-loop past 4× → excise its own calls+outputs from context (collapsed to ONE marker that keeps the last result/error), then the reasoner rebuilds a SMALL clean working context from FRESH ground truth — the touched files re-read from disk, the repeated failure, and the ONE next step — replacing the canned reframe that used to point back at the now-stale transcript. Falls back to the canned excise when the reasoner is unavailable. Fires early on purpose: a weak model ignores the "stop" nudge, and leaving the repeated calls in its context lets it copy the pattern back out — so the context is cleaned before the loop bloats to 6+ copies.
- **Rumination guard** — a self-doubt spiral mid-generation → abort the stream and re-prompt.
- **Loop-text guard** — the same assistant preamble 3× (stopword-stripped, stemmed) → re-prompt at turn end.
- **Cyclic-pattern guard** — a 2–4-step tool cycle repeated 3× → block the call and redirect.
- **Dangling-intent guard** — "now I'll do X" then stops with no tool call → re-prompt to actually act.
- **Announce-without-act escalation** — repeated stalling → escalate to "one tool call, no prose."
- **Quality gate** — empty/short/echo/refusal response → re-prompt before spending a verifier call.
- **Completion gating** — a coder's no-tool-call turn ends only if it (a) changed files this task (ground-truth gate), (b) its code passes the repo's own diagnostics (probe gate), and (c) the reasoner completion critic finds the work actually done. *There is no small-model text-shape "verifier"* — it was removed: it false-negatived on finished work ("let me verify" read as a bail) and trapped a done coder in a done→"you did nothing, act"→`ls` loop forever. Ground truth + probe + critic judge the real work instead of the phrasing.
- **Ground-truth gate** — a coder turn ends only if it actually changed files.
- **Tool-call constraint** — a bail/stall retry forces a valid (or specific) tool call at the sampler.
- **Malformed-tool-call recovery** — an unparseable `<tool_call>` (bad quote/newline escaping in a multi-line command) → re-prompt to re-issue cleanly, steering multi-line work to write_file.
- **Failed-patch → rewrite** — a failed apply_patch → a whole-file `write_file` directive; when the target file is small enough (≤24 KB, stat-checked) the write_file tool is *forced* so the model can't keep re-patching.
- **Output-truncation guard** — the server stopped at the output-token cap (`done_reason`/`finish_reason == "length"`) mid-`write_file`, so the file is cut off. Instead of silently saving the partial file (which reads back as "incomplete" and drives an endless rewrite loop, each rewrite re-truncating at the same cap), abort the write and re-prompt: *"your output was cut off — build the file in small pieces, append the rest with `edit_file`, don't rewrite the whole thing."* Bounded like the other bail retries. The model otherwise gets NO signal it was truncated — a plain "wrote N bytes" reads as success.
- **write_file-default steering** — the prompt makes whole-file write the default; apply_patch Add disabled.
- **Tunnel-vision detector** — N calls with no new well-defined target (footprint stalled) → force a step-back.
- **Read-without-write loop** — 12+ reads with zero writes this turn → `[GATHERING WITHOUT ACTING]`: name what you know, then make the first change.
- **Search-rumination guard** — a re-worded repeat of a search already made this turn → HTTP 400 before the network hit.
- **Domain-fetch steer** — a search query that names a bare domain (`api.handle.me`, not a source filename) → the first result gets a "fetch `https://<domain>` directly and parse it" hint, and a *repeat* domain search's 400 carries the same steer. Redirects a coder that circles a domain in web_search toward actually curling it.
- **Fetch exact-repeat guard** — an identical external fetch (url + find + cursor) this turn → HTTP 400; internal hosts exempt.
- **Failing-fetch guard** — N consecutive external fetches with no 2xx → a soft "stop guessing URLs" nudge; localhost exempt.

## Massages

- **write_file → shell base64 (bidirectional)** — `write_file` lowered to `printf … | base64 -d > path` (byte-exact, escaping-proof); the recorded shell call re-presented as `write_file` so the model sees only its own tool.
- **Leaked-call recovery** — tool calls emitted as text (Hermes `<tool_call>` JSON, XML `<function=…>`, Gemma-fable `<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>`) → promoted to real calls; the Gemma path also strips its `<|channel>thought…` wrapper from reasoner prose.
- **Shell-name rewrite** — `ls`/`cat`/`grep`/`git` emitted as tool names → proper `shell` calls.
- **exec_command array fix** — `cmd` given as `["bash","-lc",…]` → routed to `shell` (else it execs a `[`).
- **Shell-args normalization** — a string command wrapped to `[bash,-lc,cmd]`; a double-wrapped array unwrapped to the inner command.
- **edit_file → apply_patch** — an `edit_file`/`str_replace` find-replace → a native `apply_patch` Update hunk.
- **read_file normalization** — a `read_file(path, range)` → a `shell` `cat`/`sed`.
- **Malformed-JSON repair** — botched `write_file` args (raw newlines/quotes) → path + content recovered.
- **Tool-argument normalization** — string→JSON parse, dict passthrough, else wrapped `{value}`; alternate call shapes (`{function:{…}}` vs `{name,args}`) unified.
- **Fenced-JSON tolerance** — tool args / control-model JSON wrapped in ``` fences → extracted.
- **apply_patch normalize** — unified diff → native; missing `+`/`-`/space prefixes repaired; end marker added.
- **apply_patch hunk-header normalize** — `@@ -L,N +L,N @@` line-numbers collapsed, anchor text preserved.
- **apply_patch wrapper collapse** — multiple `*** Begin/End Patch` wrappers (one per file) → one wrapper.
- **apply_patch Add-File block fix** — Add-File blocks with stray `@@`/`-` lines stripped to `+` content only.
- **apply_patch Add → write_file** — a pure file-creating patch → a robust whole-file write.

## Context shaping

- **Own concise base prompt** — ~20 lines, write_file-first; replaces the harness's ~351-line one.
- **Tool-menu trim** — ~9 curated tools instead of the full ~120.
- **Tool cheat-sheet** — plain-language per-tool usage + examples in the prompt (built-in schemas synthesized).
- **Window auto-detect + derived budget** — read real `n_ctx` from `/props`; budget = window − reserves.
- **Real-token calibration** — learn the model's real chars→token ratio via a rise-fast/fall-slow EWMA (~1.8× initial, up to 3.5×); budget against truth, not chars/4, covering BPE undercount on code/JSON.
- **Transcript pass-through** — every turn renders VERBATIM (no per-turn collapse/fold, no stale-read drop, no output supersession); only the *current* turn's reasoning is kept (older reasoning dropped). Reads, errors, and outputs survive intact because nothing is elided — the LLM summary carries older context. Loop-excision still removes a *detected loop's* own calls.
- **Model-generated compaction** — on overflow (incoming or outgoing) or a harness compaction call: chunk the history, have the compactor LLM summarize EACH chunk in free-form prose, then one final unifying pass. The model summarizes what it sees — no deterministic state-extraction schema. Replaces the old extract→merge→refine pipeline ([compaction-reference.md](compaction-reference.md), now historical).
- **Verbatim recent tail** — all steps since the last summary pass through exact (never summarized), so the freshest opaque values (addresses, hashes, IDs) survive intact even when older mentions were summarized.
- **Post-compaction opaque-string warning** — the handoff is labelled post-compaction so a resuming model treats *summarized* high-entropy strings as suspect and re-verifies them, instead of trying to regex-pin every opaque type.
- **Boilerplate strip** — the ~3K-token Codex developer boilerplate (permissions/apps/skills) is dropped before summarizing (it demonstrably derails the summarizer) and from the verbatim render.
- **Active-turn compaction (incremental)** — summarize a long turn's middle; reuse a rolling summary keyed by content hash so it isn't re-summarized every overflow (no GPU-pegging storm).
- **Compaction hardening** — per-chunk timeout + `<think>`-strip on the summarizer so a small compactor can't wedge a turn.
- **Persist via native compaction** — feed the harness honest `total_tokens` + the real probed `n_ctx` so its *own* stock usage-driven compaction fires; no special-casing.
- **Loop-excision inline collapse** — replace an excised loop with ONE coherent marker ("tried N times, unchanged"), never a gap that reads as "I haven't acted yet."
- **System-prompt compression** — bound an oversized incoming system prompt to budget: keep head + tail, elide the middle.
- **Overflow re-trim** — a context-overflow error → re-trim to the server's real numbers and retry, no crash.
- **Last-resort drop** — drop oldest messages (keep the request, strip orphan tool-results) as the fit floor.
- **Oversized-output guard** — output over a dynamic ceiling (% of window) losslessly reduced, or omitted with a "re-run narrower / grep / find=" pointer.
- **Semantic truncation** — never blanket char-truncate; per-tool rules preserve meaning (digits/caps/symbols kept).
- **web_fetch navigation** — paginate (`cursor`, line-snapped), `find=` a section (MIME-aware), real HTTP status + body.
- **Guard observability** — loop-guard firings logged (not just queued to the TUI) so a firing guard never looks dead.
- **Browser User-Agent** — auto-add a real User-Agent so sites don't block `curl`.
- **Better errors** — patch/network errors rewritten to say what to try next.

## Probes

- **Language-aware syntax floor** — always-available `py_compile` / `node --check` over disk files → the exact `file:line` for parse errors the model can't localize.
- **Repo probe discovery** — inventory the repo's ecosystems (JS/TS, Python, Rust, Go, JVM, .NET, PHP, Ruby, Elixir) → a ranked list of SAFE diagnostic commands.
- **Package-script vetting** — read `package.json` / Make / etc. scripts and vet each body (reject install/mutate/watch/service) before offering it.
- **Safe-command ranking** — confidence → run-first tier (typecheck/build → lint → unit → full → e2e) → value → cost; package manager chosen from lockfiles, tool confidence raised by config files (`tsconfig.json`, `ruff.toml`, `mypy.ini`, …).
- **Config/glue probes** — `shellcheck` (shell), `actionlint` (CI workflows), `terraform validate` (infra), anchored at the repo root.
- **Bounded probe runner** — run the top-ranked safe probes with a hard timeout, deadlock-free capture, never mutating the workspace.
- **Diagnostic parsers** — rustc/cargo, tsc, ESLint, pytest, and generic `file:line` output → structured findings + a one-line summary.
- **Command safety classifier** — unwrap wrappers (`sudo`/`npx`/`poetry run`/…), reject installers/mutators/watch, identify probe kind — never fooled by shell syntax, filenames, or branch names.
- **Ground-truth completion gate** — on a "done" claim, run the syntax floor + the top probe **AND the top test probe** (discovery ranks typecheck/lint above tests, so a top-1-only run green-lit a repo whose tests failed — a false completion sailed through that way); broken code or failing tests block completion and the exact `file:line` becomes the re-prompt. The gate now also logs a truth-capture line (probes run, findings, floor-clean) so a passing gate is auditable, not silent.

## Reasoned guidance

- **Reasoner role routing** *(built)* — a separate, cheap local "light reasoner" role, routable with failover, distinct from the coder.
- **Structural stuck-triggers** *(built)* — the tunnel-vision / read-without-write / repeated-failure detectors decide *when* guidance is needed.
- **Plan-first** *(built)* — on a new user task the reasoner is engaged FIRST to draft a short small-step plan (small steps, one thing at a time), pinned at the top of the coder's context for the whole turn. Cached per task → one gather-and-plan pass per turn, not per step. The planner **GATHERS before it plans**: it's given a READ-ONLY tool subset (`exec_command` read-only-enforced, `read_file`, `web_fetch`, `web_search`) and loops — inspect the working dir, read files, fetch docs/search — until it emits a plan **grounded in what it actually found**, not assumptions. Read-only by construction (a write/mutate command is refused — building is the coder's job); no cap on gather calls (it self-terminates by producing the plan, with a repeated-call stuck guard). Leaked dialects (Gemma's `<|tool_call>…`) are recovered by the shared parser, so quirky reasoners gather too.
- **Completion critic** *(built)* — on a "done" claim that already passed the deterministic gates (repo diagnostics), ONE reasoner call reviews the actual work — task + a digest of recent tool outputs + the final claim + the **FRESH lint/test probe results the harness just ran** — for shortcuts, invalid assumptions (an error/404 accepted as success, a test that passes for the wrong reason), and unmet requirements; concrete issues become the re-prompt. The probe digest distinguishes tests that **ran clean** from tests that **DID NOT RUN** (import error, nothing collected, timeout) — the latter do NOT count as passing, so a suite that can't even execute no longer sails through as "done" (it looked identical to a green suite to the deterministic gate, which is silent unless a probe emits `file:line`). Bounded (≤2 blocks/turn) so it can't nag forever. **Truth-captured:** the critic logs its raw output + parse result, because a silent failure (the reasoner call timed out, or produced JSON we couldn't parse) is otherwise indistinguishable from "approved" — a false completion passed exactly this way, with no critic verdict logged at all.
- **Loop → ground truth → reasoned guidance** *(built)* — the canned loop directives are soft prompt text a 9B ignores. So when ANY of the five loop guards fires (repetition / forced-diagnosis / tunnel-vision / context-reset / read-without-write), the response is **detect → gather ground truth → reason**. One shared gatherer (`ground_truth::GroundTruth`) assembles the FRESH signal: the repeated failing action + its actual output (surfaced from the loop detector), the files the model touched this turn re-read from disk, and the dirty-only lint/syntax probe. The reasoner is called whenever there is ANY real signal (`has_signal()`) and authors the coder's next instruction from it:
  - **action-loop** (clean lint) → the *repeated action* is the ground truth: "you keep running `cat` on a directory and getting 'Is a directory' — use `ls`", or "this search returns the same result — stop and write the code". This is exactly the case the old dirty-lint-only gate dropped to a bare nudge.
  - **dirty lint** (a real `file:line`) → name the exact file:line and the one targeted fix (don't rewrite blind).
  - **no signal at all** (clean lint AND no repeated action) → the reasoner is NOT called: a groundless reasoner hallucinates (observed live: it told a coder to "add the X-API-Key header" for what was actually a runtime `TypeError`), so the detector's canned directive stands. (We never ground on a one-off *past* failing output either — a later action may have fixed it, the stale-signal footgun — but an ACTIVE repeat proves its output is current.)
  
  Fallbacks in order: reasoner output → the raw dirty-lint `file:line` → the detector's canned directive. Bounded (≤6/task).
- **Context rebuild on flail (excise)** *(built)* — the TOP escalation (a loop past the excision threshold): after the loop's own turns are excised from context, instead of a canned reframe pointing back at the now-STALE transcript, the reasoner rebuilds a SMALL (~150-word; 4 labeled parts: task / real state / why stuck / one next step) clean working context from the same fresh ground truth (repeated action + files re-read from disk). Supersedes the generic redirect that turn; falls back to the canned excise when the reasoner is unavailable.
- **Course-change reset** *(built)* — the inverse failure of the redirect: the guards were BURYING a coder that had already worked out the fix. When a loop guard fired this turn yet the coder returns a real tool call, ask the reasoner if it is GENUINELY changing course (a different tool/target/strategy — not the same approach reworded; strict, refuse-by-default, ≤6/task). A **restart** ("rewrite from scratch", "fresh implementation", "start over") is deterministically REFUSED even when the reasoner says yes — restarting throws the work away and re-enters the SAME loop from the top, so it must not reset loop state (it fooled the reasoner into granting a reset in a live session). If yes, grant a short **grace window** (3 model-returns) that suppresses the transcript-derived loop nudges so the new approach gets unobstructed turns to execute — and reset the redirect budget. Rationale: the family-A detectors re-derive from the still-loopy history every turn, so a pivot's own new reads/curls keep them firing and re-bury it (the observed "it figured out to drop `/v1` but never got a clean turn to try it"). The hard repetition guard + search/fetch guards need no reset — they only block *identical* repeats / *similar* searches, which a genuine pivot isn't.
- **Escalation ladder** *(forward)* — after K unheeded canned nudges, escalate to the reasoner / a probe / a hard stop — a guard that has fired 50× without effect is not a guard.
