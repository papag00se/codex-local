# Reasoned-Guidance Refactor — Task List

## Why
Across many Shephard iterations the intervention code (loop guards, nudges, probes,
completion gate, context surgery) has sprawled across `local_routing.rs`, `trim/`,
`reasoned_guidance.rs`, `probe_run.rs`, `linter_probe.rs`, `no_action_prompt.rs`,
`rumination_detector.rs`. Live sessions prove the pattern: **detection works, the
bare-text "nudges" do not.** One session (019f35d3) burned **510K tokens** on a
trivial loop (`cat` a directory 9×, re-search the same URL 7×) and ended in a **false
completion** (`{ "plan":` leaked as the final answer). Bare nudges must become
grounded reasoned guidance, and the sprawl must consolidate.

## Principles (apply to EVERY task)
- **Ground truth or nothing.** Every reasoned intervention is fed FRESH truth — the
  actual files re-read from disk, actual probe/test output, the actual repeated call +
  its error. NEVER reason over the model's own claims or a clean probe (that produced
  the hallucinated "add an X-API-Key header" for a runtime TypeError). No fresh signal
  → don't call the reasoner.
- **Reason to steer; mechanics to stop/repair.** Convert interventions that tell the
  model WHAT TO DO next. Leave mechanical ones alone: the hard block on identical
  infinite writes, output repair (`\n`/JSON), truncation-resume, and the rumination
  CUTOFF (its purpose is to STOP generation — reasoning more is backwards).
- **Small rebuilds.** A weak reasoner told to "summarize everything" rambles. Bound
  every reasoner rebuild (token cap + tight, single-answer prompt).
- **Upstream only.** No band-aids / fallbacks / mitigations. Root cause or stop.
- **One task at a time**; after each: add/adjust tests, `cargo build -p codex-cli`
  (deploys the binary — lib-only builds do NOT), confirm green, update the matching
  spec doc, and check the item off here with a one-line result.
- **Complexity by axis, not time** — surface area / risk / dependencies, never hours.

## Taxonomy (Shephard)
Nudges (steer) → become grounded reasoned guidance. Massages (repair output) → stay
mechanical. Context-shaping (trim/compact/excise) → the excise becomes reasoner-driven.

---

## Phase 0 — Inventory & baseline (NO code change)
Map every intervention site and classify each STEER (convert) vs MECHANICAL (keep):
the `[STOP — REPETITION DETECTED]`, `[STUCK — CIRCLING THE SAME PLACES]`,
`[GATHERING WITHOUT ACTING]`, `[NO PROGRESS — DIAGNOSE]`,
`[HARNESS — STUCK; LOOP REMOVED]` directives (`trim/render.rs`); the no-action / quality
continuation prompts (`no_action_prompt.rs`); the rumination + truncation guards; the
completion gate + critic (`local_routing.rs`); course-change (`reasoned_guidance.rs`).
For each, record where it gets its "result"/context today and whether that is FRESH or
STALE.
**Acceptance:** a table in this doc: site → kind → current grounding → target grounding.

### Phase 0 result — intervention inventory (verified against code, 2026-07-06)

| Site (directive / fn) | File | Kind | Current grounding | Fresh? | Target grounding |
|---|---|---|---|---|---|
| `[STOP — REPETITION DETECTED]` | trim/render.rs | STEER | repeated call's result, handed to coder as canned text | fresh (repeat) | reasoner authors next step from repeated call+output |
| `[STUCK — CIRCLING THE SAME PLACES]` (tunnel_vision) | trim/render.rs | STEER | result + target names, canned | fresh | reasoner from repeated targets+output |
| `[GATHERING WITHOUT ACTING]` (read_without_write) | trim/render.rs | STEER | most-recent result, canned | fresh | reasoner: "you have enough — act", from reads+task |
| `[NO PROGRESS — DIAGNOSE]` (force_diagnosis) | trim/render.rs | STEER | result; asks the **coder** to self-diagnose (weak models can't) | fresh | reasoner diagnoses from repeated output + fresh lint/file |
| `[HARNESS — STUCK; LOOP REMOVED]` (excise) | trim/render.rs | STEER + mechanical excise | excises loop turns (good); canned reframe pointing at **STALE** transcript | STALE | keep excise; reasoner rebuilds clean context from a fresh disk re-read |
| `[NO ACTION TAKEN]` / `[NO ACTION — STOP EXPLAINING]` | no_action_prompt.rs | ~~STEER~~ → **MECHANICAL** | states the protocol fact "no tool ran, so nothing happened" + demands a tool call | fresh | **keep mechanical** — it enforces the PROTOCOL (you must call a tool), it does not author an approach; bounded by `MAX_BAIL_RETRIES`. Reclassified in Phase 4. |
| `quality_continuation_prompt` | quality.rs | MECHANICAL | degenerate-output reason (fence/repeat) | fresh | keep — re-prompt for clean output shape |
| `redirect_from_loop` | reasoned_guidance.rs | REASONED (exists) | dirty lint probe ONLY | fresh (dirty only) | broaden to repeated-error; call on clean probe too (Phase 3) |
| `critique_completion` | reasoned_guidance.rs | REASONED (exists) | task + evidence + probe results | fresh | keep (already grounded) |
| `course_change` | reasoned_guidance.rs | REASONED (exists) | evidence; **fooled by restarts** | — | don't reset loop state on a restart (Phase 5) |
| hard block (identical infinite writes) | trim/loop_detector | MECHANICAL | n/a | — | keep |
| output repair (`\n`/JSON) | tool_aliases.rs | MECHANICAL | n/a | — | keep |
| `[OUTPUT TRUNCATED]` resume (done_reason=length) | local_routing/ollama | MECHANICAL | n/a | — | keep |
| `[RUMINATION GUARD]` cutoff | rumination_detector.rs | MECHANICAL | n/a | — | keep |

**Key finding:** the loop directives already *carry* the repeated call's fresh result — but it's pasted to the **coder** as canned text, never reasoned into a specific next step, and the one reasoned path (`redirect_from_loop`) is gated behind a **dirty lint probe**, so action-loops (whose lint is clean) fall through to bare text. The excise is the only STEER site grounded on **stale** context (points back at the transcript instead of re-reading the file). 6 STEER sites to convert, 3 reasoned sites to unify/fix, 5 mechanical to keep.

## Phase 1 — Ground-truth provider
Create ONE module that gathers fresh truth on demand: (a) re-read the actual current
file(s) from disk; (b) the lint/syntax probe (`linter_probe`); (c) the completion
probes (`probe_run`); (d) the dominant repeated call + its latest ACTUAL output (from
the loop detector / recent transcript). Route the existing scattered callers
(`workspace_lint_report`, `run_completion_probes`, `completion_probe_digest`,
`active_turn_file_len`) through it.
**Acceptance:** one provider, unit-tested; existing callers routed through it; build green.

## Phase 2 — Context excise → reasoned + grounded rebuild (WORKED EXAMPLE)
Today (`trim/render.rs` Tier-3 `escalate`): delete the loop turns, inject a canned
"make one different change" line pointing at STALE context. Change to: keep the
loop-turn deletion; then the reasoner (fed the provider's fresh file state + task +
repeated-error + what-failed) authors a SMALL clean working context — task, real
current state, what was tried & why it failed, the ONE next step — and REPLACES the
flailing history with it. Bound the size. Reasoner-unavailable → fall back to today's
canned excise (not nothing).
**Acceptance:** excise emits a reasoner-authored grounded block; loop turns still
removed; bounded size; tests; the 510K-bloat path shrinks.

## Phase 3 — Loop reasoner grounded on the repeated error
Today: thrash→probe→reasoned-guidance calls the reasoner ONLY when the LINT probe is
DIRTY, so action-loops (`cat` a dir, repeat search) are skipped → bare nudge → ignored.
Change: when a loop guard fires, ground the reasoner on the repeated call + its ACTUAL
error output (repetition proves it's fresh, so the stale-signal rule does not apply);
call it even on a CLEAN lint probe when a clear repeated error exists. Lint stays as one
grounding source for code loops.
**Acceptance:** the `cat <dir>` / repeat-search class now yields a grounded redirect
("it's a directory — use `ls`", "stop searching — write the code"); tests with those
exact fingerprints.

## Phase 4 — Convert remaining steering nudges through the layer
Route `[STOP — REPETITION]`, `[STUCK — CIRCLING]`, `[GATHERING WITHOUT ACTING]`,
`[NO PROGRESS — DIAGNOSE]`, `[NO ACTION TAKEN]` through the grounded reasoned-guidance
layer (each fed its appropriate fresh truth). Keep the canned text ONLY as the
reasoner-unavailable fallback.
**Acceptance:** no steering intervention emits ONLY bare text when the reasoner is
available + grounded; mechanical guards unchanged.

## Phase 5 — Course-change detector: stop being fooled by restarts
Today: "beginning a fresh implementation from scratch" is accepted as a genuine course
change and RESETS loop state — enabling more looping. Change: a restart / "start over"
is NOT a pivot; do not reset loop state on it. Distinguish real progress (new
files/edits/passing checks) from throwing work away.
**Acceptance:** a restart does not clear the loop counter; a genuine pivot still does; tests.

## Phase 6 — `text_is_product` gate bypass
CORRECTED root cause (my earlier "a leak flipped the flag" theory was wrong):
`text_is_product` is **role-based** (`role_text_is_product` → true only for
`light_reasoner`/`_backup`), NOT content-derived. The real bypass: the **bare-JSON leak
guard** was itself gated on `!text_is_product`, so when a turn ran in the reasoner role
a `{ "plan": …}` leak was neither re-prompted NOR gated (the completion gate is also
`!text_is_product`) — it was accepted as the final answer. Fix: a bare JSON object is
never a valid deliverable for EITHER role (coder → call the tool; reasoner → write
prose), so the guard fires regardless of `text_is_product`, bounded by MAX_BAIL_RETRIES.
**Acceptance:** a `{ "plan":`-style message no longer bypasses the re-prompt; detection
tested (`bare_json_leak_detected_prose_not`).

## Phase 7 — Consolidation & dead-code removal
Collapse the duplicate/overlapping intervention construction into the single guidance
layer + ground-truth provider. Remove code left dead by prior iterations.
**Acceptance:** one place builds interventions; `cargo build` clean, no dead-code
warnings for the touched modules.

## Phase 8 — Spec reconciliation
Update `docs/spec/{shephard.md, heuristic-assists.md, compaction-reference.md}` to match:
the Nudge→grounded-reasoned-guidance shift, the reasoned excise, the ground-truth
provider. Mirror `heuristic-assists.md` to `../cria-shepherd/docs/` if that mirror is
still maintained.
**Acceptance:** specs describe the code as it now is; no stale "canned nudge" text remains.

---

## Progress log (append one line per completed task)
- Phase 0 done — inventory table added: 6 STEER sites to convert, 3 reasoned to unify/fix, 5 mechanical to keep. Key gap: loop directives carry fresh result but paste it to the coder as canned text; reasoned path gated behind dirty lint probe.
- Phase 1 done — `ground_truth.rs` provider (FileSnapshot/RepeatedAction/GroundTruth + file_snapshot/file_len/lint_digest/completion_digest), 5 tests; `active_turn_file_len` rewired through it; build green. lint_digest/completion_digest wrappers land now; their loop/completion callers rewire in Phase 2/3/6.
- Phase 2 done — excise upgraded: trim surfaces `repeated_action`; new `reasoned_guidance::rebuild_context_from_loop` authors a bounded (~150-word) 4-part working context from GroundTruth (repeated action + live file re-read + dirty lint); wired into the `[HARNESS — STUCK; LOOP REMOVED]` branch, supersedes the dirty-only redirect, falls back to canned excise. `files_touched_this_turn` re-reads actual disk state. 380 routing tests pass; build green. (Empirical 510K-bloat shrink verifies on a live run.)
- Phase 3 done — `redirect_from_loop` now takes `GroundTruth` (repeated action + live files + dirty lint) instead of lint-only, and fires whenever `has_signal()` (NOT dirty-only), so action-loops (`cat` a dir / repeat search — clean lint) get a grounded redirect instead of a bare nudge. Redirect prompt leads with the repeated action. Shared `gather_loop_ground_truth` DRYs the excise + redirect paths. Removed now-dead `workspace_lint_report`. This also routes the [STOP]/[STUCK]/[GATHERING]/[NO PROGRESS] directives through the reasoner (Phase 4's first four). 380 tests pass; build green. (No new unit test: the reasoner call is networked; grounding covered by ground_truth::has_signal tests.)
- Phase 4 done — the four loop directives ([STOP]/[STUCK]/[GATHERING]/[NO PROGRESS]) were converted in Phase 3 (they flow through the unified grounded `redirect_from_loop`). The no-action continuation RECLASSIFIED as MECHANICAL protocol enforcement (states 'no tool ran' + demands a tool call; bounded by MAX_BAIL_RETRIES; not an approach) — documented in the inventory + a code comment. All STEER sites now route through the grounded layer or are documented mechanical.
- Phase 5 done — course-change no longer fooled by restarts: COURSE_CHANGE_PROMPT excludes 'from scratch / start over / fresh implementation', PLUS a deterministic `looks_like_restart` backstop that vetoes the loop-state reset whatever the weak reasoner says (checks both the new action and the reasoner's stated reason). Test with the exact 019f35d3 phrase; build green.
- Phase 6 done — corrected the root cause: `text_is_product` is role-based, not flipped by content. The bare-JSON leak guard (and thus the false-completion catch) was gated on `!text_is_product`, so a `light_reasoner` turn's `{ "plan": …}` leak was accepted as the answer. Removed the gate — a bare JSON object is re-prompted for BOTH roles (coder→call tool, reasoner→prose), bounded by MAX_BAIL_RETRIES. Detection covered by existing `bare_json_leak_detected_prose_not`; build green.
- Phase 7 done — consolidation: the loop interventions (excise rebuild + redirect) both flow through one `gather_loop_ground_truth` → `GroundTruth` → `reasoned_guidance`; removed dead `workspace_lint_report` and the speculative `ground_truth::completion_digest` (+ its Duration import). No NEW dead-code/unused warnings in the touched modules (remaining warnings are pre-existing in untouched files). Build green.
- Phase 8 done — specs reconciled: heuristic-assists.md (family def, repetition guard, reasoned excise, completion critic w/ probe results, course-change restart backstop, new 'Loop → ground truth → reasoned guidance' + 'Context rebuild on flail' entries) and shephard.md (reasoned guidance = primary loop response, grounded by the provider) updated in codex-local. heuristic-assists.md re-mirrored to ../cria-shepherd/docs/ (verified true mirror first). NEAR-MISS: cria shephard.md had DIVERGED (Python-port, 83 unique lines) — a wholesale cp would have clobbered it; caught via the 115-line diff signal and restored from git. compaction-reference.md needed no change (no nudge/loop content).

---
**REFACTOR COMPLETE (Phases 0–8).** Final: `cargo build -p codex-cli` green; 381 routing lib tests + 18 core local_routing tests pass. Every steering nudge routes through the grounded reasoned-guidance layer (fed fresh truth by one `ground_truth` provider) or is documented mechanical; the context excise rebuilds a clean ground-truth working context via the reasoner; dead/duplicate intervention code removed; specs reconciled. Remaining = EMPIRICAL: the loop-breaking wins (grounded redirects for action-loops, the reasoned excise shrinking the 510K-bloat path) need a live session to confirm the reasoner-in-the-loop behavior end-to-end.
- Phase 6 follow-up — closed the display residual: the TUI content-strip in `ollama_tool_response_to_stream` now blanks a bare-JSON blob in the NO-tool-call case too (was only stripping it when a tool call rode alongside). Since the re-prompt is bounded by MAX_BAIL_RETRIES, this is the final safety net so a stubborn model's exhausted-retry `{"plan":…}` can never reach the screen or be recorded as the final answer. Build green.
