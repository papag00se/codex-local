# Shephard — what it does

[< Spec Index](index.md) | service plan: [nudge-service.md](nudge-service.md) | the "why" + code pointers: [local-coder-massaging.md](local-coder-massaging.md)

> **Shephard** (working name) is the layer that sits between a small local model
> (9B-class) and an agent harness and keeps the two working together to do real
> agentic coding. Everything it does falls into five kinds:
>
> - **Nudges** — in-context directives that steer the *model* (the model sees them).
> - **Massages** — silent repairs of the model's *output* so the *harness* accepts it (the model never knows).
> - **Context shaping** — managing what the model sees and how much, so it fits the window and stays grounded.
> - **Probes** — Shephard makes its OWN read-only tool calls back to the harness to get ground truth, rather than only relaying the model's. Verify by *doing*, not by reading. (The first Probes — the repo-diagnostic completion gate — are **built**; see below.)
> - **Reasoned guidance** *(built — now the PRIMARY loop response)* — on a stuck trigger, hand the light reasoner FRESH ground truth (the repeated failing action + its output, the touched files re-read from disk, the lint probe) and have it author the coder's next step. Fires on ANY real signal (a repeated action counts even when lint is clean); canned nudges are only the reasoner-unavailable fallback.

The marketable catalog of every assist — name first, one line each — is in [heuristic-assists.md](heuristic-assists.md); the why + code pointers in [local-coder-massaging.md](local-coder-massaging.md).

## Principle: Shephard owns no executors
- **What it is** — never touches the workspace; a bidirectional transform on the tool-call stream. Model works with rich tools (`write_file`, `web_fetch`); harness only runs primitives (ideally just `shell`).
- **Outbound** — `write_file` → `printf %s '<base64>' | base64 -d > path` (byte-exact, escaping-proof); `web_fetch` → `curl`. `shell`, not `write_file`, is the one primitive every harness exposes — lowering to it is what ports.
- **Inbound** — re-present the recorded `shell` call + result as the original tool, so the model never sees the shell. Recognized statelessly from a `# shephard-write:<path>` sentinel (survives restarts). The old one-way `write_file → printf` failed by skipping this half.
- **Why it ports** — harness supplies only executors (Rust vehicle: `codex-core`); all intelligence is stream transforms (`codex-routing`).

## Principle: Shephard owns no rendering either
- **The rule** — every tool Shephard exposes must reduce to a primitive the harness already knows how to both *run* and *display*. A custom tool/event is a bet the harness will draw something it was never taught to.
- **Cautionary example** — the fork's custom `local_web_search` (Brave) emitted `WebSearchBegin`/`End`; searches ran and hit the rollout but the TUI never rendered them. Lower to `curl` over `shell` and it shows as an ordinary exec cell — visible everywhere.
- **Preference order** — native structured tool (`text_editor.create`, richest) → native file handler → `shell` (universal, always rendered). Never a Shephard-only tool the harness must be taught to draw.

## Principle: guard state is session-scoped
- **The rule** — anything a guard remembers (searches made this turn, URLs fetched, streaks) belongs to **one session's current turn**, never a process global. It resets on a new user turn (new task = clean slate) and never bleeds between sessions, sub-agents, or forks.
- **In Shepherd** it's just a field on the **session object** — the service already holds per-session state, so there's no keying and no eviction.
- **In this Rust fork** the tool backends are stateless module fns, so we *simulate* it: a map keyed by the harness `conversation_id`, scoped to the turn `sub_id`, capped (`guard_state::SessionTurnStore`). The map is a fork wart, not the design — the seam that makes the port trivial.

## Nudges — steer the model

In-context directives the model sees, to break loops and force progress. Full list in [heuristic-assists.md](heuristic-assists.md).

## Massages — repair the output so the harness runs it

Silent repairs of the model's output so the harness accepts it — the model never sees them. Full list in [heuristic-assists.md](heuristic-assists.md).

## Context shaping — manage what the model sees

Managing what the model sees and how much, so it fits the window and stays grounded. Full list in [heuristic-assists.md](heuristic-assists.md).

## Probes — Shephard acts for itself

Shephard makes its own **read-only** tool calls to the harness for ground truth — it picks the command *and* the output format, so it can judge the work without parsing the model's prose or being a model (*verify by doing, not reading*). The first is **built**: the repo-diagnostic ground-truth completion gate runs the code's own checks (a syntax floor + the top-ranked safe probe) on a "done" claim and refuses a false completion, handing back the exact `file:line` to fix. Full list in [heuristic-assists.md](heuristic-assists.md); design + forward notes below.

## Reasoned guidance — a context-aware redirect

On a stuck trigger, escalate to the **light reasoner** (a separate, cheap local role) and hand it FRESH ground truth — the repeated failing action + its actual output, the files the model touched re-read from disk, and the lint probe (assembled by one shared `ground_truth` provider) — to infer what the model is *trying to do* and inject a concrete new path instead of a canned nudge. This is now the **primary** response to a loop, not a forward idea: it fires on any real signal (a repeated action grounds it even when lint is clean — the `cat`-a-directory / repeat-search case), and the top escalation rebuilds the whole working context from that ground truth (the reasoned excise). It is never called on *nothing* (a clean probe with no repeated action) — a groundless reasoner hallucinates. Full list in [heuristic-assists.md](heuristic-assists.md); design notes below.

---

## Forward & design notes — Probes and Reasoned guidance

*Captured so they aren't lost; not everything here is built.*

**Why Probes matter.** They dissolve the wall "the harness can't judge X without parsing the model's noisy output or *being* a model." The eval showed the payoff: weak models rewrote a whole file 9× chasing an `IndentationError` they couldn't localize — the exact `file:line` a probe hands over is what they couldn't generate for themselves.

**Probe guardrails (apply to the built gate too).** These calls EXECUTE, so: read-only / idempotent only (never mutate the workspace); triggered, not constant (a "done" claim or a stuck trigger — each probe is real work on the box); hidden from the model's context so they never accrete. Shephard still **owns no executors** — it *borrows* the harness's, which makes it an **actor / supervisor**, not merely a stream transform.

**More probes on the same substrate (forward).**
- **Active grounding on a stuck trigger** — the structural detector says *when* (footprint stalled); a probe gets the *truth* (re-read the live file, run the failing check) and feeds fresh verified state into the nudge, so the model can't work from stale context.
- **Environment probing** — code-bug vs environment (curl the real endpoint to see if a 4xx is the code or a missing header), instead of guessing.
- **On-demand fresh-state pin / pre-flight checks.**
- **Private round-trip (open question)** — a call→result the model (and ideally the user's UI) never sees.

**Reasoner-assisted redirect (forward).** The deterministic guards fire a *fixed* nudge; the smarter successor is, on the same trigger, to ask the reasoner to (1) infer intent from the recent transcript and (2) propose a concrete new path — then inject *that*.
- **Structural detector says WHEN** (loop / tunnel-vision / read-without-write / repeated failure); the reasoner supplies a **context-aware WHAT**.
- **Kin to Probes** — probes get *ground truth* by acting, this gets a *redirect* by reasoning; best combined (probe for the real state, hand it to the reasoner so it can't invent a dead end).
- **Guardrails** — only on a confirmed stuck signal (a real model call, not every turn); the reasoner is itself a small local model, so ground it with probe truth.
- **Status** — may or may not earn its keep; noted now because the pattern-triggers it would ride on already exist.
