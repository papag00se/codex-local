# Nudge Engine as a Service (proposal)

[< Spec Index](index.md) | [Local Coder Massaging](local-coder-massaging.md)

> **Status: proposal, not built.** This captures the design discussion for
> extracting the local-model nudging layer (everything in
> [local-coder-massaging.md](local-coder-massaging.md)) out of `codex-core` and
> into a standalone HTTP service.

## Why

Today the nudging layer is inlined into `codex-core` (`local_routing.rs` drives
the loop; `codex-routing` holds the logic). That gives tight integration but
couples the nudging to this Codex fork: changing a nudge means rebuilding the
binary, and only this harness benefits.

Extracting it to a service trades integration for **portability**: any
OpenAI-compatible harness gets the nudging by pointing its model endpoint at the
service, and the nudge logic iterates/deploys independently of the harness. For
the "smarter harness carries the local model" goal, that makes the harness
reusable across agents, not just this fork. (It is partly a *return* to the
original coding-agent-router HTTP-proxy model this fork inlined — but with all
the massaging the fork has since added.)

**Enabler:** `codex-routing` is already a standalone crate with almost no
`codex-core` dependency. The service is essentially "wrap `codex-routing` in an
HTTP server and move the loop-driver out of `local_routing.rs`."

## The boundary: a "smart model" endpoint, not an advisor

Expose an **OpenAI-compatible `/v1/chat/completions` (streaming)** endpoint. The
harness sends a normal chat request (messages + tools); the service does *all*
the massaging internally — classify/route, trim, call the real LLM, handle
bail/repetition/rumination/overflow, recover leaked tool calls — and streams
back a final completion with tool calls. The harness does not know nudging is
happening; it just gets better completions.

Rationale:
- The harness already speaks OpenAI SSE → minimal harness change (point
  `ModelClient` at the service).
- **Tool *execution* stays in the harness.** The service returns tool *calls*;
  the harness runs them and sends results back next turn. The nudge loop is
  LLM-only, so the service never needs to execute tools or touch the workspace.

## Keep-alive: heartbeats + idle-timeout

The nudge loop can run for minutes (re-prompts, rumination aborts, overflow
re-trims). A plain request/response call will time out. Stream instead:

- The service holds the SSE connection open and emits **periodic heartbeats**
  (SSE comment lines or empty deltas) every few seconds while
  classifying/trimming/looping.
- The harness client uses a **read/idle timeout** (resets on each byte), NOT a
  total-request timeout. A heartbeat within the idle window keeps the connection
  alive indefinitely. Standard long-LLM-stream pattern.

**Subtlety (same tension as in-process live streaming):** the loop produces
multiple LLM responses (model bails → re-prompt → responds again). Do NOT stream
rejected attempts as final content. Stream **reasoning/heartbeats** during the
loop and only the **final accepted completion** as real content. Moving to a
service actually sidesteps the in-process live-streaming complexity: the service
buffers intermediate attempts and commits only the final, heartbeats covering
the gap.

## Session identity: a header, for the stateful nudges

Most nudges are stateless (they read the transcript sent each request). A few are
session-scoped and need an identifier:
- The **classifier cache** (route stickiness — the 3-in-a-row / 30s window).
- **`/stats`** (per-session local-vs-cloud tokens, savings).
- **budget pressure** tracking and cross-session **feedback/learning**.

Design:
- `X-Nudge-Session-Id` — the harness's conversation/thread UUID; keys the
  session state above.
- `X-Nudge-Instance` (optional) — machine/user, for multi-tenancy if the service
  is ever shared.
- **Stateless fallback:** no session id → treat each request independently (no
  cache/stickiness), so dumb harnesses still work.

## Gotchas to decide before building

1. **File access breaks if the service is remote.** The current-file-state pin
   (massaging §18) and any on-disk reads need the *workspace*. Either keep the
   service **localhost/LAN-colocated**, or have the **harness pre-pin file
   contents into the transcript** before sending. This is the biggest constraint
   — it dictates localhost-only vs anywhere. Recommend harness-pins-files so the
   service isn't permanently locked to co-location.
2. **Config relocates to the service.** The `.codex-multi` config (models,
   failover chains, trim budgets, temperatures — including the per-role
   "room to explore" settings) becomes the *service's* config. The harness sends
   only transcript + tools + session id.
3. **The deterministic dangling-intent guard** currently lives in
   `core::run_turn`, not the routing layer. Move it into the service so all
   completion logic lives in one place.
4. **Observability.** The TUI shows `↻ re-prompting` via `push_nudge` today. Over
   HTTP, emit **nudge events as a distinct SSE event type** the harness renders,
   or accept the TUI goes quieter and log them service-side.
5. **Privacy.** The full transcript (code, possibly secrets) crosses the wire
   each request. Localhost/LAN: non-issue. Beyond that: a data-egress decision.

## Recommended first cut

A **localhost/LAN OpenAI-compatible streaming service** wrapping `codex-routing`,
with: heartbeat-keepalive SSE; an `X-Nudge-Session-Id` header for the stateful
bits; the harness pre-pinning file contents (so co-location isn't permanent);
config owned by the service; tool execution and the workspace kept in the
harness. Smallest boundary that buys portability + independent iteration.

Open follow-up to spec when this is greenlit: the concrete request/response shape
(headers, SSE event types for heartbeat / nudge / final) and the harness-side
`ModelClient` change.
