# codex-local

A fork of [OpenAI's Codex CLI](https://github.com/openai/codex) focused on one thing: making **small local LLMs** (4B–30B, Q4/Q6, served by Ollama or any OpenAI-compatible server) behave as competent multi-turn agentic coders.

Stock Codex assumes a frontier model on the other end of the wire. Point it at a 9B Q4 and the wheels come off — the model picks the wrong tool, mangles patches, repeats the same failed call five times, announces "now I'll edit the file" and then stops, or spirals into "wait, actually, hmm, let me reconsider..." until `max_tokens` runs out. This fork is the orchestration layer that sits between the harness and the local model and **massages every step** so those failure modes turn into recoverable ones.

If you have a fast cloud key, stock Codex is better. If you want to run a real coding loop against your own GPU — or split work between local and cloud to preserve your subscription budget — this fork is for that.

---

## Why this exists

Local coder models in the 4–30B range fail in characteristic, *predictable* ways:

- **Tool-use discipline** — they invent tool names (`ls`, `cat`, `grep`), pass `command` as a string instead of an array, ignore half the schema.
- **Patch precision** — they emit `git diff` format instead of Codex's native `*** Begin Patch` envelope, leave `@@ -1,6 +1,6 @@` line-numbered hunk headers, drop `+` prefixes, forget the closing marker.
- **Self-correction** — after an error, they retry the *exact same broken call*. Three times. Five times.
- **Announce-without-act** — "I will update the imports and run the tests" with no tool call. Codex sees no tool call, ends the turn, the user is left holding a half-done task.
- **Rumination** — thinking-only models can spiral on the reasoning channel for ten minutes and emit empty output.
- **Context blindness** — they forget a file changed two turns ago and patch against the stale version.
- **Format leaks** — they emit a tool call as plain *text* (`<tool_call>{…}</tool_call>`, or the XML `<function=…><parameter=…>` form) instead of a structured call, or wrap a control-model JSON verdict in a ```json fence. The harness sees no tool call / can't parse the verdict, so the action — or the routing decision — silently vanishes.

Each of these has a fix. None of them are "wait for better models." All of them are layered between the model and Codex's tool dispatch so the existing harness code is unchanged.

---

## What the fork does about it

A short tour of the interventions. Full catalog with code links lives in [docs/spec/local-coder-massaging.md](docs/spec/local-coder-massaging.md).

### Tool layer

- **Tool menu trimmed from ~120 to ~10.** Small models lose attention on huge schemas. The Coder gets `shell`, `apply_patch`, `list_dir`, `view_image`, `update_plan`, `local_web_search`, `web_fetch`, `request_permissions`, `exec_command`, `write_stdin`. Per-endpoint `tool_subset = "Full"` if you have a model that can handle the firehose.
- **Plain-English tool cheat sheet** appended to the system prompt. The JSON Schema is exhaustive but small models don't read it carefully — a worked example for each tool ("if you find yourself wanting to call `ls` directly, that's wrong — wrap it as `shell` with `command: [\"bash\", \"-lc\", ...]`") gets through where the schema doesn't.
- **Shell-alias rewriting.** A tool call to `ls` / `cat` / `grep` / `git` / `pytest` / etc. is rewritten on the fly into a proper `shell` call instead of being rejected.
- **Browser User-Agent injection on `curl`.** Many sites serve garbage to `curl/8.0`. Any `curl` invocation that doesn't already set `-A` gets a real Chrome UA inserted.
- **Two new tools — `local_web_search` (Brave API) and `web_fetch` (single GET, 512KB cap).** OpenAI's built-in `web_search` is unavailable in local-only mode, and without it the model just guesses URL paths and API shapes. The tool hint explicitly tells the model to fetch real docs *before* writing code against an unfamiliar API.

### Patch layer

- **Unified-diff translation.** Models trained on `git diff` output emit `--- a/foo` / `+++ b/foo` / `@@ -L,N +L,N @@`. The normalizer detects this and rewrites it into Codex's native format, including `Add File` / `Delete File` / line-number stripping (Codex matches by context anchor, not line numbers).
- **Hybrid hunk-header normalization** — for the case where the model uses the Codex envelope but stuffs unified-diff hunk headers inside it.
- **Prefix repair + auto-`*** End Patch`** — if a body line lacks `+` / `-` / ` ` prefix, prepend `+` (treat as addition); if `*** Begin Patch` is unmatched, append the closer.
- **Directive error messages.** When a patch genuinely can't apply, the default `Failed to find context '-17,7 +17,7 @@'` is rewritten into something the model can act on. `apply_patch({})` from a confused model returns a directive with the expected shape and an escape hatch instead of `missing field input at line 1 column 2`.

### Conversation layer

- **Trim transcript for local context windows.** Active turn preserved verbatim; older turns collapsed into a synthesized state prelude (files seen, files modified, tests run, errors encountered); stale reads dropped; superseded outputs dropped; **errors are sticky** so the model can't forget a failure.
- **Live-on-disk file pin.** Files modified in the active turn get pinned at the top of the prelude with current contents (capped at 10 KB, with hash + line count). Stops the "patch fails because the model is reasoning from a stale read" loop.
- **Repetition alert (three flavors, productivity-gated).** Same `(tool_name, args_hash)` 3× in a row → `[STOP — REPETITION DETECTED]`. Same *file path* failing 3× with different commands → `[NO PROGRESS — DIAGNOSE]` forced-diagnosis. Same signature recurring *interleaved* with other work (the "run the test, edit, run again, never passing" loop) → forced-diagnosis too — but **only failing recurrences count**, so a now-passing test or a routine `ls` / `git status` re-run doesn't trip it. Past the escalation threshold the loop's calls + outputs are **excised from context** so the model can't copy them back out, and the prelude reframes to "you're stuck."

### Generation-time guards

- **Bail detector / completion verifier (escalating + ground-truth-gated).** After every Coder response with text but zero tool calls, route it through a small judge prompt (BAIL vs COMPLETE). On BAIL, inject a continuation message and re-call; if the model keeps explaining instead of acting, the nudge **escalates** to "STOP EXPLAINING — your entire next message must be a single tool call." Up to 3 retries. The judge is weak, so completion is also **ground-truth-gated**: a coder turn only ends if it actually changed files or the judge confirms COMPLETE. And a second, **deterministic** guard in core (`run_turn`) catches "Let me fix the handler:"-style announce-without-act with no judge at all — high-precision, route-agnostic, so it covers cloud routes too. **Code blocks are never actions** unless passed to `apply_patch` / `shell`.
- **Leaked tool-call recovery.** When a model emits a tool call as text — Hermes `<tool_call>` JSON, the XML `<function=…>` dialect, or fenced JSON — a single recovery pass promotes it to a real, executed call and strips it from the visible text, instead of the action vanishing. Malformed blocks (e.g. an unescaped heredoc) are detected and re-prompted rather than silently dropped. One shared implementation across the coder and reasoner paths, so a format fix can't land in only one.
- **Fenced-JSON tolerance for control models.** The classifier and verifier extract the JSON object (first `{` to last `}`) before parsing, so a model that wraps its verdict in a ```json fence doesn't silently break routing (chain-exhausted every turn) or let dangling turns finish.
- **Rumination guard (streaming).** A streaming-time phrase-counter watches the reasoning channel for 23 self-doubt markers (`actually`, `wait`, `hmm`, `let me reconsider`, `or maybe`, `scratch that`, ...). After half the token budget is burned, ≥6 markers triggers an in-flight abort (drops the SSE receiver, signals the server to stop generating, frees the slot) and a re-prompt telling the model to pick the simplest next step and take it. Same 3-retry cap as the bail detector.
- **Streaming tool-call assembly** — the tool-aware path is fully streaming so the rumination watcher can see reasoning as it happens, with a unified `StreamChunk` enum across Ollama NDJSON and OpenAI SSE.

### Wire layer

- **OpenAI-compat adapter.** A `ClientFlavor` enum (`Ollama` default, `OpenAICompat` for LM Studio / llama.cpp `server` / vLLM / LiteLLM / actual OpenAI) branches every wire op: URL (`/api/chat` vs `/v1/chat/completions`), payload shape (`options.num_ctx` + `think` vs top-level `max_tokens`), tool-call argument encoding (object vs JSON-encoded string), tool-result message role (`user` with `<tool_result>` tags vs `tool` with `tool_call_id`), streaming transport (NDJSON vs SSE), usage-token field names, startup probe.
- **Per-role `max_tokens` and `timeout_seconds`** in `config.toml`. Reasoning models can legitimately take 5–30 minutes — the original 5-min client timeout killed them mid-flight. `0` on either knob means unlimited.
- **Reasoning channel capture.** `message.thinking` (Ollama) / `reasoning_content` (OpenAI) is logged at `debug` level so weird decisions can be debugged after the fact, but never fed back to the model.
- **Network error enrichment.** `error sending request for url (...)` is replaced with `[connect] no alternative certificate subject name matches target host name 'X'` etc. — DNS, TLS, connect-refused all become distinguishable.

---

## Quickstart

Build (Rust):

```shell
cd codex-rs
cargo build --release -p codex-cli
```

Then run `codex` as usual. The fork is wire-compatible with stock Codex — same TUI, same slash commands, same `~/.codex/` config.

### Local-only mode

Drop a `config.toml` into `.codex-multi/` at the project root (or any ancestor). Minimal local-only config:

```toml
[models.light_coder]
endpoint = "http://localhost:11434"
model = "devstral-small-2:q4_k_m"
num_ctx = 16384
provider = "ollama"
reasoning = "off"

[routing]
strategy = "local_only"
```

In local-only mode, every request — including compaction — is routed to `light_coder`. No classifier call, no separate reasoner endpoint, no separate compactor. One model, one job. See [docs/spec/local-coder-massaging.md](docs/spec/local-coder-massaging.md) for what each knob does.

For LM Studio / llama.cpp / vLLM, set `provider = "openai-compat"` (or `"lmstudio"` / `"openai"`) on the endpoint — trailing `/v1` on the URL is fine either way.

### `/stats` slash command

Reports per-session routing decisions, local vs cloud token counts, and approximate cloud tokens *saved* (the stripped-down request that never hit the cloud, scored against what cloud would have paid).

---

## Routing (cloud + local)

If you have cloud credentials too, the same `config.toml` can declare cloud roles and a failover chain. A small classifier picks a route per request (fast / mini / reasoner / coder, weighted across providers); failures (rate limits, timeouts, quota, auth, quality, context overflow) are classified F1–F8 and either retried with backoff or walked down the chain. Anthropic dispatch routes through the local Claude CLI binary so subscription auth Just Works.

This is genuinely useful — it'll keep a heavy task on your subscription's preferred model and shove cheap stuff to local — but it's not the main reason this fork exists. The main reason is everything in the section above. If you want the routing internals: [orchestrator/README.md](orchestrator/README.md) and the specs under [docs/spec/](docs/spec/).

---

## Multi-agent supervisor (optional)

Beyond per-request routing, the fork adds a **`supervisor` tool** for goals too big for one turn. The model calls `supervisor(goal, verification_command?)`; control then stays inside a **deterministic loop** (the model is blocked on the tool call, so it can't bail mid-run) that:

1. **Plans** — decomposes the goal into a task list.
2. **Dispatches** — spawns a specialist sub-agent per task and waits for completion.
3. **Evaluates** — judges whether each task is actually done.
4. **Verifies** — runs `verification_command` (e.g. `pytest tests/`) and interprets the result.
5. **Retries** failures, bounded by `[supervisor] max_iterations / max_retries_per_task / timeout_seconds`.

Specialist behavior is config-driven via `[roles.coder]`, `[roles.reviewer]`, `[roles.test_runner]` (each an `instructions` + `nickname`). Planning and evaluation honor the configured `[failover]` `planning` / `evaluation` chains — so in `local_only` mode the judgment calls run on local models (`light_reasoner` → backup), and simple goals plan locally while complex ones can plan on a stronger cloud model. Details: [supervisor-integration.md](docs/spec/supervisor-integration.md) and [agent-taxonomy.md](docs/spec/agent-taxonomy.md).

---

## Relationship to upstream Codex

This is a fork of [openai/codex](https://github.com/openai/codex). It is not affiliated with or endorsed by OpenAI. The fork preserves Codex's TUI, tool dispatch, sandboxing, and rollout machinery wholesale and adds the layers described above. Upstream is a much better choice if you're running a frontier cloud model and don't need any of this.

For the original Codex docs, install instructions, and release binaries, see the upstream repo. For docs specific to this fork, start at [docs/spec/](docs/spec/).

Licensed under the [Apache-2.0 License](LICENSE).
