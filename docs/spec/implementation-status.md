# Implementation Status

[< Spec Index](index.md)

> Last updated: 2026-04-10

## What's built

### Rust crates (121 tests passing)

**`codex-rs/routing/`** (codex-routing) — 108 tests

Core routing:
- **LLM-based request classifier** (`classifier.rs`): local qwen3.5-9b:iq4_xs on the 1080 classifies every request into: light_reasoner, light_coder, cloud_fast, cloud_mini, cloud_reasoner, cloud_coder
- **Cloud tier routing**: classifier output drives model slug override — spark/mini/sonnet for secondary buckets, primary only for cloud_coder
- **Weighted model distribution**: cloud roles with multiple entries use weighted random selection from `.codex-multi/config.toml`
- **Classifier cache** (`classify_cache.rs`): skips 3-4s classifier LLM call when last 3 classifications match (30s TTL)
- Task metrics extraction (`metrics.rs`): all 27 regex patterns ported from Python reference
- Route selection algorithm (`engine.rs`): context-window filtering, LLM-assisted selection, deterministic fallback

Context and compaction:
- **Context stripping** (`context_strip.rs`): two strip levels (Reasoner: 2 turns/2K, Coder: 3 turns/4K). Removes binary blobs, think blocks, encrypted_content. Collapses poll patterns. 8 tests
- **Compaction pipeline** (`compaction/`): boilerplate-strip → normalize → split_recent_raw → chunk → summarize each chunk (LLM, **free-form prose**) → final unifying pass → assemble handoff. Model-written, not deterministic extraction. Runs entirely on local Ollama — no proxy needed
  - `normalize.rs`: strips encrypted_content, attachments, tool_result blocks; detects precompacted summaries
  - `chunking.rs`: token-budget chunking at item boundaries with overlap
  - `extract.rs`: `summarize_chunk` / `summarize_final` — free-form prose summaries via the Ollama compactor (`think=false`, 90 s per-chunk timeout)
  - `pipeline.rs`: `compact_transcript` — orchestrates the above and assembles `[post-compaction warning + summary + verbatim recent tail + current request]`
  - `models.rs`: `TranscriptChunk` only — the deterministic `ChunkExtraction` / `merge_states` / `DurableMemorySet` (5 markdown docs) were **removed** (see [compaction-reference.md](compaction-reference.md), now historical)

Failover and reliability:
- **Failover executor** (`failover.rs`): classifies failures into F1-F8 types, decides retry-same vs walk-chain vs hard-fail. 15 tests
  - F1 (rate limit): retry with backoff, honor retry-after header, cap at max wait, then walk chain
  - F2 (quota exhausted): walk chain immediately
  - F3 (model unavailable): walk chain immediately
  - F4 (model not found): walk chain with config warning
  - F5 (auth failure): hard-fail, never retry
  - F6 (timeout): retry once, then walk chain
  - F7 (quality failure): walk chain immediately
  - F8 (context overflow): walk chain to larger model
- **Quality detection** (`quality.rs`): checks local responses for empty, too short, echo, refusal, empty code fence, repetition
- **Tool-call recovery** (`tool_recovery.rs`): JSON blob recovery, embedded tool blocks, streaming partial drops

Feedback and analytics:
- **Routing feedback** (`feedback.rs`): records `RoutingOutcome` to `.codex-multi/routing_history.jsonl`, computes per-route success rates, injects `profile_context()` into classifier
- **Codebase context** (`codebase_context.rs`): auto-detects languages, file count, test frameworks, build tools. Caches in `.codex-multi/context_cache.json` (1hr TTL)
- **Budget pressure** (`budget_pressure.rs`): reads rate limit percentages, soft pressure at 50-70-90%, hard block of cloud_coder at 95%
- **Cost analytics** (`cost_analytics.rs`): persistent `usage_log.jsonl` with aggregate summaries
- **Usage tracking** (`usage.rs`): in-session per-model token tracking with bucket classification (local/secondary/primary)

Model interaction:
- **Ollama client pool** (`ollama.rs`): async with per-endpoint `tokio::Semaphore`. `chat()`, `chat_with_tools()`, `chat_stream()`. Warm model tracking per endpoint
- **Streaming** (`ollama.rs`): `chat_stream()` yields `StreamChunk::Delta` / `StreamChunk::Done` via mpsc channel
- **Tool format adapter** (`tool_format.rs`): converts Codex ToolSpec to Ollama function tool format
- **Prompt adaptation** (`prompt_adapt.rs`): per-tier scaffolding — local gets step-by-step, cloud_fast gets "be concise", frontier gets no scaffolding
- **Local dispatch** (`local_dispatch.rs`): `call_ollama_text()` for non-streaming Ollama calls

Configuration:
- **Project config** (`project_config.rs`): loads `.codex-multi/config.toml` with model roles, failover chains + behavior, supervisor settings, usage config, agent roles
- **Routing config** (`config.rs`): `RoutingConfig` with all Ollama endpoints. `from_env()` and `from_project_config()`. Multi-tier: classifier, reasoner, reasoner_backup, light_coder, compactor + cloud flags
- **Session memory** (`session_memory.rs`): saves/loads session handoffs to `.codex-multi/memory/`, prunes to 20

**`codex-rs/supervisor/`** (codex-supervisor) — 13 tests
- Task graph: deterministic state machine (Pending → Running → Evaluating → Completed/Failed/Skipped)
- Supervisor loop: bounded by iterations (default: 50), timeout (default: 2h), max retries (default: 3)
- `SupervisorJudge` trait: plan_tasks, dispatch_task (returns DispatchResult with thread ID), evaluate_completion, verify
- Dependency resolution: tasks with unmet deps wait; failed deps cascade to skip
- **Context resumption**: tracks `last_agent_thread_id` per task, retries fork from previous agent's conversation via `SpawnAgentForkMode::LastNTurns(5)`

### Codex integration

**`codex-rs/core/src/tools/handlers/supervisor.rs`** — supervisor tool handler
- `SupervisorHandler`: registered as `supervisor` tool, model calls it for complex goals
- `CodexJudge`: bridges supervisor to codex-core
  - `plan_tasks`: local Ollama with failover chain (reasoner → backup → Codex)
  - `dispatch_task`: spawns worker, waits for completion, returns thread ID; retries fork from previous context
  - `evaluate_completion`: local Ollama with failover chain, `<think>` tag handling
  - `verify`: runs subprocess in correct cwd, checks exit code

**`codex-rs/core/src/local_routing.rs`** — per-request routing
- Hooks into `ModelClientSession::stream()` — every model API call goes through the classifier
- Local routes: call Ollama directly (streaming for reasoner, non-streaming with tools for coder), translate to `ResponseEvent` stream
- Cloud routes: override `model_info.slug` for this request only (spark/mini/sonnet)
- **Context stripping**: removes binary, truncates, collapses polls, keeps only recent turns before sending to 8K local models
- **Compaction pipeline**: detects `<<<LOCAL_COMPACT>>>` sentinel, runs full normalize → chunk → extract → merge → render locally
- **Classifier cache**: skips 3-4s LLM call when recent classifications are consistent
- **Budget pressure**: injects rate limit data into classifier context, hard-blocks primary at 95%
- Loads config from `.codex-multi/config.toml`, falls back to env vars
- Health check via `/api/version` (fast, doesn't cold-load model)

**`.codex-multi/config.toml`** — project config
- Model roles: classifier, light_reasoner, light_reasoner_backup, light_coder, compactor, cloud_fast, cloud_mini, cloud_reasoner, cloud_coder
- Weighted distribution: `entries = [{provider, model, weight, reasoning}, ...]` for cloud roles
- Failover chains per task type: classification, coding, compaction, evaluation, planning, reasoning, review
- **Failover behavior**: retry_same_attempts, retry_same_backoff_ms, rate_limit_default_wait_ms, rate_limit_max_wait_ms, timeout_ms
- Supervisor behavior: max_iterations, timeout, retries, verification_command
- Usage preservation: primary_warn_threshold, prefer_secondary

### Upstream integration footprint

| File | Change |
|------|--------|
| `core/src/tools/handlers/supervisor.rs` | New: supervisor tool handler |
| `core/src/local_routing.rs` | New: per-request routing hook |
| `core/src/lib.rs` | +1 line: `mod local_routing` |
| `core/src/client.rs` | +12 lines: routing hook in `stream()` |
| `core/Cargo.toml` | +2 lines: deps |
| `tools/src/supervisor_tool.rs` | New: tool spec |
| `tools/src/tool_registry_plan_types.rs` | +1 line: enum variant |
| `tools/src/tool_registry_plan.rs` | +7 lines: register |
| `tools/src/lib.rs` | +2 lines: exports |
| `core/src/tools/spec.rs` | +3 lines: match arm |
| `core/src/tools/handlers/mod.rs` | +2 lines: module |

### Smoke tests

**`tests/smoke_multi_agent.sh`** — 15 checks
- Local routing responds to simple questions
- Response quality (non-empty, reasonable length)
- Classifier logs show activity
- Context stripping removes binary, truncates, collapses polls
- Compaction config loads correctly
- Context preserved after stripping
- Token savings vs unstripped
- Routing infrastructure intact

## Live test results

### Per-request routing (2026-04-09)
```
Request: "What is a goroutine?"
Classifier: light_reasoner, tools_potential=false (3.6s on 1080)
Route: local qwen3.5:9b on sakura:11435
Result: ✓ Correct goroutine explanation, 230 tokens
Cost: ZERO cloud tokens — entirely local
```

### Cloud tier classification (2026-04-09)
```
12 test requests classified by local LLM:
- light_reasoner: 4/12 (simple questions, yes/no, architecture)
- light_coder: 3/12 (file reads, docstrings, renames)
- cloud_fast: 2/12 (unit test fix, single-file refactor)
- cloud_mini: 2/12 (Playwright E2E, multi-file investigation)
- cloud_reasoner: 1/12 (security review)
- cloud_coder: 1/12 (full app debug)
All classifications correct. Every tier hit.
```

### Supervisor tool (2026-04-08)
```
Goal: "Create calculator.py + test_calculator.py + run tests"
Result: ✓ Both files created, 5 tests passing (including edge cases)

Goal: "Create math_utils.py with is_prime + tests"
Result: ✓ 12 parametrized pytest tests passing
```

## What's next

| # | Item | Status |
|---|------|--------|
| 1 | Wire failover executor into request flow | DONE — local + cloud paths walk failover chains |
| 2 | Local coder multi-turn tool loop reliability | DONE — sticky routing prevents mid-loop rerouting |
| 3 | Supervisor tool model guidance | DONE — directive description with examples |
| 4 | Observability — routing decisions in TUI | Logged via tracing only |

## Build instructions

```bash
cd codex-rs

# Set up build environment (WSL without libssl-dev)
source routing/build-env.sh

# Build
cargo build -p codex-cli

# Run tests (121 total: 108 routing + 13 supervisor)
cargo test -p codex-routing -p codex-supervisor

# Run smoke tests
bash tests/smoke_multi_agent.sh

# Run with routing enabled (needs .codex-multi/config.toml in cwd)
RUST_LOG=codex_core::local_routing=info,codex_routing=info ./target/debug/codex
```

## Git log (recent)

```
7300c0ff6 Strengthen supervisor tool description — directive language, examples
d2d75081a Sticky routing for local coder tool loops — prevent mid-loop rerouting
4a5928084 Wire failover executor into request flow — local and cloud paths
a31832961 Failover executor — classifies failures F1-F8, decides retry vs chain-walk
4e19e0f9a Failover behavior config — retry, rate limit, timeout parameters
9f6248099 Full compaction pipeline in Rust — no proxy needed
38f855d3b Expanded smoke test: 15 checks for routing, stripping, compaction
0ab00c723 Context stripping for local models — fit conversations in 8K context
069cb503d G14: Dynamic budget pressure — routing shifts based on rate limit data
ea4ac2ff9 G3,G4,G8,G9: Cross-session memory, prompt adaptation, classifier cache, cost analytics
df641074d G5: Streaming from local Ollama models
811aa9939 G1,G2,G6,G7: Routing feedback, codebase context, warm GPU, quality detection
742db265d Cloud tier routing + weighted distribution + classifier robustness
4bd210eac Multi-agent orchestration: routing, supervisor, per-request local model routing
```
