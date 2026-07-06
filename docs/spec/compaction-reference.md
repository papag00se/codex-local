# Compaction Algorithm Reference

[< Spec Index](index.md) | [Product Index](../product/index.md)

> **⚠️ Historical — this describes the DETERMINISTIC pipeline that the local-coder implementation REPLACED.** The shipped compaction produces a **model-written, free-form** handoff (chunk → LLM-summarize each chunk → final unifying pass → verbatim recent tail), not the `ChunkExtraction` → `merge_states` → `DurableMemorySet` extraction below. The rigid schema, deterministic merge, and 5 markdown docs were removed after the free-form path measurably outperformed them on real sessions. For the implemented design see [local-coder-massaging.md §35](local-coder-massaging.md) and `codex-rs/routing/src/compaction/`. This document is retained as the **algorithm-history reference** for the original `coding-agent-router` port — the chunking/normalize/token-budget mechanics still inform the current pipeline, and the extraction/merge rules record *why* the deterministic approach was tried.
>
> **Migration source:** `coding-agent-router/app/compaction/` (service.py, extractor.py, refiner.py, chunking.py, merger.py, normalize.py, storage.py, handoff.py, models.py, structured_output.py, durable_memory.py, prompts.py) plus `app/prompts/compaction_extraction_system.md` and `app/prompts/compaction_refinement_system.md`
>
> This document preserves the complete *original* compaction pipeline — every algorithm step, threshold, data model, prompt, and merge rule.

---

## Purpose

Long-running agent sessions accumulate large conversation histories that exceed model context windows. The compaction pipeline:

1. Normalizes raw transcript into compactable items
2. Chunks items by prompt-budget boundaries
3. Extracts durable state from each chunk via a compactor LLM
4. Merges chunk states deterministically
5. Refines merged state with recent raw turns
6. Renders durable memory files and a structured handoff

The output is a compact representation of the session that preserves task state, decisions, errors, and plans — enabling a new agent to continue work without seeing the full history.

---

## Pipeline overview

```
Raw transcript items
    │
    ▼
1. NORMALIZE — strip encrypted_content, attachments, tool_result blocks,
               detect precompacted summaries, preserve newest raw turn
    │
    ▼
2. SPLIT RECENT RAW — keep last ~8000 tokens of raw turns aside for refinement
    │
    ▼
3. CHUNK — split compactable items at transcript-item boundaries using
           prompt-budget estimation (not raw token count)
    │
    ▼
4. EXTRACT — for each chunk, call compactor LLM with structured output schema
             to extract durable state (objective, repo_state, files, errors, etc.)
    │
    ▼
5. MERGE — deterministically combine all chunk extractions:
           latest-non-empty for scalars, shallow-merge for dicts,
           deduplicated-reverse for lists
    │
    ▼
6. REFINE — iteratively feed recent raw turns (≤8000 tokens per iteration)
            to compactor LLM, merge each extraction onto accumulated state
    │
    ▼
7. RENDER — produce 5 durable memory markdown files + structured SessionHandoff
    │
    ▼
8. PERSIST — save all artifacts to disk under state/compaction/<session_id>/
```

---

## Data models

### TranscriptChunk

```python
class TranscriptChunk(BaseModel):
    chunk_id: int                          # Unique chunk identifier (1-based)
    start_index: int                       # Index of first item in source transcript
    end_index: int                         # Index after last item (exclusive)
    token_count: int                       # Total tokens in chunk items
    overlap_from_previous_tokens: int = 0  # Tokens shared with previous chunk
    items: List[Dict[str, Any]] = []       # The transcript items in this chunk
```

### ChunkExtraction

```python
class ChunkExtraction(BaseModel):
    chunk_id: int                                    # Source chunk ID
    objective: str = ""                              # Latest task objective
    repo_state: Dict[str, Any] = {}                  # Concrete repo facts (key→value)
    files_touched: List[str] = []                    # Real file paths acted on
    commands_run: List[str] = []                      # Shell commands executed
    errors: List[str] = []                           # Concrete failures
    accepted_fixes: List[str] = []                   # Applied solutions
    rejected_ideas: List[str] = []                   # Rejected approaches
    constraints: List[str] = []                      # Instructions constraining future work
    environment_assumptions: List[str] = []          # Infrastructure assumptions
    pending_todos: List[str] = []                    # Outstanding tasks
    unresolved_bugs: List[str] = []                  # Known open bugs
    test_status: List[str] = []                      # Test outcomes
    external_references: List[str] = []              # URLs, endpoints, services
    latest_plan: List[str] = []                      # Current plan steps
    source_token_count: int = 0                      # Tokens in source content
```

### MergedState

Same fields as ChunkExtraction plus:
```python
    merged_chunk_count: int = 0   # Number of chunks merged into this state
```

### SessionHandoff

```python
class SessionHandoff(BaseModel):
    stable_task_definition: str = ""                 # = objective
    repo_state: Dict[str, Any] = {}                  # = repo_state
    key_decisions: List[str] = []                    # = accepted_fixes
    unresolved_work: List[str] = []                  # = pending_todos + unresolved_bugs
    latest_plan: List[str] = []                      # = latest_plan
    failures_to_avoid: List[str] = []                # = errors + rejected_ideas
    recent_raw_turns: List[Dict[str, Any]] = []      # Uncompacted conversation tail
    current_request: str = ""                        # Latest user request
```

### DurableMemorySet

Five markdown documents:

```python
class DurableMemorySet(BaseModel):
    task_state: str = ""          # Objective, files, commands, test status
    decisions: str = ""           # Accepted fixes, constraints
    failures_to_avoid: str = ""   # Errors, rejected ideas
    next_steps: str = ""          # Pending TODOs, latest plan
    session_handoff: str = ""     # Stable task definition, unresolved work
```

### CodexHandoffFlow

```python
class CodexHandoffFlow(BaseModel):
    durable_memory: List[Dict[str, str]] = []     # [{name: "TASK_STATE.md", content: "..."}, ...]
    structured_handoff: Dict[str, Any] = {}       # SessionHandoff fields (sans recent_raw_turns)
    recent_raw_turns: List[Dict[str, Any]] = []   # Uncompacted tail
    current_request: str = ""                     # Latest request
```

---

## Step 1: Normalization

**Source:** `compaction/normalize.py`

### What normalization does

1. **Strips `encrypted_content`** recursively from all items
2. **Removes attachment blocks** — types: `image`, `input_image`, `localImage`, `local_image`, `file`, `input_file`
3. **Drops historical `tool_result` and `function_call_output` blocks** from compactable items (these are tool outputs, not needed for state extraction)
4. **Detects precompacted summaries** — text starting with `"Another language model started to solve this problem and produced a summary of its thinking process."` or containing both `"## Thread Summary for Continuation"` and `"Latest Real User Intent"` — these are carried forward raw, not re-compacted
5. **Strips Codex bootstrap `instructions`** block (text starting with `"You are Codex"`) for inline compaction
6. **Preserves the newest top-level turn raw** — the last non-None item is set aside uncompacted

### Token budget for individual items

Items exceeding `max_item_tokens` (= `_extraction_hard_chunk_tokens()`, derived from `COMPACTOR_TARGET_CHUNK_TOKENS`, default 10,000) are **skipped from chunk extraction** and carried forward raw into the recent_raw_turns list.

### Annotation

Each item gets a `_compaction_index` metadata field tracking its original position. This is stripped before final output. Sorting by `_compaction_index` preserves chronological order after merges.

---

## Step 2: Split recent raw turns

**Source:** `compaction/chunking.py` → `split_recent_raw_turns()`

Walk backwards from the end of compactable items. Accumulate item tokens until `COMPACTOR_KEEP_RAW_TOKENS` (default: 8,000) is reached.

```python
# Pseudocode
kept_tokens = 0
split_index = len(items)
for index from len(items)-1 down to 0:
    item_tokens = estimate_tokens(items[index])
    if kept_tokens > 0 and kept_tokens + item_tokens > keep_tokens:
        break
    kept_tokens += item_tokens
    split_index = index
    if kept_tokens >= keep_tokens:
        break
return items[:split_index], items[split_index:]   # (compactable, recent_raw)
```

---

## Step 3: Chunking

**Source:** `compaction/chunking.py` → `chunk_transcript_items_by_prompt()`

Chunks are created at **transcript item boundaries** (never splitting a message mid-content). The chunking algorithm uses **prompt-budget estimation** — it estimates the full extraction request size (system prompt + payload + schema) for each candidate chunk, not just the raw transcript token count.

### Algorithm

```python
for each candidate end position:
    token_total += item_tokens[end]
    if max_chunk_tokens set and token_total > max_chunk_tokens:
        break
    prompt_tokens = estimate_extraction_request_tokens(candidate_chunk)
    if prompt_tokens > max_prompt_tokens:
        break
    best_end = end + 1
    if prompt_tokens >= target_prompt_tokens:
        break
```

### Overlap

Chunks share overlap with their predecessor. Walk backwards from the chunk end, accumulating tokens up to `COMPACTOR_OVERLAP_TOKENS` (default: 1,500).

### Skipped items

If a single item cannot fit the extraction prompt budget even alone, it is **skipped** (added to `skipped_items` list) and merged back into recent_raw_turns. This prevents the pipeline from failing on oversized items.

### Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `COMPACTOR_TARGET_CHUNK_TOKENS` | 10,000 | Hard chunk-content limit |
| `COMPACTOR_MAX_CHUNK_TOKENS` | 10,000 | Legacy compat; effective limit is min of both |
| `COMPACTOR_MAX_PROMPT_TOKENS` | 12,256 | Hard ceiling for full extraction request |
| `COMPACTOR_OVERLAP_TOKENS` | 1,500 | Overlap between consecutive chunks |

---

## Step 4: Extraction

**Source:** `compaction/extractor.py`, `compaction/prompts.py`

### What happens

For each chunk, call the compactor LLM with:
- **System prompt:** The extraction system prompt (see below)
- **User content:** A compact JSON payload containing the chunk events
- **Response format:** Strict JSON schema (`chunk_extraction_response_schema()`)
- **Think:** disabled
- **Temperature:** `COMPACTOR_TEMPERATURE` (default: 0.0 — deterministic)

### Event compaction

Before sending to the LLM, transcript items are compacted into a dense event stream:

| Event key | Meaning |
|-----------|---------|
| `r` | Role: `u` = user, `a` = assistant |
| `k` | Kind: `msg`, `cmd`, `plan`, `call`, `poll`, `stdin`, `out` |
| `c` | Main text or command content |
| `wd` | Working directory (only when changed) |
| `sid` | PTY session ID (for `poll`/`stdin`) |
| `n` | Tool name (for generic `call`) |
| `a` | Compact tool arguments (for generic `call`) |
| `steps` | Plan steps (for `plan`) |

Special tool-call compaction:
- `exec_command` → `{"k": "cmd", "c": "<command>", "wd": "<dir if changed>"}`
- `write_stdin` with chars → `{"k": "stdin", "sid": "...", "c": "..."}`
- `write_stdin` without chars → `{"k": "poll", "sid": "..."}` (empty polls compacted)
- `update_plan` → `{"k": "plan", "steps": [...]}` (or dropped if no steps)
- Others → `{"k": "call", "n": "<name>", "a": <compact_args>}`

Compact args strip keys: `yield_time_ms`, `max_output_tokens`, and any empty values.

### Context budget

```python
target_prompt_tokens = num_ctx - minimum_response_tokens - slack
                     = 16384 - 1024 - 256 = 15104  (with defaults)

# If target can't fit, try burst context:
burst_prompt_tokens = burst_num_ctx - minimum_response_tokens - slack
                    = 17408 - 1024 - 256 = 16128  (with defaults)
```

| Constant | Value | Purpose |
|----------|-------|---------|
| `_TOKEN_ESTIMATION_SLACK` | 256 | Buffer for token estimation error |
| `_MIN_EXTRACTION_RESPONSE_TOKENS` | 1,024 | Minimum tokens reserved for model output |
| `COMPACTOR_NUM_CTX` | 16,384 | Normal context window |
| `COMPACTOR_BURST_NUM_CTX` | 17,408 | Burst context for large chunks |

### Extraction system prompt

```
You are extracting durable coding-session state for a later Codex handoff.

Return exactly one JSON object and nothing else.
Do not use markdown fences.
Do not explain your answer.
Do not include prose before or after the JSON.

This extraction is chunk-local:
- use only facts present in this chunk
- prefer newer facts over older facts inside the chunk
- if unsure, omit the fact instead of guessing
- empty strings, empty arrays, and empty objects are valid

Input notes:
- chunk.events is an ordered compact event stream
- event keys: r (role), k (kind), c (content), wd (workdir), sid (session),
  n (tool name), a (tool args), steps (plan steps)
- poll means the agent checked an existing PTY session without sending input
- chronology matters; use event order

Field rules:
- objective: latest stable task objective visible in the chunk
- repo_state: concrete repo facts only, emitted as {"key":"...","value":"..."} entries
- files_touched: real file paths mentioned or acted on
- commands_run: shell commands that were actually run
- errors: concrete failures, parser errors, bad outputs
- accepted_fixes: fixes already applied or clearly accepted
- rejected_ideas: ideas explicitly rejected or shown to fail
- constraints: instructions or requirements that constrain future work
- environment_assumptions: concrete environment/infrastructure assumptions
- pending_todos: remaining concrete tasks
- unresolved_bugs: still-open bugs or failure modes
- test_status: concrete test outcomes
- external_references: endpoints, hosts, credentials, services, model tags
- latest_plan: most recent active plan steps if present, otherwise []
- source_token_count: copy from chunk metadata

If the chunk contains a failure and a later fix, include both.
If a command failed, record under errors, not accepted_fixes.
If a file path appears in an error/tool payload and is relevant, include it.
```

### Response schema

Strict JSON schema with `additionalProperties: false`. The `repo_state` field uses an array-of-entries format:
```json
{"repo_state": [{"key": "branch", "value": "main"}, {"key": "runtime", "value": "python3.11"}]}
```
This is normalized back to a dict at runtime.

---

## Step 5: Deterministic merge

**Source:** `compaction/merger.py`

### Merge rules

| Field | Strategy | Detail |
|-------|----------|--------|
| `objective` | Latest non-empty | Last non-empty string wins |
| `repo_state` | Shallow dict merge | Later values overwrite earlier for same key |
| `latest_plan` | Latest non-empty list | Last non-empty list wins entirely |
| All list fields | Deduplicate reverse | Process groups in reverse order; case-insensitive dedup; newer entries take priority; original casing preserved |
| `merged_chunk_count` | Sum | Sum all counts (ChunkExtraction counts as 1) |

### List deduplication algorithm

```python
def _merge_unique(groups):
    seen = set()
    merged = []
    for group in reversed(list(groups)):   # Process newest first
        for item in group:
            key = item.strip().lower()
            if key == "" or key in seen:
                continue
            seen.add(key)
            merged.append(item)            # Preserve original casing
    return merged
```

This means: if chunk 3 says `"test_auth passing"` and chunk 1 says `"test_auth failing"`, only the chunk 3 version survives (newer wins).

---

## Step 6: Refinement

**Source:** `compaction/refiner.py`, `compaction/service.py`

### Purpose

The merged state from chunk extraction may miss facts from the most recent raw turns (which were split out in Step 2). Refinement feeds these recent turns to the compactor LLM in bounded iterations.

### Algorithm

```python
state = merged
remaining = list(recent_raw_turns)
iteration = 0

# Check if base prompt alone exceeds budget
base_prompt_tokens = estimate_refinement_request_tokens(state, [], current_request, repo_context)
if base_prompt_tokens > max_prompt_tokens:
    return merged  # Skip refinement entirely

while remaining:
    # Take items that fit budget
    chunk_items, next_index, skipped = take_items_with_prompt_budget(
        items=remaining,
        target_prompt_tokens=max_prompt_tokens,
        max_prompt_tokens=max_prompt_tokens,
        target_item_tokens=8000,   # _REFINEMENT_RECENT_RAW_TARGET_TOKENS
        item_token_counter=estimate_tokens,
        prompt_token_counter=lambda items: estimate_refinement_request_tokens(state, items, ...)
    )

    if skipped:
        remaining = remaining[1:]  # Skip oversized item
        continue

    if not chunk_items:
        break

    iteration += 1
    recent_state = refiner.refine_state(state, chunk_items, current_request, repo_context)

    # Check if extraction had any effect
    if not _recent_state_has_effect(recent_state):
        # No useful facts extracted — skip
        remaining = remaining[next_index:]
        continue

    state = merge_states([state, recent_state])
    state.merged_chunk_count = original_merged_chunk_count  # Preserve count
    remaining = remaining[next_index:]

return state
```

### Effect detection

A refinement extraction "has effect" if any of these fields are truthy: `objective`, `repo_state`, `files_touched`, `commands_run`, `errors`, `accepted_fixes`, `rejected_ideas`, `constraints`, `environment_assumptions`, `pending_todos`, `unresolved_bugs`, `test_status`, `external_references`, `latest_plan`.

### Refinement system prompt

```
You are extracting durable coding-session state from recent raw transcript events
for a later Codex handoff.

Return exactly one JSON object and nothing else.

This is a recent-state extraction pass, not a diff or patch pass.
- extract state only from recent_events and current_request
- do not infer facts from older transcript history
- prefer newer facts over older facts
- never invent facts, file paths, commands, errors, or plans
- if unsure, leave the field empty
```

### Refinement budget

| Constant | Value | Purpose |
|----------|-------|---------|
| `_REFINEMENT_RECENT_RAW_TARGET_TOKENS` | 8,000 | Max tokens per refinement iteration |
| `_MIN_REFINEMENT_RESPONSE_TOKENS` | 512 | Minimum response budget |
| `_TOKEN_ESTIMATION_SLACK` | 256 | Buffer |

---

## Step 7: Rendering

**Source:** `compaction/durable_memory.py`, `compaction/handoff.py`

### SessionHandoff construction

```python
SessionHandoff(
    stable_task_definition = state.objective,
    repo_state = state.repo_state,
    key_decisions = state.accepted_fixes,
    unresolved_work = [*state.pending_todos, *state.unresolved_bugs],
    latest_plan = state.latest_plan,
    failures_to_avoid = [*state.errors, *state.rejected_ideas],
    recent_raw_turns = recent_raw_turns,
    current_request = current_request,
)
```

### DurableMemorySet construction

Five markdown documents rendered from the merged state:

| File | Contents |
|------|----------|
| `TASK_STATE.md` | `# Task State` → Objective, Files Touched, Commands Run, Test Status |
| `DECISIONS.md` | `# Decisions` → Accepted Fixes, Constraints |
| `FAILURES_TO_AVOID.md` | `# Failures to Avoid` → Errors, Rejected Ideas |
| `NEXT_STEPS.md` | `# Next Steps` → Pending TODOs, Latest Plan |
| `SESSION_HANDOFF.md` | `# Session Handoff` → Stable Task Definition, Unresolved Work |

Each section is rendered as `## Heading` followed by bulleted items, or `- none` if empty.

### CodexHandoffFlow construction

```python
CodexHandoffFlow(
    durable_memory = [
        {"name": "TASK_STATE.md", "content": task_state_md},
        {"name": "DECISIONS.md", "content": decisions_md},
        {"name": "FAILURES_TO_AVOID.md", "content": failures_md},
        {"name": "NEXT_STEPS.md", "content": next_steps_md},
        {"name": "SESSION_HANDOFF.md", "content": session_handoff_md},
    ],
    structured_handoff = {handoff fields except recent_raw_turns and current_request},
    recent_raw_turns = handoff.recent_raw_turns,
    current_request = current_request,
)
```

### Handoff validation

Before rendering, `validate_codex_handoff_flow()` checks:
1. All durable_memory items have non-empty `name`
2. Structured handoff fields validate as SessionHandoff
3. Both structured_handoff and recent_raw_turns are JSON-serializable

If validation fails, the router falls back to plain `system + prompt` instead of crashing.

### Prompt templates

**compacted_flow.md** (used for machine consumption):
```
Durable memory:

{{DURABLE_MEMORY_BLOCKS}}

Structured handoff:

{{STRUCTURED_HANDOFF}}

Recent raw turns:

{{RECENT_RAW_TURNS}}

Current request:

{{CURRENT_REQUEST}}
```

**codex_support_prompt.md** (used for Codex CLI handoff):
```
{{SYSTEM_SECTION}}Durable memory:

{{DURABLE_MEMORY_BLOCKS}}

Structured handoff:

{{STRUCTURED_HANDOFF}}

Recent raw turns:

{{RECENT_RAW_TURNS}}

Current request:

{{CURRENT_REQUEST}}
```

---

## Step 8: Persistence

**Source:** `compaction/storage.py`

All artifacts saved under `COMPACTION_STATE_DIR/<session_id>/`:

```
state/compaction/<session_id>/
├── chunks/
│   ├── chunk-1.json          # ChunkExtraction for chunk 1
│   ├── chunk-2.json          # ChunkExtraction for chunk 2
│   └── ...
├── merged-state.json         # MergedState after chunk merge
├── refined-state.json        # MergedState after refinement
├── handoff.json              # SessionHandoff
├── TASK_STATE.md             # Durable memory
├── DECISIONS.md
├── FAILURES_TO_AVOID.md
├── NEXT_STEPS.md
└── SESSION_HANDOFF.md
```

### Refresh-if-needed optimization

`refresh_if_needed()` short-circuits: if `estimate_tokens(items) < _extraction_hard_chunk_tokens()` and a saved handoff exists, return the saved handoff without recompacting. This avoids redundant compaction for short sessions.

---

## Structured output schema

**Source:** `compaction/structured_output.py`

### Normalization rules

Model output is normalized before validation:

| Field | Normalization |
|-------|--------------|
| `repo_state` | If dict → pass through. If list of `{key, value}` entries → convert to dict. If list of strings → join as `summary`. If scalar → wrap as `{summary: value}`. |
| `test_status` | If dict → format as `"key: value"` pairs. If list → normalize as string list. |
| `latest_plan` | If list of dicts with `step`/`status` → format as `"step [status]"`. If strings → pass through. |
| All other list fields | Normalize each item: dicts → `"key: value; ..."`, scalars → stringify, booleans → `"true"/"false"`. |

### JSON Schema

The extraction response schema uses **strict mode** (`additionalProperties: false`, all fields required). The `repo_state` field is schema'd as an array of `{key: string, value: string}` objects (normalized to dict at runtime).

---

## Compactor model configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `COMPACTOR_OLLAMA_BASE_URL` | `http://127.0.0.1:11435` | Ollama instance |
| `COMPACTOR_MODEL` | `qwen3.5:9b` | Compactor model |
| `COMPACTOR_NUM_CTX` | 16,384 | Normal context |
| `COMPACTOR_BURST_NUM_CTX` | 17,408 | Burst context for large chunks |
| `COMPACTOR_TEMPERATURE` | 0.0 | Deterministic extraction |
| `COMPACTOR_TIMEOUT_SECONDS` | 1,800 | Timeout |
| `COMPACTION_STATE_DIR` | `state/compaction` | Artifact storage root |
| `INLINE_COMPACT_SENTINEL` | `<<<LOCAL_COMPACT>>>` | Trigger for inline compaction |

---

## Migration notes

1. **The compaction pipeline is self-contained.** It depends only on: OllamaClient, estimate_tokens, prompt_loader, config settings. All can be provided by the orchestrator.

2. **The Ollama client used by compaction must respect per-endpoint serialization** — same `asyncio.Semaphore(1)` as the routing Ollama client. The compactor may share an Ollama instance with the coder (both default to port 11435).

3. **The prompt files must be migrated verbatim** — the extraction and refinement system prompts encode carefully tested instructions. Do not paraphrase or simplify them.

4. **The structured output schema must be preserved exactly** — the `additionalProperties: false` and entry-array format for `repo_state` were chosen because they work reliably with qwen3.5:9b. Changing the schema may break extraction.

5. **The normalization rules handle real model output quirks** — models sometimes return `repo_state` as a list of entries instead of a dict, `latest_plan` as dicts with `step`/`status` keys instead of strings, etc. The normalization layer must be preserved.

6. **Refresh-if-needed is a critical optimization** — without it, every compaction call reprocesses the full transcript. The short-circuit check (`tokens < hard_chunk_tokens and handoff exists`) must be preserved.
