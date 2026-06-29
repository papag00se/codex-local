# Routing Logic Reference

[< Spec Index](index.md) | [Product Index](../product/index.md)

> **Migration source:** `coding-agent-router/app/router.py`, `task_metrics.py`, `config.py`, `tool_adapter.py`, `clients/ollama_client.py`, `clients/codex_client.py`, `prompts/router_system.md`, `prompts/router_task.md`
>
> This document preserves every deliberated heuristic, threshold, fallback path, and configuration knob from the coding-agent-router routing system. Nothing in this document should be discarded during implementation — every value was chosen through testing.

## Route set

Three execution backends:

| Route | Backend | Tool support | Default model |
|-------|---------|-------------|---------------|
| `local_coder` | Ollama | Structured tool calls (with leaked-tool-call recovery — Hermes/XML/malformed) | `qwen3-coder:30b-a3b-q4_K_M` |
| `local_reasoner` | Ollama | Plain text only (no tool surface) | `qwen3:14b` |
| `codex_cli` | Codex CLI subprocess | Full Codex tool loop | `gpt-5.4` via `openai` provider |

## Route selection algorithm

Exact algorithm from `RoutingService.route()`:

```
1. Build routing digest (metrics payload) from system + prompt + metadata
2. If preferred_backend is set → return it immediately (confidence 1.0)
3. Build available list from enabled backends:
   - enable_local_coder → 'local_coder'
   - enable_local_reasoner → 'local_reasoner'  
   - enable_codex_cli and client configured → 'codex_cli'
4. If no backends available → return 'local_reasoner' (confidence 0.0)
5. Begin eligibility filtering on a copy of available list:
   a. If backend_request_tokens > REASONER_NUM_CTX → remove 'local_reasoner'
   b. If backend_request_tokens > CODER_NUM_CTX → remove 'local_coder'
6. Compute router_payload_tokens from the digest
7. If router_request_tokens > ROUTER_NUM_CTX:
   → return 'codex_cli' if available, else fallback order (confidence 1.0)
   → reason: "router request exceeds router context window"
8. If eligible list is empty:
   → return 'codex_cli' if available, else fallback order (confidence 1.0)
   → reason: "local context windows exceeded"
9. If exactly one eligible backend remains:
   → return it (confidence 1.0)
   → reason: "only one route fits context limits"
10. Multiple eligible → ask router model for JSON decision:
    - Model: ROUTER_MODEL (default: qwen3:8b-q4_K_M)
    - Temperature: 0.0 (deterministic)
    - Context: ROUTER_NUM_CTX (default: 8192)
    - Format: JSON
    - Think: disabled
    - System prompt: "Return JSON only with keys: route, confidence, reason. Pick exactly one route from the available routes."
    - User content: the full routing digest as JSON
11. Parse router response:
    - Extract route, confidence, reason from JSON
    - If route not in eligible list → use fallback order
    - If JSON parse fails → use fallback order (confidence 0.0, reason: "router JSON parse fallback")
```

## Deterministic fallback order

```python
def _fallback_route(available):
    if 'local_coder' in available: return 'local_coder'
    if 'local_reasoner' in available: return 'local_reasoner'
    if 'codex_cli' in available: return 'codex_cli'
    return available[0] if available else 'local_reasoner'
```

This order is intentional: coder is preferred because it supports tools. Reasoner is second because it's local/free. Codex CLI is the cloud fallback.

## Token estimation

Two estimation methods coexist:

### Quick estimate (used for routing decisions)
```python
def estimate_tokens(value):
    text = json.dumps(value) if not isinstance(value, str) else value
    return max(1, (len(text) + 3) // 4)
```

### Tiktoken estimate (used for Spark/mini rewriting, OpenAI token counting)
```python
def estimate_model_tokens(value, *, model):
    text = json.dumps(value) if not isinstance(value, str) else value
    encoding = tiktoken.get_encoding('o200k_base')  # for gpt-5.x / codex
    return len(encoding.encode(text))
```

The quick estimate is used for routing because the router model itself has limited context — a rough estimate is sufficient for "does this fit?" decisions. The tiktoken estimate is used when precise token counting matters (Spark cap at 114,688 tokens).

## Routing digest construction

`build_routing_digest()` produces the payload sent to the router model:

```json
{
  "task": "Choose exactly one route from available_routes for this request. Return JSON only with keys route, confidence, reason. route must exactly match one entry in available_routes.",
  "available_routes": ["local_coder", "local_reasoner", "codex_cli"],
  "user_prompt": "<the latest user message>",
  "trajectory": [<prior messages excluding the latest>],
  "metrics": {
    "backend_request_tokens": 4200,
    "reasoner_context_limit": 16384,
    "coder_context_limit": 16384,
    "router_context_limit": 8192,
    "router_payload_tokens": 1800,
    "router_request_tokens": 1800,
    "user_prompt_chars": 340,
    "user_prompt_lines": 8,
    "user_prompt_tokens": 85,
    "trajectory_chars": 12000,
    "trajectory_lines": 200,
    "trajectory_tokens": 3000,
    "message_count": 12,
    "user_message_count": 6,
    "assistant_message_count": 5,
    "tool_message_count": 1,
    "tool_call_count": 3,
    "command_count": 5,
    "command_output_tokens": 800,
    "file_reference_count": 14,
    "unique_file_reference_count": 7,
    "code_block_count": 4,
    "json_block_count": 1,
    "diff_line_count": 23,
    "error_line_count": 2,
    "stack_trace_count": 1,
    "prior_failure_count": 0,
    "question_count": 2,
    "metadata_key_count": 3
  }
}
```

## Task metrics extraction

27 metrics extracted from every request by `extract_task_metrics()`:

| Metric | Source | How computed |
|--------|--------|-------------|
| `user_prompt_chars` | User prompt | `len(prompt)` |
| `user_prompt_lines` | User prompt | `text.count('\n') + 1` |
| `user_prompt_tokens` | User prompt | Quick estimate |
| `trajectory_chars` | Trajectory (prior messages) | `len(json.dumps(trajectory))` |
| `trajectory_lines` | Trajectory | Line count of stringified |
| `trajectory_tokens` | Trajectory | Quick estimate |
| `message_count` | Trajectory | Count dicts with 'role' key |
| `user_message_count` | Trajectory | role == 'user' |
| `assistant_message_count` | Trajectory | role == 'assistant' |
| `tool_message_count` | Trajectory | role in ('tool', 'function') |
| `tool_call_count` | Combined text | Regex: `tool_call\|tool_calls\|function_call\|recipient_name` |
| `command_count` | Combined text | Regex: lines starting with `$ ` or common command names |
| `command_output_tokens` | Trajectory | Sum tokens of stdout/stderr/output/result fields |
| `file_reference_count` | Combined text | Regex: paths matching known extensions |
| `unique_file_reference_count` | Combined text | Deduplicated case-insensitive |
| `code_block_count` | Combined text | Count ``` pairs / 2 |
| `json_block_count` | Combined text | Count ```json or ```javascript blocks |
| `diff_line_count` | Combined text | Lines starting with +/- (not +++ or ---) |
| `error_line_count` | Combined text | Lines containing error/exception/failed/failure/traceback/panic/fatal |
| `stack_trace_count` | Combined text | Traceback patterns, "at file:line" patterns |
| `prior_failure_count` | Trajectory attempts | Count items with status in FAILURE_STATUSES |
| `question_count` | User prompt | Count `?` characters |
| `metadata_key_count` | Metadata dict | `len(metadata)` |

### Recognized file extensions
```
py, js, ts, tsx, jsx, md, yml, yaml, json, toml, go, java, rb, php, rs,
cpp, c, h, sql, sh, bash, html, css, scss, vue, svelte, kt, kts
```

### Failure statuses (for prior_failure_count)
```
error, failed, failure, timeout, timed_out, cancelled, canceled,
low_confidence, malformed_output
```

### Command detection patterns
Lines starting with: `$ `, `bash `, `sh `, `zsh `, `fish `, `python `, `python3 `, `node `, `npm `, `pnpm `, `yarn `, `uv `, `pytest `, `git `, `rg `, `sed `, `cat `, `ls `, `curl `, `ollama `

## Configuration knobs (all with defaults)

### Router model
| Variable | Default | Purpose |
|----------|---------|---------|
| `ROUTER_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama instance for the router model |
| `ROUTER_MODEL` | `qwen3:8b-q4_K_M` | Model that makes routing decisions |
| `ROUTER_NUM_CTX` | `8192` | Context window for router model |
| `ROUTER_TEMPERATURE` | `0.0` | Temperature (deterministic) |
| `ROUTER_TIMEOUT_SECONDS` | `1800` | Timeout for router inference |

### Local coder
| Variable | Default | Purpose |
|----------|---------|---------|
| `CODER_OLLAMA_BASE_URL` | `http://127.0.0.1:11435` | Ollama instance for coder |
| `CODER_MODEL` | `qwen3-coder:30b-a3b-q4_K_M` | Coder model |
| `CODER_NUM_CTX` | `16384` | Coder context window |
| `CODER_TEMPERATURE` | `0.1` | Coder temperature |
| `CODER_TIMEOUT_SECONDS` | `1800` | Coder timeout |
| `ENABLE_LOCAL_CODER` | `true` | Enable/disable coder route |

### Local reasoner
| Variable | Default | Purpose |
|----------|---------|---------|
| `REASONER_OLLAMA_BASE_URL` | `http://127.0.0.1:11436` | Ollama instance for reasoner |
| `REASONER_MODEL` | `qwen3:14b` | Reasoner model |
| `REASONER_NUM_CTX` | `16384` | Reasoner context window |
| `REASONER_TEMPERATURE` | `0.1` | Reasoner temperature |
| `REASONER_TIMEOUT_SECONDS` | `1800` | Reasoner timeout |
| `ENABLE_LOCAL_REASONER` | `true` | Enable/disable reasoner route |

### Codex CLI
| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_CODEX_CLI` | `false` | Enable/disable Codex CLI route |
| `CODEX_CMD` | `codex` | Codex binary path |
| `CODEX_WORKDIR` | `.` | Default working directory |
| `CODEX_EXEC_MODEL_PROVIDER` | `openai` | Model provider for codex exec |
| `CODEX_EXEC_MODEL` | `gpt-5.4` | Model for codex exec |
| `CODEX_TIMEOUT_SECONDS` | `1800` | Subprocess timeout |

### Connection pooling
| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_CONNECT_TIMEOUT_SECONDS` | `5` | TCP connect timeout |
| `OLLAMA_POOL_CONNECTIONS` | `8` | HTTP connection pool size |
| `OLLAMA_POOL_MAXSIZE` | `16` | Max pool size |

## Ollama client serialization

The Ollama client serializes requests per-endpoint using file locks:

```python
lock_path = state/ollama_locks/<sha256(base_url)>.lock
fcntl.flock(lock_file, LOCK_EX)  # Blocks until lock acquired
try:
    # Send request to Ollama
finally:
    fcntl.flock(lock_file, LOCK_UN)
```

This ensures only one request hits each Ollama instance at a time. Ollama struggles with concurrent requests — this was discovered through testing.

**Migration note:** In the orchestrator, replace file locking with `asyncio.Semaphore(1)` per Ollama endpoint. Same semantics, no filesystem coordination needed since only the orchestrator talks to Ollama.

## Tool call recovery for local models

Local models routinely emit tool calls as **text** instead of the structured `tool_calls` field — the server's chat template (even `--jinja`) doesn't recognize the finetune's format. The Rust implementation is `codex_routing::tool_recovery::recover_tool_calls`, the **single** recovery entry point shared by the coder path (`light_coder`) and the reasoner path (`light_reasoner`). (It previously existed as two divergent implementations plus a buried third; a format fix that landed in only one let `light_reasoner` leaks slip through. Unified — see [Local Coder Massaging §21](local-coder-massaging.md).)

### Recovery strategy (`recover_tool_calls`)

```
1. If message already has structured tool_calls → return as-is
2. Strategy 0 — leaked <tool_call> blocks (via tool_aliases::parse_leaked_tool_calls):
   - Hermes JSON:   <tool_call>{"name":..,"arguments":{..}}</tool_call>
   - XML-function:  <tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>
                    (also <function name="NAME">; numerics coerced; multi-line cmds preserved)
   - Malformed:     a <tool_call> block whose JSON won't parse → detected, NOT executed;
                    caller re-prompts to re-issue cleanly (prefer write_file over heredocs)
3. Strategy 1 — whole content as a JSON blob (stripping ``` fences); "tool_calls" key → normalize
4. Strategy 2 — embedded tool blocks: split by double-newline; per paragraph,
   strip [USER]/[ASSISTANT] prefixes, parse JSON; type="tool_use"+name → tool call;
   type="tool_result" → drop (echoed context); else keep as text
5. Return cleaned text + recovered tool calls
```

The coder path bridges the recovered `ToolCall`s to the Ollama wire shape (`tool_call_to_wire` → `translate_native_tool_calls`) so recovered and structured calls share the same normalization + shell-alias translation; the reasoner path emits them directly. Either way there is exactly one recovery pass.

### Streaming recovery (`recover_stream_ollama_message`)

Same as above but with additional handling:
- During streaming, the last paragraph may be an incomplete JSON blob
- `_looks_like_partial_tool_block()` detects this: starts with `{` and contains `"type"` + one of `tool_use`/`tool_result`/`tool_calls`
- Partial tool blocks are dropped from streaming text output (will be recovered in the final message)

### Tool call normalization

```python
# Arguments can arrive as:
# - dict → use directly
# - string → try json.loads, fallback to {"raw": string}
# - other → wrap in {"value": other}
# 
# Function can arrive as:
# - {"function": {"name": ..., "arguments": ...}} → standard
# - {"name": ..., "arguments": ...} → wrap in function key
```

### Stream deduplication

During streaming, tool calls are deduplicated by signature:
```python
signature = json.dumps({"name": name, "arguments": arguments}, sort_keys=True)
```
If a tool call with the same signature was already yielded, it's skipped. This prevents duplicate tool calls when both the chunk-level and recovery-level parsers find the same call.

## Codex CLI prompt construction

When routing to `codex_cli`, the prompt may be enriched with compaction handoff state:

```python
def _build_codex_cli_prompt(system, prompt, metadata):
    session_id = metadata.get('compaction_session_id') or metadata.get('session_id')
    if session_id:
        try:
            flow = compaction_service.build_codex_handoff_flow(session_id)
            if flow is not None:
                return render_codex_support_prompt(flow, system=system, current_request=current_request)
        except Exception:
            logger.exception('failed to render compaction handoff')
    # Fallback: plain concatenation
    return f'{system}\n\n{prompt}'.strip()
```

## Working directory resolution

Codex CLI working directory is resolved from metadata with this priority:

```
1. metadata['cwd']
2. metadata['workdir']
3. metadata['project_path']
4. metadata['repo_path']
5. metadata['repo_context']['cwd']
6. CODEX_WORKDIR setting (default: '.')
```

## Input normalization (Anthropic → routing format)

`anthropic_messages_to_prompt()` flattens Anthropic-format requests:

```
Input: AnthropicMessagesRequest with system, messages
Output: {
  system: flattened system text,
  prompt: all messages formatted as "[ROLE]\ncontent" joined by \n\n,
  user_prompt: content of the last user message,
  trajectory: all messages except the last
}
```

Content flattening: text blocks extracted as text, non-text blocks JSON-stringified.
