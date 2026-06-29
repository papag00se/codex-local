# Supervisor Integration

[< Spec Index](index.md) | [Integration Model](integration-model.md) | [Design Principles](design-principles.md)

## The integration approach: supervisor as a tool

The supervisor loop integrates into Codex as a **tool handler** — the thinnest possible integration point.

### How it works

1. The model receives a complex goal from the user
2. The model decides this needs decomposition (this is what frontier models already do naturally)
3. The model calls a `supervisor` tool with the goal
4. The tool handler runs the deterministic supervisor loop
5. Inside the loop, sub-agents are spawned via existing `AgentControl`
6. **The model never gets control back until the loop terminates**
7. The tool returns a summary of what was accomplished

The model cannot ask "should I continue?" because it's waiting for a tool call to return. The loop is deterministic code running inside the tool handler.

### Why this is the thinnest integration

| What we touch in upstream code | Where | Size |
|---|---|---|
| One tool handler file | `codex-rs/core/src/tools/handlers/supervisor.rs` | ~200 lines |
| Register the tool in the tool spec | `codex-rs/tools/src/tool_spec.rs` or equivalent | ~10 lines |
| Add crate dependencies | `codex-rs/core/Cargo.toml` | 2 lines |

Everything else — the supervisor loop, task graph, routing engine, metrics, tool-call recovery — lives in `codex-routing` and `codex-supervisor` crates that don't depend on codex-core.

### Upstream merge safety

When upstream changes `codex-core`:
- `codex-routing` and `codex-supervisor` are untouched (zero dependency on codex-core)
- The tool handler may need adjustment if `AgentControl`, `Session`, or `ToolHandler` APIs change
- This is the same risk as any other tool handler in the codebase — no worse, no better

### The SupervisorJudge implementation (IMPLEMENTED)

The tool handler creates a `CodexJudge` that bridges the supervisor trait to codex-core. It makes judgment calls **two** ways: spawn a full Codex sub-agent (`spawn_and_wait`), or call a local model endpoint directly (`call_endpoint` / `call_with_failover`). Which one is used depends on the call and the goal's complexity.

```rust
struct CodexJudge {
    session: Arc<Session>,
    turn: Arc<TurnContext>,
    routing_config: RoutingConfig,                            // local endpoints per role
    ollama_pool: Arc<OllamaClientPool>,                       // for direct local calls
    failover: codex_routing::project_config::FailoverChains,  // configured chains
}

// Spawn-a-sub-agent primitive: used for worker dispatch and complex-goal planning.
async fn spawn_and_wait(&self, prompt: &str) -> Result<String, String> {
    let thread_id = agent_control.spawn_agent(config, Op::UserInput { ... }, None).await?;
    let mut status_rx = agent_control.subscribe_status(thread_id).await?;
    while !is_final(&status) { status_rx.changed().await; }
    // return Completed(message) or error
}

impl SupervisorJudge for CodexJudge {
    // Planning: classify the goal. Simple goals plan on the LOCAL `planning`
    // failover chain (call_with_failover, reasoner → reasoner_backup fallback);
    // complex goals plan on a stronger model via a spawned sub-agent.
    async fn plan_tasks(&self, goal: &str) -> Vec<Task> { /* … parse_plan(output, goal) */ }

    // Worker dispatch: spawn a specialist sub-agent for the task, wait for it.
    async fn dispatch_task(&self, task: &Task) -> Result<String, String> {
        self.spawn_and_wait(&task.description).await
    }

    // Completion judgment: run on the LOCAL `evaluation` failover chain.
    async fn evaluate_completion(&self, task: &Task, output: &str) -> bool { /* … */ }

    // Deterministic: run the verification command, check exit code.
    async fn verify(&self, task: &Task, cmd: &str) -> bool { /* bash -c, success */ }
}
```

**Failover-chain-driven judgment.** Planning and evaluation resolve their model list from the project's `[failover]` `planning` / `evaluation` chains via two helpers:
- `local_endpoint_for_role(role)` — maps a chain role name (`light_reasoner`, `light_coder`, `compactor`, …) to a local endpoint; cloud roles return `None` (the supervisor reaches cloud only through `spawn_and_wait`).
- `local_chain(chain)` — resolves a chain name to the ordered list of usable local endpoints; empty for an all-cloud chain, in which case the caller falls back to `reasoner → reasoner_backup`.

So the same `[failover]` config that drives per-request routing also drives the supervisor's internal judgment, and `local_only` keeps planning/evaluation entirely on local models. Specialist sub-agents pick up their persona from `[roles.coder]` / `[roles.reviewer]` / `[roles.test_runner]` (`instructions` + `nickname`). The code is at `codex-rs/core/src/tools/handlers/supervisor.rs`.

### What the model sees

The supervisor tool is defined in the tool spec like any other tool:

```json
{
    "name": "supervisor",
    "description": "Run a multi-agent supervised workflow to accomplish a complex goal. The supervisor decomposes the goal into tasks, routes each to the best model, dispatches specialist agents, verifies results, and retries failures. Use this for goals that require multiple files, tests, or sequential steps.",
    "parameters": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The engineering goal to accomplish"
            },
            "verification_command": {
                "type": "string",
                "description": "Optional command to verify results (e.g., 'pytest tests/')"
            }
        },
        "required": ["goal"]
    }
}
```

The model calls it when the goal is complex. For simple goals ("fix the typo in README.md"), the model handles it directly — no supervisor involved.

### Process flow

```
User: "Add rate limiting with Redis and write tests"
  │
  ▼
Codex main agent (model turn):
  Model thinks: "This is complex — I'll use the supervisor tool"
  Model calls: supervisor(goal="Add rate limiting with Redis and write tests",
                          verification_command="pytest tests/")
  │
  ▼
Supervisor tool handler (deterministic loop — model is blocked):
  │
  ├─ Plan: ask model to decompose → 3 tasks
  ├─ Route: codex-routing picks model per task
  ├─ Dispatch: spawn sub-agent for task 1 → wait for completion
  ├─ Evaluate: ask model "is task 1 done?" → yes
  ├─ Dispatch: spawn sub-agent for task 2 → wait for completion
  ├─ Evaluate: ask model "is task 2 done?" → yes
  ├─ Dispatch: spawn sub-agent for task 3 → wait for completion
  ├─ Evaluate: ask model "is task 3 done?" → yes
  ├─ Verify: run "pytest tests/" → ask model to interpret → pass
  └─ Return: "3/3 tasks completed, all tests passing"
  │
  ▼
Model receives tool result: "3/3 tasks completed, all tests passing"
Model responds to user: "Done. I added rate limiting with Redis..."
```

The model never gets a chance to ask "should I continue?" — it's waiting for the tool to return.
