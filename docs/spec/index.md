# Codex Multi-Agent — Technical Specification

> Product documentation: [docs/product/index.md](../product/index.md)

## Architecture and Technical Specification
- **[Design Principles](design-principles.md) — Deterministic control, intelligent judgment. The core mantra.**
- **[Integration Model](integration-model.md) — How multi-agent orchestration fits into the existing Codex agent system**
- **[Supervisor Integration](supervisor-integration.md) — The supervisor-as-tool approach: thinnest possible integration**
- **[Implementation Status](implementation-status.md) — What's built, tested, and working right now**
- **[Gaps and Future Work](gaps.md) — Known gaps, prioritized, with implementation notes**
- [System Architecture Overview](system-architecture.md) — Layers, run flow, deterministic vs agentic split
- [Core Architectural Principles](architectural-principles.md) — Bounded loops, event-driven orchestration, durable state, verification-first
- [System Context Diagram](system-context-diagram.md) — ASCII diagrams of component interactions
- [Logical Components](logical-components.md) — 15 components with purpose, interfaces, failure modes
- [Agent Taxonomy](agent-taxonomy.md) — Agent roles, tools, autonomy limits, backend preferences
- [Routing Architecture](routing-architecture.md) — Unified routing model, scoring, fallback, budget/privacy-aware routing
- [Routing Logic Reference](routing-logic-reference.md) — **Preserved from coding-agent-router:** every heuristic, threshold, fallback path, config knob
- [Compaction Reference](compaction-reference.md) — **Preserved from coding-agent-router:** full compaction pipeline algorithm
- [Local Coder Massaging](local-coder-massaging.md) — Every orchestration-layer intervention that props up small local coder models: problem, fix, code pointer, log signal
- [Nudge Engine as a Service](nudge-service.md) — **Proposal:** extract the massaging layer into a standalone OpenAI-compatible streaming service (heartbeat keep-alive, session header, harness keeps tool execution + workspace)
- [Content Reduction (`content_reduce`)](content-reduce.md) — **Proposal:** MIME-aware, lossless-first reducer that bounds large tool outputs at the source (HTML→text, JSON/YAML tree-walk, guarded prose stripper), plus `find=` selection + `cursor=` pagination — replaces trim's destructive truncation
- [Provider Abstraction](provider-abstraction.md) — Provider types, capability schema, adapter interface
- [State Model](state-model.md) — SQLite schema, task state machine, entity relationships
- [Event Model](event-model.md) — 20+ event types with producers, consumers, payloads, idempotency
- [Verification and Safety](verification-safety.md) — What gets verified, approval policies, retry/hard-fail rules
- [Repository Isolation](repository-isolation.md) — Git worktrees, branch strategy, merge/conflict handling
- [Observability](observability.md) — Logs, metrics, traces, run summaries, replay strategy
- [Security Model](security-model.md) — Trust boundaries, secrets, subprocess sandboxing, least privilege
- [Technology Decisions](technology-decisions.md) — 10 technology decision records with alternatives and tradeoffs
- [Risks and Failure Modes](risks-and-failure-modes.md) — 13 risks with impact, likelihood, and mitigations

## Execution Plan
- [Project Structure](project-structure.md) — Monorepo layout, relationship with coding-agent-router
- [MVP Definition](mvp-definition.md) — Smallest useful v1 with success criteria
- [Milestone Plan](milestone-plan.md) — M0–M9 milestones with deliverables, dependencies, exit criteria
- [Task Backlog](task-backlog.md) — 55 tasks grouped by milestone with IDs, deps, complexity
- [Implementation Order](implementation-order.md) — 5-week build order with critical path
- [Initial Interfaces and Contracts](initial-interfaces.md) — Typed schemas with sample JSON payloads
- [Operational Model](operational-model.md) — How runs start, persist, verify, approve, recover from crashes
- [Testing Strategy](testing-strategy.md) — Unit, integration, replay, routing, contract, and crash recovery tests
- [Rollout Strategy](rollout-strategy.md) — Incremental trust-building from single-agent to full multi-agent
- [Future Expansion](future-expansion.md) — Growth paths: more providers, browser agents, distributed execution

## Implementation Scaffolding
- [Starter File Tree](starter-file-tree.md) — Concrete directory and file layout
- [Initial Data Models](initial-data-models.md) — Pydantic models ready to copy into code
- [Orchestrator Pseudocode](orchestrator-pseudocode.md) — Supervisor loop, dispatch, verification, retry, crash recovery
- [Provider Adapter Interface](provider-adapter-interface.md) — ABC + CodexCliAdapter skeleton
- [Service/Process List](service-process-list.md) — Runtime processes, startup/shutdown, IPC protocol
- [Week-1 Execution Checklist](week1-checklist.md) — Day-by-day checklist for first week of implementation
