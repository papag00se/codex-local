//! codex-routing: Task routing engine for multi-agent orchestration.
//!
//! Ported from coding-agent-router (Python). Reference docs:
//! - docs/spec/routing-logic-reference.md
//! - docs/spec/design-principles.md
//!
//! This crate provides:
//! - Task metrics extraction (27 regex-based features)
//! - Route selection algorithm (context filtering → single-eligible fast path → LLM selection → fallback)
//! - Ollama HTTP client with per-endpoint serialization
//! - Routing configuration

pub mod budget_pressure;
pub mod classifier;
pub mod classify_cache;
pub mod claude_cli;
pub mod codebase_context;
pub mod compaction;
pub mod completion_verifier;
pub mod config;
pub mod content_reduce;
pub mod context_strip;
pub mod cost_analytics;
pub mod curl_ua;
pub mod engine;
pub mod failover;
pub mod feedback;
pub mod local_dispatch;
pub mod local_web_search;
pub mod loop_detector;
pub mod metrics;
pub mod ollama;
pub mod project_config;
pub mod prompt_adapt;
pub mod prompt_local;
pub mod quality;
pub mod rumination_detector;
pub mod session_memory;
pub mod tool_aliases;
pub mod tool_format;
pub mod tool_recovery;
pub mod trim;
pub mod usage;
pub mod web_fetch;

pub use classifier::{ClassifyResult, RouteTarget, classify_request};
pub use config::RoutingConfig;
pub use engine::{RouteDecision, route_task};
pub use local_dispatch::{OllamaTextResponse, call_ollama_text};
pub use metrics::{TaskMetrics, estimate_tokens, extract_task_metrics};
pub use ollama::OllamaClientPool;
pub use tool_recovery::{
    RecoveredMessage, ToolCall, recover_tool_calls, recover_tool_calls_streaming,
};
