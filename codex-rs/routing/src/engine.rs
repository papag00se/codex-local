//! Route selection engine — ported from coding-agent-router/app/router.py.
//!
//! Exact port of `RoutingService.route()`. Every decision path, fallback,
//! and threshold matches the Python reference.
//! See docs/spec/routing-logic-reference.md.

use crate::config::RoutingConfig;
use crate::metrics::{TaskMetrics, estimate_tokens, extract_task_metrics};
use crate::ollama::OllamaClientPool;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use tracing::{info, warn};

/// The result of a routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    pub route: String,
    pub confidence: f64,
    pub reason: String,
    pub metrics: TaskMetrics,
}

/// Router task prompt — sent as part of the digest to the router model.
const ROUTER_TASK: &str = "Choose exactly one route from available_routes for this request.\n\
     Return JSON only with keys route, confidence, reason.\n\
     route must exactly match one entry in available_routes.";

/// Router system prompt — the system message for the router model.
const ROUTER_SYSTEM: &str = "Return JSON only with keys: route, confidence, reason.\n\
     Pick exactly one route from the available routes.";

/// Deterministic fallback order — coder preferred (has tools),
/// then reasoner (local/free), then codex_cli (cloud).
fn fallback_route(available: &[String]) -> String {
    for preferred in &["local_coder", "local_reasoner", "codex_cli"] {
        if available.iter().any(|r| r == preferred) {
            return preferred.to_string();
        }
    }
    available
        .first()
        .cloned()
        .unwrap_or_else(|| "local_reasoner".to_string())
}

/// Build the routing digest — the JSON payload sent to the router model.
fn build_routing_digest(
    system: &str,
    prompt: &str,
    user_prompt: &str,
    trajectory_json: &str,
    metadata_key_count: usize,
    config: &RoutingConfig,
    eligible: &[String],
) -> JsonValue {
    let metrics = extract_task_metrics(user_prompt, trajectory_json, metadata_key_count);

    let backend_request_text = if system.is_empty() {
        prompt.to_string()
    } else {
        format!("{system}\n\n{prompt}")
    };
    let backend_request_tokens = estimate_tokens(&backend_request_text);

    // Store these in a serde_json::Value since TaskMetrics doesn't have these fields
    // (they're routing-specific additions to the digest)
    serde_json::json!({
        "task": ROUTER_TASK,
        "available_routes": eligible,
        "user_prompt": user_prompt,
        "trajectory": trajectory_json,
        "metrics": {
            "user_prompt_chars": metrics.user_prompt_chars,
            "user_prompt_lines": metrics.user_prompt_lines,
            "user_prompt_tokens": metrics.user_prompt_tokens,
            "trajectory_chars": metrics.trajectory_chars,
            "trajectory_lines": metrics.trajectory_lines,
            "trajectory_tokens": metrics.trajectory_tokens,
            "message_count": metrics.message_count,
            "user_message_count": metrics.user_message_count,
            "assistant_message_count": metrics.assistant_message_count,
            "tool_message_count": metrics.tool_message_count,
            "tool_call_count": metrics.tool_call_count,
            "command_count": metrics.command_count,
            "command_output_tokens": metrics.command_output_tokens,
            "file_reference_count": metrics.file_reference_count,
            "unique_file_reference_count": metrics.unique_file_reference_count,
            "code_block_count": metrics.code_block_count,
            "json_block_count": metrics.json_block_count,
            "diff_line_count": metrics.diff_line_count,
            "error_line_count": metrics.error_line_count,
            "stack_trace_count": metrics.stack_trace_count,
            "prior_failure_count": metrics.prior_failure_count,
            "question_count": metrics.question_count,
            "metadata_key_count": metrics.metadata_key_count,
            "backend_request_tokens": backend_request_tokens,
            "reasoner_context_limit": config.reasoner.trim_budget,
            "coder_context_limit": config.coder.trim_budget,
            "router_context_limit": config.router.trim_budget,
        }
    })
}

/// Route a request to the best available backend.
///
/// This is the exact algorithm from `RoutingService.route()` in the Python reference:
///
/// 1. If preferred_backend set → return it (confidence 1.0)
/// 2. Build available list from enabled backends
/// 3. Filter by context window
/// 4. If router digest too large → fallback
/// 5. If no eligible → fallback
/// 6. If exactly one eligible → return it
/// 7. If multiple → ask router LLM
/// 8. If LLM fails → deterministic fallback
pub async fn route_task(
    system: &str,
    prompt: &str,
    user_prompt: &str,
    trajectory_json: &str,
    metadata_key_count: usize,
    preferred_backend: Option<&str>,
    config: &RoutingConfig,
    ollama: &OllamaClientPool,
) -> RouteDecision {
    let metrics = extract_task_metrics(user_prompt, trajectory_json, metadata_key_count);

    // Step 1: Preferred backend override
    if let Some(preferred) = preferred_backend {
        return RouteDecision {
            route: preferred.to_string(),
            confidence: 1.0,
            reason: "preferred backend override".into(),
            metrics,
        };
    }

    // Step 2: Build available backends
    let mut available = Vec::new();
    if config.coder.enabled {
        available.push("local_coder".to_string());
    }
    if config.reasoner.enabled {
        available.push("local_reasoner".to_string());
    }
    if config.codex_cli_enabled {
        available.push("codex_cli".to_string());
    }

    if available.is_empty() {
        return RouteDecision {
            route: "local_reasoner".to_string(),
            confidence: 0.0,
            reason: "no configured backends".into(),
            metrics,
        };
    }

    // Step 3: Filter by context window
    let backend_text = if system.is_empty() {
        prompt.to_string()
    } else {
        format!("{system}\n\n{prompt}")
    };
    let backend_request_tokens = estimate_tokens(&backend_text);

    let mut eligible = available.clone();
    if backend_request_tokens > config.reasoner.trim_budget {
        eligible.retain(|r| r != "local_reasoner");
    }
    if backend_request_tokens > config.coder.trim_budget {
        eligible.retain(|r| r != "local_coder");
    }

    // Step 4: Build digest and check router payload size
    let digest = build_routing_digest(
        system,
        prompt,
        user_prompt,
        trajectory_json,
        metadata_key_count,
        config,
        &eligible,
    );
    let digest_str = serde_json::to_string(&digest).unwrap_or_default();
    let router_request_tokens = estimate_tokens(&digest_str);

    if router_request_tokens > config.router.trim_budget {
        let route = if available.contains(&"codex_cli".to_string()) {
            "codex_cli".to_string()
        } else {
            fallback_route(&eligible)
        };
        return RouteDecision {
            route,
            confidence: 1.0,
            reason: "router request exceeds router context window".into(),
            metrics,
        };
    }

    // Step 5: No eligible local routes
    if eligible.is_empty() {
        let route = if available.contains(&"codex_cli".to_string()) {
            "codex_cli".to_string()
        } else {
            fallback_route(&available)
        };
        return RouteDecision {
            route,
            confidence: 1.0,
            reason: "local context windows exceeded".into(),
            metrics,
        };
    }

    // Step 6: Exactly one eligible
    if eligible.len() == 1 {
        return RouteDecision {
            route: eligible[0].clone(),
            confidence: 1.0,
            reason: "only one route fits context limits".into(),
            metrics,
        };
    }

    // Step 7: Multiple eligible — ask router LLM
    info!(
        eligible = ?eligible,
        router_tokens = router_request_tokens,
        "Asking router model to select route"
    );

    let router_ep = crate::config::OllamaEndpoint {
        base_url: config.router.base_url.clone(),
        model: config.router.model.clone(),
        trim_budget: config.router.trim_budget,
        temperature: config.router.temperature,
        timeout_seconds: config.router.timeout_seconds,
        enabled: true,
        think: false,
        tool_subset: crate::config::ToolSubset::Focused,
        flavor: crate::config::ClientFlavor::Ollama,
        max_tokens: None,
        top_p: None,
        top_k: None,
        repeat_penalty: None,
        tool_choice: None,
    };
    let response = ollama
        .chat(
            &router_ep,
            vec![serde_json::json!({"role": "user", "content": digest_str})],
            Some(ROUTER_SYSTEM),
            Some("json"),
        )
        .await;

    // Step 8: Parse response or fallback
    let Some(response) = response else {
        let route = fallback_route(&eligible);
        return RouteDecision {
            route,
            confidence: 0.0,
            reason: "router model unreachable, using fallback".into(),
            metrics,
        };
    };

    let content = response
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");

    match serde_json::from_str::<JsonValue>(content) {
        Ok(parsed) => {
            let route = parsed
                .get("route")
                .and_then(|r| r.as_str())
                .unwrap_or("codex_cli")
                .to_string();
            let confidence = parsed
                .get("confidence")
                .and_then(|c| c.as_f64())
                .unwrap_or(0.0);
            let reason = parsed
                .get("reason")
                .and_then(|r| r.as_str())
                .unwrap_or("")
                .to_string();

            if eligible.iter().any(|r| r == &route) {
                RouteDecision {
                    route,
                    confidence,
                    reason,
                    metrics,
                }
            } else {
                warn!(
                    route = %route,
                    eligible = ?eligible,
                    "Router model returned invalid route, using fallback"
                );
                RouteDecision {
                    route: fallback_route(&eligible),
                    confidence: 0.0,
                    reason: format!("router returned invalid route '{route}', using fallback"),
                    metrics,
                }
            }
        }
        Err(_) => {
            warn!("Router model returned non-JSON, using fallback");
            RouteDecision {
                route: fallback_route(&eligible),
                confidence: 0.0,
                reason: "router JSON parse fallback".into(),
                metrics,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RoutingConfig;

    fn test_config() -> RoutingConfig {
        RoutingConfig::default()
    }

    #[test]
    fn test_preferred_backend_override() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = test_config();
        let ollama = OllamaClientPool::new();

        let decision = rt.block_on(route_task(
            "",
            "Fix a bug",
            "Fix a bug",
            "",
            0,
            Some("codex_cli"),
            &config,
            &ollama,
        ));

        assert_eq!(decision.route, "codex_cli");
        assert_eq!(decision.confidence, 1.0);
        assert_eq!(decision.reason, "preferred backend override");
    }

    #[test]
    fn test_no_backends_available() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut config = test_config();
        config.coder.enabled = false;
        config.reasoner.enabled = false;
        config.codex_cli_enabled = false;
        let ollama = OllamaClientPool::new();

        let decision = rt.block_on(route_task(
            "",
            "Fix a bug",
            "Fix a bug",
            "",
            0,
            None,
            &config,
            &ollama,
        ));

        assert_eq!(decision.route, "local_reasoner");
        assert_eq!(decision.confidence, 0.0);
    }

    #[test]
    fn test_single_eligible_route() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut config = test_config();
        config.reasoner.enabled = false;
        config.codex_cli_enabled = false;
        // Only coder is enabled
        let ollama = OllamaClientPool::new();

        let decision = rt.block_on(route_task(
            "",
            "Fix a bug",
            "Fix a bug",
            "",
            0,
            None,
            &config,
            &ollama,
        ));

        assert_eq!(decision.route, "local_coder");
        assert_eq!(decision.confidence, 1.0);
        assert_eq!(decision.reason, "only one route fits context limits");
    }

    #[test]
    fn test_context_window_filtering() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut config = test_config();
        config.coder.trim_budget = 10; // Very small — will be filtered
        config.reasoner.trim_budget = 10; // Very small — will be filtered
        config.codex_cli_enabled = true;
        let ollama = OllamaClientPool::new();

        let long_prompt = "x".repeat(100); // ~25 tokens, exceeds ctx of 10
        let decision = rt.block_on(route_task(
            "",
            &long_prompt,
            &long_prompt,
            "",
            0,
            None,
            &config,
            &ollama,
        ));

        // Both local routes filtered, codex_cli is the only option
        assert_eq!(decision.route, "codex_cli");
    }

    #[test]
    fn test_fallback_order() {
        assert_eq!(
            fallback_route(&[
                "local_coder".into(),
                "local_reasoner".into(),
                "codex_cli".into()
            ]),
            "local_coder"
        );
        assert_eq!(
            fallback_route(&["local_reasoner".into(), "codex_cli".into()]),
            "local_reasoner"
        );
        assert_eq!(fallback_route(&["codex_cli".into()]), "codex_cli");
        assert_eq!(fallback_route(&[]), "local_reasoner");
    }
}
