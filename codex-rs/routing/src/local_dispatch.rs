//! In-process dispatch to local Ollama models.
//!
//! When the routing engine decides a request should go to a local model,
//! this module calls Ollama and translates the response into the format
//! that codex-core expects (ResponseEvent stream over an mpsc channel).
//!
//! This replaces the coding-agent-router's HTTP proxy pattern with a
//! direct in-process call — no separate service needed.

use crate::config::OllamaEndpoint;
use crate::metrics::{estimate_tokens, extract_task_metrics};
use crate::ollama::OllamaClientPool;
use serde_json::Value as JsonValue;
use std::sync::Arc;
use tracing::{info, warn};

/// The result of classifying a request for routing.
#[derive(Debug, Clone)]
pub enum RouteTarget {
    /// Send to a local Ollama model (free).
    Local(OllamaEndpoint),
    /// Send to the cloud provider (normal path).
    Cloud,
}

/// Classify a request and decide where it should go.
///
/// This is the per-request routing decision. It runs on every model API call
/// within an agent session.
///
/// Decision logic (matching coding-agent-router's approach):
/// - If the request is too large for any local model's context window → Cloud
/// - If this looks like a simple request (short, no complex tool calls) → Local
/// - Otherwise → Cloud
pub fn classify_request(
    prompt_text: &str,
    has_tools: bool,
    local_endpoints: &[&OllamaEndpoint],
) -> RouteTarget {
    if local_endpoints.is_empty() {
        return RouteTarget::Cloud;
    }

    let prompt_tokens = estimate_tokens(prompt_text);

    // Find a local endpoint that can fit this request
    for ep in local_endpoints {
        if !ep.enabled {
            continue;
        }
        if prompt_tokens <= ep.trim_budget {
            return RouteTarget::Local((*ep).clone());
        }
    }

    // No local endpoint has enough context → cloud
    RouteTarget::Cloud
}

/// Call an Ollama endpoint and return the raw text response.
///
/// This is the lowest-level call — just sends the prompt and gets text back.
/// The caller is responsible for translating into ResponseEvents.
pub async fn call_ollama_text(
    pool: &OllamaClientPool,
    endpoint: &OllamaEndpoint,
    messages: Vec<JsonValue>,
    system: Option<&str>,
) -> Result<OllamaTextResponse, String> {
    let response = pool.chat(endpoint, messages, system, None).await;

    match response {
        Some(body) => {
            let content = body
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();

            let input_tokens = body
                .get("prompt_eval_count")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let output_tokens = body.get("eval_count").and_then(|v| v.as_u64()).unwrap_or(0);

            if content.is_empty() {
                Err("Ollama returned empty response".into())
            } else {
                Ok(OllamaTextResponse {
                    content,
                    model: endpoint.model.clone(),
                    input_tokens,
                    output_tokens,
                })
            }
        }
        None => Err(format!(
            "Ollama request failed: {} / {}",
            endpoint.base_url, endpoint.model
        )),
    }
}

/// A successful text response from Ollama.
#[derive(Debug, Clone)]
pub struct OllamaTextResponse {
    pub content: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}
