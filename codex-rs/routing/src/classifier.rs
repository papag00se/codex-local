//! LLM-based request classifier for per-request routing.
//!
//! Uses a local classifier model (e.g., qwen3.5-9b:iq4_xs on a 1080)
//! to decide where each request should go and whether tools are needed.
//!
//! See docs/spec/design-principles.md — the LLM makes the judgment call,
//! deterministic code handles the control flow.

use crate::config::{OllamaEndpoint, RoutingConfig};
use crate::ollama::OllamaClientPool;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

const CLASSIFIER_PROMPT: &str = "\
You are a REQUEST CLASSIFIER. You do NOT execute requests. You ONLY classify them.

Routes (cheapest first):
- light_reasoner: Free. Questions, explanations, yes/no, summaries.
- light_coder: Free with tools. Single file reads, grep, small single-line edits only.
- cloud_fast: Cheap cloud. Use for: ANY test-and-fix loop involving a single test file, single-file refactors, applying known patterns, rename across one file. This is the go-to for simple coding tasks that are too complex for the free local model.
- cloud_mini: Medium cloud. Use for: multi-file edits, integration/E2E/Playwright/browser tests, investigations spanning 2+ files, dependency changes.
- cloud_reasoner: Strong cloud. Code review, architecture, cross-file security analysis, complex planning.
- cloud_coder: Strongest (conserve). ONLY for: large-scale refactors, complex multi-step debugging across many files, tasks that failed on cheaper models.

CLASSIFY this request. Return ONLY JSON: {\"route\": \"...\", \"tools_potential\": true/false, \"reason\": \"...\"}

Available tools: ";

/// The result of classifying a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyResult {
    pub route: RouteTarget,
    pub tools_potential: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteTarget {
    #[serde(rename = "light_reasoner")]
    LightReasoner,
    #[serde(rename = "light_coder")]
    LightCoder,
    #[serde(rename = "cloud_fast")]
    CloudFast,
    #[serde(rename = "cloud_mini")]
    CloudMini,
    #[serde(rename = "cloud_reasoner")]
    CloudReasoner,
    #[serde(rename = "cloud_coder")]
    CloudCoder,
}

/// Raw JSON response from the classifier LLM.
#[derive(Deserialize)]
struct ClassifierResponse {
    route: String,
    tools_potential: Option<bool>,
    reason: Option<String>,
}

/// Classify a request using the local classifier LLM.
///
/// Calls the classifier model (cheap local LLM) to decide:
/// 1. Which model tier should handle this request
/// 2. Whether the model will likely need to call tools
///
/// Falls back to CloudCoder if the classifier is unreachable or returns garbage.
pub async fn classify_request(
    prompt_text: &str,
    tool_names: &[&str],
    recent_tool_call_count: usize,
    recent_turn_count: usize,
    config: &RoutingConfig,
    pool: &OllamaClientPool,
) -> ClassifyResult {
    classify_request_with_context(
        prompt_text,
        tool_names,
        recent_tool_call_count,
        recent_turn_count,
        config,
        pool,
        "",
        "",
    )
    .await
}

/// Classify a request with additional context from routing history and codebase.
///
/// Thin wrapper over [`classify_with_endpoint`] using the configured
/// `classifier` endpoint, applying the static fallback on failure. Kept for
/// callers that don't need classifier failover (e.g. the supervisor).
pub async fn classify_request_with_context(
    prompt_text: &str,
    tool_names: &[&str],
    recent_tool_call_count: usize,
    recent_turn_count: usize,
    config: &RoutingConfig,
    pool: &OllamaClientPool,
    routing_profile: &str,
    codebase_context: &str,
) -> ClassifyResult {
    classify_with_endpoint(
        prompt_text,
        tool_names,
        recent_tool_call_count,
        recent_turn_count,
        &config.classifier,
        pool,
        routing_profile,
        codebase_context,
    )
    .await
    .unwrap_or_else(|| fallback("classifier failed"))
}

/// Classify a request using a specific classifier endpoint. Returns `None` when
/// that endpoint can't produce a usable classification (disabled, unreachable,
/// timed out, or unparseable) so the caller can fail over to the next role in
/// the `classification` chain. `Some` only on a successful parse.
pub async fn classify_with_endpoint(
    prompt_text: &str,
    tool_names: &[&str],
    recent_tool_call_count: usize,
    recent_turn_count: usize,
    classifier_ep: &OllamaEndpoint,
    pool: &OllamaClientPool,
    routing_profile: &str,
    codebase_context: &str,
) -> Option<ClassifyResult> {
    if !classifier_ep.enabled {
        return None;
    }

    // Build the classifier prompt — minimal context, fast
    let tools_str = tool_names.join(", ");
    let mut extra_context = String::new();
    if !codebase_context.is_empty() {
        extra_context.push_str(codebase_context);
        extra_context.push('\n');
    }
    if !routing_profile.is_empty() {
        extra_context.push_str(routing_profile);
        extra_context.push('\n');
    }
    let user_content = format!(
        "{CLASSIFIER_PROMPT}{tools_str}\n\
         {extra_context}\
         Recent context: {recent_tool_call_count} tool calls in last {recent_turn_count} turns\n\
         Request to classify: {prompt_text}",
    );

    // Bounded timeout for the classifier — it must stay snappy (it gates every
    // turn), but a hard 10s was too tight for a slow local server (e.g. a model
    // split onto a slow second GPU): the classifier would time out every turn
    // while the coder, with a longer timeout, succeeded. Honor the role's
    // configured `timeout_seconds`, clamped to a sane window so a misconfigured
    // value can't either fail instantly or stall the turn for minutes.
    let classify_timeout = classifier_ep.timeout_seconds.clamp(15, 60);
    let mut classifier_override = classifier_ep.clone();
    classifier_override.temperature = 0.0; // Deterministic
    classifier_override.timeout_seconds = classify_timeout;
    classifier_override.think = Some(false); // Never reason on classify
    let classify_future = pool.chat(
        &classifier_override,
        vec![serde_json::json!({"role": "user", "content": user_content})],
        None,
        Some("json"),
    );

    let response = match tokio::time::timeout(
        std::time::Duration::from_secs(classify_timeout),
        classify_future,
    )
    .await
    {
        Ok(r) => r,
        Err(_) => {
            warn!(
                timeout_s = classify_timeout,
                "Classifier timed out; advancing chain"
            );
            return None;
        }
    };

    let Some(body) = response else {
        warn!("Classifier LLM unreachable, falling back to cloud_coder");
        return None;
    };

    let content = body
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");

    // Strip <think>...</think> tags some models add, then pull out the JSON
    // object itself — newer models (e.g. Ornith) wrap it in a ```json fence,
    // which `serde_json::from_str` would otherwise reject.
    let content = strip_think_tags(content);
    let content = extract_json_object(content.trim());

    // Parse the JSON response
    let parsed: ClassifierResponse = match serde_json::from_str(content) {
        Ok(p) => p,
        Err(_) => {
            // Try to salvage: sometimes the model returns JSON but with extra fields
            // or returns a tool call instead of a classification
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(content) {
                if let Some(route) = v.get("route").and_then(|r| r.as_str()) {
                    ClassifierResponse {
                        route: route.to_string(),
                        tools_potential: v.get("tools_potential").and_then(|t| t.as_bool()),
                        reason: v.get("reason").and_then(|r| r.as_str()).map(String::from),
                    }
                } else {
                    warn!(
                        content = %&content[..content.len().min(200)],
                        "Classifier returned JSON without 'route' field, falling back"
                    );
                    return None;
                }
            } else {
                warn!(
                    content = %&content[..content.len().min(200)],
                    "Classifier returned non-JSON, falling back"
                );
                return None;
            }
        }
    };

    let route = match parsed.route.as_str() {
        "light_reasoner" => RouteTarget::LightReasoner,
        "light_coder" => RouteTarget::LightCoder,
        "cloud_fast" => RouteTarget::CloudFast,
        "cloud_mini" => RouteTarget::CloudMini,
        "cloud_reasoner" => RouteTarget::CloudReasoner,
        "cloud_coder" => RouteTarget::CloudCoder,
        other => {
            warn!(route = %other, "Classifier returned unknown route, falling back to cloud_coder");
            RouteTarget::CloudCoder
        }
    };

    let result = ClassifyResult {
        route,
        tools_potential: parsed.tools_potential.unwrap_or(true),
        reason: parsed.reason.unwrap_or_default(),
    };

    info!(
        route = ?result.route,
        tools_potential = result.tools_potential,
        reason = %result.reason,
        "Request classified"
    );

    Some(result)
}

/// Strip `<think>...</think>` blocks from model output.
pub fn strip_think_tags(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            result = format!("{}{}", &result[..start], &result[end + 8..]);
        } else {
            result = result[..start].to_string();
            break;
        }
    }
    result
}

/// Extract the outermost JSON object from a model response. Models differ in how
/// cleanly they emit JSON: some return a bare object, others wrap it in a
/// markdown code fence (```json … ```) or surround it with prose. Slicing from
/// the first `{` to the last `}` tolerates all of those without a brittle
/// fence-stripping ladder. Returns the input unchanged if no object is present.
pub fn extract_json_object(text: &str) -> &str {
    match (text.find('{'), text.rfind('}')) {
        (Some(start), Some(end)) if end > start => &text[start..=end],
        _ => text,
    }
}

fn fallback(reason: &str) -> ClassifyResult {
    ClassifyResult {
        route: RouteTarget::CloudCoder,
        tools_potential: true,
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ClientFlavor, ToolSubset};

    #[test]
    fn extract_json_object_unwraps_fences_and_prose() {
        // The exact Ornith failure: JSON wrapped in a ```json fence.
        let fenced = "```json\n{\"route\": \"light_coder\", \"tools_potential\": true}\n```";
        let obj = extract_json_object(fenced);
        let v: serde_json::Value = serde_json::from_str(obj).expect("parses after unwrap");
        assert_eq!(v["route"], "light_coder");
        // Bare JSON is unchanged; prose around the object is stripped.
        assert_eq!(extract_json_object("{\"a\":1}"), "{\"a\":1}");
        assert_eq!(
            extract_json_object("Sure, here it is: {\"route\":\"cloud_fast\"} done"),
            "{\"route\":\"cloud_fast\"}"
        );
        // No object → returned unchanged (caller still fails gracefully).
        assert_eq!(extract_json_object("no json here"), "no json here");
    }

    fn disabled_endpoint() -> OllamaEndpoint {
        OllamaEndpoint {
            base_url: "http://127.0.0.1:1".to_string(),
            model: "m".to_string(),
            trim_budget: 2048,
            temperature: 0.0,
            timeout_seconds: 1,
            enabled: false,
            think: Some(false),
            tool_subset: ToolSubset::Focused,
            flavor: ClientFlavor::OpenAICompat,
            max_tokens: None,
            output_reserve: None,
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            tool_choice: None,
        }
    }

    #[tokio::test]
    async fn classify_with_endpoint_returns_none_when_disabled() {
        // The failover contract: a disabled (or otherwise unusable) endpoint
        // must yield `None`, not a fallback route, so `classify_via_chain` can
        // advance to the next role in the classification chain instead of
        // prematurely settling on CloudCoder.
        let pool = OllamaClientPool::new();
        let result = classify_with_endpoint(
            "do something",
            &["shell"],
            0,
            0,
            &disabled_endpoint(),
            &pool,
            "",
            "",
        )
        .await;
        assert!(result.is_none());
    }
}
