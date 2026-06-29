//! Failover executor — handles request failures and walks failover chains.
//!
//! Classifies failures into types (F1-F8), applies the appropriate strategy
//! (retry-same, walk-chain, hard-fail), and returns the next model to try.
//!
//! Deterministic control flow with no LLM involvement.
//! See docs/spec/design-principles.md.

use crate::project_config::{FailoverBehavior, FailoverChains, ModelRole, ProjectConfig};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, warn};

/// Why a request failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureType {
    /// F1: Provider returned 429. May have retry-after header.
    RateLimit,
    /// F2: Quota/budget exhausted. Waiting won't help.
    QuotaExhausted,
    /// F3: Model temporarily unavailable (503/502).
    ModelUnavailable,
    /// F4: Model not found (404). Config error.
    ModelNotFound,
    /// F5: Auth failure (401/403). Hard-fail.
    AuthFailure,
    /// F6: Request timed out.
    Timeout,
    /// F7: Response failed quality check.
    QualityFailure,
    /// F8: Request too large for model's context window.
    ContextOverflow,
    /// F9: A role in the failover chain couldn't be resolved — e.g. a cloud role
    /// skipped under `local_only`, or a chain referencing an undefined/disabled
    /// role. Benign: walk the chain. This is NOT a server "model not found" (404)
    /// and must not be reported as a config error about a model name.
    RoleUnresolvable,
}

/// What the failover executor decided to do.
#[derive(Debug, Clone)]
pub enum FailoverAction {
    /// Retry the same model after waiting.
    RetrySame { wait: Duration, attempt: u32 },
    /// Try the next model in the failover chain.
    NextInChain { model_role: String, reason: String },
    /// Hard failure — don't retry.
    HardFail { reason: String },
    /// Chain exhausted — all models tried and failed.
    ChainExhausted { chain_name: String },
}

/// Classify an HTTP status code + error info into a FailureType.
pub fn classify_failure(
    status_code: Option<u16>,
    error_message: &str,
    is_quality_failure: bool,
    is_context_overflow: bool,
) -> FailureType {
    if is_quality_failure {
        return FailureType::QualityFailure;
    }
    if is_context_overflow {
        return FailureType::ContextOverflow;
    }

    match status_code {
        Some(429) => {
            let lower = error_message.to_lowercase();
            if lower.contains("quota")
                || lower.contains("insufficient")
                || lower.contains("usage limit")
            {
                FailureType::QuotaExhausted
            } else {
                FailureType::RateLimit
            }
        }
        Some(401) | Some(403) => FailureType::AuthFailure,
        Some(404) => FailureType::ModelNotFound,
        Some(502) | Some(503) | Some(504) => FailureType::ModelUnavailable,
        Some(408) => FailureType::Timeout,
        None => {
            let lower = error_message.to_lowercase();
            if lower.contains("timeout") || lower.contains("timed out") {
                FailureType::Timeout
            } else if lower.contains("connection refused") || lower.contains("unreachable") {
                FailureType::ModelUnavailable
            } else {
                FailureType::ModelUnavailable // Default for unknown errors
            }
        }
        _ => FailureType::ModelUnavailable,
    }
}

/// Determine what to do after a failure.
///
/// `current_model_role`: the role name of the model that failed (e.g., "light_reasoner")
/// `chain`: the failover chain being used (e.g., the "reasoning" chain)
/// `attempt`: how many times we've already retried the current model (0 = first failure)
/// `retry_after_ms`: if the provider gave a retry-after hint
pub fn decide_action(
    failure: FailureType,
    current_model_role: &str,
    chain_name: &str,
    chain: &[String],
    attempt: u32,
    retry_after_ms: Option<u64>,
    behavior: &FailoverBehavior,
) -> FailoverAction {
    match failure {
        // F5: Auth failure — never retry
        FailureType::AuthFailure => {
            warn!(model = current_model_role, "Auth failure — hard-fail");
            FailoverAction::HardFail {
                reason: format!("Authentication failed for {current_model_role}"),
            }
        }

        // F1: Rate limit — retry with backoff, then walk chain
        FailureType::RateLimit => {
            if attempt < behavior.retry_same_attempts {
                let wait = retry_after_ms
                    .unwrap_or(behavior.rate_limit_default_wait_ms)
                    .min(behavior.rate_limit_max_wait_ms);
                info!(
                    model = current_model_role,
                    attempt = attempt + 1,
                    wait_ms = wait,
                    "Rate limited — retrying same model"
                );
                FailoverAction::RetrySame {
                    wait: Duration::from_millis(wait),
                    attempt: attempt + 1,
                }
            } else {
                walk_chain(
                    current_model_role,
                    chain_name,
                    chain,
                    "rate limit retries exhausted",
                )
            }
        }

        // F6: Timeout — retry once, then walk chain
        FailureType::Timeout => {
            if attempt < 1 {
                info!(model = current_model_role, "Timeout — retrying once");
                FailoverAction::RetrySame {
                    wait: Duration::from_millis(behavior.retry_same_backoff_ms),
                    attempt: attempt + 1,
                }
            } else {
                walk_chain(current_model_role, chain_name, chain, "timeout after retry")
            }
        }

        // F2, F3, F4, F7, F8: Walk chain immediately
        FailureType::QuotaExhausted => {
            walk_chain(current_model_role, chain_name, chain, "quota exhausted")
        }
        FailureType::ModelUnavailable => {
            walk_chain(current_model_role, chain_name, chain, "model unavailable")
        }
        FailureType::ModelNotFound => {
            warn!(model = current_model_role, "Model not found — check config");
            walk_chain(
                current_model_role,
                chain_name,
                chain,
                "model not found (config error?)",
            )
        }
        FailureType::RoleUnresolvable => {
            // Not a model/name error — a role just didn't resolve (cloud skipped
            // under local_only, or an undefined/disabled role in the chain).
            info!(
                role = current_model_role,
                "Role not resolvable (cloud disabled by local_only, or unconfigured) — walking chain"
            );
            walk_chain(
                current_model_role,
                chain_name,
                chain,
                "role not resolvable (cloud disabled or unconfigured)",
            )
        }
        FailureType::QualityFailure => walk_chain(
            current_model_role,
            chain_name,
            chain,
            "quality check failed",
        ),
        FailureType::ContextOverflow => walk_chain(
            current_model_role,
            chain_name,
            chain,
            "context overflow — need larger model",
        ),
    }
}

/// Find the next model in the chain after the current one.
fn walk_chain(
    current_model_role: &str,
    chain_name: &str,
    chain: &[String],
    reason: &str,
) -> FailoverAction {
    // Find current position in chain
    let current_pos = chain.iter().position(|r| r == current_model_role);

    let next = match current_pos {
        Some(pos) => {
            // Get the next one in the chain
            chain.get(pos + 1)
        }
        None => {
            // Current model isn't in the chain — try the first one
            chain.first()
        }
    };

    match next {
        Some(next_role) => {
            info!(
                from = current_model_role,
                to = next_role.as_str(),
                reason = reason,
                chain = chain_name,
                "Failing over to next model in chain"
            );
            FailoverAction::NextInChain {
                model_role: next_role.clone(),
                reason: reason.into(),
            }
        }
        None => {
            warn!(
                chain = chain_name,
                reason = reason,
                "Failover chain exhausted — all models tried"
            );
            FailoverAction::ChainExhausted {
                chain_name: chain_name.into(),
            }
        }
    }
}

/// Convenience: get the failover chain for a classifier route.
pub fn chain_for_route(route: &str, chains: &FailoverChains) -> Vec<String> {
    match route {
        "light_reasoner" | "LightReasoner" => chains.reasoning.clone(),
        "light_coder" | "LightCoder" => chains.coding.clone(),
        "cloud_fast" | "CloudFast" => chains.coding.clone(),
        "cloud_mini" | "CloudMini" => chains.coding.clone(),
        "cloud_reasoner" | "CloudReasoner" => chains.reasoning.clone(),
        "cloud_coder" | "CloudCoder" => chains.coding.clone(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_behavior() -> FailoverBehavior {
        FailoverBehavior::default()
    }

    fn test_chain() -> Vec<String> {
        vec![
            "light_reasoner".into(),
            "light_reasoner_backup".into(),
            "cloud_reasoner".into(),
            "cloud_coder".into(),
        ]
    }

    // --- Failure classification ---

    #[test]
    fn test_classify_429_as_rate_limit() {
        assert_eq!(
            classify_failure(Some(429), "too many requests", false, false),
            FailureType::RateLimit,
        );
    }

    #[test]
    fn test_classify_429_quota() {
        assert_eq!(
            classify_failure(Some(429), "quota exceeded", false, false),
            FailureType::QuotaExhausted,
        );
    }

    #[test]
    fn test_classify_401_auth() {
        assert_eq!(
            classify_failure(Some(401), "", false, false),
            FailureType::AuthFailure,
        );
    }

    #[test]
    fn test_classify_503_unavailable() {
        assert_eq!(
            classify_failure(Some(503), "", false, false),
            FailureType::ModelUnavailable,
        );
    }

    #[test]
    fn test_classify_quality() {
        assert_eq!(
            classify_failure(None, "", true, false),
            FailureType::QualityFailure,
        );
    }

    #[test]
    fn test_classify_context_overflow() {
        assert_eq!(
            classify_failure(None, "", false, true),
            FailureType::ContextOverflow,
        );
    }

    #[test]
    fn test_classify_timeout_from_message() {
        assert_eq!(
            classify_failure(None, "connection timed out", false, false),
            FailureType::Timeout,
        );
    }

    // --- Failover actions ---

    #[test]
    fn test_auth_hard_fails() {
        let action = decide_action(
            FailureType::AuthFailure,
            "cloud_coder",
            "reasoning",
            &test_chain(),
            0,
            None,
            &default_behavior(),
        );
        assert!(matches!(action, FailoverAction::HardFail { .. }));
    }

    #[test]
    fn test_rate_limit_retries_then_walks() {
        let chain = test_chain();
        let b = default_behavior();

        // First failure: retry same
        let a1 = decide_action(
            FailureType::RateLimit,
            "light_reasoner",
            "reasoning",
            &chain,
            0,
            None,
            &b,
        );
        assert!(matches!(a1, FailoverAction::RetrySame { .. }));

        // Second failure: retry same (attempt < 2)
        let a2 = decide_action(
            FailureType::RateLimit,
            "light_reasoner",
            "reasoning",
            &chain,
            1,
            None,
            &b,
        );
        assert!(matches!(a2, FailoverAction::RetrySame { .. }));

        // Third failure: walk chain (attempt >= 2)
        let a3 = decide_action(
            FailureType::RateLimit,
            "light_reasoner",
            "reasoning",
            &chain,
            2,
            None,
            &b,
        );
        match a3 {
            FailoverAction::NextInChain { model_role, .. } => {
                assert_eq!(model_role, "light_reasoner_backup");
            }
            _ => panic!("Expected NextInChain"),
        }
    }

    #[test]
    fn test_timeout_retries_once() {
        let chain = test_chain();
        let b = default_behavior();

        let a1 = decide_action(
            FailureType::Timeout,
            "cloud_reasoner",
            "reasoning",
            &chain,
            0,
            None,
            &b,
        );
        assert!(matches!(a1, FailoverAction::RetrySame { .. }));

        let a2 = decide_action(
            FailureType::Timeout,
            "cloud_reasoner",
            "reasoning",
            &chain,
            1,
            None,
            &b,
        );
        match a2 {
            FailoverAction::NextInChain { model_role, .. } => {
                assert_eq!(model_role, "cloud_coder");
            }
            _ => panic!("Expected NextInChain"),
        }
    }

    #[test]
    fn test_quality_walks_immediately() {
        let chain = test_chain();
        let b = default_behavior();

        let action = decide_action(
            FailureType::QualityFailure,
            "light_reasoner",
            "reasoning",
            &chain,
            0,
            None,
            &b,
        );
        match action {
            FailoverAction::NextInChain { model_role, .. } => {
                assert_eq!(model_role, "light_reasoner_backup");
            }
            _ => panic!("Expected NextInChain"),
        }
    }

    #[test]
    fn test_chain_exhausted() {
        let chain = test_chain();
        let b = default_behavior();

        // Last model in chain fails
        let action = decide_action(
            FailureType::ModelUnavailable,
            "cloud_coder",
            "reasoning",
            &chain,
            0,
            None,
            &b,
        );
        assert!(matches!(action, FailoverAction::ChainExhausted { .. }));
    }

    #[test]
    fn test_rate_limit_respects_retry_after() {
        let chain = test_chain();
        let b = default_behavior();

        let action = decide_action(
            FailureType::RateLimit,
            "light_reasoner",
            "reasoning",
            &chain,
            0,
            Some(2000),
            &b,
        );
        match action {
            FailoverAction::RetrySame { wait, .. } => {
                assert_eq!(wait, Duration::from_millis(2000));
            }
            _ => panic!("Expected RetrySame"),
        }
    }

    #[test]
    fn test_rate_limit_caps_wait() {
        let chain = test_chain();
        let b = default_behavior();

        // retry-after says 60s, but max is 30s
        let action = decide_action(
            FailureType::RateLimit,
            "light_reasoner",
            "reasoning",
            &chain,
            0,
            Some(60000),
            &b,
        );
        match action {
            FailoverAction::RetrySame { wait, .. } => {
                assert_eq!(wait, Duration::from_millis(30000));
            }
            _ => panic!("Expected RetrySame"),
        }
    }

    #[test]
    fn test_model_not_in_chain_starts_from_beginning() {
        let chain = test_chain();
        let b = default_behavior();

        let action = decide_action(
            FailureType::ModelUnavailable,
            "unknown_model",
            "reasoning",
            &chain,
            0,
            None,
            &b,
        );
        match action {
            FailoverAction::NextInChain { model_role, .. } => {
                assert_eq!(model_role, "light_reasoner"); // First in chain
            }
            _ => panic!("Expected NextInChain"),
        }
    }
}
