//! Per-request quality detection for local model responses.
//!
//! Quick checks before returning a local response. If the response
//! looks like garbage, discard it and fall back to cloud.
//! This is NOT an LLM judgment — it's fast deterministic checks
//! for obvious failures that don't need LLM intelligence to detect.

/// Check if a local model response is acceptable quality.
/// Returns None if OK, or Some(reason) if the response should be discarded.
pub fn check_response_quality(response_text: &str, prompt_text: &str) -> Option<String> {
    let text = response_text.trim();

    // Empty or near-empty response
    if text.is_empty() {
        return Some("empty response".into());
    }
    if text.len() < 5 {
        return Some(format!("response too short ({} chars)", text.len()));
    }

    // Response is just the prompt echoed back
    let prompt_trimmed = prompt_text.trim();
    if !prompt_trimmed.is_empty() && text == prompt_trimmed {
        return Some("response echoes the prompt".into());
    }

    // Response is a refusal or error from the model itself
    let lower = text.to_lowercase();
    if lower.starts_with("i cannot")
        || lower.starts_with("i can't")
        || lower.starts_with("i'm unable")
        || lower.starts_with("as an ai")
        || lower.starts_with("i don't have access")
    {
        return Some("model refusal detected".into());
    }

    // Response is just a code fence with nothing inside
    if text.starts_with("```") && text.ends_with("```") {
        let inner = text
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        // Allow code fences with actual content
        if inner.is_empty() || inner.lines().all(|l| l.trim().is_empty()) {
            return Some("empty code fence".into());
        }
    }

    // Excessive repetition (model stuck in a loop). Split on a char boundary
    // near byte 100 — a raw `text[..100]` slice panics when 100 lands inside a
    // multi-byte UTF-8 sequence.
    if text.len() > 200 {
        let split = text
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= 100)
            .unwrap_or(text.len());
        let (first, rest) = text.split_at(split);
        if !first.is_empty() && rest.contains(first) {
            return Some("repetition detected".into());
        }
    }

    None
}

/// Re-prompt to inject when a local response is discarded by
/// [`check_response_quality`]. Names the specific defect so the model corrects
/// it rather than reproducing it.
pub fn quality_continuation_prompt(reason: &str) -> String {
    format!(
        "Your previous response was rejected: {reason}. \
         Do not repeat it. Produce a proper response now — either call the appropriate tool to make progress, \
         or give a substantive, complete answer to the user's request."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_good_response() {
        assert!(check_response_quality("This is a helpful answer.", "What is X?").is_none());
    }

    #[test]
    fn test_empty_response() {
        assert!(check_response_quality("", "What is X?").is_some());
        assert!(check_response_quality("   ", "What is X?").is_some());
    }

    #[test]
    fn test_too_short() {
        assert!(check_response_quality("hi", "What is X?").is_some());
    }

    #[test]
    fn test_echo() {
        assert!(check_response_quality("What is X?", "What is X?").is_some());
    }

    #[test]
    fn test_refusal() {
        assert!(check_response_quality("I cannot help with that.", "How to X?").is_some());
        assert!(
            check_response_quality("As an AI, I don't have opinions.", "What do you think?")
                .is_some()
        );
    }

    #[test]
    fn test_empty_code_fence() {
        assert!(check_response_quality("```\n\n```", "Write code").is_some());
    }

    #[test]
    fn test_code_fence_with_content() {
        assert!(check_response_quality("```python\nprint('hello')\n```", "Write code").is_none());
    }

    #[test]
    fn test_repetition() {
        let repeated = "The answer is 42. ".repeat(20);
        assert!(check_response_quality(&repeated, "What is the answer?").is_some());
    }
}
