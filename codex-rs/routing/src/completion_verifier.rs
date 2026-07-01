//! Verify whether a local model's no-tool-call response actually completes
//! the user's task, or whether it's an "announcement-then-bail" — the model
//! says "now I'll do X" and then ends the turn without doing X.
//!
//! Asks the classifier endpoint (small, fast, already-warm) for a binary
//! judgment. The classifier is well-suited for this kind of structured
//! short-output decision.

use crate::OllamaClientPool;
use crate::config::OllamaEndpoint;
use serde::Deserialize;
use tracing::warn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionVerdict {
    /// The agent's response is a legitimate completion (task done, deferring
    /// appropriately to the user, or asking a clarifying question).
    Complete,
    /// The agent announced an action but didn't take it, or otherwise
    /// terminated the turn prematurely.
    Bail,
    /// Verifier was unreachable or returned unparseable output. Treat as
    /// Complete (don't intervene if we can't judge).
    Unclear,
}

const VERIFIER_SYSTEM_PROMPT: &str = "\
You audit AI coding agents to detect when they end a turn prematurely.\n\
Return JSON only with no markdown fencing.\n\
\n\
The agent COMPLETED the task if any of the following are true:\n\
- The agent describes a concrete result it produced or tested (e.g. \"created file X with Y content\")\n\
- The agent tells the user the task is done and explains the outcome\n\
- The agent asks the user a clarifying question that genuinely blocks progress\n\
- The agent reports an error or limitation that prevents proceeding\n\
\n\
The agent BAILED if any of the following are true:\n\
- The agent says it WILL do something but the message ends without showing the action was performed\n\
- The agent describes its plan/intent but no tool was actually invoked to execute it\n\
- The agent says \"now I'll X\" or \"let me X\" or \"next I'll X\" and stops\n\
- The agent restates findings but doesn't apply them when applying them was the obvious next step\n\
\n\
**Code blocks are never actions.** A ```...``` fence containing source code is a *suggestion* — it only counts as completed work if it was passed to a tool like `write_file` or written to disk via `shell`. An agent that pastes code without a tool call BAILED, regardless of how confident the prose is.\n\
\n\
**The closing matters most.** Read the final sentence or paragraph carefully. If it contains \"I will X\", \"I'll X\", \"now I'll X\", \"next I will X\", \"let me X\", \"I'm going to X\", or any other future-tense first-person action language, and no tool call was produced, it is BAIL — even if earlier paragraphs describe findings or results. A correctly-completed response ends with a statement of what *was* done, not what *will* be done. A long message that opens with findings and ends with announced intent is still a bail — the opening findings don't launder the closing bail.";

#[derive(Deserialize)]
struct VerifierResponse {
    verdict: String,
    #[serde(default)]
    #[allow(dead_code)]
    reason: String,
}

/// Ask the classifier whether the agent's final response is a real completion.
///
/// `user_message` is the most recent user request driving the current turn.
/// `agent_message` is the agent's final text output for this turn.
pub async fn verify_completion(
    user_message: &str,
    agent_message: &str,
    classifier: &OllamaEndpoint,
    pool: &OllamaClientPool,
) -> CompletionVerdict {
    if agent_message.trim().is_empty() {
        // Empty messages can't be judged; let upstream handle.
        return CompletionVerdict::Unclear;
    }

    let user_payload = serde_json::json!({
        "user_request": user_message,
        "agent_final_message": agent_message,
        "task": "Did the agent COMPLETE the task or BAIL? Reply with JSON only.",
        "schema": {"verdict": "complete | bail", "reason": "<one short sentence>"},
    });
    let user_payload_str = match serde_json::to_string(&user_payload) {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "failed to serialize completion verifier payload");
            return CompletionVerdict::Unclear;
        }
    };

    let mut verify_ep = classifier.clone();
    verify_ep.temperature = 0.0;
    verify_ep.think = false; // Snap-judgment; reasoning tokens only add latency
    let response = pool
        .chat(
            &verify_ep,
            vec![serde_json::json!({"role": "user", "content": user_payload_str})],
            Some(VERIFIER_SYSTEM_PROMPT),
            Some("json"),
        )
        .await;

    let Some(body) = response else {
        warn!("Completion verifier classifier unreachable");
        return CompletionVerdict::Unclear;
    };

    let content = body
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let stripped = crate::classifier::strip_think_tags(content);
    // Pull out the JSON object so a ```json fence (newer models like Ornith wrap
    // their output) doesn't make a real verdict look unparseable — which would
    // default to Complete and let a dangling-intent turn finish.
    let json = crate::classifier::extract_json_object(stripped.trim());

    let parsed: Result<VerifierResponse, _> = serde_json::from_str(json);
    match parsed {
        Ok(resp) => match resp.verdict.to_lowercase().as_str() {
            "complete" => CompletionVerdict::Complete,
            "bail" => CompletionVerdict::Bail,
            other => {
                warn!(verdict = %other, "verifier returned unknown verdict");
                CompletionVerdict::Unclear
            }
        },
        Err(e) => {
            warn!(error = %e, content = %&stripped[..stripped.len().min(200)], "verifier returned non-JSON");
            CompletionVerdict::Unclear
        }
    }
}

/// Build the continuation prompt to inject when verification flags a bail.
/// Appended as a `user`-role message before re-calling the local model.
///
/// `attempt` is how many times we've *already* re-prompted in this turn (0 on
/// the first). The first nudge is a plain reminder; once the model has ignored
/// it and kept emitting prose, the nudge escalates to a hard tool-call-only
/// constraint — the failure mode is a model that explains the fix repeatedly
/// without ever emitting the patch.
pub fn continuation_prompt(prior_response: &str, attempt: usize) -> String {
    let prior = prior_response.chars().take(400).collect::<String>();
    if attempt == 0 {
        format!(
            "[VERIFIER] You called NO tools last turn, so nothing actually happened — no file was written and no command ran. Your previous text was: \"{prior}\"\n\n\
             If that text contained code or file contents, it was NOT saved to disk. You MUST now re-issue it as a tool call: `write_file` to create a new file, or `edit_file`/`apply_patch` to change an existing one. Do NOT paste the code as text again — pasted code is discarded.\n\
             If the task is genuinely already done, restate the concrete result (which file you wrote, which command you ran, which output you got).\n\
             If you cannot proceed, explain WHY and what you need.\n\n\
             Act via a tool call now — do not just describe what you would do."
        )
    } else {
        // The model has now described the action `attempt`+ times without doing
        // it. Stop accepting prose: constrain the entire next message to one
        // tool call. Repetition of the prior text is what we must break.
        format!(
            "[VERIFIER — STOP EXPLAINING] You have now described what you would do {n} times without doing it. Nothing has changed on disk and the task is still NOT done.\n\n\
             Your ENTIRE next message must be a SINGLE tool call and nothing else — no prose, no analysis, no restating the plan or the problem. \
             To change a file use `apply_patch` or `edit_file`; to create one use `write_file`; to run/verify use `exec_command`. \
             Pick the one tool call that moves the task forward and emit ONLY that. Do not explain it first.\n\n\
             (Your last text, which did nothing, was: \"{prior}\")",
            n = attempt + 1,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_agent_message_returns_unclear() {
        // No live test of the network path; just verify the empty-input branch.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let endpoint = OllamaEndpoint {
            base_url: "http://127.0.0.1:9999".into(),
            model: "x".into(),
            trim_budget: 4096,
            temperature: 0.0,
            timeout_seconds: 5,
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
        let pool = OllamaClientPool::new();
        let verdict = rt.block_on(verify_completion("hi", "", &endpoint, &pool));
        assert_eq!(verdict, CompletionVerdict::Unclear);
    }

    #[test]
    fn continuation_prompt_includes_prior_response_excerpt() {
        let prompt = continuation_prompt("Now I'll create the file:", 0);
        assert!(prompt.contains("Now I'll create the file:"));
        assert!(prompt.contains("VERIFIER"));
        // Must direct the model to actually call a write tool, not re-narrate.
        assert!(prompt.contains("write_file"));
        assert!(prompt.contains("tool call"));
    }

    #[test]
    fn continuation_prompt_escalates_after_first_ignore() {
        // First nudge is the gentle reminder; later ones hard-constrain to a
        // single tool call so a model that keeps explaining gets cut off.
        let first = continuation_prompt("Let me fix this:", 0);
        let escalated = continuation_prompt("Let me fix this:", 2);
        assert!(!first.contains("STOP EXPLAINING"));
        assert!(escalated.contains("STOP EXPLAINING"));
        assert!(escalated.contains("ENTIRE next message must be a SINGLE tool call"));
        // It tells the model how many times it has now stalled (2 prior + this).
        assert!(escalated.contains('3'));
    }
}
