//! Continuation prompt for a local model that ended a turn with NO tool call.
//!
//! Injected as a `user`-role message before re-calling the model when it produced
//! prose but changed nothing (the coder "did nothing" case). It tells the model
//! that pasted code/plans were discarded and that it must act via a tool call.
//!
//! This module used to also host a small-model **completion verifier** that judged
//! done-vs-bail from the text. That verifier was removed: it false-negatived on
//! finished tasks — "let me verify" read as a bail — and trapped a done coder in a
//! done→"you did nothing, act"→`ls` loop (the model even personified the `[VERIFIER]`
//! tag below and reasoned "the verifier is correct"). Completion is now decided by
//! ground truth (files actually changed) plus the probe gate and the reasoner
//! completion critic, which judge the real work rather than the phrasing.

/// Build the continuation prompt to inject when a coder ended a turn with no tool
/// call and no file change. Appended as a `user`-role message before re-calling.
///
/// `attempt` is how many times we've *already* re-prompted in this turn (0 on the
/// first). The first nudge is a plain reminder; once the model has ignored it and
/// kept emitting prose, the nudge escalates to a hard tool-call-only constraint —
/// the failure mode is a model that explains the fix repeatedly without ever
/// emitting the patch.
pub fn continuation_prompt(prior_response: &str, attempt: usize) -> String {
    let prior = prior_response.chars().take(400).collect::<String>();
    if attempt == 0 {
        format!(
            "[NO ACTION TAKEN] You called NO tools last turn, so nothing actually happened — no file was written and no command ran. Your previous text was: \"{prior}\"\n\n\
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
            "[NO ACTION — STOP EXPLAINING] You have now described what you would do {n} times without doing it. Nothing has changed on disk and the task is still NOT done.\n\n\
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
    fn continuation_prompt_includes_prior_response_excerpt() {
        let prompt = continuation_prompt("Now I'll create the file:", 0);
        assert!(prompt.contains("Now I'll create the file:"));
        assert!(prompt.contains("NO ACTION TAKEN"));
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
