//! Compaction pipeline — transcript to a model-written handoff summary.
//!
//! Flow: strip boilerplate -> normalize -> split off the recent raw tail ->
//! chunk the rest -> summarize EACH chunk (free-form, model) -> one final
//! unifying pass -> assemble [warning + summary + verbatim recent tail].
//!
//! The summary is the model's own prose (not deterministic state extraction) —
//! validated to produce an accurate, resumable handoff where the deterministic
//! path produced "No compactable content." The recent tail is kept VERBATIM (never
//! summarized), so the freshest exact values survive intact, and the handoff is
//! labelled post-compaction so a resuming model treats summarized opaque strings
//! (addresses, hashes) with suspicion.

use super::chunking::{chunk_items, split_recent_raw};
use super::extract::{summarize_chunk, summarize_final};
use super::normalize::normalize_transcript;
use crate::config::OllamaEndpoint;
use crate::ollama::OllamaClientPool;
use tracing::{info, warn};

/// Configuration for the compaction pipeline.
pub struct CompactionConfig {
    pub target_chunk_tokens: usize,
    pub max_chunk_tokens: usize,
    pub overlap_tokens: usize,
    pub keep_raw_tokens: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            target_chunk_tokens: 10_000,
            max_chunk_tokens: 10_000,
            overlap_tokens: 1_500,
            keep_raw_tokens: 8_000,
        }
    }
}

/// Run the full compaction pipeline. Returns a rendered handoff summary suitable
/// for injection as the model's context.
pub async fn compact_transcript(
    items: &[serde_json::Value],
    current_request: &str,
    pool: &OllamaClientPool,
    endpoint: &OllamaEndpoint,
    config: &CompactionConfig,
) -> Result<String, String> {
    // Strip the Codex developer boilerplate AND the harness compaction directive
    // before anything else — see `is_boilerplate` / `is_compaction_directive`.
    let items: Vec<serde_json::Value> = items
        .iter()
        .filter(|m| !is_boilerplate(m) && !is_compaction_directive(m))
        .cloned()
        .collect();

    info!(items = items.len(), "Starting compaction pipeline");

    let normalized = normalize_transcript(&items, config.target_chunk_tokens);
    let (compactable, recent_raw) =
        split_recent_raw(&normalized.compactable_items, config.keep_raw_tokens);
    let chunks = chunk_items(
        &compactable,
        config.target_chunk_tokens,
        config.max_chunk_tokens,
        config.overlap_tokens,
    );
    info!(
        chunks = chunks.len(),
        recent_raw = recent_raw.len(),
        "Compaction: chunked transcript"
    );

    // Summarize each chunk (free-form). A chunk that fails/times out is skipped —
    // the recent tail is still preserved below, so we never lose everything.
    let mut summaries: Vec<String> = Vec::new();
    for chunk in &chunks {
        match summarize_chunk(chunk, pool, endpoint).await {
            Ok(s) if !s.trim().is_empty() => summaries.push(s),
            Ok(_) => {}
            Err(e) => warn!(chunk_id = chunk.chunk_id, error = %e, "chunk summary failed — skipping"),
        }
    }

    // Merge the per-chunk summaries into ONE handoff. 0 -> nothing; 1 -> use it as
    // is; 2+ -> a final unifying pass.
    let summary_body = match summaries.len() {
        0 => String::new(),
        1 => summaries.into_iter().next().unwrap_or_default(),
        _ => {
            let combined = summaries
                .iter()
                .enumerate()
                .map(|(i, s)| format!("[Portion {}]\n{s}", i + 1))
                .collect::<Vec<_>>()
                .join("\n\n");
            summarize_final(&combined, pool, endpoint)
                .await
                .unwrap_or(combined)
        }
    };

    // Preserve the recent tail VERBATIM (exact — never summarized).
    let all_recent: Vec<serde_json::Value> = [
        recent_raw,
        normalized.precompacted_items,
        normalized.preserved_tail,
    ]
    .concat();
    let recent_text = render_recent_turns(&all_recent);

    let out = assemble_handoff(&summary_body, &recent_text, current_request);
    info!(summary_len = out.len(), "Compaction complete");
    Ok(out)
}

/// Assemble the final handoff text: the post-compaction warning + the model
/// summary, then the verbatim recent tail, then the current request.
fn assemble_handoff(summary: &str, recent_text: &str, current_request: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !summary.trim().is_empty() {
        parts.push(format!(
            "[COMPACTED SUMMARY of the work so far. NOTE: this is a post-compaction summary — opaque \
             high-entropy strings (addresses, hashes, IDs, tokens) may have been corrupted when \
             summarized; re-verify them against the source before relying on them.]\n\n{summary}"
        ));
    }
    if !recent_text.trim().is_empty() {
        parts.push(format!("[RECENT TURNS — verbatim and exact]\n\n{recent_text}"));
    }
    if !current_request.trim().is_empty() {
        parts.push(format!("# Current Request\n{current_request}"));
    }
    let joined = parts.join("\n\n");
    if joined.trim().is_empty() {
        "No compactable content.".to_string()
    } else {
        joined
    }
}

/// The Codex "developer" boilerplate (permissions / collaboration-mode /
/// apps / skills / plugins) — ~3K tokens of cloud-oriented instructions the local
/// model never uses. Beyond wasting tokens, it demonstrably derails the summarizer
/// on the first chunk. Drop it before summarizing.
fn is_boilerplate(item: &serde_json::Value) -> bool {
    item.get("content")
        .and_then(|c| c.as_str())
        .is_some_and(|c| {
            c.contains("<permissions instructions>")
                || c.contains("<apps_instructions>")
                || c.contains("<skills_instructions>")
        })
}

/// The harness compaction *directive* — the synthesized user message that asks
/// the model to summarize (`<<<LOCAL_COMPACT>>>` sentinel, or the built-in
/// `CONTEXT CHECKPOINT COMPACTION` template). It is an instruction to compact,
/// NOT conversation content, so it must never be summarized or copied into the
/// handoff's verbatim recent tail.
///
/// Leaving it in was the session-break: `render_recent_turns` copied the
/// directive verbatim into `[RECENT TURNS — verbatim and exact]`, so the handoff
/// itself carried the sentinel. The compacted summary became the next turn's last
/// user message, `is_compaction_request` (local_routing.rs) re-matched it on a
/// normal sampling call, and route_request handed the summary back as a plain text
/// turn — ending the turn in `task_complete` and abandoning the in-progress task.
/// Stripping the directive here breaks that self-perpetuating re-trigger.
fn is_compaction_directive(item: &serde_json::Value) -> bool {
    item.get("content")
        .and_then(|c| c.as_str())
        .is_some_and(|c| {
            c.contains("<<<LOCAL_COMPACT>>>") || c.contains("CONTEXT CHECKPOINT COMPACTION")
        })
}

/// Serialize recent transcript turns (`{role, content}` dicts) into readable text
/// for the verbatim tail of the handoff.
fn render_recent_turns(items: &[serde_json::Value]) -> String {
    items
        .iter()
        .filter_map(|m| {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("message");
            let content = m.get("content").and_then(|c| c.as_str())?;
            let content = content.trim();
            if content.is_empty() {
                None
            } else {
                Some(format!("[{role}]\n{content}"))
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_handoff_labels_summary_and_keeps_recent_verbatim() {
        let out = assemble_handoff("did some work", "[user]\nwrote foo.py", "continue");
        assert!(out.contains("post-compaction"), "opaque-string warning present");
        assert!(out.contains("did some work"));
        assert!(out.contains("verbatim and exact"));
        assert!(out.contains("wrote foo.py"));
        assert!(out.contains("Current Request"));
    }

    #[test]
    fn assemble_handoff_empty_reports_nothing() {
        assert_eq!(assemble_handoff("  ", "  ", "  "), "No compactable content.");
    }

    #[test]
    fn is_boilerplate_detects_codex_developer_message() {
        let dev = serde_json::json!({"role":"developer","content":"<permissions instructions>\nFilesystem..."});
        let real = serde_json::json!({"role":"user","content":"write a lambda handler"});
        assert!(is_boilerplate(&dev));
        assert!(!is_boilerplate(&real));
    }

    #[test]
    fn is_compaction_directive_detects_both_sentinels() {
        let legacy = serde_json::json!({
            "role": "user",
            "content": "<<<LOCAL_COMPACT>>> Summarize the thread for continuation."
        });
        let template = serde_json::json!({
            "role": "user",
            "content": "CONTEXT CHECKPOINT COMPACTION\n\nYou are summarizing..."
        });
        assert!(is_compaction_directive(&legacy));
        assert!(is_compaction_directive(&template));
    }

    #[test]
    fn is_compaction_directive_ignores_real_summarize_request() {
        // A genuine user task that merely mentions summarizing must NOT be
        // mistaken for the harness directive — no sentinel, no false match.
        let real = serde_json::json!({
            "role": "user",
            "content": "Please summarize this thread and write it to notes.md"
        });
        assert!(!is_compaction_directive(&real));
    }

    #[test]
    fn pipeline_filter_strips_directive_from_items() {
        // The line-50 filter is the single chokepoint: after it, the directive
        // is gone from EVERYTHING (chunking, summary, and the verbatim tail), so
        // the handoff can never carry the sentinel that re-triggers compaction.
        let items = vec![
            serde_json::json!({"role":"user","content":"write a lambda handler"}),
            serde_json::json!({"role":"assistant","content":"wrote handler.py"}),
            serde_json::json!({"role":"user","content":"<<<LOCAL_COMPACT>>> Summarize the thread."}),
        ];
        let kept: Vec<_> = items
            .iter()
            .filter(|m| !is_boilerplate(m) && !is_compaction_directive(m))
            .collect();
        assert_eq!(kept.len(), 2);
        let tail = render_recent_turns(
            &kept.iter().map(|m| (*m).clone()).collect::<Vec<_>>(),
        );
        assert!(!tail.contains("<<<LOCAL_COMPACT>>>"));
        assert!(tail.contains("lambda handler"));
    }

    #[test]
    fn render_recent_turns_preserves_content_skips_empty() {
        let items = vec![
            serde_json::json!({"role":"user","content":"wrote 4838 bytes to src/lambda_handler.py"}),
            serde_json::json!({"role":"assistant","content":"   "}),
        ];
        let out = render_recent_turns(&items);
        assert!(out.contains("lambda_handler.py"));
        assert_eq!(out.matches("[assistant]").count(), 0);
    }
}
