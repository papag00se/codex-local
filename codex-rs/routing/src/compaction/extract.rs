//! Chunk summarization via the compactor LLM.
//!
//! Each transcript chunk is summarized into faithful PROSE (free-form), not a
//! rigid JSON `ChunkExtraction`. The model is markedly better at the narrative —
//! objective, files, endpoints, state, what's broken — than at filling a schema,
//! and the prose is what the resuming model actually reads. Validated against the
//! real Ada-handle session: the deterministic path produced "No compactable
//! content"; the free-form path produced an accurate, resumable handoff.
//!
//! Caveat baked into the pipeline's output (see `pipeline::assemble_handoff`): a
//! model can silently corrupt opaque high-entropy strings (addresses, hashes) when
//! it re-types them, so the handoff is labelled as post-compaction and the recent
//! tail is preserved verbatim.

use super::models::TranscriptChunk;
use crate::config::OllamaEndpoint;
use crate::ollama::OllamaClientPool;
use std::time::Duration;
use tracing::{info, warn};

/// Hard deadline for a single chunk summarization. A small local compactor can
/// wedge in a repetition loop on a dense chunk (observed pegging the box for 8+
/// min); the deadline turns that into a skipped chunk, not a frozen turn.
const SUMMARIZE_TIMEOUT: Duration = Duration::from_secs(90);

const CHUNK_SUMMARY_PROMPT: &str = "\
You are compacting a coding agent's transcript so another model can resume the task without redoing work. \
Summarize THIS PORTION faithfully and concretely. Capture: the task/goal; every file created or edited (its path and what it \
contains); key facts learned (correct API endpoints/URLs, response schemas, field names, values); commands run and their results; \
errors encountered and how they were fixed; and anything still broken or unresolved. Preserve specifics — paths, endpoints, exact \
values. Do NOT invent details or add commentary. Output only the summary.";

const FINAL_SUMMARY_PROMPT: &str = "\
You are writing the FINAL handoff summary for a coding agent that will resume the task. Below are ordered summaries of consecutive \
portions of the transcript. Merge them into ONE coherent handoff with sections: OBJECTIVE; CURRENT STATE (files created/edited with \
paths + what each contains); KEY FACTS (correct API endpoints/URLs, schemas, field names); WHAT WORKS; WHAT'S BROKEN / UNRESOLVED; \
NEXT STEP. Preserve all concrete specifics. Do not invent or drop details. Output only the handoff.";

/// Summarize one transcript chunk into faithful prose via the compactor LLM.
pub async fn summarize_chunk(
    chunk: &TranscriptChunk,
    pool: &OllamaClientPool,
    endpoint: &OllamaEndpoint,
) -> Result<String, String> {
    let text = render_chunk_text(&chunk.items);
    if text.trim().is_empty() {
        return Ok(String::new());
    }
    info!(
        chunk_id = chunk.chunk_id,
        items = chunk.items.len(),
        "Summarizing chunk"
    );
    call_summarizer(CHUNK_SUMMARY_PROMPT, &text, pool, endpoint, chunk.chunk_id).await
}

/// Merge the per-chunk summaries into ONE handoff via a final compactor pass.
pub async fn summarize_final(
    combined: &str,
    pool: &OllamaClientPool,
    endpoint: &OllamaEndpoint,
) -> Result<String, String> {
    if combined.trim().is_empty() {
        return Ok(String::new());
    }
    call_summarizer(FINAL_SUMMARY_PROMPT, combined, pool, endpoint, usize::MAX).await
}

async fn call_summarizer(
    system: &str,
    user: &str,
    pool: &OllamaClientPool,
    endpoint: &OllamaEndpoint,
    chunk_id: usize,
) -> Result<String, String> {
    let mut ep = endpoint.clone();
    ep.temperature = 0.2;
    ep.think = Some(false); // a snap summary; reasoning tokens only add latency
    let response = match tokio::time::timeout(
        SUMMARIZE_TIMEOUT,
        pool.chat(
            &ep,
            vec![serde_json::json!({"role": "user", "content": user})],
            Some(system),
            None,
        ),
    )
    .await
    {
        Ok(r) => r,
        Err(_) => {
            warn!(
                chunk_id,
                timeout_s = SUMMARIZE_TIMEOUT.as_secs(),
                "Compactor summarization timed out — skipping (mechanical fallback)"
            );
            return Err("summarization timed out".into());
        }
    };
    let Some(body) = response else {
        return Err("compactor LLM unreachable".into());
    };
    let content = body
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    let content = crate::classifier::strip_think_tags(content).trim().to_string();
    if content.is_empty() {
        return Err("compactor returned an empty summary".into());
    }
    Ok(content)
}

/// Render chunk items (`{role, content}` dicts) to readable text for the
/// summarizer. Empty-content items (bare tool-call shells, blank narration) are
/// skipped.
fn render_chunk_text(items: &[serde_json::Value]) -> String {
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
    fn render_chunk_text_joins_role_and_content_skips_empty() {
        let items = vec![
            serde_json::json!({"role": "user", "content": "resolve ada handle goose"}),
            serde_json::json!({"role": "assistant", "content": "  "}), // skipped
            serde_json::json!({"role": "user", "content": "wrote 4838 bytes to src/lambda_handler.py"}),
        ];
        let out = render_chunk_text(&items);
        assert!(out.contains("[user]"));
        assert!(out.contains("resolve ada handle goose"));
        assert!(out.contains("lambda_handler.py"));
        assert_eq!(out.matches("[assistant]").count(), 0, "empty content skipped");
    }
}
