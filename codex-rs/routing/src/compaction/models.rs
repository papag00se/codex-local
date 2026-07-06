//! Compaction data models — the transcript chunk fed to the summarizer.

use serde::{Deserialize, Serialize};

/// A chunk of transcript items handed to the compactor LLM for summarization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptChunk {
    pub chunk_id: usize,
    pub start_index: usize,
    pub end_index: usize,
    pub token_count: usize,
    pub overlap_from_previous_tokens: usize,
    pub items: Vec<serde_json::Value>,
}
