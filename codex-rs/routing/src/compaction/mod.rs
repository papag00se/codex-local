//! Compaction pipeline — summarize a long transcript into a model-written handoff.
//!
//! Flow (see `pipeline::compact_transcript`): strip boilerplate -> normalize ->
//! split off a verbatim recent tail -> chunk the rest -> summarize each chunk with
//! the compactor LLM -> one final unifying pass -> assemble handoff.

pub mod chunking;
pub mod extract;
pub mod models;
pub mod normalize;
pub mod pipeline;

pub use models::TranscriptChunk;
pub use pipeline::{CompactionConfig, compact_transcript};
