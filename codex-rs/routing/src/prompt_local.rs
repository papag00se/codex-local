//! Our own concise base system prompt for LOCAL models, replacing the Codex
//! base prompt entirely on local routes.
//!
//! Why: the Codex base prompt is ~351 lines and teaches `apply_patch` heavily
//! (a dedicated section plus scattered directives). A small local model follows
//! that volume of instruction and emits `apply_patch` no matter what shorter
//! hints we add — and the bulk also eats ~3–5k tokens of context every turn.
//! This prompt is deliberately short and guiding: role, agentic persistence,
//! act-don't-narrate, `write_file`-first editing (no patch tool), and the curated
//! tools. Per-tool argument shapes are still appended separately (the tool hint).
//!
//! Portability: the text lives in `local_coder_prompt.md` so the Python service
//! can load the exact same file — the prompt is a data asset, not code.

/// The local-model base system prompt. Used for local coder/reasoner routes in
/// place of `prompt.base_instructions.text`.
pub const LOCAL_CODER_SYSTEM_PROMPT: &str = include_str!("local_coder_prompt.md");
