//! Intermediate representation: parse a slice of `ResponseItem` into typed
//! `TrimItem` values tagged with a `turn_id` and (where relevant) a
//! deterministic tool-call signature.

use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;

use super::signatures::signature_for_call;

/// Per-item typed representation used by the trimmer.
#[derive(Debug, Clone)]
pub enum TrimItem {
    User {
        turn_id: u32,
        text: String,
    },
    AssistantText {
        turn_id: u32,
        text: String,
    },
    Reasoning {
        turn_id: u32,
        text: String,
    },
    ToolCall {
        turn_id: u32,
        call_id: String,
        tool_name: String,
        /// Raw arguments (function tools: JSON string; custom tools: raw string).
        args: String,
        /// Deterministic dedup key, e.g. `read_file::path=src/auth.py`.
        signature: String,
    },
    ToolOutput {
        turn_id: u32,
        call_id: String,
        /// Tool name carried over from the matching call (resolved during parse).
        tool_name: String,
        /// Same signature as the matching call (resolved during parse).
        signature: String,
        /// True iff the output represents a successful invocation.
        /// Heuristic: shell exit code 0, no error markers in text payload.
        success: bool,
        content: String,
    },
    /// Anything we don't model explicitly (e.g. ImageGenerationCall, GhostSnapshot)
    /// — preserved opaquely so the active-turn passthrough remains lossless.
    Other {
        turn_id: u32,
        original: ResponseItem,
    },
}

impl TrimItem {
    pub fn turn_id(&self) -> u32 {
        match self {
            Self::User { turn_id, .. }
            | Self::AssistantText { turn_id, .. }
            | Self::Reasoning { turn_id, .. }
            | Self::ToolCall { turn_id, .. }
            | Self::ToolOutput { turn_id, .. }
            | Self::Other { turn_id, .. } => *turn_id,
        }
    }
}

/// Parsed transcript: a flat list of `TrimItem` values plus the highest turn_id
/// observed (the active turn).
#[derive(Debug, Clone, Default)]
pub struct ParsedTranscript {
    pub items: Vec<TrimItem>,
    pub max_turn_id: u32,
}

impl ParsedTranscript {
    /// The turn id of the active (most recent) turn. Returns 0 if there are no
    /// user messages yet.
    pub fn active_turn_id(&self) -> u32 {
        self.max_turn_id
    }
}

/// Parse a `ResponseItem` slice into `TrimItem` values.
///
/// Turn ids start at 0 and increment on each *user* message. Items appearing
/// before the first user message belong to turn 0 (system context, project
/// context, etc.).
///
/// Tool outputs are matched to their preceding tool calls by `call_id` so we
/// can carry the tool name and signature onto the output.
pub fn parse(items: &[ResponseItem]) -> ParsedTranscript {
    let mut out = Vec::with_capacity(items.len());
    let mut current_turn: u32 = 0;
    let mut user_seen = false;

    // call_id -> (tool_name, signature) so we can resolve outputs to their calls.
    let mut call_table: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();

    for item in items.iter() {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let text = collect_message_text(content);
                if role == "user" {
                    if user_seen {
                        current_turn = current_turn.saturating_add(1);
                    }
                    user_seen = true;
                    if !text.is_empty() {
                        out.push(TrimItem::User {
                            turn_id: current_turn,
                            text,
                        });
                    }
                } else if role == "assistant" || role == "system" || role == "developer" {
                    if !text.is_empty() {
                        out.push(TrimItem::AssistantText {
                            turn_id: current_turn,
                            text,
                        });
                    }
                }
            }
            ResponseItem::Reasoning {
                summary, content, ..
            } => {
                let text = collect_reasoning_text(summary, content.as_deref());
                if !text.is_empty() {
                    out.push(TrimItem::Reasoning {
                        turn_id: current_turn,
                        text,
                    });
                }
            }
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                let signature = signature_for_call(name, arguments);
                call_table.insert(call_id.clone(), (name.clone(), signature.clone()));
                out.push(TrimItem::ToolCall {
                    turn_id: current_turn,
                    call_id: call_id.clone(),
                    tool_name: name.clone(),
                    args: arguments.clone(),
                    signature,
                });
            }
            ResponseItem::CustomToolCall {
                name,
                input,
                call_id,
                ..
            } => {
                let signature = signature_for_call(name, input);
                call_table.insert(call_id.clone(), (name.clone(), signature.clone()));
                out.push(TrimItem::ToolCall {
                    turn_id: current_turn,
                    call_id: call_id.clone(),
                    tool_name: name.clone(),
                    args: input.clone(),
                    signature,
                });
            }
            ResponseItem::LocalShellCall {
                call_id,
                id,
                action,
                ..
            } => {
                let key = call_id.clone().or_else(|| id.clone()).unwrap_or_default();
                let args = serde_json::to_string(action).unwrap_or_default();
                let signature = signature_for_call("local_shell", &args);
                call_table.insert(key.clone(), ("local_shell".to_string(), signature.clone()));
                out.push(TrimItem::ToolCall {
                    turn_id: current_turn,
                    call_id: key,
                    tool_name: "local_shell".to_string(),
                    args,
                    signature,
                });
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                let (tool_name, signature) = call_table
                    .get(call_id)
                    .cloned()
                    .unwrap_or_else(|| ("unknown".to_string(), format!("unknown::{call_id}")));
                let content = output.body.to_text().unwrap_or_default();
                let success = derive_success(output.success, &content);
                out.push(TrimItem::ToolOutput {
                    turn_id: current_turn,
                    call_id: call_id.clone(),
                    tool_name,
                    signature,
                    success,
                    content,
                });
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                let (tool_name, signature) = call_table
                    .get(call_id)
                    .cloned()
                    .unwrap_or_else(|| ("unknown".to_string(), format!("unknown::{call_id}")));
                let content = output.body.to_text().unwrap_or_default();
                let success = derive_success(output.success, &content);
                out.push(TrimItem::ToolOutput {
                    turn_id: current_turn,
                    call_id: call_id.clone(),
                    tool_name,
                    signature,
                    success,
                    content,
                });
            }
            other => {
                out.push(TrimItem::Other {
                    turn_id: current_turn,
                    original: other.clone(),
                });
            }
        }
    }

    ParsedTranscript {
        items: out,
        max_turn_id: current_turn,
    }
}

/// Derive whether a tool output represents a success.
///
/// The Codex `FunctionCallOutputPayload.success` field is hardcoded `true`
/// by some tool handlers (notably the shell handler) regardless of the
/// underlying command's exit code — the actual exit lives inside the content
/// as `metadata.exit_code`. Trust the explicit `Some(false)` when a handler
/// sets it; otherwise look inside the content for a non-zero exit code.
fn derive_success(explicit: Option<bool>, content: &str) -> bool {
    if let Some(success) = explicit {
        // An explicit `false` is authoritative. An explicit `true` may be
        // unreliable for shell-style outputs, so still check the metadata.
        if !success {
            return false;
        }
    }
    if let Some(exit_code) = parse_exit_code(content)
        && exit_code != 0
    {
        return false;
    }
    true
}

/// Try to extract `metadata.exit_code` (or top-level `exit_code`) from a tool
/// output that contains structured shell-result JSON.
fn parse_exit_code(content: &str) -> Option<i64> {
    let parsed: serde_json::Value = serde_json::from_str(content).ok()?;
    parsed
        .get("metadata")
        .and_then(|m| m.get("exit_code"))
        .and_then(|c| c.as_i64())
        .or_else(|| parsed.get("exit_code").and_then(|c| c.as_i64()))
}

fn collect_message_text(content: &[ContentItem]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            ContentItem::InputText { text } => Some(text.as_str()),
            ContentItem::OutputText { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn collect_reasoning_text(
    summary: &[codex_protocol::models::ReasoningItemReasoningSummary],
    content: Option<&[codex_protocol::models::ReasoningItemContent]>,
) -> String {
    use codex_protocol::models::ReasoningItemContent;
    use codex_protocol::models::ReasoningItemReasoningSummary;

    let mut parts: Vec<&str> = Vec::new();
    for s in summary {
        let ReasoningItemReasoningSummary::SummaryText { text } = s;
        parts.push(text.as_str());
    }
    if let Some(items) = content {
        for c in items {
            match c {
                ReasoningItemContent::ReasoningText { text } => parts.push(text.as_str()),
                ReasoningItemContent::Text { text } => parts.push(text.as_str()),
            }
        }
    }
    parts.join("\n")
}
