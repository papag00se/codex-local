//! Tool-call recovery for local models — ported from coding-agent-router/app/tool_adapter.py.
//!
//! Local models (devstral, qwen3-coder) sometimes emit tool calls as embedded JSON
//! in their text output instead of using the structured tool_calls field.
//! This module recovers those into structured tool calls.
//!
//! See docs/spec/routing-logic-reference.md § "Tool call recovery for local models".

use serde_json::Value as JsonValue;
/// A recovered tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: Option<String>,
    pub name: String,
    pub arguments: JsonValue,
}

/// Result of recovering tool calls from a message.
#[derive(Debug, Clone)]
pub struct RecoveredMessage {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Recover tool calls from an Ollama message that may have embedded JSON.
///
/// Algorithm (matching Python `recover_ollama_message`):
/// 1. If message already has tool_calls → return as-is
/// 2. Try parsing entire content as a JSON blob with "tool_calls" key
/// 3. If that fails, try recovering embedded tool blocks (paragraphs split by \n\n)
pub fn recover_tool_calls(content: &str, existing_tool_calls: bool) -> RecoveredMessage {
    if existing_tool_calls {
        return RecoveredMessage {
            content: content.to_string(),
            tool_calls: Vec::new(),
        };
    }

    // Strategy 0: leaked `<tool_call>` blocks (Hermes JSON or XML-function),
    // via the shared tool_aliases parser — single source of truth so the
    // reasoner path recovers the same formats as the coder path.
    if let Some(recovered) = recover_leaked_tool_call_blocks(content) {
        return recovered;
    }

    // Strategy 1: Try parsing entire content as a JSON blob
    if let Some(recovered) = try_parse_json_blob(content) {
        if let Some(tool_calls) = extract_tool_calls_from_blob(&recovered) {
            let text = recovered
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            return RecoveredMessage {
                content: text,
                tool_calls,
            };
        }
    }

    // Strategy 2: Try recovering embedded tool blocks
    recover_embedded_tool_blocks(content, /*streaming=*/ false)
}

/// Recover leaked `<tool_call>…</tool_call>` blocks via the shared
/// [`crate::tool_aliases`] parser, which handles Hermes JSON, malformed JSON,
/// and the XML-function (`<function=NAME><parameter=KEY>`) dialect. Returns
/// `None` when there's nothing to recover so callers fall through to their
/// other strategies. This is the one place both the coder and reasoner paths
/// converge on for `<tool_call>` recovery.
fn recover_leaked_tool_call_blocks(content: &str) -> Option<RecoveredMessage> {
    if !crate::tool_aliases::has_leaked_tool_call(content) {
        return None;
    }
    let tool_calls: Vec<ToolCall> = crate::tool_aliases::parse_leaked_tool_calls(content)
        .iter()
        .filter_map(tool_call_from_wire)
        .collect();
    if tool_calls.is_empty() {
        return None;
    }
    Some(RecoveredMessage {
        content: crate::tool_aliases::strip_leaked_tool_calls(content),
        tool_calls,
    })
}

/// Convert a tool_aliases wire call (`{"function":{"name","arguments":<string>}}`)
/// into a [`ToolCall`].
fn tool_call_from_wire(v: &JsonValue) -> Option<ToolCall> {
    let func = v.get("function")?;
    let name = func.get("name")?.as_str()?.to_string();
    let arguments = match func.get("arguments") {
        Some(JsonValue::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| JsonValue::Object(serde_json::Map::new()))
        }
        Some(other) => other.clone(),
        None => JsonValue::Object(serde_json::Map::new()),
    };
    Some(ToolCall {
        id: None,
        name,
        arguments,
    })
}

/// Streaming variant — partial tool blocks at the end are dropped.
pub fn recover_tool_calls_streaming(content: &str, existing_tool_calls: bool) -> RecoveredMessage {
    if existing_tool_calls {
        return RecoveredMessage {
            content: content.to_string(),
            tool_calls: Vec::new(),
        };
    }

    // Strategy 0: leaked `<tool_call>` blocks (shared tool_aliases parser).
    if let Some(recovered) = recover_leaked_tool_call_blocks(content) {
        return recovered;
    }

    // Strategy 1: JSON blob
    if let Some(recovered) = try_parse_json_blob(content) {
        if let Some(tool_calls) = extract_tool_calls_from_blob(&recovered) {
            let text = recovered
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            return RecoveredMessage {
                content: text,
                tool_calls,
            };
        }
    }

    // Strategy 2: Embedded blocks (streaming mode drops partial blocks at end)
    recover_embedded_tool_blocks(content, /*streaming=*/ true)
}

/// Try parsing content as a JSON blob, stripping markdown fences if present.
fn try_parse_json_blob(content: &str) -> Option<JsonValue> {
    let text = content.trim();

    // Strip markdown fences
    let text = if text.starts_with("```") {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() >= 3 {
            lines[1..lines.len() - 1].join("\n")
        } else {
            return None;
        }
    } else {
        text.to_string()
    };

    let text = text.trim();
    if !text.starts_with('{') {
        return None;
    }

    serde_json::from_str::<JsonValue>(text)
        .ok()
        .filter(|v| v.is_object())
}

/// Extract tool_calls from a parsed JSON blob.
fn extract_tool_calls_from_blob(blob: &JsonValue) -> Option<Vec<ToolCall>> {
    let raw_calls = blob.get("tool_calls")?.as_array()?;
    let calls: Vec<ToolCall> = raw_calls
        .iter()
        .filter_map(|tc| normalize_tool_call(tc))
        .collect();
    if calls.is_empty() { None } else { Some(calls) }
}

/// Normalize a single tool call from various formats.
///
/// Handles:
/// - {"function": {"name": ..., "arguments": ...}} (standard)
/// - {"name": ..., "arguments": ...} (flattened)
/// - arguments as string → try JSON parse, fallback to {"raw": string}
fn normalize_tool_call(tc: &JsonValue) -> Option<ToolCall> {
    let obj = tc.as_object()?;

    let (name, arguments, id) = if let Some(func) = obj.get("function").and_then(|f| f.as_object())
    {
        let name = func.get("name")?.as_str()?.to_string();
        let args = normalize_arguments(func.get("arguments"));
        let id = obj.get("id").and_then(|i| i.as_str()).map(String::from);
        (name, args, id)
    } else {
        let name = obj.get("name")?.as_str()?.to_string();
        let args = normalize_arguments(obj.get("arguments"));
        let id = obj.get("id").and_then(|i| i.as_str()).map(String::from);
        (name, args, id)
    };

    if name.is_empty() {
        return None;
    }

    Some(ToolCall {
        id,
        name,
        arguments,
    })
}

/// Normalize arguments: string → try parse as JSON, dict → use directly, else wrap.
fn normalize_arguments(args: Option<&JsonValue>) -> JsonValue {
    match args {
        None => JsonValue::Object(serde_json::Map::new()),
        Some(JsonValue::Object(map)) => JsonValue::Object(map.clone()),
        Some(JsonValue::String(s)) => match serde_json::from_str::<JsonValue>(s) {
            Ok(JsonValue::Object(map)) => JsonValue::Object(map),
            _ => serde_json::json!({"raw": s}),
        },
        Some(other) => serde_json::json!({"value": other}),
    }
}

/// Recover embedded tool blocks from paragraphs separated by \n\n.
fn recover_embedded_tool_blocks(content: &str, streaming: bool) -> RecoveredMessage {
    let paragraphs: Vec<&str> = content.split("\n\n").collect();
    let mut kept: Vec<&str> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for (index, paragraph) in paragraphs.iter().enumerate() {
        let candidate = paragraph.trim();

        // Strip [USER]/[ASSISTANT] prefixes
        let candidate =
            if candidate.starts_with("[USER]\n") || candidate.starts_with("[ASSISTANT]\n") {
                candidate.splitn(2, '\n').nth(1).unwrap_or("").trim()
            } else {
                candidate
            };

        let parsed = try_parse_json_blob(candidate);

        match parsed {
            Some(ref obj) if is_tool_use_block(obj) => {
                // Extract as tool call
                let name = obj.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let input = obj
                    .get("input")
                    .cloned()
                    .filter(|v| v.is_object())
                    .unwrap_or(JsonValue::Object(serde_json::Map::new()));
                let id = obj.get("id").and_then(|i| i.as_str()).map(String::from);
                tool_calls.push(ToolCall {
                    id,
                    name: name.to_string(),
                    arguments: input,
                });
            }
            Some(ref obj) if is_tool_result_block(obj) => {
                // Drop tool_result blocks (echoed context)
            }
            None if streaming
                && index == paragraphs.len() - 1
                && looks_like_partial_tool_block(paragraph) =>
            {
                // Streaming: drop partial tool blocks at the end
            }
            _ => {
                kept.push(paragraph);
            }
        }
    }

    RecoveredMessage {
        content: kept.join("\n\n").trim().to_string(),
        tool_calls,
    }
}

fn is_tool_use_block(obj: &JsonValue) -> bool {
    obj.get("type").and_then(|t| t.as_str()) == Some("tool_use")
        && obj.get("name").and_then(|n| n.as_str()).is_some()
}

fn is_tool_result_block(obj: &JsonValue) -> bool {
    obj.get("type").and_then(|t| t.as_str()) == Some("tool_result")
}

/// Check if text looks like an incomplete JSON tool block (streaming).
fn looks_like_partial_tool_block(content: &str) -> bool {
    let text = content.trim();
    let text = if text.starts_with("[USER]\n") || text.starts_with("[ASSISTANT]\n") {
        text.splitn(2, '\n').nth(1).unwrap_or("")
    } else {
        text
    };
    if !text.starts_with('{') {
        return false;
    }
    text.contains("\"type\"")
        && (text.contains("tool_use")
            || text.contains("tool_result")
            || text.contains("tool_calls"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passthrough_plain_text() {
        let result = recover_tool_calls("Hello, just text", false);
        assert_eq!(result.content, "Hello, just text");
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_passthrough_existing_tool_calls() {
        let result = recover_tool_calls("anything", true);
        assert_eq!(result.content, "anything");
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_recover_xml_function_leak_on_reasoner_path() {
        // The reasoner path (chat_stream) calls recover_tool_calls. It must now
        // recover the XML-function dialect Ornith leaks — previously it didn't,
        // so a tool call on light_reasoner silently leaked as text.
        let content = "Sure.\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\npytest -q 2>&1 | tail -5\n</parameter>\n</function>\n</tool_call>";
        let result = recover_tool_calls(content, false);
        assert_eq!(result.tool_calls.len(), 1, "reasoner must recover XML leak");
        assert_eq!(result.tool_calls[0].name, "exec_command");
        assert_eq!(
            result.tool_calls[0].arguments["cmd"],
            "pytest -q 2>&1 | tail -5"
        );
        assert_eq!(result.content, "Sure."); // block stripped from visible text
    }

    #[test]
    fn test_recover_json_blob_with_tool_calls() {
        let content = r#"{"content": "thinking...", "tool_calls": [{"function": {"name": "shell", "arguments": {"cmd": "pwd"}}}]}"#;
        let result = recover_tool_calls(content, false);
        assert_eq!(result.content, "thinking...");
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "shell");
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({"cmd": "pwd"})
        );
    }

    #[test]
    fn test_recover_embedded_tool_use() {
        let content = "Some text\n\n{\"type\": \"tool_use\", \"name\": \"exec_command\", \"input\": {\"cmd\": \"ls\"}}\n\nMore text";
        let result = recover_tool_calls(content, false);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "exec_command");
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({"cmd": "ls"})
        );
        assert!(result.content.contains("Some text"));
        assert!(result.content.contains("More text"));
    }

    #[test]
    fn test_tool_result_blocks_stripped_with_tool_use() {
        let content = "text\n\n{\"type\": \"tool_use\", \"name\": \"shell\", \"input\": {\"cmd\": \"ls\"}}\n\n{\"type\": \"tool_result\", \"content\": \"output\"}\n\nmore text";
        let result = recover_tool_calls(content, false);
        assert_eq!(result.tool_calls.len(), 1);
        assert!(!result.content.contains("tool_result"));
    }

    #[test]
    fn test_recover_fenced_json() {
        let content = "```json\n{\"type\": \"tool_use\", \"name\": \"read_file\", \"input\": {\"path\": \"test.py\"}}\n```";
        // The fenced block is the entire content, so it's parsed as a JSON blob
        // But it has type=tool_use not tool_calls, so it goes through embedded recovery
        let result = recover_tool_calls(content, false);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "read_file");
    }

    #[test]
    fn test_arguments_string_parsed() {
        let content = r#"{"content": "", "tool_calls": [{"function": {"name": "test", "arguments": "{\"key\": \"value\"}"}}]}"#;
        let result = recover_tool_calls(content, false);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(
            result.tool_calls[0].arguments,
            serde_json::json!({"key": "value"})
        );
    }

    #[test]
    fn test_streaming_partial_block_dropped() {
        let content = "text\n\n{\"type\": \"tool_use\", \"name\": \"incomplete";
        let result = recover_tool_calls_streaming(content, false);
        // Partial block at end should be dropped in streaming mode
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content, "text");
    }

    #[test]
    fn test_multiple_tool_calls() {
        let content = "thinking\n\n{\"type\": \"tool_use\", \"name\": \"cmd1\", \"input\": {\"a\": 1}}\n\n{\"type\": \"tool_use\", \"name\": \"cmd2\", \"input\": {\"b\": 2}}\n\ndone";
        let result = recover_tool_calls(content, false);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "cmd1");
        assert_eq!(result.tool_calls[1].name, "cmd2");
        assert!(result.content.contains("thinking"));
        assert!(result.content.contains("done"));
    }

    #[test]
    fn test_normalize_arguments_dict() {
        let args = serde_json::json!({"key": "value"});
        assert_eq!(
            normalize_arguments(Some(&args)),
            serde_json::json!({"key": "value"})
        );
    }

    #[test]
    fn test_normalize_arguments_string() {
        let args = serde_json::json!("{\"key\": \"value\"}");
        assert_eq!(
            normalize_arguments(Some(&args)),
            serde_json::json!({"key": "value"})
        );
    }

    #[test]
    fn test_normalize_arguments_bad_string() {
        let args = serde_json::json!("not json");
        assert_eq!(
            normalize_arguments(Some(&args)),
            serde_json::json!({"raw": "not json"})
        );
    }

    #[test]
    fn test_normalize_arguments_none() {
        let result = normalize_arguments(None);
        assert_eq!(result, serde_json::json!({}));
    }

    #[test]
    fn test_assistant_prefix_stripped() {
        let content = "text\n\n[ASSISTANT]\n{\"type\": \"tool_use\", \"name\": \"shell\", \"input\": {\"cmd\": \"ls\"}}";
        let result = recover_tool_calls(content, false);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "shell");
    }
}
