//! Transcript normalization — ported from compaction/normalize.py.
//! Strips encrypted_content, attachments, tool_result blocks.
//! Detects precompacted summaries. Preserves newest turn raw.

use serde_json::Value as JsonValue;

/// Result of normalizing a transcript.
pub struct NormalizedTranscript {
    /// Items that can be compacted.
    pub compactable_items: Vec<JsonValue>,
    /// Items that are precompacted summaries (carried forward raw).
    pub precompacted_items: Vec<JsonValue>,
    /// The newest top-level turn, preserved uncompacted.
    pub preserved_tail: Vec<JsonValue>,
}

/// Normalize transcript items for compaction.
pub fn normalize_transcript(items: &[JsonValue], max_item_tokens: usize) -> NormalizedTranscript {
    let mut compactable = Vec::new();
    let mut precompacted = Vec::new();
    let mut preserved_tail = Vec::new();

    // Sanitize all items
    let sanitized: Vec<Option<JsonValue>> = items.iter().map(sanitize_item).collect();

    // Find the last non-None item — preserve it raw
    let last_idx = sanitized
        .iter()
        .enumerate()
        .rev()
        .find(|(_, item)| item.is_some())
        .map(|(i, _)| i);

    for (i, item) in sanitized.into_iter().enumerate() {
        let Some(item) = item else { continue };

        if Some(i) == last_idx {
            preserved_tail.push(item);
            continue;
        }

        if is_precompacted_summary(&item) {
            precompacted.push(item);
        } else {
            // Check size — skip oversized items
            let item_tokens =
                crate::metrics::estimate_tokens(&serde_json::to_string(&item).unwrap_or_default());
            if item_tokens <= max_item_tokens {
                compactable.push(item);
            }
            // Oversized items are silently dropped (carried in recent_raw if needed)
        }
    }

    NormalizedTranscript {
        compactable_items: compactable,
        precompacted_items: precompacted,
        preserved_tail,
    }
}

/// Sanitize a single item — strip encrypted_content, attachments.
fn sanitize_item(item: &JsonValue) -> Option<JsonValue> {
    let obj = item.as_object()?;
    let mut clean = serde_json::Map::new();

    for (key, value) in obj {
        if key == "encrypted_content" {
            continue;
        }
        if let Some(content) = value.as_array() {
            // Filter content blocks
            let filtered: Vec<JsonValue> = content
                .iter()
                .filter_map(|block| sanitize_block(block))
                .collect();
            if filtered.is_empty() {
                continue;
            }
            clean.insert(key.clone(), JsonValue::Array(filtered));
        } else {
            clean.insert(key.clone(), strip_encrypted(value));
        }
    }

    if clean.is_empty() {
        None
    } else {
        Some(JsonValue::Object(clean))
    }
}

/// Sanitize a content block — remove attachments, tool_result blocks.
fn sanitize_block(block: &JsonValue) -> Option<JsonValue> {
    let obj = block.as_object()?;
    let block_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");

    // Remove attachment types
    match block_type {
        "image" | "input_image" | "localImage" | "local_image" | "file" | "input_file" => {
            return None;
        }
        _ => {}
    }

    // Remove tool_result and function_call_output from compactable items
    match block_type {
        "tool_result" | "function_call_output" => {
            return None;
        }
        _ => {}
    }

    // Strip encrypted_content from the block
    Some(strip_encrypted(block))
}

/// Recursively strip encrypted_content keys.
fn strip_encrypted(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Object(obj) => {
            let clean: serde_json::Map<String, JsonValue> = obj
                .iter()
                .filter(|(k, _)| k.as_str() != "encrypted_content")
                .map(|(k, v)| (k.clone(), strip_encrypted(v)))
                .collect();
            JsonValue::Object(clean)
        }
        JsonValue::Array(arr) => JsonValue::Array(arr.iter().map(strip_encrypted).collect()),
        other => other.clone(),
    }
}

/// Check if an item is a precompacted summary.
fn is_precompacted_summary(item: &JsonValue) -> bool {
    let content = item.get("content");

    // Check string content
    if let Some(text) = content.and_then(|c| c.as_str()) {
        return is_precompacted_text(text);
    }

    // Check array content for text blocks
    if let Some(blocks) = content.and_then(|c| c.as_array()) {
        for block in blocks {
            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                if is_precompacted_text(text) {
                    return true;
                }
            }
        }
    }

    false
}

fn is_precompacted_text(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.starts_with("Another language model started to solve this problem")
        || (trimmed.contains("## Thread Summary for Continuation")
            && trimmed.contains("Latest Real User Intent"))
        // Our OWN handoff wrappers (see `pipeline::assemble_handoff`). Without
        // these, a second compaction re-summarizes a prior local summary
        // (summary-of-a-summary), nesting the handoff. Recognizing them carries
        // the prior summary forward verbatim instead of re-chunking it.
        || trimmed.starts_with("[COMPACTED SUMMARY")
        || trimmed.starts_with("[RECENT TURNS")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_preserves_last_item() {
        let items = vec![
            serde_json::json!({"role": "user", "content": "hello"}),
            serde_json::json!({"role": "assistant", "content": "hi"}),
            serde_json::json!({"role": "user", "content": "thanks"}),
        ];
        let result = normalize_transcript(&items, 10000);
        assert_eq!(result.preserved_tail.len(), 1);
        assert_eq!(result.compactable_items.len(), 2);
    }

    #[test]
    fn test_strips_encrypted_content() {
        let items = vec![
            serde_json::json!({"role": "user", "content": "hello", "encrypted_content": "secret"}),
        ];
        let result = normalize_transcript(&items, 10000);
        // The only item is preserved_tail (last item)
        assert!(result.preserved_tail[0].get("encrypted_content").is_none());
    }

    #[test]
    fn test_detects_precompacted() {
        let items = vec![
            serde_json::json!({"role": "assistant", "content": "Another language model started to solve this problem and produced a summary of its thinking process."}),
            serde_json::json!({"role": "user", "content": "continue"}),
        ];
        let result = normalize_transcript(&items, 10000);
        assert_eq!(result.precompacted_items.len(), 1);
    }

    #[test]
    fn test_detects_our_own_handoff_wrapper_as_precompacted() {
        // A prior LOCAL compaction summary re-entering compaction must be carried
        // forward, NOT re-summarized (summary-of-a-summary nesting).
        let items = vec![
            serde_json::json!({"role": "user", "content": "[COMPACTED SUMMARY of the work so far. NOTE: this is a post-compaction summary…]\n\nOBJECTIVE: build X"}),
            serde_json::json!({"role": "user", "content": "keep going"}),
        ];
        let result = normalize_transcript(&items, 10000);
        assert_eq!(result.precompacted_items.len(), 1);
        assert!(is_precompacted_text("[RECENT TURNS — verbatim and exact]\n\n[user]\nhi"));
    }
}
