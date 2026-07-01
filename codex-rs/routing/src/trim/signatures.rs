//! Deterministic signature keys for tool calls.
//!
//! Two calls share a signature when the trimmer should treat the later one as
//! superseding the earlier — e.g. two `read_file` calls on the same path,
//! or two `grep_files` calls with the same query. Signature derivation never
//! calls an LLM and never panics on malformed input.

use serde_json::Value as JsonValue;

/// Build a signature for the given tool name and raw argument string.
///
/// Returns a string of the form `tool_name::key=value[,key=value...]` for tools
/// that have a meaningful dedup key, or `tool_name::<call-id-hash>` for tools
/// that should never be deduplicated.
pub fn signature_for_call(tool_name: &str, args_raw: &str) -> String {
    let parsed: Option<JsonValue> = serde_json::from_str(args_raw).ok();
    let key_part = match tool_name {
        // File reads — dedup by path.
        "text_editor" => editor_signature(parsed.as_ref()),
        "view_image" => string_field(parsed.as_ref(), &["path", "image_url", "url"])
            .map(|v| format!("path={v}"))
            .unwrap_or_else(|| "path=?".to_string()),
        // List/search — dedup by primary input.
        "list_dir" => string_field(parsed.as_ref(), &["path", "dir"])
            .map(|v| format!("path={v}"))
            .unwrap_or_else(|| "path=?".to_string()),
        "grep_files" => grep_signature(parsed.as_ref()),
        // apply_patch / shell variants — dedup by full command/diff so that
        // re-running the same exact command supersedes the earlier output.
        "shell" | "shell_command" | "exec_command" | "local_shell" => {
            shell_signature(parsed.as_ref())
        }
        "apply_patch" => apply_patch_signature(parsed.as_ref()),
        // Writes/edits — dedup by PATH so writing different files isn't a "repeat",
        // but re-writing the same file is. (edit also keys on the snippet, so
        // distinct edits to one file stay distinct.)
        "write_file" | "create_file" => {
            string_field(parsed.as_ref(), &["path", "file_path", "file", "filename"])
                .map(|v| format!("path={v}"))
                .unwrap_or_else(|| "path=?".to_string())
        }
        "edit_file" | "str_replace" => {
            let path = string_field(parsed.as_ref(), &["path", "file_path", "file", "filename"])
                .unwrap_or_default();
            let snippet: String = string_field(parsed.as_ref(), &["old_string", "old_str", "old"])
                .unwrap_or_default()
                .chars()
                .take(80)
                .collect();
            format!("path={path}|old={snippet}")
        }
        "read_file" | "cat_file" => {
            let path =
                string_field(parsed.as_ref(), &["path", "file", "file_path"]).unwrap_or_default();
            let n = |k: &str| {
                parsed
                    .as_ref()
                    .and_then(|v| v.get(k))
                    .and_then(JsonValue::as_i64)
                    .unwrap_or(0)
            };
            format!("path={path}|{}-{}", n("start_line"), n("end_line"))
        }
        // Web fetch — dedup by url + the navigation args, so fetching a doc and
        // then `find=`-ing different sections (or paging with `cursor`) is NOT a
        // repeat; only an identical re-fetch of the same view is.
        "web_fetch" => {
            let url = string_field(parsed.as_ref(), &["url"]).unwrap_or_default();
            let find = string_field(parsed.as_ref(), &["find"]).unwrap_or_default();
            let cursor = string_field(parsed.as_ref(), &["cursor"]).unwrap_or_default();
            format!("url={url}|find={find}|cursor={cursor}")
        }
        "local_web_search" | "web_search" => string_field(parsed.as_ref(), &["query", "q"])
            .map(|v| format!("query={v}"))
            .unwrap_or_else(|| "query=?".to_string()),
        "write_stdin" => "live=stdin".to_string(),
        // Everything else: each call is unique.
        _ => "unique".to_string(),
    };
    format!("{tool_name}::{key_part}")
}

fn string_field(parsed: Option<&JsonValue>, keys: &[&str]) -> Option<String> {
    let v = parsed?;
    for k in keys {
        if let Some(s) = v.get(*k).and_then(|v| v.as_str()) {
            return Some(s.to_string());
        }
    }
    None
}

fn editor_signature(parsed: Option<&JsonValue>) -> String {
    // text_editor accepts either {command, path} or freeform variants. The
    // path is the meaningful dedup key. Two reads of the same path supersede
    // each other; a write to that path invalidates earlier reads (handled in
    // `rules` via stale-after-modify, not here).
    let Some(v) = parsed else {
        return "path=?".to_string();
    };
    let path = v
        .get("path")
        .and_then(|p| p.as_str())
        .or_else(|| v.get("file_path").and_then(|p| p.as_str()))
        .unwrap_or("?");
    let command = v.get("command").and_then(|c| c.as_str()).unwrap_or("read");
    format!("command={command},path={path}")
}

fn grep_signature(parsed: Option<&JsonValue>) -> String {
    let Some(v) = parsed else {
        return "query=?".to_string();
    };
    let query = v
        .get("query")
        .or_else(|| v.get("pattern"))
        .and_then(|q| q.as_str())
        .unwrap_or("?");
    let path = v
        .get("path")
        .or_else(|| v.get("dir"))
        .and_then(|p| p.as_str())
        .unwrap_or(".");
    format!("query={query},path={path}")
}

fn shell_signature(parsed: Option<&JsonValue>) -> String {
    let Some(v) = parsed else {
        return "cmd=?".to_string();
    };
    // Shell tools come in many shapes; try the common fields in order.
    let cmd = v
        .get("command")
        .or_else(|| v.get("cmd"))
        .or_else(|| v.get("argv"))
        .map(stringify_command)
        .unwrap_or_else(|| "?".to_string());
    format!("cmd={cmd}")
}

fn apply_patch_signature(parsed: Option<&JsonValue>) -> String {
    // Patch operations are typically unique — two identical patches in a row
    // are still independent operations. Use the hash of the input rather than
    // its full text so the signature stays small.
    let Some(v) = parsed else {
        return "patch=?".to_string();
    };
    let input = v
        .get("input")
        .or_else(|| v.get("patch"))
        .and_then(|p| p.as_str())
        .unwrap_or("");
    if input.is_empty() {
        return "patch=empty".to_string();
    }
    format!("patch={}", short_hash(input))
}

fn stringify_command(v: &JsonValue) -> String {
    if let Some(s) = v.as_str() {
        return s.to_string();
    }
    if let Some(arr) = v.as_array() {
        return arr
            .iter()
            .filter_map(|p| p.as_str())
            .collect::<Vec<_>>()
            .join(" ");
    }
    v.to_string()
}

/// Tiny non-cryptographic hash for signature keys. We only need stability
/// within a session, not collision resistance.
fn short_hash(s: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Extract the file path from a tool call's signature, if it has one.
/// Used by stale-after-modify detection in `rules.rs`.
pub fn path_from_signature(signature: &str) -> Option<&str> {
    // Signatures look like "tool_name::key=value,key=value". Find the
    // first `path=` segment after the `::`.
    let Some((_tool, kv)) = signature.split_once("::") else {
        return None;
    };
    for pair in kv.split(',') {
        if let Some(rest) = pair.strip_prefix("path=") {
            return Some(rest);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_fetch_find_navigation_is_not_a_repeat() {
        // Fetching a doc then find=-ing different sections is navigation, not a
        // loop — each must be a DISTINCT signature so the repetition guard doesn't
        // fire. Only an identical re-fetch of the same view repeats.
        let plain = signature_for_call("web_fetch", r#"{"url":"https://x/openapi.json"}"#);
        let holders = signature_for_call(
            "web_fetch",
            r#"{"url":"https://x/openapi.json","find":"holders"}"#,
        );
        let handle = signature_for_call(
            "web_fetch",
            r#"{"url":"https://x/openapi.json","find":"Handle"}"#,
        );
        assert_ne!(plain, holders);
        assert_ne!(holders, handle);
        // Identical view → same signature (a real re-fetch loop is still caught).
        let holders2 = signature_for_call(
            "web_fetch",
            r#"{"url":"https://x/openapi.json","find":"holders"}"#,
        );
        assert_eq!(holders, holders2);
    }

    #[test]
    fn writes_to_different_files_are_not_a_repeat() {
        let a = signature_for_call(
            "write_file",
            r#"{"path":"tests/__init__.py","content":"x"}"#,
        );
        let b = signature_for_call("write_file", r#"{"path":"tests/test_h.py","content":"y"}"#);
        let a2 = signature_for_call(
            "write_file",
            r#"{"path":"tests/__init__.py","content":"z"}"#,
        );
        assert_ne!(a, b, "different files must be distinct");
        assert_eq!(
            a, a2,
            "same file (any content) repeats — re-writing one file is the loop signal"
        );
    }

    #[test]
    fn shell_signature_stable_for_identical_and_distinct_for_different() {
        // The loop guard blocks re-execution by comparing consecutive
        // signatures, so byte-identical exec_command args MUST yield the same
        // signature, and a changed command MUST yield a different one.
        let a = signature_for_call("exec_command", r#"{"cmd":"python3 run.py"}"#);
        let b = signature_for_call("exec_command", r#"{"cmd":"python3 run.py"}"#);
        let c = signature_for_call("exec_command", r#"{"cmd":"python3 run.py --fix"}"#);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn write_stdin_signature_is_constant() {
        // write_stdin's signature is intentionally constant — which is exactly
        // why the dispatcher's loop guard exempts it (repeated stdin writes are
        // legitimate, not a stuck loop).
        let a = signature_for_call("write_stdin", r#"{"text":"yes\n"}"#);
        let b = signature_for_call("write_stdin", r#"{"text":"no\n"}"#);
        assert_eq!(a, b);
    }
}
