//! Tool spec for `web_fetch` — single-GET HTTP fetch with a browser-like
//! User-Agent.
//!
//! Dispatched locally (no cloud round-trip). The handler lives in
//! `codex-core` (see `tools/handlers/web_fetch.rs`).

use crate::JsonSchema;
use crate::ResponsesApiTool;
use crate::ToolSpec;
use std::collections::BTreeMap;

pub const WEB_FETCH_TOOL_NAME: &str = "web_fetch";

pub fn create_web_fetch_tool() -> ToolSpec {
    let properties = BTreeMap::from([
        (
            "url".to_string(),
            JsonSchema::string(Some(
                "Absolute http:// or https:// URL to fetch. file://, ftp:// and other schemes are rejected."
                    .to_string(),
            )),
        ),
        (
            "user_agent".to_string(),
            JsonSchema::string(Some(
                "Optional User-Agent header. If omitted, a current Brave-style desktop Chrome UA is used so ordinary websites respond as they would to a real browser."
                    .to_string(),
            )),
        ),
        (
            "find".to_string(),
            JsonSchema::string(Some(
                "Optional. Jump straight to the part of the page you need: a keyword or field name (e.g. \"handles/{handle}\" or \"authentication\"). Returns a small, in-context slice — for JSON/YAML the matching section with its path and any referenced schema inlined; for text the matching section. Prefer this over reading the whole page when you know what you're looking for."
                    .to_string(),
            )),
        ),
        (
            "cursor".to_string(),
            JsonSchema::string(Some(
                "Optional. If a previous fetch said \"More remains\", pass the cursor token it gave you (copy it verbatim) with the SAME url to read the next page."
                    .to_string(),
            )),
        ),
    ]);

    ToolSpec::Function(ResponsesApiTool {
        name: WEB_FETCH_TOOL_NAME.to_string(),
        description:
            "Fetch a web page over HTTP(S) and return its content as text. Use this to read a specific URL (documentation, a blog post, an API spec) without running curl. Large pages are cleaned (HTML stripped to text, JSON/YAML minified) and returned one page at a time; if more remains you'll get a `cursor` token to continue. When you know what you're looking for, pass `find=\"<keyword>\"` to jump straight to the relevant section instead of paging."
                .to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            properties,
            Some(vec!["url".to_string()]),
            Some(false.into()),
        ),
        output_schema: None,
    })
}
