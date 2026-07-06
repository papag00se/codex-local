//! In-process per-request routing to local Ollama models.
//!
//! Hooks into ModelClientSession::stream() to intercept requests that can
//! be handled by a local model (free) instead of the cloud provider.
//!
//! Uses a local classifier LLM to decide where each request goes.
//! See docs/spec/design-principles.md — the LLM makes the judgment call,
//! deterministic code handles the control flow.

#[derive(Default)]
struct StreamToolCallAcc {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

use crate::client_common::Prompt;
use crate::client_common::ResponseStream;
use codex_api::ResponseEvent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use codex_routing::OllamaClientPool;
use codex_routing::classifier::RouteTarget;
use codex_routing::config::OllamaEndpoint;
use codex_routing::config::RoutingConfig;
use codex_routing::failover::FailoverAction;
use codex_routing::failover::FailureType;
use codex_routing::failover::{self};
use codex_routing::local_dispatch::OllamaTextResponse;
use std::sync::Arc;
use tokio::sync::OnceCell;
use tokio::sync::mpsc;
use tracing::info;
use tracing::warn;

/// Tools exposed to the LightCoder route — same in regular and local-only
/// modes. Curated to fit comfortably in a small local model's context window
/// and attention budget. Do not expand without a deliberate reason.
///
/// Names here must exactly match what's actually registered in the Codex tool
/// registry (see `codex-rs/tools/src/`). Names not present are silently
/// dropped by the filter — the model would then see fewer tools than intended,
/// which is how the first cut of this list went wrong.
///
/// Excluded by design: MCP connectors (`mcp__*`), multi-agent orchestration
/// (`spawn_*`, `wait_*`, `supervisor`, …), `js_repl`, `code_mode_*`, and
/// dynamic-tool plumbing. Cloud routes still see all of these.
///
/// Reads + greps + listings happen via `shell` (e.g. `cat`, `rg`, `ls`); there
/// is no dedicated `text_editor`/`grep_files`/`read_file` tool in this Codex
/// install. `list_dir` is kept alongside `shell ls` because it's safer and
/// cloud models use it natively.
// Note: `apply_patch` is intentionally NOT exposed to local coders. Its
// unified-diff/hunk format (hunk headers, per-line prefixes, context lines the
// model must reproduce from memory) is the single biggest source of stuck
// edit loops for small models. Instead we inject the synthesized content-based
// `edit_file` / `write_file` tools (see `synthetic_local_edit_tools`), which
// `tool_aliases` translates back into `apply_patch` bodies — so the model gets
// a forgiving "find this snippet / write this file" interface while we keep the
// battle-tested apply_patch executor underneath. `shell` stays as a fallback.
// MODEL-FACING tool names, ordered as the model should prefer them — specific
// tools first, generic `shell` LAST so it reaches for write_file/edit_file/
// web_search/web_fetch before a raw command. `web_search` is what we present for
// Codex's Brave `local_web_search` (mapped to the native name by `native_tool_name`
// for the `prompt.tools` filter; renamed back for display in `present_local_tools`,
// which also governs the final order the model actually sees).
const LIGHT_CODER_TOOL_NAMES: &[&str] = &[
    "list_dir",
    "view_image",
    "update_plan",
    "web_search",
    "web_fetch",
    "request_permissions",
    "exec_command",
    "write_stdin",
    "shell",
];

/// Map a model-facing curated name ([`LIGHT_CODER_TOOL_NAMES`]) to the native Codex
/// registry name it resolves to when filtering `prompt.tools`. Identity for every
/// tool except `web_search`, which we present to the model but Codex registers as
/// `local_web_search` (the Brave backend); [`present_local_tools`] does the reverse
/// rename on the pulled tool for display.
fn native_tool_name(curated: &str) -> &str {
    match curated {
        "web_search" => "local_web_search",
        other => other,
    }
}

/// Synthetic, content-based editing tools exposed to Focused local coders in
/// place of `apply_patch`. They are not real Codex tools — `translate_one_native_call`
/// rewrites them into `apply_patch` bodies before dispatch (see
/// `tool_aliases::normalize_edit_file_call` / `normalize_write_file_call`). The
/// schemas deliberately steer the model to COPY `old_string` from the file
/// content pinned in its prompt rather than recall it.
fn synthetic_local_edit_tools() -> Vec<serde_json::Value> {
    vec![
        // write_file FIRST — it is the default edit path for local models. A whole
        // -file write has nothing to match, so it can't fail the way a patch does.
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create OR completely overwrite a file with its full content. This is the DEFAULT, most reliable way to write or change a file — you supply the entire file, so there is nothing to match and nothing to fail. Keep files small and focused (prefer several small modules over one big file) so a full rewrite stays cheap.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path of the file to write (created or overwritten)."},
                        "content": {"type": "string", "description": "The ENTIRE file content to write."}
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace one exact snippet in an existing file, for when rewriting the whole file would be wasteful. Copy `old_string` VERBATIM from the file content pinned in your prompt — do not retype it from memory; it must match exactly and be unique. Set `new_string` to \"\" to delete. If an edit won't apply, fall back to write_file with the whole file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit."},
                        "old_string": {"type": "string", "description": "The exact existing text to replace, copied verbatim from the pinned file content."},
                        "new_string": {"type": "string", "description": "The replacement text. An empty string deletes old_string."}
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            }
        }),
    ]
}

/// Read-only file reader exposed to every local role (coder and reasoner). The
/// reasoner has no shell/exec, so this is its only way to read file CONTENTS;
/// the harness runs `cat`/`sed` under the hood, so it cannot mutate anything.
fn synthetic_local_read_tools() -> Vec<serde_json::Value> {
    vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents (read-only). Optionally pass start_line/end_line (1-based, inclusive) to read just a range; omit them to read the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to read."},
                    "start_line": {"type": "integer", "description": "Optional 1-based first line to read."},
                    "end_line": {"type": "integer", "description": "Optional 1-based last line to read."}
                },
                "required": ["path"]
            }
        }
    })]
}

/// Translate a slice of native Ollama tool calls, rewriting any whose name is
/// a recognized shell-command alias (e.g. `ls`, `git`, `cat`) into a proper
/// `shell` invocation. Calls whose name is already a registered Codex tool
/// pass through unchanged.
fn translate_native_tool_calls(raw_calls: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    raw_calls
        .into_iter()
        .map(translate_one_native_call)
        .collect()
}

/// Inbound half of the write_file massage: rewrite recorded `shell` calls that WE
/// synthesized from `write_file` (the base64 lowering in [`translate_one_native_call`])
/// back into `write_file` calls, so the model only ever sees its own high-level
/// tool — never the shell substrate underneath. Without this the model finds a
/// base64 shell blob where it called write_file and loses the thread (the exact
/// failure the old one-way translation hit). Runs BEFORE trim so state-extraction
/// and current-file pinning still recognize the write. Idempotent — real
/// write_file calls and every other item pass through untouched.
fn represent_shell_writes(items: &[ResponseItem]) -> Vec<ResponseItem> {
    items
        .iter()
        .cloned()
        .map(represent_shell_write_item)
        .collect()
}

/// The per-item transform behind [`represent_shell_writes`]: if `item` is a `shell`
/// call WE synthesized from `write_file` (it carries the `# shephard-write:`
/// sentinel), rewrite it back into the `write_file` the model actually made. Used
/// both to rebuild the model's prompt AND to re-present the RECORDED item for the
/// rollout/TUI (`handle_output_item_done`) — otherwise the TUI re-renders a 30 KB
/// base64 shell line every frame and freezes. Idempotent: real `write_file` calls
/// and every other item pass through untouched.
pub(crate) fn represent_shell_write_item(item: ResponseItem) -> ResponseItem {
    if let ResponseItem::FunctionCall {
        id,
        name,
        namespace,
        arguments,
        call_id,
    } = &item
        && name == "shell"
        && let Some(cmd) = shell_command_str(arguments)
        && let Some((path, content)) = codex_routing::tool_aliases::parse_shephard_write(&cmd)
    {
        ResponseItem::FunctionCall {
            id: id.clone(),
            name: "write_file".to_string(),
            namespace: namespace.clone(),
            arguments: serde_json::json!({ "path": path, "content": content }).to_string(),
            call_id: call_id.clone(),
        }
    } else {
        item
    }
}

/// Re-present recorded `local_web_search` calls as `web_search` in the history
/// shown to the model. The local coder is shown the Brave tool as `web_search`
/// (see `present_local_tools`); `translate_one_native_call` routes the name back
/// to the registered `local_web_search` handler for dispatch, so the RECORDED call
/// carries the internal name. Swap it back so the model sees its own tool name in
/// its history, consistent with the tool list. Idempotent.
fn represent_web_search_names(items: Vec<ResponseItem>) -> Vec<ResponseItem> {
    items
        .into_iter()
        .map(|item| match item {
            ResponseItem::FunctionCall {
                id,
                name,
                namespace,
                arguments,
                call_id,
            } if name == "local_web_search" => ResponseItem::FunctionCall {
                id,
                name: "web_search".to_string(),
                namespace,
                arguments,
                call_id,
            },
            other => other,
        })
        .collect()
}

/// Extract the executed command string from a `shell` call's arguments — the
/// `{command: ["bash","-lc", CMD]}` array form (take the last string), or a bare
/// `{command: "CMD"}` string.
fn shell_command_str(arguments: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(arguments).ok()?;
    let c = v.get("command")?;
    if let Some(s) = c.as_str() {
        return Some(s.to_string());
    }
    c.as_array()?
        .iter()
        .filter_map(|x| x.as_str())
        .last()
        .map(str::to_string)
}

/// Convert a recovered [`codex_routing::tool_recovery::ToolCall`] into the Ollama
/// wire shape that [`translate_native_tool_calls`] consumes, so recovered and
/// structured calls flow through the exact same normalization + shell-alias
/// translation. The one bridge between the shared recovery type and the coder
/// path's wire format.
fn tool_call_to_wire(tc: &codex_routing::tool_recovery::ToolCall) -> serde_json::Value {
    serde_json::json!({
        "function": { "name": tc.name, "arguments": tc.arguments }
    })
}

/// Set `name` + `arguments` on a tool call in either wire shape
/// (`{"function": {…}}` or flat `{…}`).
fn set_call_arguments(call: &mut serde_json::Value, name: &str, arguments: &str) {
    let entry = if call.get("function").is_some() {
        call.get_mut("function").and_then(|f| f.as_object_mut())
    } else {
        call.as_object_mut()
    };
    if let Some(obj) = entry {
        obj.insert(
            "name".to_string(),
            serde_json::Value::String(name.to_string()),
        );
        obj.insert(
            "arguments".to_string(),
            serde_json::Value::String(arguments.to_string()),
        );
    }
}

fn translate_one_native_call(mut call: serde_json::Value) -> serde_json::Value {
    // Ollama wraps the call as either {"function": {"name", "arguments"}} or
    // a flat {"name", "arguments"}. Normalize.
    let func_obj_path: &[&str] = if call.get("function").is_some() {
        &["function"]
    } else {
        &[]
    };

    let name = func_obj_path
        .iter()
        .fold(Some(&call), |v, k| v.and_then(|v| v.get(*k)))
        .and_then(|v| v.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if name.is_empty() {
        return call;
    }

    let raw_arguments = func_obj_path
        .iter()
        .fold(Some(&call), |v, k| v.and_then(|v| v.get(*k)))
        .and_then(|v| v.get("arguments"));
    let raw_args_str: Option<String> = raw_arguments.and_then(|v| v.as_str().map(str::to_string));
    let mut args_value: serde_json::Value = raw_arguments
        .map(|v| {
            if let Some(s) = v.as_str() {
                serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
            } else {
                v.clone()
            }
        })
        .unwrap_or(serde_json::Value::Null);

    // The local coder is shown Codex's Brave search as `web_search` (see
    // `present_local_tools`); route that name back to the registered
    // `local_web_search` handler so the call dispatches. Args pass through.
    if name == "web_search" {
        let args = raw_args_str
            .clone()
            .unwrap_or_else(|| args_value.to_string());
        set_call_arguments(&mut call, "local_web_search", &args);
        return call;
    }

    // A small model often emits file content the JSON parser rejects (raw
    // newlines, bare quotes). Repair it here so we have clean `{path, content}`
    // BEFORE lowering write_file to the shell+base64 massage below (the match's
    // `write_file` arm). Valid args fall through untouched.
    if matches!(name.as_str(), "write_file" | "create_file")
        && args_value.is_null()
        && let Some(raw) = raw_args_str.as_deref()
        && let Some(repaired) = codex_routing::tool_aliases::recover_write_file_args(raw)
    {
        info!(
            tool = %name,
            bytes = repaired.get("content").and_then(|c| c.as_str()).map(str::len).unwrap_or(0),
            "Recovered botched write_file JSON arguments (raw newlines / unescaped quotes)"
        );
        args_value = repaired;
    }

    // Try shell-alias / shell-shape translation first; then the content-based
    // editing tools we expose to local coders (edit_file / write_file), which
    // we rewrite into apply_patch bodies; finally fall back to apply_patch
    // normalization. All rewrite `args` in place to a real Codex tool.
    let translated = codex_routing::tool_aliases::translate_to_shell_call(&name, &args_value)
        .or_else(|| match name.as_str() {
            "edit_file" | "str_replace" => {
                codex_routing::tool_aliases::normalize_edit_file_call(&args_value)
            }
            // `write_file`/`create_file` are lowered to the agent-agnostic
            // shell+base64 substrate — `shell` is the one primitive every harness
            // exposes, base64 makes the write byte-exact and escaping-proof. The
            // inbound `represent_shell_writes` pass re-presents the recorded shell
            // call AS write_file, so the model only ever sees its own tool. The
            // real handler (handlers/write_file.rs) stays registered as the
            // degradation fallback (used only if translation yields no path).
            "write_file" | "create_file" => {
                codex_routing::tool_aliases::write_file_to_base64_shell(&args_value)
            }
            "read_file" | "cat_file" => {
                codex_routing::tool_aliases::normalize_read_file_call(&args_value)
            }
            // exec_command with a shell-style array `cmd` → route to shell (else
            // the runner tries to exec a program named `[` → "No such file").
            "exec_command" => {
                codex_routing::tool_aliases::normalize_exec_command_array(&args_value)
            }
            // apply_patch is being phased out for local models (it chronically
            // fails — the 9B can't produce matching context). A pure Add File is
            // equivalent to writing the whole file, so route it to the robust
            // write_file handler (which overwrites, so it can't hit "Cannot add:
            // already exists"). Next, a DOUBLE-ESCAPED whole-file Update (the
            // reasoning-tuned Fabliq collapses the file onto one `-`/`+` line with
            // literal `\n`) is rewritten to write_file — but ONLY after verifying the
            // reconstructed old content matches the file on disk, so a partial patch
            // can never silently truncate. Everything else normalizes as before;
            // failed Updates get steered to a write_file rewrite by the trim layer.
            "apply_patch" => codex_routing::tool_aliases::apply_patch_add_to_write_file(
                &args_value,
            )
            .or_else(|| collapsed_update_to_write_file(&args_value))
            .or_else(|| expand_collapsed_update_patch(&args_value))
            .or_else(|| codex_routing::tool_aliases::normalize_apply_patch_call(&args_value)),
            _ => None,
        });
    let Some(translated) = translated else {
        return call;
    };

    info!(
        from = %name,
        to = %translated.name,
        command_line = %translated.command_line,
        "Translated tool call (native)"
    );

    set_call_arguments(&mut call, translated.name, &translated.args.to_string());
    call
}

/// Build a short usage hint listing the tools the local model can call. This
/// gets appended to the system prompt for the LightCoder route only — small
/// models otherwise emit shell command names (`ls`, `rg`, `cat`) as tool
/// names, or guess at the arg shape. Naming the wrapper explicitly and
/// showing concrete examples closes most of the failure modes.
/// Build the tool set sent to a LightCoder local model: the configured subset
/// (Focused curated list, or the Full catalog) plus the synthetic content-based
/// editing tools for the Focused case. Built BEFORE trimming so its token
/// footprint can be reserved from the context budget — the schemas are sent to
/// the model but aren't part of the trimmed messages, so without reserving for
/// them the real prompt overflows the window.
fn build_local_tools(
    prompt: &Prompt,
    endpoint: &OllamaEndpoint,
    names: &[&str],
    include_edit_tools: bool,
) -> Vec<serde_json::Value> {
    let tool_json: Vec<serde_json::Value> = match endpoint.tool_subset {
        codex_routing::config::ToolSubset::Focused => prompt
            .tools
            .iter()
            .filter(|t| names.iter().any(|n| native_tool_name(n) == t.name()))
            .filter_map(|t| serde_json::to_value(t).ok())
            .collect(),
        codex_routing::config::ToolSubset::Full => prompt
            .tools
            .iter()
            .filter_map(|t| serde_json::to_value(t).ok())
            .collect(),
    };
    let mut ollama_tools = codex_routing::tool_format::to_ollama_tools(&tool_json);
    if matches!(
        endpoint.tool_subset,
        codex_routing::config::ToolSubset::Focused
    ) {
        // Read access for EVERY local role — the reasoner has no shell/exec, so
        // this is its only way to read file contents.
        ollama_tools.extend(synthetic_local_read_tools());
        // Edit tools only for roles allowed to mutate the workspace.
        if include_edit_tools {
            ollama_tools.extend(synthetic_local_edit_tools());
        }
        present_local_tools(&mut ollama_tools);
    }
    ollama_tools
}

/// Final shaping of the local coder's tool list, applied after the native +
/// synthetic tools are assembled (Focused subset only). The usage hint is derived
/// from THIS list (names + order — see the `tool_names` collection at the call
/// site), so shaping here is the single authoritative source for BOTH the schema
/// the model receives and the hint:
/// 1. Codex's Brave `local_web_search` is presented to the model as plain
///    `web_search` — the only web search this path exposes. Inbound,
///    `translate_one_native_call` routes the name back to the registered
///    `local_web_search` handler; `represent_web_search_names` re-presents it in
///    history. Codex's native cloud `web_search` is never sent here.
/// 2. `shell` is moved LAST so the model reaches for the specific tools
///    (`write_file`/`edit_file`/`web_search`/`web_fetch`) before the generic shell.
///
/// (`web_fetch` already advertises its `find`/`cursor` navigation params in its
/// own schema — see `web_fetch_tool.rs` — so the schema needs nothing here; the
/// usage hint is what surfaces them to the model.)
fn present_local_tools(tools: &mut Vec<serde_json::Value>) {
    // 1. Present the Brave search tool to the model as `web_search`.
    for tool in tools.iter_mut() {
        let is_brave_search =
            tool.pointer("/function/name").and_then(|v| v.as_str()) == Some("local_web_search");
        if is_brave_search && let Some(n) = tool.pointer_mut("/function/name") {
            *n = serde_json::Value::String("web_search".to_string());
        }
    }
    // 2. `shell` last so the model prefers the specific tools before falling back.
    if let Some(pos) = tools
        .iter()
        .position(|t| t.pointer("/function/name").and_then(|v| v.as_str()) == Some("shell"))
    {
        let shell = tools.remove(pos);
        tools.push(shell);
    }
}

fn build_tool_hint(tool_names: &[&str]) -> String {
    let has = |name: &str| tool_names.contains(&name);
    let mut lines = vec!["You have ONLY the following tools. You MUST call them by these exact names with the exact argument shape shown in the examples — never invent tool names, never guess at argument shapes.".to_string()];

    if has("write_file") || has("edit_file") {
        lines.push(
            "CRITICAL — writing files: to create or change a file you MUST call `write_file` (new file) or `edit_file` (existing file).".to_string(),
        );
    }

    for name in tool_names {
        let block = match *name {
            "write_file" => {
                "- `write_file`: Create OR overwrite a file with its FULL contents. Args: `{\"path\": \"<file>\", \"content\": \"<entire file>\"}`. This is the DEFAULT and most reliable way to write or change a file — you supply the whole file, so there is nothing to match and nothing to fail. Keep files SMALL and focused (prefer several small modules over one big file) so rewriting a whole file stays cheap."
            }
            "edit_file" => {
                "- `edit_file`: Replace one exact snippet in an existing file (use when rewriting the whole file would be wasteful). Args: `{\"path\": \"<file>\", \"old_string\": \"<exact text to replace>\", \"new_string\": \"<replacement>\"}`. Copy `old_string` VERBATIM from the file content pinned in your prompt — do NOT retype it from memory, and include enough surrounding text to make it unique. To delete code, set `new_string` to \"\". If an edit won't apply, fall back to `write_file` with the whole file."
            }
            "read_file" => {
                "- `read_file`: Read a file's contents (read-only). Args: `{\"path\": \"<file>\"}`, or `{\"path\": \"<file>\", \"start_line\": 40, \"end_line\": 80}` for a 1-based inclusive range. Use this to inspect code/config before reasoning about it."
            }
            "list_dir" => {
                "- `list_dir`: List directory contents (safer alternative to `shell ls`). Args: `{\"dir_path\": \"/abs/path\"}`. Path must be absolute."
            }
            "view_image" => {
                "- `view_image`: View a local image file. Args: `{\"path\": \"/abs/path/to/image.png\"}`."
            }
            "update_plan" => {
                "- `update_plan`: Track a multi-step task plan. Args: `{\"plan\": [{\"status\": \"in_progress\", \"step\": \"...\"}]}`."
            }
            "web_search" => {
                "- `web_search`: Search the web via Brave; returns titles, URLs, and short descriptions. Args: `{\"query\": \"<search terms>\", \"count\": 10}` (count optional, 1-20). Pair this with `web_fetch` to read a specific result."
            }
            "web_fetch" => {
                "- `web_fetch`: Fetch a single http(s) URL and return the page body as text. Use this BEFORE writing code against an unfamiliar API or library — read the docs page rather than guessing the endpoint shape. Args: `{\"url\": \"https://...\"}`. For a long page, narrow it: add `\"find\": \"<text>\"` to return ONLY the matching section, or pass `\"cursor\": \"<token>\"` from a previous response to page through. Body is capped at 512KB; binary responses return a placeholder."
            }
            "request_permissions" => {
                "- `request_permissions`: Ask for sandbox escalation when a command would otherwise be blocked (network access for `npm install`/`pip install`/`apt`, writing to a path outside cwd, etc.). Call this BEFORE the command that would fail, with a short justification."
            }
            "exec_command" => {
                "- `exec_command`: Start a long-running shell command with streaming output. Use INSTEAD OF `shell` for: dev servers (`npm run dev`), watch processes, anything that runs more than ~5 seconds, or any command you might need to send input to. Returns a session id you can pair with `write_stdin`."
            }
            "write_stdin" => {
                "- `write_stdin`: Send input to a shell session previously started by `exec_command` (e.g. answer an interactive prompt from `npm init`). Args include the session id and the text to write."
            }
            "shell" => {
                "- `shell`: Run any shell command. Use this for `ls`, `cat`, `rg`, `grep`, `find`, `mkdir`, `rm`, `cd`, `pwd`, build/test commands, package installs, writing files via heredoc — anything you would type at a terminal.\n  REQUIRED ARG SHAPE: `command` MUST be a JSON array of strings, ALWAYS prefixed with `[\"bash\", \"-lc\", \"<your command line>\"]`.\n  Correct example: `{\"command\": [\"bash\", \"-lc\", \"ls -la\"]}`.\n  WRONG: `{\"command\": \"ls -la\"}` (must be an array).\n  WRONG: `{\"command\": [\"bash\", \"-lc\", \"[bash, -lc, ls]\"]}` (do NOT nest the bash invocation; the third element is your literal shell command)."
            }
            _ => continue,
        };
        lines.push(block.to_string());
    }

    if has("shell") {
        lines.push(
            "If you find yourself wanting to call a command like `ls`, `rg`, `cat`, `git`, or `pytest` directly, that is wrong — wrap it as `shell` with `command: [\"bash\", \"-lc\", \"<the command>\"]`.".to_string(),
        );
    }

    lines.join("\n\n")
}

/// Global routing state — initialized lazily on first use.
static ROUTING_STATE: OnceCell<Option<RoutingState>> = OnceCell::const_new();

struct RoutingState {
    config: RoutingConfig,
    project_config: codex_routing::project_config::ProjectConfig,
    pool: Arc<OllamaClientPool>,
    usage: codex_routing::usage::UsageTracker,
    feedback: std::sync::Mutex<codex_routing::feedback::FeedbackStore>,
    codebase_context: codex_routing::codebase_context::CodebaseContext,
    classify_cache: std::sync::Mutex<codex_routing::classify_cache::ClassifyCache>,
    budget: Arc<codex_routing::budget_pressure::BudgetState>,
    claude_sessions: codex_routing::claude_cli::ClaudeSessionTracker,
    /// Single-entry cache of the most recent older-turn compaction summary.
    /// Keyed by hash of the older-turn message contents; reused as long as
    /// the older history is unchanged from request to request within a
    /// session. Prevents recompacting the same history each turn.
    inline_compact_cache: std::sync::Mutex<Option<InlineCompactCacheEntry>>,
    /// Rolling cache for ACTIVE-turn compaction. The active turn is append-only, so
    /// once we've summarized its first `prefix_len` steps we reuse that summary and
    /// only LLM-compact the new tail — instead of re-summarizing the whole (growing)
    /// turn from scratch on every overflow. This is what kills the compaction storm
    /// (observed: 13 from-scratch active-turn compactions / 60 chunk calls in one
    /// stuck turn, ~half the wall-clock).
    active_compact_cache: std::sync::Mutex<Option<ActiveCompactEntry>>,
    /// Pending local-model "nudge" notices — each time a guard intervenes
    /// (repetition guard, rumination guard, quality gate, probe gate, completion
    /// critic) it queues a one-line message here. The TUI drains these via
    /// [`drain_route_notices`] and renders them as history lines so guard
    /// interventions are visible to the user instead of buried in the logs.
    nudges: std::sync::Mutex<Vec<String>>,
}

impl RoutingState {
    /// Queue a one-line notice that a local-model guard fired. Best-effort:
    /// if the lock is poisoned the notice is simply dropped.
    fn push_nudge(&self, message: String) {
        if let Ok(mut queue) = self.nudges.lock() {
            queue.push(message);
        }
    }
}

#[derive(Clone)]
struct InlineCompactCacheEntry {
    older_content_hash: u64,
    summary_message: serde_json::Value,
}

/// One rolling active-turn summary. Covers the first `prefix_len` messages of the
/// active-turn middle (identified by `prefix_hash`); the next compaction reuses it
/// and only folds in `middle[prefix_len..]`.
#[derive(Clone)]
struct ActiveCompactEntry {
    prefix_len: usize,
    prefix_hash: u64,
    summary: String,
}

/// How to compact the active-turn `middle`, given any cached rolling summary.
#[derive(Debug, PartialEq)]
enum ActiveCompactPlan {
    /// No reusable prefix — compact the whole middle (cold start, or the leading
    /// messages changed, e.g. trim dropped one).
    Full,
    /// Reuse `summary` for `middle[..from]`; only LLM-compact `middle[from..]`.
    Incremental { summary: String, from: usize },
}

/// Robust per-message hash for the active-turn prefix: role + content + tool_calls
/// (the plain [`hash_messages`] ignores tool_calls, which is most of an active
/// turn). Two prefixes hash equal only when their leading steps are byte-identical.
fn hash_prefix(messages: &[serde_json::Value]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    for m in messages {
        m.get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .hash(&mut h);
        m.get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .hash(&mut h);
        if let Some(tc) = m.get("tool_calls") {
            tc.to_string().hash(&mut h);
        }
    }
    h.finish()
}

/// Decide how to compact the active-turn `middle`, reusing a prior rolling summary
/// for the unchanged leading prefix. The active turn is append-only across overflow
/// rounds (the model appends tool calls/outputs; it never rewrites history), so once
/// the first `prefix_len` steps are summarized we only need to fold in the new tail
/// — turning each re-compaction from O(whole turn) into O(delta).
fn plan_active_compaction(
    middle: &[serde_json::Value],
    cache: Option<&ActiveCompactEntry>,
) -> ActiveCompactPlan {
    if let Some(c) = cache
        && c.prefix_len > 0
        && c.prefix_len <= middle.len()
        && hash_prefix(&middle[..c.prefix_len]) == c.prefix_hash
    {
        return ActiveCompactPlan::Incremental {
            summary: c.summary.clone(),
            from: c.prefix_len,
        };
    }
    ActiveCompactPlan::Full
}

/// Initialize the global routing state.
/// Loads from `.codex-multi/config.toml` in the current directory, falling
/// back to environment variables for anything not in the config file.
/// Called once, lazily. Returns None if local routing is not configured.
async fn get_routing_state() -> &'static Option<RoutingState> {
    ROUTING_STATE
        .get_or_init(|| async {
            // Load project config from .codex-multi/config.toml if it exists
            let cwd = std::env::current_dir().unwrap_or_default();
            let project_config = codex_routing::project_config::ProjectConfig::load(&cwd);
            let config = RoutingConfig::from_project_config(&project_config);
            let pool = Arc::new(OllamaClientPool::new());

            // Check if the classifier endpoint is reachable via a fast HTTP
            // GET that doesn't require loading a model into GPU memory
            // (a chat request would take 30s on cold start). The probe path
            // depends on flavor: Ollama exposes `/api/version` (returns
            // `{"version": "0.x.y"}`); OpenAI-compat servers expose
            // `/v1/models` (returns the loaded model list). LM Studio,
            // llama.cpp, and vLLM all support `/v1/models`.
            let version_url = match config.classifier.flavor {
                codex_routing::config::ClientFlavor::Ollama => format!(
                    "{}/api/version",
                    config.classifier.base_url.trim_end_matches('/')
                ),
                codex_routing::config::ClientFlavor::OpenAICompat => {
                    let base = config
                        .classifier
                        .base_url
                        .trim_end_matches('/')
                        .trim_end_matches("/v1");
                    format!("{base}/v1/models")
                }
            };
            let initial_reachable = pool
                .client()
                .get(&version_url)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
                .is_ok();

            // Always create the routing state, even if the classifier is
            // unreachable at startup. Reachability is re-checked per request
            // in `route_request` (the classifier's own fallback returns
            // `CloudCoder` when it can't be reached, so subsequent requests
            // degrade gracefully without locking the entire session into
            // "no routing"). This avoids the failure mode where a single
            // flaky network blip at session start sends every request to
            // cloud for the rest of the session.
            if initial_reachable {
                info!(
                    classifier_url = %config.classifier.base_url,
                    classifier_model = %config.classifier.model,
                    "Per-request routing enabled — classifier LLM reachable at startup"
                );
            } else {
                info!(
                    classifier_url = %config.classifier.base_url,
                    "Classifier LLM not reachable at startup; routing state created anyway, will retry per request"
                );
            }

            let usage = codex_routing::usage::UsageTracker::new(
                project_config.usage.primary_warn_threshold,
            );
            let feedback = std::sync::Mutex::new(
                codex_routing::feedback::FeedbackStore::new(&cwd),
            );
            let codebase_context = codex_routing::codebase_context::CodebaseContext::detect(&cwd);
            let classify_cache = std::sync::Mutex::new(
                codex_routing::classify_cache::ClassifyCache::new(),
            );
            let budget = Arc::new(codex_routing::budget_pressure::BudgetState::new());
            let claude_sessions = codex_routing::claude_cli::ClaudeSessionTracker::new();
            Some(RoutingState {
                config,
                project_config,
                pool,
                usage,
                feedback,
                codebase_context,
                classify_cache,
                budget,
                claude_sessions,
                inline_compact_cache: std::sync::Mutex::new(None),
                active_compact_cache: std::sync::Mutex::new(None),
                nudges: std::sync::Mutex::new(Vec::new()),
            })
        })
        .await
}

/// Record session usage to `.codex-multi/usage_log.jsonl`.
/// Called at session exit from the TUI.
pub async fn record_session_usage() {
    let Some(state) = get_routing_state().await.as_ref() else {
        return;
    };

    let cwd = std::env::current_dir().unwrap_or_default();
    let analytics = codex_routing::cost_analytics::CostAnalytics::new(&cwd);

    let local = state.usage.local_usage();
    let secondary = state.usage.secondary_usage();
    let primary = state.usage.primary_usage();
    let total_requests = local.request_count + secondary.request_count + primary.request_count;
    let total_tokens = local.total_tokens() + secondary.total_tokens() + primary.total_tokens();

    if total_requests == 0 {
        return; // Nothing to record
    }

    let savings_pct = if total_requests > 0 {
        ((local.request_count + secondary.request_count) as f64 / total_requests as f64) * 100.0
    } else {
        0.0
    };

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let summary = codex_routing::cost_analytics::SessionUsageSummary {
        session_id: format!("session_{timestamp}"),
        timestamp,
        duration_seconds: 0, // We don't track session duration
        local_requests: local.request_count,
        local_tokens: local.total_tokens(),
        secondary_requests: secondary.request_count,
        secondary_tokens: secondary.total_tokens(),
        primary_requests: primary.request_count,
        primary_tokens: primary.total_tokens(),
        total_requests,
        total_tokens,
        estimated_savings_pct: savings_pct,
    };

    analytics.record_session(&summary);
}

/// Get usage summary string. Returns None if routing is not active.
/// Called from the TUI `/stats` command.
pub async fn usage_summary() -> Option<String> {
    let state = get_routing_state().await.as_ref()?;
    Some(state.usage.summary())
}

/// One-line live readout (current route + most recent local tok/s) for the
/// TUI status line. None if routing is not active or nothing dispatched yet.
pub async fn live_readout() -> Option<String> {
    let state = get_routing_state().await.as_ref()?;
    state.usage.live_readout()
}

/// Drain queued local-model guard notices (oldest first) for the TUI to render.
/// Each call returns the notices accumulated since the previous drain and clears
/// the queue. Empty when routing is inactive or no guard has fired.
pub async fn drain_route_notices() -> Vec<String> {
    let Some(state) = get_routing_state().await.as_ref() else {
        return Vec::new();
    };
    match state.nudges.lock() {
        Ok(mut queue) => std::mem::take(&mut *queue),
        Err(_) => Vec::new(),
    }
}

/// Record cloud model usage (called from client.rs after cloud responses).
pub(crate) async fn record_cloud_usage(model: &str, input_tokens: u64, output_tokens: u64) {
    if let Some(state) = get_routing_state().await.as_ref() {
        state.usage.record(model, input_tokens, output_tokens);
    }
}

/// Update budget state from rate limit headers (called after cloud responses).
/// primary_pct and secondary_pct are 0.0-100.0.
pub(crate) async fn update_budget(
    primary_pct: f64,
    secondary_pct: f64,
    primary_reset: Option<u64>,
) {
    if let Some(state) = get_routing_state().await.as_ref() {
        state
            .budget
            .update(primary_pct, secondary_pct, primary_reset);
    }
}

/// Result of per-request routing.
pub(crate) enum RouteResult {
    /// Request handled locally — use this stream.
    Local(ResponseStream),
    /// Request should go to cloud, but with this model override.
    /// The slug replaces model_info.slug for this request only.
    /// Carries failover chain context so cloud errors can walk the chain.
    CloudOverride {
        slug: String,
        failover_ctx: Option<CloudFailoverCtx>,
    },
    /// No routing — use the default cloud model.
    Default,
}

/// Cloud failover context — passed to stream() so it can retry on HTTP errors.
#[derive(Clone)]
pub(crate) struct CloudFailoverCtx {
    pub role_name: String,
    pub chain_name: String,
    pub chain: Vec<String>,
    pub behavior: codex_routing::project_config::FailoverBehavior,
}

/// What a role name resolves to.
enum ResolvedRole {
    /// Local Ollama model.
    Local(OllamaEndpoint),
    /// Cloud model via OpenAI Responses API (slug override).
    Cloud(String),
    /// Anthropic model via Claude CLI subprocess.
    ClaudeExec { model: String },
}

/// Map a classifier route to the failover chain name.
fn chain_name_for_route(route: &RouteTarget) -> &'static str {
    match route {
        RouteTarget::LightReasoner => "reasoning",
        RouteTarget::LightCoder => "coding",
        RouteTarget::CloudFast | RouteTarget::CloudMini => "coding",
        RouteTarget::CloudReasoner => "reasoning",
        RouteTarget::CloudCoder => "coding",
    }
}

/// Returns true if local_only mode is requested, even when no RoutingState
/// has been initialized (e.g., classifier endpoint unreachable at startup).
/// Reads from the env var only — config.toml requires RoutingState to load.
fn local_only_env() -> bool {
    matches!(
        std::env::var("CODEX_LOCAL_ONLY")
            .unwrap_or_default()
            .trim()
            .to_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Effective `local_only` for callers OUTSIDE the routing layer (e.g. the
/// auto-compaction fork). Prefers the resolved `RoutingState` config — which is
/// `env OR .codex-multi/config.toml routing.local_only` — and falls back to the
/// env var only if `RoutingState` hasn't loaded yet. This is why compaction can
/// honor a config-file `local_only = true`, which [`local_only_env`] alone misses.
pub(crate) async fn is_local_only() -> bool {
    if let Some(state) = get_routing_state().await.as_ref() {
        return state.config.local_only;
    }
    local_only_env()
}

/// Inside the failover loop, decide whether to surface a local-only error
/// (when local_only is on) or fall through to the default cloud path.
fn cloud_fallback_or_local_error(state: &RoutingState, reason: &str) -> RouteResult {
    if state.config.local_only {
        local_only_error(reason)
    } else {
        RouteResult::Default
    }
}

/// Build a RouteResult that surfaces a local-only error to the user as an
/// assistant message, instead of silently falling through to cloud.
fn local_only_error(reason: &str) -> RouteResult {
    let message = format!(
        "Local-only mode is enabled, but no local model can serve this request: {reason}.\n\nThe request was not sent to any cloud provider. To allow cloud dispatch, disable local-only mode (unset CODEX_LOCAL_ONLY, remove --local-only, or set `local_only = false` in `.codex-multi/config.toml` under [routing])."
    );
    warn!(reason = %reason, "local_only: surfacing error to user");
    RouteResult::Local(ollama_response_to_stream(OllamaTextResponse {
        content: message,
        model: "local-only".to_string(),
        input_tokens: 0,
        output_tokens: 0,
    }))
}

/// Map a classifier route to its role name in the failover chain.
/// Inverse of `role_name_for_route` for local request roles: maps a resolved
/// role name back to the `RouteTarget` it represents, so downstream logic
/// (notably tool attachment) reflects the role actually being served after a
/// failover walk — not the stale classified route. A cloud route that's
/// unresolvable in local_only falls back to `light_coder`; without this, the
/// coder would run with `route == CloudMini` and therefore no tools. Returns
/// `None` for roles that aren't local request routes (classifier/compactor,
/// cloud roles), leaving the caller's original route intact.
fn route_target_for_role(role: &str) -> Option<RouteTarget> {
    match role {
        "light_reasoner" | "light_reasoner_backup" => Some(RouteTarget::LightReasoner),
        "light_coder" => Some(RouteTarget::LightCoder),
        _ => None,
    }
}

/// Whether a text-only turn counts as a completed turn for this role. Every
/// local role is a **full coder** — same tools, same ability to edit, same
/// streaming/recovery/overflow path — so this is the ONE axis on which coder and
/// reasoner differ in behavior: a reasoner's product can be text (analysis,
/// answers, plans), so it finishes unless it *bailed*; a coder must take an
/// action. Everything else that distinguishes a "reasoner" — temperature,
/// reasoning budget, `max_tokens` ("room to explore") — is per-endpoint config,
/// not behavior, and is set in the role's `[models.*]` block.
fn role_text_is_product(role: &str) -> bool {
    matches!(role, "light_reasoner" | "light_reasoner_backup")
}

/// True when `content` is essentially a bare JSON object rather than prose — a weak
/// model leaking a structured tool call (commonly `update_plan`) into the content
/// channel. Requires it to START with `{` after trimming AND carry a `"…":` key, so
/// ordinary prose that merely mentions a brace isn't caught. Tolerant of truncation
/// (the object may be cut off) — a partial leak is still a leak.
fn looks_like_bare_json_object(content: &str) -> bool {
    let t = content.trim();
    t.starts_with('{') && t.contains("\":")
}

/// A bare-JSON `update_plan` the model dumped as CONTENT instead of calling the tool:
/// `{"plan":[{"step":…,"status":…}], "explanation"?:…}`. Validates by deserializing as
/// the real tool args (so a synthesized call can't fail downstream) and returns the
/// arguments JSON to feed a genuine update_plan call. `None` when it isn't a well-formed
/// plan object.
fn recover_bare_update_plan(content: &str) -> Option<String> {
    let trimmed = content.trim();
    let args: codex_protocol::plan_tool::UpdatePlanArgs = serde_json::from_str(trimmed).ok()?;
    if args.plan.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

/// Whether to force `tool_choice="required"` on a re-prompt retry.
///
/// The local-coder massaging escalation (probe gate / completion critic /
/// rumination / quality) re-prompts a model that gave us a no-tool-call response
/// when we needed an action. By the retry the prose nudge has already failed, so we
/// enforce the tool call at the sampler. Conditions:
/// - `continuation_count > 0`: only on a retry, never the first call — so normal
///   text-completion and natural loop termination are never blocked.
/// - `use_tools`: nothing to constrain to without a tool set.
/// - `!text_is_product`: a reasoner's text IS the product; never force it to a
///   tool call.
/// - `configured.is_none()`: an explicit operator `tool_choice` always wins.
fn enforce_tool_call_on_retry(
    continuation_count: usize,
    use_tools: bool,
    text_is_product: bool,
    configured: &Option<String>,
) -> bool {
    continuation_count > 0 && use_tools && !text_is_product && configured.is_none()
}

/// Synthetic tools we advertise to local models but which have NO real
/// dispatcher handler — they only function via [`translate_native_tool_calls`].
/// If one of these names survives translation, the model's arguments failed to
/// parse (almost always botched JSON escaping of large/multiline file content);
/// dispatching it would yield an opaque `unsupported call: <tool>`. Detecting it
/// lets the loop re-prompt under a grammar constraint so the server re-emits the
/// call with valid JSON escaping. Returns the offending tool name.
fn surviving_untranslated_synthetic(calls: &[serde_json::Value]) -> Option<String> {
    // Synthetic tools that are TRANSLATED into a real Codex tool (apply_patch /
    // shell). If translation didn't fire, the call kept its synthetic name and
    // has NO handler to dispatch to — that only happens when the args were
    // malformed, so we re-prompt. `write_file`/`create_file` are deliberately
    // ABSENT: they now have real handlers (handlers/write_file.rs) and are never
    // translated, so they always keep their name — listing them here made every
    // write_file call look "malformed" and re-prompt instead of dispatching.
    // Their botched-escaping case is repaired upstream in `translate_one_native_call`.
    const SYNTHETIC: &[&str] = &["edit_file", "read_file", "str_replace", "cat_file"];
    calls.iter().find_map(|c| {
        let name = c
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .or_else(|| c.get("name").and_then(|n| n.as_str()))?;
        SYNTHETIC.contains(&name).then(|| name.to_string())
    })
}

fn role_name_for_route(route: &RouteTarget) -> &'static str {
    match route {
        RouteTarget::LightReasoner => "light_reasoner",
        RouteTarget::LightCoder => "light_coder",
        RouteTarget::CloudFast => "cloud_fast",
        RouteTarget::CloudMini => "cloud_mini",
        RouteTarget::CloudReasoner => "cloud_reasoner",
        RouteTarget::CloudCoder => "cloud_coder",
    }
}

/// Resolve a role name from the failover chain to a concrete endpoint or cloud slug.
fn resolve_role(role_name: &str, state: &RoutingState) -> Option<ResolvedRole> {
    // Global local-only backstop: a cloud role never resolves when cloud is
    // disabled, so no cloud dispatch can happen regardless of what any failover
    // chain contains. This complements the per-chain cloud-strip; together they
    // make "local_only = no cloud inference, ever" true at two layers.
    if state.config.local_only && codex_routing::project_config::is_cloud_role(role_name) {
        return None;
    }
    match role_name {
        "light_reasoner" => {
            if state.config.reasoner.enabled {
                Some(ResolvedRole::Local(state.config.reasoner.clone()))
            } else {
                None
            }
        }
        "light_reasoner_backup" => {
            if state.config.reasoner_backup.enabled {
                Some(ResolvedRole::Local(state.config.reasoner_backup.clone()))
            } else {
                None
            }
        }
        "light_coder" => {
            if state.config.light_coder.enabled {
                Some(ResolvedRole::Local(state.config.light_coder.clone()))
            } else {
                None
            }
        }
        "compactor" => {
            if state.config.compactor.enabled {
                Some(ResolvedRole::Local(state.config.compactor.clone()))
            } else {
                None
            }
        }
        // Cloud roles — resolve via weighted selection, dispatch by provider
        "cloud_fast" | "cloud_mini" | "cloud_reasoner" | "cloud_coder" => {
            match pick_cloud_model_with_provider(&state.project_config, role_name) {
                Some((slug, provider)) if provider == "anthropic" => {
                    Some(ResolvedRole::ClaudeExec { model: slug })
                }
                Some((slug, _)) => Some(ResolvedRole::Cloud(slug)),
                None => None,
            }
        }
        // Classifier itself — not a useful failover target
        "classifier" => None,
        _ => None,
    }
}

/// Classify by walking the `classification` failover chain, trying each role's
/// local endpoint until one returns a usable result. Cloud roles are skipped
/// (classification runs on the local Ollama pool, not a cloud provider) and are
/// stripped entirely in local_only. If the whole chain is exhausted, fall back
/// to a default route: `LightCoder` in local_only (nothing to escalate to),
/// otherwise `CloudCoder`.
#[allow(clippy::too_many_arguments)]
async fn classify_via_chain(
    prompt_text: &str,
    tool_names: &[&str],
    recent_tool_call_count: usize,
    recent_turn_count: usize,
    state: &RoutingState,
    routing_profile: &str,
    codebase_context: &str,
) -> codex_routing::classifier::ClassifyResult {
    let mut chain = state
        .project_config
        .failover_chain("classification")
        .to_vec();
    if state.config.local_only {
        chain.retain(|role| !codex_routing::project_config::is_cloud_role(role));
    }

    for role in &chain {
        // Map the chain role to a local classifier endpoint. `classifier` isn't
        // a `resolve_role` target (it's not a failover destination for normal
        // requests), so handle it explicitly; other local roles resolve
        // normally. Cloud roles can't classify here, so they're skipped.
        let endpoint = match role.as_str() {
            "classifier" if state.config.classifier.enabled => {
                Some(state.config.classifier.clone())
            }
            "classifier" => None,
            other => match resolve_role(other, state) {
                Some(ResolvedRole::Local(ep)) => Some(ep),
                _ => None,
            },
        };
        let Some(endpoint) = endpoint else { continue };

        if let Some(result) = codex_routing::classifier::classify_with_endpoint(
            prompt_text,
            tool_names,
            recent_tool_call_count,
            recent_turn_count,
            &endpoint,
            &state.pool,
            routing_profile,
            codebase_context,
        )
        .await
        {
            if role != "classifier" {
                info!(via = %role, route = ?result.route, "Classified via failover role");
            }
            return result;
        }
        warn!(role = %role, "Classification role failed; advancing classification chain");
    }

    let route = if state.config.local_only {
        RouteTarget::LightCoder
    } else {
        RouteTarget::CloudCoder
    };
    info!(
        ?route,
        "Classification chain exhausted; using default route"
    );
    codex_routing::classifier::ClassifyResult {
        route,
        tools_potential: true,
        reason: "classification chain exhausted".to_string(),
    }
}

/// Check if a prompt is a compaction request.
///
/// Two recognizers:
///   1. The legacy `<<<LOCAL_COMPACT>>>` sentinel — kept for callers that
///      explicitly want our specialized local pipeline.
///   2. The opening line of Codex's built-in compaction prompt template
///      (`"CONTEXT CHECKPOINT COMPACTION"`, see
///      `core/templates/compact/prompt.md`). When Codex's `run_compact_task`
///      fires (auto-compact at token limit, or manual `/compact`), the
///      synthesized user message starts with that line. Detecting it lets
///      our specialized pipeline take over for local sessions instead of
///      asking the local model to do the whole summarization itself.
fn is_compaction_request(prompt: &Prompt) -> bool {
    let text = extract_last_message(prompt);
    text.contains("<<<LOCAL_COMPACT>>>") || text.contains("CONTEXT CHECKPOINT COMPACTION")
}

/// Route a request: local model, cloud with model override, or default.
///
/// Called from ModelClientSession::stream() on every model API call.
pub(crate) async fn route_request(prompt: &Prompt) -> RouteResult {
    // Compaction requests: run the full compaction pipeline locally.
    // Detects <<<LOCAL_COMPACT>>> sentinel and runs normalize → chunk →
    // extract → merge → render on local Ollama. No proxy needed.
    if is_compaction_request(prompt) {
        if let Some(state) = get_routing_state().await.as_ref() {
            // Resolve the compaction endpoint by walking the `compaction`
            // failover chain (cloud-stripped in local_only), taking the first
            // available local role — e.g. compactor → light_reasoner. Falls back
            // to the compactor/Coder if the chain yields nothing. The chain is
            // the control surface; local_only just removes cloud entries.
            let mut compaction_chain = state.project_config.failover_chain("compaction").to_vec();
            if state.config.local_only {
                compaction_chain.retain(|role| !codex_routing::project_config::is_cloud_role(role));
            }
            let endpoint: OllamaEndpoint = compaction_chain
                .iter()
                .find_map(|role| match resolve_role(role, state) {
                    Some(ResolvedRole::Local(ep)) if ep.enabled => Some(ep),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    if state.config.compactor.enabled {
                        state.config.compactor.clone()
                    } else {
                        state.config.light_coder.clone()
                    }
                });
            if endpoint.enabled {
                info!(
                    model = %endpoint.model,
                    "Compaction request detected — running full pipeline locally"
                );

                // Phase 3: feed the compaction pipeline through the same
                // role-aware trimmer used for per-request routing. The
                // compactor inherits all the dedup/stale-removal/error-
                // preservation logic for free, and we stop maintaining two
                // parallel cleanup paths.
                let project_instructions = extract_project_instructions(prompt);
                let trim_input = codex_routing::trim::TrimInput {
                    items: &prompt.input,
                    system_prompt: &prompt.base_instructions.text,
                    user_instructions: project_instructions.as_deref(),
                    flavor: endpoint.flavor,
                    // Compaction summarizes history; leave its system prompt
                    // alone so the summary keeps full instruction fidelity.
                    system_budget_pct: 0,
                    // Compaction has no loop-guard prelude to suppress.
                    suppress_loop_alerts: false,
                };
                // Resolve the compactor's real window the SAME way the coder path
                // does. The compactor's `trim_budget` is usually UNSET → 0, which
                // means AUTO ("the budget logic resolves it"). Passing that 0 raw to
                // `trim_for_local` gave it a 0-token budget, so it truncated ALL
                // tool-output content to nothing — the compaction pipeline then saw
                // "No compactable content" and returned an empty summary, and the
                // model, told nothing had been done, restarted the task from scratch.
                // Resolving to the true window feeds the pipeline the actual work.
                let compact_server_ctx = resolve_server_ctx(&endpoint, state).await;
                let compact_window = effective_window(
                    endpoint.trim_budget,
                    endpoint.output_reserve,
                    compact_server_ctx,
                );
                let trimmed = codex_routing::trim::trim_for_local(&trim_input, compact_window);
                info!(
                    trim_summary = %trimmed.summary.to_log_line(),
                    compact_window,
                    "Trimmed transcript for compaction input"
                );
                // The compaction pipeline expects bare `{role, content}`
                // dicts — extract those from the trimmed messages and drop
                // any tool-call shapes the compactor can't ingest.
                let items: Vec<serde_json::Value> = trimmed
                    .messages
                    .iter()
                    .filter(|m| m.get("content").and_then(|c| c.as_str()).is_some())
                    .cloned()
                    .collect();

                // The `# Current Request` anchor for the handoff must be the REAL
                // prior user task — NOT the compaction directive. When the harness
                // fires compaction, the newest user message IS the directive
                // ("<<<LOCAL_COMPACT>>> Summarize the thread for continuation…");
                // using it here hijacks the resuming model into "summarize" and
                // abandons the real work (the observed session-break). Walk back to
                // the last genuine user task instead; empty if none (the handoff
                // then omits the section and the verbatim recent tail carries it).
                let current_request = extract_real_user_task(prompt).unwrap_or_default();

                let compaction_config = codex_routing::compaction::CompactionConfig::default();

                match codex_routing::compaction::compact_transcript(
                    &items,
                    &current_request,
                    &state.pool,
                    &endpoint,
                    &compaction_config,
                )
                .await
                {
                    Ok(summary) => {
                        info!(summary_len = summary.len(), "Compaction pipeline complete");
                        // Return the compacted summary as a text response
                        let response = codex_routing::local_dispatch::OllamaTextResponse {
                            content: summary,
                            model: endpoint.model.clone(),
                            input_tokens: 0, // Pipeline doesn't track total
                            output_tokens: 0,
                        };
                        return RouteResult::Local(ollama_response_to_stream(response));
                    }
                    Err(e) => {
                        warn!(error = %e, "Compaction pipeline failed, falling back to cloud");
                        // Fall through to normal routing
                    }
                }
            }
        }
    }

    let state = match get_routing_state().await.as_ref() {
        Some(s) => s,
        None => {
            if local_only_env() {
                return local_only_error("classifier endpoint is not reachable");
            }
            return RouteResult::Default;
        }
    };

    // Extract the last user message for classification
    let prompt_text = extract_last_message(prompt);
    if prompt_text.is_empty() {
        return cloud_fallback_or_local_error(state, "request had no user message text");
    }

    // Classify the request to pick the route (and thus the failover chain).
    // The classifier itself fails over across the `classification` chain — see
    // `classify_via_chain`, which is cloud-stripped in local_only. A cache
    // short-circuits repeat classifications. `local_only` no longer bypasses
    // the classifier: it just means the chosen route resolves to local roles
    // (cloud is stripped from every chain), so the configured roles/failovers
    // remain the control surface even with cloud disabled.
    let classification = {
        // Extract tool names (just names, not full schemas)
        let tool_names: Vec<&str> = prompt.tools.iter().map(|t| t.name()).collect();

        // Count recent tool calls from conversation history
        let (tool_call_count, turn_count) = count_recent_activity(prompt);

        // G8: Check classifier cache — skip the classifier LLM call if confident
        let cached_classification = state
            .classify_cache
            .lock()
            .ok()
            .and_then(|cache| cache.try_cached());

        if let Some(cached) = cached_classification {
            info!(
                route = ?cached.route,
                reason = %cached.reason,
                "Using cached classification (skipping classifier LLM)"
            );
            cached
        } else {
            let routing_profile = state
                .feedback
                .lock()
                .map(|f| f.profile_context())
                .unwrap_or_default();
            let codebase_ctx = state.codebase_context.classifier_context();

            // G14: Add budget pressure to classifier context
            let budget_ctx = state.budget.pressure_context();
            let full_context = if budget_ctx.is_empty() {
                codebase_ctx.clone()
            } else {
                format!("{codebase_ctx}\n{budget_ctx}")
            };

            let result = classify_via_chain(
                &prompt_text,
                &tool_names,
                tool_call_count,
                turn_count,
                state,
                &routing_profile,
                &full_context,
            )
            .await;

            // Record in cache for future requests
            if let Ok(mut cache) = state.classify_cache.lock() {
                cache.record(&result);
            }

            result
        }
    };

    // G14: Hard-block primary if budget is critical (deterministic, not LLM)
    let route =
        if state.budget.should_block_primary() && classification.route == RouteTarget::CloudCoder {
            warn!(
                primary_used = state.budget.primary_used(),
                "Primary budget critical — downgrading cloud_coder to cloud_reasoner"
            );
            RouteTarget::CloudReasoner
        } else {
            classification.route
        };

    // (No reasoner→coder override: every local role is a full coder now, so a
    // request classified as LightReasoner is served with the full tool set and
    // can edit just like the coder — only its config and text-completion
    // behavior differ. Nothing to "rescue" by switching the route.)

    // --- Determine the failover chain for this route ---
    let chain_name = chain_name_for_route(&route);
    let initial_role = role_name_for_route(&route);
    let mut chain = state.project_config.failover_chain(chain_name).to_vec();
    if state.config.local_only {
        chain.retain(|role| !codex_routing::project_config::is_cloud_role(role));
        // Stripping cloud can collapse a chain to a single local entry (e.g.
        // reasoning = [light_reasoner, cloud_*] -> [light_reasoner]), turning any
        // single failure into a dead turn. Coder and reasoner now run the same
        // full-tool path, so each is a valid backup for the other. Append the
        // standard local roles (deduped) so a local_only chain always has a
        // fallback. Unconfigured roles are simply skipped when the chain walks.
        for backup in ["light_coder", "light_reasoner"] {
            if !chain.iter().any(|r| r == backup) {
                chain.push(backup.to_string());
            }
        }
    }
    let behavior = &state.project_config.failover.behavior;

    // Walk the failover chain starting from the classified route.
    let mut current_role = initial_role.to_string();
    let mut attempt: u32 = 0;

    loop {
        // Resolve the current role to a concrete endpoint or cloud slug
        let resolved = resolve_role(&current_role, state);

        match resolved {
            Some(ResolvedRole::Cloud(slug)) => {
                info!(
                    route = %current_role,
                    model = %slug,
                    reason = %classification.reason,
                    "Routing to cloud model (override)"
                );
                state.usage.set_current_route(&current_role, &slug);
                return RouteResult::CloudOverride {
                    slug,
                    failover_ctx: Some(CloudFailoverCtx {
                        role_name: current_role.clone(),
                        chain_name: chain_name.to_string(),
                        chain: chain.clone(),
                        behavior: behavior.clone(),
                    }),
                };
            }
            Some(ResolvedRole::ClaudeExec { model }) => {
                info!(
                    route = %current_role,
                    model = %model,
                    reason = %classification.reason,
                    "Routing to Anthropic via Claude CLI"
                );
                state.usage.set_current_route(&current_role, &model);

                let prompt_text = extract_last_message(prompt);
                let claude_binary = state.project_config.cli.claude.clone();
                let cwd = std::env::current_dir().ok();

                // Use conversation-level session key for context resumption
                let session_key = "main";
                let resume_id = state.claude_sessions.get_session(session_key);

                let result = codex_routing::claude_cli::invoke_claude(
                    &claude_binary,
                    &model,
                    &prompt_text,
                    resume_id.as_deref(),
                    cwd.as_deref(),
                )
                .await;

                match result {
                    Ok(cli_result) => {
                        // Track session for resumption
                        if let Some(ref sid) = cli_result.session_id {
                            state.claude_sessions.set_session(session_key, sid);
                        }

                        // Record usage
                        state.usage.record(
                            &cli_result.model,
                            cli_result.input_tokens,
                            cli_result.output_tokens,
                        );

                        let response = OllamaTextResponse {
                            content: cli_result.content,
                            model: cli_result.model,
                            input_tokens: cli_result.input_tokens,
                            output_tokens: cli_result.output_tokens,
                        };
                        return RouteResult::Local(ollama_response_to_stream(response));
                    }
                    Err(e) => {
                        warn!(error = %e, "Claude CLI failed");
                        let action = failover::decide_action(
                            FailureType::ModelUnavailable,
                            &current_role,
                            chain_name,
                            &chain,
                            attempt,
                            None,
                            behavior,
                        );
                        match action {
                            FailoverAction::NextInChain { model_role, reason } => {
                                info!(from = %current_role, to = %model_role, reason = %reason, "Failover from Claude CLI");
                                current_role = model_role;
                                attempt = 0;
                                continue;
                            }
                            FailoverAction::RetrySame {
                                wait,
                                attempt: next,
                            } => {
                                tokio::time::sleep(wait).await;
                                attempt = next;
                                continue;
                            }
                            _ => {
                                return cloud_fallback_or_local_error(
                                    state,
                                    "Claude CLI dispatch failed and no failover chain entry resolved",
                                );
                            }
                        }
                    }
                }
            }
            Some(ResolvedRole::Local(endpoint)) => {
                state
                    .usage
                    .set_current_route(&current_role, &endpoint.model);
                // The failover walk may have landed on a different role than the
                // classifier picked (e.g. a cloud route, unresolvable in
                // local_only, falls back to light_coder). Derive the route from
                // the role actually being served so tool attachment is correct —
                // otherwise a coder serving a cloud-classified request runs with
                // no tools and can never write a file.
                let effective_route = route_target_for_role(&current_role).unwrap_or(route);
                // The role being served — not the route enum — decides the tool
                // set and completion behavior inside try_local_model (via
                // `local_role_profile`). Coder and reasoner run the same path.
                let result = try_local_model(
                    prompt,
                    &endpoint,
                    &effective_route,
                    &classification,
                    &current_role,
                    state,
                )
                .await;

                match result {
                    Ok(stream) => return RouteResult::Local(stream),
                    Err(failure_type) => {
                        // Local model failed — consult the failover executor
                        let action = failover::decide_action(
                            failure_type,
                            &current_role,
                            chain_name,
                            &chain,
                            attempt,
                            None, // no retry-after for local models
                            behavior,
                        );

                        match action {
                            FailoverAction::RetrySame {
                                wait,
                                attempt: next_attempt,
                            } => {
                                info!(
                                    model = %current_role,
                                    wait_ms = wait.as_millis() as u64,
                                    attempt = next_attempt,
                                    "Failover: retrying same local model"
                                );
                                tokio::time::sleep(wait).await;
                                attempt = next_attempt;
                                continue;
                            }
                            FailoverAction::NextInChain { model_role, reason } => {
                                info!(
                                    from = %current_role,
                                    to = %model_role,
                                    reason = %reason,
                                    "Failover: walking to next model in chain"
                                );
                                current_role = model_role;
                                attempt = 0;
                                continue;
                            }
                            FailoverAction::HardFail { reason } => {
                                warn!(reason = %reason, "Failover: hard fail");
                                return cloud_fallback_or_local_error(
                                    state,
                                    &format!("local model failover hard-failed: {reason}"),
                                );
                            }
                            FailoverAction::ChainExhausted { chain_name } => {
                                warn!(chain = %chain_name, "Failover: chain exhausted, using default");
                                return cloud_fallback_or_local_error(
                                    state,
                                    &format!("local failover chain '{chain_name}' exhausted"),
                                );
                            }
                        }
                    }
                }
            }
            None => {
                // Role can't be resolved: a cloud role under local_only (already
                // stripped from the chain, but defensive) or an undefined/disabled
                // role. This is NOT a server 404 — use RoleUnresolvable so it isn't
                // logged as a "model not found (config error?)" name problem.
                let action = failover::decide_action(
                    FailureType::RoleUnresolvable,
                    &current_role,
                    chain_name,
                    &chain,
                    0,
                    None,
                    behavior,
                );
                match action {
                    FailoverAction::NextInChain { model_role, reason } => {
                        info!(
                            from = %current_role,
                            to = %model_role,
                            reason = %reason,
                            "Failover: role unresolvable, walking chain"
                        );
                        current_role = model_role;
                        attempt = 0;
                        continue;
                    }
                    _ => {
                        return cloud_fallback_or_local_error(
                            state,
                            "no role in failover chain could be resolved",
                        );
                    }
                }
            }
        }
    }
}

/// Build the "new response" digest handed to the course-change reasoner: the
/// coder's reasoning plus the tool call it is about to make, each bounded so the
/// check stays cheap. Char-boundary safe.
fn summarize_new_action(reasoning: &str, tool_calls: &[serde_json::Value]) -> String {
    fn clip(s: &str, max_chars: usize) -> String {
        if s.chars().count() <= max_chars {
            s.to_string()
        } else {
            let taken: String = s.chars().take(max_chars).collect();
            format!("{taken}…")
        }
    }
    let calls: Vec<String> = tool_calls
        .iter()
        .map(|c| {
            let f = c.get("function");
            let name = f
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("?");
            let args = f
                .and_then(|f| f.get("arguments"))
                .map(|a| match a.as_str() {
                    Some(s) => s.to_string(),
                    None => a.to_string(),
                })
                .unwrap_or_default();
            format!("{name}({})", clip(&args, 300))
        })
        .collect();
    let flat: String = reasoning.split_whitespace().collect::<Vec<_>>().join(" ");
    format!(
        "REASONING: {}\nTOOL CALL: {}",
        clip(&flat, 1200),
        calls.join("; ")
    )
}

/// One-clause description of the loop the coder is stuck in, derived from which
/// guard directive is present in the guidance — context for the course-change
/// reasoner.
fn loop_summary_from_guidance(guidance: &str) -> String {
    let s = if guidance.contains("[GATHERING WITHOUT ACTING]") {
        "reading/searching repeatedly without making any change"
    } else if guidance.contains("[STUCK — CIRCLING THE SAME PLACES]") {
        "circling the same files/URLs without progress"
    } else if guidance.contains("[NO PROGRESS — DIAGNOSE") {
        "retrying the same failing goal without diagnosing it"
    } else if guidance.contains("[HARNESS — STUCK") {
        "stuck in a loop it has repeatedly ignored"
    } else {
        "repeating the same action without progress"
    };
    s.to_string()
}

/// Try executing a request on a local Ollama model.
/// Returns Ok(ResponseStream) on success, Err(FailureType) on failure.
///
/// This is the only entry point that hands a transcript to a local model. It
/// runs `trim_for_local` to produce a role-aware, deduplicated prompt; the
/// same trimming logic is also used by the compaction pipeline. Local models
/// are not gimped in any mode-specific way — the only thing that varies is
/// routing (whether we even reach this function vs. dispatching to cloud).
async fn try_local_model(
    prompt: &Prompt,
    endpoint: &OllamaEndpoint,
    route: &RouteTarget,
    classification: &codex_routing::classifier::ClassifyResult,
    role: &str,
    state: &RoutingState,
) -> Result<ResponseStream, FailureType> {
    // Authoritative record of what ACTUALLY served this request. The turn's
    // nominal model (e.g. "gpt-5.5") still shows in `turn_context` and otel spans
    // because the local path deliberately suppresses `ServerModel` to avoid
    // tripping Codex's server-model-mismatch reroute (see
    // `ollama_tool_response_to_stream`). So this event — and the status indicator,
    // fed by `usage.set_current_route` — are where the real local model appears.
    info!(
        effective_model = %endpoint.model,
        role = %role,
        route = ?route,
        reason = %classification.reason,
        "Routing to local model (free)"
    );

    let model_name = endpoint.model.clone();

    // Estimate what the cloud model would have processed (savings metric).
    let pre_trim_tokens = estimate_prompt_tokens(prompt);
    state.usage.record_savings(pre_trim_tokens as u64);

    // Pull AGENTS.md / CLAUDE.md content out of the conversation so it can
    // be pinned into the persistent context block at the top of the prelude
    // (always visible, never aged out, distinct from rolling content). It's
    // also still preserved as a user message via the trim's user-message rule.
    let project_instructions = extract_project_instructions(prompt);

    // Trim the transcript with role-aware semantic compression.
    // Build the tool set up front so we can reserve its token budget: the
    // schemas are sent to the model but added *after* trimming, so the trimmer
    // must subtract them from trim_budget or the real prompt overflows the window.
    // Every local role is a full coder: same tools, same ability to edit. The
    // only behavioral difference is whether a text-only turn completes the turn
    // (`role_text_is_product`); "room to explore" is per-endpoint config. Coder
    // and reasoner both run THIS path with the full tool set.
    let text_is_product = role_text_is_product(role);
    let local_tools = build_local_tools(
        prompt,
        endpoint,
        LIGHT_CODER_TOOL_NAMES,
        /* include_edit_tools */ true,
    );
    let use_tools = !local_tools.is_empty();
    let tool_reserve_tokens = if local_tools.is_empty() {
        0
    } else {
        codex_routing::metrics::estimate_tokens(
            &serde_json::to_string(&local_tools).unwrap_or_default(),
        )
    };

    // Local models get OUR OWN concise base prompt, NOT Codex's ~351-line
    // apply_patch-heavy one. The Codex prompt teaches `apply_patch` so heavily a
    // 9B emits it no matter what shorter hints we add, and the bulk also costs
    // ~3–5k tokens every turn. Ours is small and write_file-first, so there's
    // nothing to summarize (the old Stage-2 system-prompt compression is moot).
    // See codex_routing::prompt_local (the text is a portable .md asset).
    let system_prompt_ref: &str = codex_routing::prompt_local::LOCAL_CODER_SYSTEM_PROMPT;

    // Inbound half of the write_file massage: re-present the `shell` base64 writes
    // we synthesized last turn back as `write_file`, so the model sees only its own
    // tool and so trim's state-extraction/file-pinning recognize the writes. Done
    // before trim consumes the transcript.
    let represented_input = represent_web_search_names(represent_shell_writes(&prompt.input));

    // Post-course-change grace: if the reasoner confirmed a genuine pivot on a
    // recent turn, spend one turn of grace now — trim will drop this turn's loop
    // nudges so the new approach can run unobstructed. Consumed ONCE per model
    // return (here, before the retry loop), keyed by the task text.
    let grace_task = extract_last_message(prompt);
    let suppress_loop = codex_routing::reasoned_guidance::consume_loop_grace(&grace_task);

    let trim_input = codex_routing::trim::TrimInput {
        items: &represented_input,
        system_prompt: system_prompt_ref,
        user_instructions: project_instructions.as_deref(),
        flavor: endpoint.flavor,
        system_budget_pct: state.project_config.routing.system_budget_pct,
        suppress_loop_alerts: suppress_loop,
    };
    // Size the budget from the server's REAL context window (detected from
    // /props, cached) minus reserves — not the hand-set trim_budget. Then scale by
    // the model's LEARNED real÷estimate ratio (the chars/4 estimate undercounts
    // dense content ~1.8–2.8×). trim_budget is now an optional cap (0 = full
    // window). Falls back to trim_budget if the window can't be detected.
    let server_ctx = resolve_server_ctx(endpoint, state).await;
    // Publish the coder's real window so the harness can override the model's
    // (over-large) advertised context window and let native auto-compaction fire.
    if let Some(n) = server_ctx {
        CODER_CONTEXT_WINDOW.store(n, std::sync::atomic::Ordering::Relaxed);
    }
    let window = effective_window(endpoint.trim_budget, endpoint.output_reserve, server_ctx);
    // Tool schemas tokenize at the model's learned ratio and are NOT part of
    // trim's chars/4 estimate. Reserve them in REAL tokens out of the real window
    // FIRST, then calibrate the remainder to estimate space. The old path
    // subtracted their estimate INSIDE the estimate-space budget (so it got
    // ÷ DEFAULT'd too), under-reserving them ~ratio× and letting the real prompt
    // overflow. See docs/spec/local-coder-massaging §12.
    let ratio = observed_token_ratio(&endpoint.model);
    let real_tool_reserve = (tool_reserve_tokens as f64 * ratio) as usize;
    let fit_budget =
        calibrated_trim_budget(&endpoint.model, window.saturating_sub(real_tool_reserve));
    let trimmed = codex_routing::trim::trim_for_local(&trim_input, fit_budget);
    info!(
        trim_summary = %trimmed.summary.to_log_line(),
        server_ctx = server_ctx.unwrap_or(0),
        window,
        fit_budget,
        real_tool_reserve,
        observed_token_ratio = ratio,
        "Trimmed transcript for local model"
    );

    // If the trimmed transcript still exceeds the fit budget, summarize the bulk
    // (older prelude or the active turn's own middle) via the compaction pipeline
    // and replace it with a single summary message. Cached by hash so we don't
    // recompact identical history each turn.
    let trimmed = maybe_inline_compact(trimmed, fit_budget, state).await;

    // ── End-of-prompt guidance ────────────────────────────────────────────────
    // Everything the harness wants the model to DO or KNOW this turn is assembled
    // into `guidance_parts` and injected as the FINAL message below — NOT the system
    // prompt — because a small model attends to the END of the context far more than
    // the beginning (recency). The base prompt stays as the stable system frame.
    let mut guidance_parts: Vec<String> = Vec::new();
    // The trim's prelude — now just the loop directive, the AGENTS.md pin, and the
    // patch-rewrite nudge. (World-state + file/manifest pins were dropped; the LLM
    // summary and the verbatim transcript carry file state now.)
    if !trimmed.guidance.trim().is_empty() {
        guidance_parts.push(trimmed.guidance.clone());
    }

    // Reasoned guidance (assist #1 — PLAN FIRST): on a new user task, engage the
    // light reasoner FIRST to draft a small-step plan for this low-context model,
    // and pin it at the very top of the coder's prompt for the whole turn. The
    // plan is cached per task, so the reasoner runs ONCE per user turn (not per
    // step) — the deliberate ceiling, since it shares the local GPU with the
    // coder. Coder routes only: a reasoner's own text IS its product, so we never
    // plan for the planner.
    // `fresh_plan` carries the plan out to the stream builder so it's persisted to
    // the rollout (as a Reasoning item) once per task, alongside the coder's own
    // per-turn reasoning — see `ollama_tool_response_to_stream`.
    let mut fresh_plan: Option<String> = None;
    // How many times the completion critic (assist #2) has sent the model back
    // this turn — bounds it so it can't block a "done" claim forever.
    let mut critic_blocks: u32 = 0;
    if !text_is_product && state.config.reasoner.enabled {
        let task = extract_last_message(prompt);
        if let Some((plan, fresh)) = codex_routing::reasoned_guidance::plan_for_task(
            state.pool.as_ref(),
            &state.config.reasoner,
            &task,
            &state.project_config.search.brave_api_key,
        )
        .await
        {
            // Surface the reasoner's plan in the TUI once per task (the same
            // push_nudge channel the loop guards use), so plan-first is visible
            // instead of hiding in the system prompt. `fresh` is only true on the
            // draft step, not the cached reuses across the rest of the turn.
            if fresh {
                state.push_nudge(format!("Reasoned guidance — plan-first:\n{plan}"));
                fresh_plan = Some(plan.clone());
            }
            guidance_parts.push(plan);
        }
    }

    // Surface the loop guards so the user sees the coaching happen, in
    // escalation order:
    //  - context reset → loop excised from context + reframed (nudge+block ignored)
    //  - exact repeat → STOP directive
    //  - thrash (same goal, varying commands, still failing) → forced diagnosis
    //
    // ALSO log it (not just push_nudge → TUI): the guard notices go to the TUI
    // queue, which never reaches the tracing log — so from the logs alone a guard
    // looks like it "never fired" even when it fires every turn. (That gap led to
    // a real misdiagnosis: a loop where context-reset WAS firing repeatedly read as
    // "escalation never fires" because we were grepping the log for TUI-only text.)
    let rep_count = trimmed.repetition_count.map(|c| c as i64).unwrap_or(-1);
    // Set to a short human description of the loop when a repetition/loop guard
    // fires. When set, thrash → probe → reasoned guidance kicks off below: run the
    // read-only lint probe and hand its result to the reasoner to author the
    // coder's next step, grounded in the ACTUAL workspace errors.
    let mut loop_summary: Option<&str> = None;
    if trimmed.guidance.contains("[HARNESS — STUCK; LOOP REMOVED") {
        warn!(
            guard = "context_reset",
            repetition_count = rep_count,
            "Loop guard fired: excised the loop from context and reframed (advisory + STOP both ignored)"
        );
        state.push_nudge(
            "Context-reset guard fired — model ignored the nudges and the hard block; excised the loop from its context and reframed it to the unsolved step".to_string(),
        );
        loop_summary =
            Some("repeating a loop it was already warned about and had excised from context");
    } else if trimmed.guidance.contains("[STOP — REPETITION DETECTED]") {
        info!(
            guard = "repetition_stop",
            repetition_count = rep_count,
            "Loop guard fired: injected a STOP directive (identical call repeated)"
        );
        state.push_nudge(
            "Repetition guard fired — model was repeating an identical tool call; injected a stop directive".to_string(),
        );
        loop_summary = Some("repeating the exact same tool call with the same arguments");
    } else if trimmed.guidance.contains("[NO PROGRESS — DIAGNOSE") {
        warn!(
            guard = "forced_diagnosis",
            repetition_count = rep_count,
            "Loop guard fired: forcing a diagnosis before the next action (thrash, still failing)"
        );
        state.push_nudge(
            "Forced-diagnosis guard fired — model was thrashing; requiring it to read the failure and state the root cause before acting".to_string(),
        );
        loop_summary = Some("thrashing on the same goal with different commands, and still failing");
    } else if trimmed
        .guidance
        .contains("[STUCK — CIRCLING THE SAME PLACES]")
    {
        warn!(
            guard = "tunnel_vision",
            repetition_count = rep_count,
            "Loop guard fired: footprint stopped expanding (circling a fixed set of targets)"
        );
        state.push_nudge(
            "Tunnel-vision guard fired — model was circling the same targets without touching anything new; forced a step-back".to_string(),
        );
        loop_summary = Some("circling the same files/targets without touching anything new");
    } else if trimmed.guidance.contains("[GATHERING WITHOUT ACTING]") {
        info!(
            guard = "read_without_write",
            repetition_count = rep_count,
            "Loop guard fired: many reads, no writes — nudged to act on what it has"
        );
        state.push_nudge(
            "Read-without-write guard fired — model was searching/fetching/reading without ever acting; nudged it to make a concrete change".to_string(),
        );
        loop_summary =
            Some("reading, searching, and fetching repeatedly without making any concrete change");
    }
    // Reasoned Guidance — CONTEXT REBUILD ON FLAIL (the excise, done right). The excise
    // is the TOP escalation: the model ignored every softer nudge, and the canned
    // reframe points it back at the now-STALE transcript. Instead, rebuild a clean
    // working context from FRESH ground truth — the repeated failing action (from the
    // trim layer) + the files as they are on disk NOW — and let the reasoner author the
    // one next step. On success this SUPERSEDES the generic dirty-only redirect below;
    // on any failure (reasoner off / silent / no signal) the canned excise stands.
    if trimmed.guidance.contains("[HARNESS — STUCK; LOOP REMOVED")
        && state.config.reasoner.enabled
    {
        let task = extract_last_message(prompt);
        let gt = gather_loop_ground_truth(prompt, trimmed.repeated_action.clone()).await;
        if gt.has_signal()
            && let Some(rebuild) = codex_routing::reasoned_guidance::rebuild_context_from_loop(
                state.pool.as_ref(),
                &state.config.reasoner,
                &task,
                &gt,
            )
            .await
        {
            warn!(
                guard = "context_rebuild",
                repetition_count = rep_count,
                "Excise upgraded: reasoner rebuilt a clean working context from fresh ground truth (repeated action + live files)"
            );
            state.push_nudge(
                "Context rebuild — the reasoner rebuilt your working context from the ACTUAL files on disk and chose the next step".to_string(),
            );
            guidance_parts.push(rebuild);
            loop_summary = None; // supersede the generic dirty-only redirect this turn
        }
    }

    // Reasoned Guidance assist #3 — THRASH → PROBE → REASONED GUIDANCE. A canned
    // loop directive is soft prompt text a 9B routinely ignores. When ANY of the
    // five loop guards fired, run the read-only lint/syntax PROBE for ground truth,
    // then ask the reasoner to author the coder's NEXT INSTRUCTION grounded in the
    // real errors — the exact file:line to fix, or (on a clean probe) reasoning past
    // syntax toward the actual cause. Fallbacks in escalation order: reasoner output
    // → raw probe grounding (dirty only) → the canned directive already in the
    // prompt. Bounded per task inside `redirect_from_loop`.
    if let Some(loop_summary) = loop_summary {
        // THRASH → GROUND TRUTH → REASONED GUIDANCE. The reasoner is grounded on the
        // FRESH bundle (`gather_loop_ground_truth`): the repeated failing action + the
        // dirty-only lint + the actual files on disk. The repeated action is the signal
        // the old dirty-only gate DROPPED — an action-loop (`cat` a directory,
        // re-search the same query) has a CLEAN lint, so the reasoner was never called
        // and only a bare nudge fired, which the model ignores. We now call the reasoner
        // whenever there is ANY real signal (`has_signal()`), and NEVER on nothing (a
        // groundless reasoner once hallucinated "add an X-API-Key header") or on the
        // model's own claims.
        let gt = gather_loop_ground_truth(prompt, trimmed.repeated_action.clone()).await;
        let mut grounded = false;
        if state.config.reasoner.enabled && gt.has_signal() {
            let task = extract_last_message(prompt);
            let evidence = recent_evidence(prompt, 2800);
            if let Some(redirect) = codex_routing::reasoned_guidance::redirect_from_loop(
                state.pool.as_ref(),
                &state.config.reasoner,
                &task,
                loop_summary,
                &gt,
                &evidence,
            )
            .await
            {
                warn!(
                    guard = "reasoner_redirect",
                    "Loop redirect: reasoner authored the coder's next step from fresh ground truth (repeated action + lint + live files)"
                );
                state.push_nudge(format!(
                    "Reasoner redirect — you're looping; the reasoner read the ground truth and chose your next step:\n{redirect}"
                ));
                guidance_parts.push(redirect);
                grounded = true;
            }
        }
        // Reasoner unavailable / silent: if the workspace fails its syntax floor, still
        // hand over the exact file:line. A clean probe with no repeated signal → the
        // detector's canned directive already in the prompt stands (we add nothing).
        if !grounded
            && let Some(lint_text) = gt.lint_digest.clone()
        {
            warn!(
                guard = "stuck_grounding",
                "Probe grounding (reasoner unavailable): workspace fails its syntax floor — attached the exact file:line"
            );
            state.push_nudge(
                "Probe grounding fired — the workspace fails its own syntax check; handed the model the exact file:line to fix".to_string(),
            );
            guidance_parts.push(format!(
                "[GROUND TRUTH — the repo's own checks fail] Fix these exact problems; go to the reported line, do not rewrite whole files:\n{lint_text}"
            ));
        }
    }

    // Assemble the final guidance block; injected as the LAST message before the
    // coder generates (see the tool-capable send path below), so a recency-biased
    // small model reads it right before it acts.
    let end_guidance = guidance_parts.join("\n\n");
    // Did a loop/repetition guard put a directive in the guidance this turn? If
    // so, the coder's next response is a candidate for a course-change check: it
    // saw a "you're looping" nudge, and if it now genuinely pivots we want to
    // pause the guards rather than re-bury the pivot. (During an active grace
    // window the alert is already suppressed, so this is false — we don't re-fire
    // the check while the pivot is still running.)
    let loop_guard_fired = [
        "[STOP — REPETITION DETECTED]",
        "[GATHERING WITHOUT ACTING]",
        "[STUCK — CIRCLING THE SAME PLACES]",
        "[NO PROGRESS — DIAGNOSE",
        "[HARNESS — STUCK",
    ]
    .iter()
    .any(|marker| end_guidance.contains(marker));
    // A failed apply_patch means the model's edit never landed — usually because it
    // patched a STALE view (the loop keeps rewriting the file, so the patch context no
    // longer matches disk). Steering it to write_file with a prelude directive does NOT
    // work — a weak model ignores prose. So we FORCE the write_file tool at the sampler
    // whenever a patch failed: the model literally cannot emit another stale patch, and
    // write_file OVERWRITES, so the edit lands regardless of how out-of-date its view is.
    // No size gate: the old "big files get the directive only" fallback was exactly the
    // steering that does nothing — i.e. no floor at all. A forced rewrite that truncates a
    // huge file is caught by the output-truncation guard, strictly better than a no-op.
    let force_write_file = trimmed.patch_rewrite_path.is_some();
    if let Some(path) = trimmed.patch_rewrite_path.as_ref() {
        state.push_nudge(format!(
            "Patch failed to apply — forcing a write_file rewrite of {} (the edit never landed; write_file overwrites regardless of how stale the patch was)",
            path.rsplit('/').next().unwrap_or(path),
        ));
    }

    let codex_routing::trim::TrimResult {
        system: trimmed_system,
        messages,
        ..
    } = trimmed;

    // LightCoder route gets a curated tool subset — applied identically in
    // regular and local-only modes. The full Codex tool catalog is ~120
    // schemas (MCP connectors, multi-agent orchestration, dynamic tools, …),
    // which exceeds the local model's context window and overwhelms its
    // attention. We expose only the tools a coding model actually needs to
    // execute work in the workspace. Cloud routes still receive the full set.
    //
    // Adding a new tool to this list is a deliberate decision: keep it
    // focused on capabilities the local model can use successfully.
    if use_tools {
        // Tool set was assembled before trimming (so its token footprint could
        // be reserved from the context budget) — see `build_local_tools`.
        let ollama_tools = local_tools;

        // Append a tool-usage hint to the system prompt. Small local models
        // habitually emit shell command names (`ls`, `rg`, `cat`) as tool
        // names because that's how their training data shaped them. Telling
        // them explicitly which tool wraps which capability avoids the
        // hallucination loop without restricting what they can do.
        let tool_names: Vec<&str> = ollama_tools
            .iter()
            .filter_map(|t| {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
            })
            .collect();
        let hint = build_tool_hint(&tool_names);
        // Guidance goes in the SYSTEM prompt, not a trailing user message. Placing
        // it at the end (its own turn) made the model over-attend to the static plan
        // and RESTART step 1 every turn, stranding its real progress in the
        // lost-in-the-middle zone it ignores (observed: identical "let me start by
        // listing the directory" reasoning 67× in one turn). Kept at the front.
        let guidance_block = if end_guidance.trim().is_empty() {
            String::new()
        } else {
            format!("{end_guidance}\n\n")
        };
        let mut system = Some(format!("{guidance_block}{trimmed_system}\n\n{hint}"));

        let dropped_tool_names: Vec<&str> = match endpoint.tool_subset {
            codex_routing::config::ToolSubset::Focused => LIGHT_CODER_TOOL_NAMES
                .iter()
                .copied()
                .filter(|name| !tool_names.contains(name))
                .collect(),
            codex_routing::config::ToolSubset::Full => Vec::new(),
        };
        info!(
            tool_count = ollama_tools.len(),
            available_in_prompt = prompt.tools.len(),
            subset = ?endpoint.tool_subset,
            text_is_product = text_is_product,
            tools_passed = ?tool_names,
            tools_dropped = ?dropped_tool_names,
            "Passing tool set to local model"
        );

        // Loop here so we can re-call the model up to MAX_BAIL_RETRIES times
        // when it ends a turn with prose and no tool call (a "bail") — see the
        // body of the loop for details. Set to 3 so a model that's making genuine
        // progress (e.g. ran a probe but hasn't applied the result yet)
        // gets enough nudges to land the change before we give up.
        const MAX_BAIL_RETRIES: usize = 3;
        let mut effective_messages = messages.clone();
        let mut continuation_count = 0usize;
        let last_user_message = extract_last_message(prompt);
        // Shrinks if the server reports the prompt overflowed its context, so we
        // re-trim to fit and retry rather than crashing to a tools-less role.
        let mut coder_budget = fit_budget;

        loop {
            // Streaming coder call with in-flight rumination detection.
            // See rumination_detector.rs for the heuristic; the watcher
            // here aborts the HTTP connection (by dropping the receiver)
            // when the detector flags a loop, then re-prompts with a
            // guard directive so we don't burn 10 min of reasoning on a
            // model that's stuck self-interrupting.
            // On a re-prompt retry, ENFORCE the tool call at the sampler instead
            // of merely asking for it in prose: the model already emitted a
            // no-tool-call response we rejected, and prose nudges get ignored.
            // `tool_choice="required"` makes the server grammar-constrain a valid
            // call. Scoped to this retry only (a fresh turn starts at
            // continuation_count=0), so normal text-completion and loop
            // termination are untouched. Reasoners (text is their product) are
            // never forced; an explicit operator `tool_choice` always wins.
            let mut forced_endpoint;
            // A stuck patch forces the write_file tool specifically (the rewrite
            // remedy); otherwise the retry-constraint may force "required" (any
            // tool). The specific force wins — it's a targeted recovery.
            let forced_tool_choice: Option<&str> = if force_write_file {
                Some("write_file")
            } else if enforce_tool_call_on_retry(
                continuation_count,
                use_tools,
                text_is_product,
                &endpoint.tool_choice,
            ) {
                Some("required")
            } else {
                None
            };
            let call_endpoint = if let Some(tc) = forced_tool_choice {
                forced_endpoint = endpoint.clone();
                forced_endpoint.tool_choice = Some(tc.to_string());
                &forced_endpoint
            } else {
                endpoint
            };
            // Our estimate for the exact prompt about to be sent — paired with the
            // server's real token count (success usage or overflow) to learn the
            // per-model ratio.
            let sent_estimate =
                estimate_combined_tokens(system.as_deref().unwrap_or(""), &effective_messages);
            let mut stream_rx = match state
                .pool
                .chat_with_tools_stream(
                    call_endpoint,
                    effective_messages.clone(),
                    system.as_deref(),
                    Some(ollama_tools.clone()),
                )
                .await
            {
                Ok(rx) => rx,
                Err(codex_routing::ollama::SendError::ContextOverflow {
                    n_ctx,
                    n_prompt_tokens,
                }) => {
                    // Ground truth: the server just told us the REAL token count for
                    // a prompt we estimated at `sent_estimate`. Learn the ratio so the
                    // NEXT turn budgets correctly and never reaches this overflow. (The
                    // overflow itself is surfaced below; no separate calibration nudge.)
                    let _ = record_token_ratio(
                        &endpoint.model,
                        n_prompt_tokens,
                        sent_estimate + tool_reserve_tokens,
                    );
                    // Learn the real window too (in case /props was unreachable).
                    record_server_ctx(&endpoint.base_url, n_ctx);
                    // The tokenized prompt overflowed the server's context window.
                    // The server hands us the real numbers, so re-trim to fit and
                    // retry the SAME coder — context maxing out must NOT crash the
                    // turn into a tools-less role. Token estimates undercount dense
                    // content (addresses, JSON, code), so scale the budget by how
                    // far we actually overshot rather than trusting the estimate.
                    const OVERFLOW_MARGIN: u64 = 2048;
                    const MIN_CODER_BUDGET: usize = 4096;
                    // Reserve output room the SAME way `effective_window` does —
                    // the input-side `output_reserve` (NOT the hard-cap `max_tokens`,
                    // which is usually unset). Missing this left 0 output room on
                    // the retry → immediate re-truncation.
                    let reserve = endpoint
                        .output_reserve
                        .unwrap_or(CTX_OUTPUT_RESERVE_DEFAULT) as u64
                        + real_tool_reserve as u64
                        + OVERFLOW_MARGIN;
                    // Aim for 80% of the window, not 100%. We scale the MESSAGE
                    // budget, but the overflow figure is the REAL total — which
                    // includes the fixed system prompt + tool schemas that don't
                    // shrink with the budget. The 20% headroom absorbs that
                    // nonlinearity so the re-trim fits on the first retry instead
                    // of dancing at the boundary across several slow round-trips
                    // (each oversized re-prefill risks a timeout on a CPU box).
                    let target = n_ctx
                        .saturating_mul(4)
                        .saturating_div(5)
                        .saturating_sub(reserve);
                    let scaled =
                        (coder_budget as u64).saturating_mul(target) / n_prompt_tokens.max(1);
                    let new_budget = (scaled as usize)
                        .min(coder_budget.saturating_sub(1024))
                        .max(MIN_CODER_BUDGET);
                    if new_budget >= coder_budget {
                        warn!(
                            n_ctx,
                            n_prompt_tokens,
                            "Context overflow with prompt already at the floor; failing over"
                        );
                        return Err(FailureType::ContextOverflow);
                    }
                    warn!(
                        old_budget = coder_budget,
                        new_budget,
                        n_ctx,
                        n_prompt_tokens,
                        "Coder prompt overflowed server context — re-trimming smaller and retrying same endpoint"
                    );
                    state.push_nudge(format!(
                        "Context overflow ({n_prompt_tokens} tok > {n_ctx} ctx) — re-trimmed the prompt to fit and retried; no crash"
                    ));
                    coder_budget = new_budget;
                    // coder_budget is already the MESSAGE budget (tool schemas are
                    // reserved separately in real tokens via `reserve` above), so
                    // don't subtract the tool estimate again here.
                    let re = codex_routing::trim::trim_for_local(&trim_input, coder_budget);
                    let re = maybe_inline_compact(re, coder_budget, state).await;
                    let re_guidance = if re.guidance.trim().is_empty() {
                        String::new()
                    } else {
                        format!("{}\n\n", re.guidance)
                    };
                    let re_system = format!("{re_guidance}{}", re.system);
                    system = Some(format!("{re_system}\n\n{hint}"));
                    effective_messages = re.messages;
                    continue;
                }
                Err(_) => {
                    // Transient stream-start blips are retried inside the pool (same
                    // server — failover can't route around them); reaching here means
                    // it stayed unavailable, so let the chain walk.
                    warn!("Local coder stream failed to start");
                    return Err(FailureType::ModelUnavailable);
                }
            };

            let detector =
                codex_routing::rumination_detector::RuminationDetector::from_reasoning_budget(
                    endpoint.output_reserve,
                );

            let mut content = String::new();
            let mut reasoning = String::new();
            let mut tool_call_acc: std::collections::BTreeMap<usize, StreamToolCallAcc> =
                std::collections::BTreeMap::new();
            let mut input_tokens = 0u64;
            let mut output_tokens = 0u64;
            let mut reasoning_tokens_seen = 0u64;
            let mut prompt_ms = 0u64;
            let mut gen_ms = 0u64;
            // Re-run the rumination regex at most every N bytes of new
            // reasoning so a long stream doesn't quadratic-scan itself.
            const RUMINATION_CHECK_STRIDE: usize = 500;
            let mut next_check_at = RUMINATION_CHECK_STRIDE;

            let mut rumination_trigger: Option<(usize, usize)> = None;
            let mut stream_ended_cleanly = false;
            // Set when the server stopped at the output-token cap (done_reason /
            // finish_reason == "length"). A turn that truncated mid-write_file
            // produced a cut-off file — see the output-truncation guard below.
            let mut output_truncated = false;

            while let Some(chunk) = stream_rx.recv().await {
                match chunk {
                    codex_routing::ollama::StreamChunk::Delta(text) => {
                        content.push_str(&text);
                    }
                    codex_routing::ollama::StreamChunk::ReasoningDelta(text) => {
                        reasoning.push_str(&text);
                        if reasoning.len() >= next_check_at {
                            next_check_at = reasoning.len() + RUMINATION_CHECK_STRIDE;
                            // Prefer the server's reported reasoning-token
                            // count when the usage chunk has already landed;
                            // otherwise estimate from char count. Most SSE
                            // servers only emit usage in the final chunk,
                            // so the estimate is what actually fires the
                            // budget gate mid-stream.
                            let tokens = if reasoning_tokens_seen > 0 {
                                reasoning_tokens_seen as usize
                            } else {
                                codex_routing::rumination_detector::estimate_reasoning_tokens(
                                    &reasoning,
                                )
                            };
                            let marker_count =
                                codex_routing::rumination_detector::count_rumination_markers(
                                    &reasoning,
                                );
                            let budget_gate = detector.budget_gate();
                            let gated = tokens >= budget_gate;
                            info!(
                                reasoning_chars = reasoning.len(),
                                reasoning_tokens = tokens,
                                budget_gate,
                                marker_count,
                                threshold = detector.threshold(),
                                gated,
                                "Rumination watch"
                            );
                            if gated && marker_count >= detector.threshold() {
                                rumination_trigger = Some((marker_count, tokens));
                                break;
                            }
                        }
                    }
                    codex_routing::ollama::StreamChunk::ToolCallDelta {
                        index,
                        id,
                        name,
                        arguments_delta,
                    } => {
                        let acc = tool_call_acc.entry(index).or_default();
                        if let Some(v) = id {
                            acc.id = Some(v);
                        }
                        if let Some(v) = name {
                            acc.name = Some(v);
                        }
                        acc.arguments.push_str(&arguments_delta);
                    }
                    codex_routing::ollama::StreamChunk::Done {
                        input_tokens: it,
                        output_tokens: ot,
                        reasoning_tokens: rt,
                        prompt_ms: pm,
                        gen_ms: gm,
                        truncated,
                    } => {
                        input_tokens = it;
                        output_tokens = ot;
                        reasoning_tokens_seen = rt;
                        prompt_ms = pm;
                        gen_ms = gm;
                        output_truncated = truncated;
                        stream_ended_cleanly = true;
                        break;
                    }
                }
            }

            // Dropping stream_rx here (end of scope or explicit) closes the
            // TCP connection when the stream task next tries to send,
            // which signals LM Studio / Ollama to stop generating.
            drop(stream_rx);

            if let Some((hits, rumination_tokens)) = rumination_trigger {
                info!(
                    hits,
                    reasoning_tokens = rumination_tokens,
                    reasoning_len = reasoning.len(),
                    continuation_count,
                    "Rumination guard aborted local coder; re-prompting"
                );
                if continuation_count >= MAX_BAIL_RETRIES {
                    warn!("Rumination guard hit retry cap; returning last partial response");
                    // Fall through to assemble whatever we got.
                } else {
                    let guard = codex_routing::rumination_detector::continuation_prompt(
                        hits,
                        rumination_tokens,
                    );
                    effective_messages.push(serde_json::json!({
                        "role": "user",
                        "content": guard,
                    }));
                    state.push_nudge(format!(
                        "Rumination guard fired — model was looping in its reasoning ({hits} repeated markers, ~{rumination_tokens} reasoning tokens); re-prompting to break out"
                    ));
                    continuation_count += 1;
                    continue;
                }
            }

            if !stream_ended_cleanly && rumination_trigger.is_none() {
                warn!("Local coder stream closed without Done; treating as unavailable");
                return Err(FailureType::ModelUnavailable);
            }

            // Output-truncation guard. The server stopped at the output-token cap
            // (done_reason / finish_reason == "length"), not a natural stop. If the
            // model was mid-way through a whole-file `write_file`, the file it would
            // produce is cut off — the exact footgun where the model then re-reads an
            // "incomplete" file and rewrites it forever, each rewrite truncating again
            // at the same cap. Don't emit the partial write: abort it and steer the
            // model to build the file incrementally (which stays under the cap). The
            // model gets NO other signal that its output was cut — a plain
            // "wrote N bytes" reads as success — so this is the only place it learns.
            if output_truncated && continuation_count < MAX_BAIL_RETRIES {
                if let Some(path) = truncated_write_path(&tool_call_acc) {
                    warn!(
                        %path,
                        output_tokens,
                        "Output truncated mid-write; re-prompting for incremental write"
                    );
                    let guard = format!(
                        "[OUTPUT TRUNCATED — YOUR LAST WRITE WAS CUT OFF]\n\
                         Your previous response hit the output token limit (~{output_tokens} tokens) partway through writing `{path}`, so the file's content was cut off mid-way. That is why re-reading the file shows it ending abruptly — it is NOT a corruption you can fix by rewriting the whole file: a full rewrite will hit the SAME limit and truncate at the same place.\n\
                         Instead, build the file in SMALL pieces: write only the FIRST portion with write_file, then APPEND each remaining section with edit_file (one small edit at a time). Keep every single tool call well under the output limit."
                    );
                    effective_messages.push(serde_json::json!({
                        "role": "user",
                        "content": guard,
                    }));
                    state.push_nudge(format!(
                        "Output-truncation guard fired — the write to {path} was cut off at the token limit (~{output_tokens} tok); steering to incremental writes"
                    ));
                    continuation_count += 1;
                    continue;
                }
            }

            if !reasoning.is_empty() {
                // INFO (not debug) so it survives the default `codex_core=info`
                // filter, and flattened to a single line so the tail parser can
                // pick it up. This log is the ONLY window into the model's
                // thinking: reasoning is deliberately never recorded to the rollout
                // or fed back into the model's context, so without this it's
                // invisible even though it drives every decision.
                let flat: String = reasoning.split_whitespace().collect::<Vec<_>>().join(" ");
                let shown = if flat.len() > 4000 {
                    let mut end = 4000;
                    while end > 0 && !flat.is_char_boundary(end) {
                        end -= 1;
                    }
                    format!("{}… [{} chars total]", &flat[..end], flat.len())
                } else {
                    flat
                };
                tracing::info!(
                    reasoning_len = reasoning.len(),
                    reasoning_tokens = reasoning_tokens_seen,
                    reasoning = %shown,
                    "Local coder reasoning channel"
                );
            }

            // Assemble tool calls from the accumulator in Ollama wire shape.
            let raw_tool_calls: Vec<serde_json::Value> = tool_call_acc
                .into_values()
                .map(|acc| {
                    serde_json::json!({
                        "function": {
                            "name": acc.name.unwrap_or_default(),
                            "arguments": acc.arguments,
                        }
                    })
                })
                .collect();
            let mut native_tool_calls = translate_native_tool_calls(raw_tool_calls);

            // Recover tool calls the model emitted as TEXT instead of structured
            // calls — Hermes `<tool_call>` JSON, the XML-function dialect, fenced
            // JSON blobs, or embedded tool_use blocks — when the server template
            // doesn't parse them. Without this the model's action silently
            // vanishes and the turn can be mistaken for a completion. This is the
            // SINGLE recovery entry point (`tool_recovery::recover_tool_calls`),
            // shared with the reasoner path so a format fix can never again land
            // in only one of the two.
            if native_tool_calls.is_empty() && !content.trim().is_empty() {
                let recovered = codex_routing::tool_recovery::recover_tool_calls(&content, false);
                if !recovered.tool_calls.is_empty() {
                    warn!(
                        recovered = recovered.tool_calls.len(),
                        "Recovered leaked tool call(s) from text — server didn't parse the model's tool-call blocks (check the chat template / --jinja)"
                    );
                    let wire: Vec<serde_json::Value> =
                        recovered.tool_calls.iter().map(tool_call_to_wire).collect();
                    native_tool_calls = translate_native_tool_calls(wire);
                    content = recovered.content;
                }
            }

            // Bare-JSON `update_plan` leak: the model dumps `{"plan":[{"step","status"}]}`
            // as content instead of calling the tool. It's raw JSON with no dialect
            // wrapper, so `recover_tool_calls` above misses it — and re-prompting it does
            // NOT work: the model repeats the same JSON until the bail budget is gone and
            // the turn dies empty (session 019f38a5). Convert it to a REAL update_plan call
            // so the plan updates, the turn makes progress, and the retry budget survives
            // for the completion critic instead of being burned on a leak it can't escape.
            if native_tool_calls.is_empty()
                && looks_like_bare_json_object(&content)
                && let Some(args) = recover_bare_update_plan(&content)
            {
                warn!(
                    "Recovered a bare-JSON update_plan (leaked as content) into a real update_plan call"
                );
                native_tool_calls = translate_native_tool_calls(vec![serde_json::json!({
                    "function": { "name": "update_plan", "arguments": args }
                })]);
                content = String::new();
            }

            info!(
                content_len = content.len(),
                native_tool_calls = native_tool_calls.len(),
                reasoning_tokens = reasoning_tokens_seen,
                continuation_count,
                "Local coder response received"
            );

            // Record local usage for /stats (with timing for tok/s).
            state
                .usage
                .record_timed(&model_name, input_tokens, output_tokens, prompt_ms, gen_ms);
            // Learn the real÷estimate token ratio from this response's actual
            // prompt_tokens, so the next turn budgets against truth, not chars/4.
            // Measure against the FULL prompt estimate (messages + tool schemas) so
            // the ratio is pure tokenizer density, not skewed by the tool fraction.
            let full_estimate = sent_estimate + tool_reserve_tokens;
            if let Some(ratio) = record_token_ratio(&endpoint.model, input_tokens, full_estimate) {
                state.push_nudge(format!(
                    "Calibrated context budget — this model packs ~{ratio:.1}× the tokens our estimate assumed ({input_tokens} real vs ~{full_estimate} est)"
                ));
            }

            // A TRANSLATED synthetic tool (edit_file/read_file/str_replace/
            // cat_file) that survived translation means its JSON arguments didn't
            // parse — and unlike write_file/create_file it has NO real handler to
            // fall back to, so dispatching gives an opaque "unsupported call".
            // Instead re-prompt grammar-constrained (enforce_tool_call_on_retry
            // forces tool_choice once continuation_count > 0). write_file's botched
            // content is recovered upstream in translate_one_native_call, so it
            // never reaches here. This is the tool-call constraint (§25).
            if continuation_count < MAX_BAIL_RETRIES
                && let Some(tool) = surviving_untranslated_synthetic(&native_tool_calls)
            {
                warn!(
                    tool = %tool,
                    "Synthetic tool survived translation (malformed args) — re-prompting under constraint"
                );
                state.push_nudge(format!(
                    "{tool} arguments were malformed JSON — re-prompting with a tool-call constraint"
                ));
                effective_messages.push(serde_json::json!({
                    "role": "user",
                    "content": format!(
                        "Your `{tool}` call had malformed JSON arguments — most likely an escaping \
                         error in the file content (unescaped quotes or raw newlines). Re-issue the \
                         SAME call as valid JSON: escape every double-quote as \\\" and every newline \
                         as \\n inside the JSON string values. Emit only the tool call."
                    ),
                }));
                continuation_count += 1;
                continue;
            }

            // COURSE-CHANGE RESET (Reasoned Guidance #4). A loop guard fired this
            // turn, yet the coder just produced a real tool call. Ask the reasoner
            // whether this is a GENUINE pivot (not the same approach reworded); if
            // so, grant a short grace window so the transcript-derived guards don't
            // instantly re-bury the new approach next turn — the exact failure
            // where the coder worked out "drop the /v1" but never got an
            // unobstructed turn to try it. Strict + bounded inside `course_change`.
            if loop_guard_fired
                && !native_tool_calls.is_empty()
                && state.config.reasoner.enabled
                && codex_routing::reasoned_guidance::course_change(
                    &state.pool,
                    &state.config.reasoner,
                    &last_user_message,
                    &loop_summary_from_guidance(&end_guidance),
                    &summarize_new_action(&reasoning, &native_tool_calls),
                )
                .await
            {
                codex_routing::reasoned_guidance::grant_loop_grace(&last_user_message);
                codex_routing::reasoned_guidance::reset_redirect_budget(&last_user_message);
                state.push_nudge(
                    "Reasoner confirmed a genuine course change — loop guards paused for a few \
                     turns so the new approach can run"
                        .to_string(),
                );
            }

            // Deterministic quality gate: discard obviously-broken text-only
            // responses (empty, too short, prompt echo, model refusal, empty
            // code fence, degenerate repetition) and re-prompt. Cheap and runs
            // before the reasoner completion critic so we don't spend a reasoner
            // call judging garbage. Only applies to text-only responses — a tool
            // call is the model making progress.
            if native_tool_calls.is_empty()
                && continuation_count < MAX_BAIL_RETRIES
                && let Some(reason) =
                    codex_routing::quality::check_response_quality(&content, &last_user_message)
            {
                warn!(reason = %reason, "Local response failed quality check; re-prompting");
                state.push_nudge(format!("Quality gate fired — {reason}; re-prompting"));
                let continuation = codex_routing::quality::quality_continuation_prompt(&reason);
                effective_messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": content,
                }));
                effective_messages.push(serde_json::json!({
                    "role": "user",
                    "content": continuation,
                }));
                continuation_count += 1;
                continue;
            }

            // No-tool-call handling, gated on GROUND TRUTH. The quality gate
            // above catches broken output; this catches "produced prose instead
            // of acting". The decision rests on what the model *did* — files
            // actually modified this task, witnessed in the transcript, not
            // timestamps:
            //
            //  - A coder that changed nothing is bailing/bluffing no matter how
            //    confident it sounds. Nudge it to act, and NEVER let it finish on
            //    an unbacked "done". This is the hard guard against "claimed done
            //    but wrote nothing".
            //  - A coder that DID change files may finish — but only if its code
            //    passes the repo's own diagnostics (probe gate) AND the reasoner
            //    completion critic finds the work actually complete. A reasoner
            //    (text-only by design) finishes here. Anything blocked re-prompts.
            //    (There is no small-model text-shape "verifier" anymore — it
            //    false-negatived on finished work and trapped a done coder in a
            //    done→act→`ls` loop; ground truth + probe + critic replace it.)
            // A `<tool_call>` block still present here means recovery above
            // couldn't parse it — the model tried to call a tool but emitted
            // MALFORMED JSON (commonly a heredoc / multi-line command whose
            // quotes or newlines it couldn't escape inside the JSON string). It
            // was NOT executed and must NEVER be mistaken for a completion.
            // Re-prompt to re-issue it cleanly, and steer multi-line work to
            // write_file instead of an inline heredoc.
            if native_tool_calls.is_empty()
                && continuation_count < MAX_BAIL_RETRIES
                && codex_routing::tool_aliases::has_leaked_tool_call(&content)
            {
                warn!(
                    "Local model emitted an unparseable <tool_call> (malformed JSON); re-prompting"
                );
                state.push_nudge(
                    "Malformed tool call — the model's `<tool_call>` JSON didn't parse (bad escaping); re-prompting it to re-issue cleanly".to_string(),
                );
                let continuation = "Your last turn contained a `<tool_call>` block, but its JSON did not parse \
                    — usually broken escaping of quotes or newlines in a multi-line command — so it was NOT \
                    executed and nothing happened. Re-issue it as a single valid tool call. For ANY multi-line \
                    script or heredoc, FIRST write the script to a file with the `write_file` tool, then run \
                    that file with a one-line command. Do NOT inline a heredoc inside a command argument.";
                effective_messages
                    .push(serde_json::json!({"role": "assistant", "content": content}));
                effective_messages
                    .push(serde_json::json!({"role": "user", "content": continuation}));
                continuation_count += 1;
                continue;
            }

            // Bare-JSON tool-call leak. A weak model (Fabliq) emits a STRUCTURED tool
            // call — commonly `update_plan` — as a JSON object in the CONTENT channel
            // instead of a real tool call. It isn't one of the recognized leaked-call
            // dialects (`<tool_call>`/XML/Gemma), so it leaks to the TUI raw (a `{` and
            // JSON-escaped `\n`) AND gets mistaken for the final answer (a plan-as-JSON
            // masquerading as completed work → false `task_complete`). A bare JSON
            // object is NEVER a valid deliverable for EITHER role — a coder should call
            // the tool, a reasoner should write prose — so this fires regardless of
            // `text_is_product`. Gating it on `!text_is_product` let a `light_reasoner`
            // turn emit `{ "plan": …}` and have it ACCEPTED as the final answer (the
            // false completion that ended session 019f35d3, since the completion gate is
            // also skipped for a text-product role). Bounded by MAX_BAIL_RETRIES.
            if native_tool_calls.is_empty()
                && continuation_count < MAX_BAIL_RETRIES
                && looks_like_bare_json_object(&content)
            {
                warn!(
                    text_is_product,
                    "Local model emitted a bare JSON object as content (leaked tool call / plan-as-JSON); re-prompting"
                );
                state.push_nudge(
                    "Bare-JSON leak — the model dumped a JSON object as text instead of a real action or answer; re-prompting".to_string(),
                );
                let continuation = "Your last turn was a raw JSON object, not a real action or answer — \
                    it was NOT executed and nothing happened. A bare `{...}` is never a valid response. \
                    Either make a REAL tool call (e.g. `update_plan`, `write_file`, `exec_command`), or \
                    write your actual answer in plain prose. Never output a JSON object as your response.";
                effective_messages.push(serde_json::json!({"role": "assistant", "content": content}));
                effective_messages.push(serde_json::json!({"role": "user", "content": continuation}));
                continuation_count += 1;
                continue;
            }

            // The completion gate (probe + critic) below runs on EVERY text-only
            // completion candidate — NOT only while re-prompt budget remains. Gating
            // the CHECK on `continuation_count < MAX_BAIL_RETRIES` was a real bug: a
            // turn that thrashed enough to exhaust its budget had its (often FALSE)
            // "done" accepted with no probe and no critic — the more a model flailed,
            // the more likely its false completion sailed through unchecked. The budget
            // now bounds only the RE-PROMPT action (below), never the verification.
            if native_tool_calls.is_empty() && !content.trim().is_empty() {
                let did_real_work =
                    !codex_routing::trim::files_modified_in_active_turn(&prompt.input).is_empty();

                // Set by the linter Probe below when the code on disk doesn't pass
                // its own checker; used as the (more specific) re-prompt text.
                let mut probe_nudge: Option<String> = None;
                // Set by the completion critic (assist #2) when the reasoner judges a
                // passed-the-gates "done" claim not actually complete.
                let mut critique_nudge: Option<String> = None;

                let reprompt: Option<&str> = if !text_is_product && !did_real_work {
                    Some(
                        "Coder used no tools and changed nothing — nudging it to act via a tool call",
                    )
                } else if last_user_message.trim().is_empty() {
                    None
                } else {
                    // No small-model text-shape verifier here. A coder that DID real
                    // work (ground truth, checked above) and stopped is a completion
                    // CANDIDATE; the real gates are the probe gate (its code passes the
                    // repo's own checks) and the reasoner completion critic (the work
                    // is actually done — no shortcuts). A reasoner's text IS its
                    // product, so it finishes here; empty/broken text was already
                    // rejected by the quality gate. Removing the verifier kills the loop
                    // it caused: a FINISHED coder whose phrasing ("let me verify") read
                    // as a bail could never escape — it kept being told "you did
                    // nothing, act" and answered with pointless `ls` forever. The
                    // smarter reasoner critic below now owns the "is it really done?"
                    // judgment.
                    let mut finish = true;
                    // Fresh lint/test probe results, captured for the completion critic
                    // below EVEN WHEN the deterministic gate passes — so a "done" claim
                    // whose tests never ran (no file:line findings, so the gate is silent)
                    // reaches the critic as ground truth instead of a blind judgment.
                    let mut probe_digest = String::new();
                    // Ground-truth completion gate (the Probe system): a coder is not
                    // "done" while its code fails the repo's OWN diagnostics. Runs the
                    // top-ranked SAFE probe AND the top TEST probe (discovery ranks
                    // typecheck/lint above tests, so a top-1 run would green-light a repo
                    // whose tests fail — a false completion sailed through exactly that
                    // way) plus the always-available syntax floor, only when we were
                    // about to accept completion. The exact file:line becomes the
                    // re-prompt, so the model fixes the reported line, not blind rewrites.
                    if finish && !text_is_product {
                        if let Ok(dir) = std::env::current_dir() {
                            let checked = tokio::task::spawn_blocking(move || {
                                let report = codex_routing::probe_run::run_completion_probes(
                                    &dir,
                                    std::time::Duration::from_secs(45),
                                );
                                let floor = codex_routing::linter_probe::run_linter_probe(&dir);
                                (report, floor)
                            })
                            .await;
                            if let Ok((report, floor)) = checked {
                                // TRUTH CAPTURE: which probes ran + how many findings +
                                // whether the syntax floor was clean — so a completion
                                // that passed the gate is auditable, not a silent event.
                                info!(
                                    probes_run = report.results.len(),
                                    commands = ?report.selected.iter().map(|c| c.command.join(" ")).collect::<Vec<_>>(),
                                    findings = report.results.iter().map(|r| r.findings.len()).sum::<usize>(),
                                    floor_clean = floor.is_clean(),
                                    "Completion probe gate ran (truth capture)"
                                );
                                // Capture the digest regardless of the block outcome — the
                                // critic needs "ran clean" vs "did not run" even when the
                                // gate found nothing structured to block on.
                                probe_digest = codex_routing::probe_run::completion_probe_digest(
                                    &report, &floor,
                                );
                                if let Some(n) = codex_routing::probe_run::completion_block_nudge(
                                    &report, &floor,
                                ) {
                                    warn!(
                                        probes = report.results.len(),
                                        "Probe gate blocked completion — repo diagnostics failed"
                                    );
                                    probe_nudge = Some(n);
                                    finish = false;
                                }
                            }
                        }
                    }
                    // Reasoned Guidance assist #2 — COMPLETION CRITIC. The gates above
                    // are deterministic and cannot catch semantic shortcuts: tests
                    // passing for the wrong reason, an error accepted as success, a
                    // skipped requirement. On a "done" claim that passed them, spend ONE
                    // reasoner call to review the actual work against the task. Bounded by
                    // `critic_blocks` so it can't nag forever.
                    if finish
                        && !text_is_product
                        && state.config.reasoner.enabled
                        && critic_blocks < COMPLETION_CRITIC_MAX_BLOCKS
                    {
                        let mut evidence = recent_evidence(prompt, 3200);
                        if !content.trim().is_empty() {
                            evidence.push_str(&format!(
                                "\n\nFINAL CLAIM (model declared the task done): {}",
                                truncate_ev(&content, 500)
                            ));
                        }
                        if let Some(critique) =
                            codex_routing::reasoned_guidance::critique_completion(
                                state.pool.as_ref(),
                                &state.config.reasoner,
                                &last_user_message,
                                &evidence,
                                &probe_digest,
                            )
                            .await
                        {
                            warn!(
                                "Completion critic (reasoner) blocked completion — shortcuts/assumptions/gaps"
                            );
                            state.push_nudge(
                                "Completion critic — the reasoner found the task not fully done; sending it back with specifics".to_string(),
                            );
                            critique_nudge = Some(critique);
                            critic_blocks += 1;
                            finish = false;
                        }
                    }
                    if finish {
                        None
                    } else {
                        Some(
                            "Completion gate fired — repo diagnostics or the reviewer flagged it; re-prompting",
                        )
                    }
                };

                if let Some(notice) = reprompt {
                    if continuation_count < MAX_BAIL_RETRIES {
                        warn!("Re-prompting local model after a no-tool-call turn: {notice}");
                        state.push_nudge(notice.to_string());
                        // Prefer a GROUNDED re-prompt: the completion probe's exact
                        // errors, else the completion critic's issues. Only if neither
                        // fired do we use the no-action continuation — which is a
                        // deliberately MECHANICAL protocol nudge (it states "no tool ran,
                        // so nothing happened" and demands a tool call; it does not author
                        // an approach), bounded by MAX_BAIL_RETRIES. See the Phase 4
                        // reclassification in docs/spec/reasoned-guidance-refactor.md.
                        let continuation = match probe_nudge.take().or_else(|| critique_nudge.take()) {
                            Some(n) => n,
                            None => codex_routing::no_action_prompt::continuation_prompt(
                                &content,
                                continuation_count,
                            ),
                        };
                        effective_messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": content,
                        }));
                        effective_messages.push(serde_json::json!({
                            "role": "user",
                            "content": continuation,
                        }));
                        continuation_count += 1;
                        continue;
                    } else {
                        // Re-prompt budget spent. The gate STILL ran (it is no longer
                        // skipped on exhaustion — that was the bug). We can't re-prompt
                        // forever, but we must NOT accept SILENTLY: surface the flagged
                        // "done" so a false completion is visible, not mistaken for clean.
                        let flagged = probe_nudge.take().or_else(|| critique_nudge.take());
                        warn!(
                            continuation_count,
                            gate_flagged = flagged.is_some(),
                            "Completion accepted despite the completion gate flagging it — re-prompt budget exhausted"
                        );
                        if let Some(f) = flagged {
                            state.push_nudge(format!(
                                "Completion gate flagged this 'done' but the re-prompt budget is spent — \
                                 accepting with UNRESOLVED issues:\n{f}"
                            ));
                        }
                        // Fall through to accept (the return below).
                    }
                }
            }

            return Ok(ollama_tool_response_to_stream(
                content,
                native_tool_calls,
                reasoning.clone(),
                fresh_plan.clone(),
                input_tokens,
                output_tokens,
            ));
        }
    }

    // Every local role — coder and reasoner alike — is served on the unified
    // tool-capable streaming path above, which always returns. Reaching here
    // means the role resolved to an EMPTY tool set, which is a config error,
    // not a normal state — so fail over rather than run a tools-less turn.
    warn!(role = %role, "Local role resolved to an empty tool set — failing over");
    Err(FailureType::ModelUnavailable)
}

/// Pick a cloud model from the project config's weighted entries for a role.
/// Returns None if no config exists for this role.
fn pick_cloud_model(
    pc: &codex_routing::project_config::ProjectConfig,
    role_name: &str,
) -> Option<String> {
    pick_cloud_model_with_provider(pc, role_name).map(|(slug, _)| slug)
}

/// Pick a cloud model and its provider from the project config.
/// Returns (model_slug, provider_name) or None.
fn pick_cloud_model_with_provider(
    pc: &codex_routing::project_config::ProjectConfig,
    role_name: &str,
) -> Option<(String, String)> {
    use codex_routing::project_config::ModelRole;

    let role = pc.get_model(role_name)?;
    match role {
        ModelRole::Single {
            provider, model, ..
        } => Some((model.clone(), provider.clone())),
        ModelRole::Weighted { entries } => {
            if entries.is_empty() {
                return None;
            }
            let total_weight: u32 = entries.iter().map(|e| e.weight).sum();
            if total_weight == 0 {
                return Some((entries[0].model.clone(), entries[0].provider.clone()));
            }
            let mut pick = rand_u32() % total_weight;
            for entry in entries {
                if pick < entry.weight {
                    return Some((entry.model.clone(), entry.provider.clone()));
                }
                pick -= entry.weight;
            }
            Some((entries[0].model.clone(), entries[0].provider.clone()))
        }
    }
}

/// Handle a cloud API error by consulting the failover executor.
/// Returns Some(new_slug) if we should retry with a different model,
/// or None if we should propagate the error.
///
/// Called from client.rs when a cloud request fails with an HTTP error.
pub(crate) async fn handle_cloud_failover(
    ctx: &mut CloudFailoverCtx,
    status_code: Option<u16>,
    error_message: &str,
    attempt: &mut u32,
    retry_after_ms: Option<u64>,
) -> Option<String> {
    let failure_type = failover::classify_failure(
        status_code,
        error_message,
        false, // not a quality failure (we don't check cloud response quality)
        false, // not context overflow (would need specific detection)
    );

    let action = failover::decide_action(
        failure_type,
        &ctx.role_name,
        &ctx.chain_name,
        &ctx.chain,
        *attempt,
        retry_after_ms,
        &ctx.behavior,
    );

    match action {
        FailoverAction::RetrySame {
            wait,
            attempt: next_attempt,
        } => {
            info!(
                model = %ctx.role_name,
                wait_ms = wait.as_millis() as u64,
                attempt = next_attempt,
                "Cloud failover: retrying same model"
            );
            tokio::time::sleep(wait).await;
            *attempt = next_attempt;
            // Return the same slug — caller should retry the request
            let state = get_routing_state().await.as_ref()?;
            pick_cloud_model(&state.project_config, &ctx.role_name)
        }
        FailoverAction::NextInChain { model_role, reason } => {
            info!(
                from = %ctx.role_name,
                to = %model_role,
                reason = %reason,
                "Cloud failover: walking to next model in chain"
            );
            // Update context for potential future failures
            ctx.role_name = model_role.clone();
            *attempt = 0;

            // Resolve the next role — only cloud models (local would need
            // a full re-route which we don't do from the cloud path)
            let state = get_routing_state().await.as_ref()?;
            pick_cloud_model(&state.project_config, &model_role)
        }
        FailoverAction::HardFail { reason } => {
            warn!(reason = %reason, "Cloud failover: hard fail");
            None
        }
        FailoverAction::ChainExhausted { chain_name } => {
            warn!(chain = %chain_name, "Cloud failover: chain exhausted");
            None
        }
    }
}

/// Simple random u32 — no external crate dependency.
fn rand_u32() -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    std::time::Instant::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    hasher.finish() as u32
}

// --- Response translation ---

/// Convert an Ollama text response into a ResponseStream that codex-core expects.
///
/// Event sequence must be: Created → OutputItemAdded → OutputItemDone → Completed.
/// Do NOT send ServerModel (triggers reroute detection when model name differs).
/// Do NOT send OutputTextDelta before OutputItemAdded (panics).
fn ollama_response_to_stream(response: OllamaTextResponse) -> ResponseStream {
    let (tx, rx) = mpsc::channel(16);

    tokio::spawn(async move {
        // 1. Created
        let _ = tx.send(Ok(ResponseEvent::Created)).await;

        let message = ResponseItem::Message {
            id: Some("local_msg_0".to_string()),
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText {
                text: response.content,
            }],
            end_turn: Some(true),
            phase: None,
        };

        // 2. OutputItemAdded — registers the item so deltas/done can reference it
        let _ = tx
            .send(Ok(ResponseEvent::OutputItemAdded(message.clone())))
            .await;

        // 3. OutputItemDone — the complete message
        let _ = tx.send(Ok(ResponseEvent::OutputItemDone(message))).await;

        // 4. Completed with usage
        let _ = tx
            .send(Ok(ResponseEvent::Completed {
                response_id: "local_response".to_string(),
                token_usage: Some(local_token_usage(
                    response.input_tokens as i64,
                    response.output_tokens as i64,
                )),
            }))
            .await;
    });

    ResponseStream { rx_event: rx }
}

/// Convert an Ollama response with native tool_calls to a ResponseStream.
/// Handles both native Ollama tool_calls and embedded JSON tool calls.
/// Build a rollout `Reasoning` item. Used to persist the local model's thinking
/// (and the plan-first block) to the rollout for observability — a tail script's
/// `--reasoning`, after-the-fact audit — WITHOUT feeding it back to the model:
/// the trim treats old reasoning as single-use exhaust and drops it from the next
/// prompt (see `codex_routing::trim::rules`). `label` becomes the reasoning
/// summary so the tail can tell a plan from ordinary thinking.
fn reasoning_item(label: &str, text: &str) -> ResponseItem {
    ResponseItem::Reasoning {
        id: String::new(),
        summary: vec![
            codex_protocol::models::ReasoningItemReasoningSummary::SummaryText {
                text: label.to_string(),
            },
        ],
        content: Some(vec![
            codex_protocol::models::ReasoningItemContent::ReasoningText {
                text: text.to_string(),
            },
        ]),
        encrypted_content: None,
    }
}

fn ollama_tool_response_to_stream(
    content: String,
    native_tool_calls: Vec<serde_json::Value>,
    reasoning: String,
    plan: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
) -> ResponseStream {
    // A bare-JSON blob — e.g. `update_plan` emitted as `{"plan":…}` — is NEVER prose,
    // whether or not a real tool call rides alongside it. Blank it so it can't stream to
    // the TUI as raw JSON with escaped `\n` (and can't be recorded as the final answer).
    // The no-tool-call case is normally caught UPSTREAM by a re-prompt, but that is
    // bounded by MAX_BAIL_RETRIES; this is the final safety net that keeps a stubborn
    // model's exhausted-retry `{"plan":…}` off the screen (the false completion that
    // ended session 019f35d3). A genuine answer never starts with a bare `{ "…":` — and
    // if the model still emits one after 3 "write prose, not JSON" nudges, empty is
    // strictly better than showing garbage as the answer.
    let content = if looks_like_bare_json_object(&content) {
        String::new()
    } else {
        content
    };

    let (tx, rx) = mpsc::channel(16);

    tokio::spawn(async move {
        let _ = tx.send(Ok(ResponseEvent::Created)).await;

        // Persist the plan-first block (once per task) and the model's reasoning
        // to the rollout as Reasoning items — visible to a tail / audit, then
        // dropped from the next prompt by the trim ("single-use exhaust"), so the
        // model's context is unchanged. Plan first so it reads before the thinking.
        if let Some(plan) = plan.filter(|p| !p.is_empty()) {
            let item = reasoning_item("Plan-first (reasoned guidance)", &plan);
            let _ = tx
                .send(Ok(ResponseEvent::OutputItemAdded(item.clone())))
                .await;
            let _ = tx.send(Ok(ResponseEvent::OutputItemDone(item))).await;
        }
        if !reasoning.is_empty() {
            let item = reasoning_item("Reasoning", &reasoning);
            let _ = tx
                .send(Ok(ResponseEvent::OutputItemAdded(item.clone())))
                .await;
            let _ = tx.send(Ok(ResponseEvent::OutputItemDone(item))).await;
        }

        // Emit text content if any
        if !content.is_empty() {
            let text_msg = ResponseItem::Message {
                id: Some("local_msg_0".to_string()),
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: content.clone(),
                }],
                end_turn: Some(native_tool_calls.is_empty()),
                phase: None,
            };
            let _ = tx
                .send(Ok(ResponseEvent::OutputItemAdded(text_msg.clone())))
                .await;
            let _ = tx.send(Ok(ResponseEvent::OutputItemDone(text_msg))).await;
        }

        // Emit native tool calls from Ollama
        for (i, tc) in native_tool_calls.iter().enumerate() {
            let func = tc.get("function").unwrap_or(tc);
            let name = func
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("unknown")
                .to_string();
            let call_id = tc
                .get("id")
                .and_then(|id| id.as_str())
                .map(String::from)
                .unwrap_or_else(|| format!("local_call_{i}"));
            let arguments = func
                .get("arguments")
                .map(|a| {
                    if a.is_string() {
                        a.as_str().unwrap_or("{}").to_string()
                    } else {
                        serde_json::to_string(a).unwrap_or_else(|_| "{}".into())
                    }
                })
                .unwrap_or_else(|| "{}".into());

            let func_call = ResponseItem::FunctionCall {
                id: Some(format!("local_fc_{i}")),
                name,
                namespace: None,
                arguments,
                call_id,
            };
            let _ = tx
                .send(Ok(ResponseEvent::OutputItemAdded(func_call.clone())))
                .await;
            let _ = tx.send(Ok(ResponseEvent::OutputItemDone(func_call))).await;
        }

        // Text-leaked tool calls are recovered upstream (the single
        // `tool_recovery::recover_tool_calls` pass before this stream is built),
        // so by here `native_tool_calls` is already complete — no second,
        // divergent recovery here.

        let _ = tx
            .send(Ok(ResponseEvent::Completed {
                response_id: "local_response".to_string(),
                token_usage: Some(local_token_usage(input_tokens as i64, output_tokens as i64)),
            }))
            .await;
    });

    ResponseStream { rx_event: rx }
}

/// Convert an Ollama response with potential tool calls into a ResponseStream.
/// Runs tool-call recovery to extract embedded function calls.
#[allow(dead_code)]
fn ollama_response_to_stream_with_tools(response: OllamaTextResponse) -> ResponseStream {
    let (tx, rx) = mpsc::channel(16);

    tokio::spawn(async move {
        let _ = tx.send(Ok(ResponseEvent::Created)).await;

        let recovered = codex_routing::tool_recovery::recover_tool_calls(&response.content, false);

        if recovered.tool_calls.is_empty() {
            // No tool calls — just text
            let message = ResponseItem::Message {
                id: Some("local_msg_0".to_string()),
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: response.content,
                }],
                end_turn: Some(true),
                phase: None,
            };
            let _ = tx
                .send(Ok(ResponseEvent::OutputItemAdded(message.clone())))
                .await;
            let _ = tx.send(Ok(ResponseEvent::OutputItemDone(message))).await;
        } else {
            // Has tool calls
            if !recovered.content.is_empty() {
                let text_msg = ResponseItem::Message {
                    id: Some("local_msg_0".to_string()),
                    role: "assistant".to_string(),
                    content: vec![ContentItem::OutputText {
                        text: recovered.content,
                    }],
                    end_turn: None,
                    phase: None,
                };
                let _ = tx
                    .send(Ok(ResponseEvent::OutputItemAdded(text_msg.clone())))
                    .await;
                let _ = tx.send(Ok(ResponseEvent::OutputItemDone(text_msg))).await;
            }

            for (i, tc) in recovered.tool_calls.iter().enumerate() {
                let call_id = tc.id.clone().unwrap_or_else(|| format!("local_call_{i}"));
                let arguments =
                    serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".into());

                let func_call = ResponseItem::FunctionCall {
                    id: Some(format!("local_fc_{i}")),
                    name: tc.name.clone(),
                    namespace: None,
                    arguments,
                    call_id,
                };
                let _ = tx
                    .send(Ok(ResponseEvent::OutputItemAdded(func_call.clone())))
                    .await;
                let _ = tx.send(Ok(ResponseEvent::OutputItemDone(func_call))).await;
            }
        }

        let _ = tx
            .send(Ok(ResponseEvent::Completed {
                response_id: "local_response".to_string(),
                token_usage: Some(local_token_usage(
                    response.input_tokens as i64,
                    response.output_tokens as i64,
                )),
            }))
            .await;
    });

    ResponseStream { rx_event: rx }
}

// --- Prompt extraction helpers ---

/// Extract the last user message from the prompt.
/// This is what the classifier sees — just the current request, not full history.
fn extract_last_message(prompt: &Prompt) -> String {
    for item in prompt.input.iter().rev() {
        if let ResponseItem::Message { role, content, .. } = item {
            if role == "user" {
                let text: String = content
                    .iter()
                    .filter_map(|c| match c {
                        ContentItem::InputText { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    return text;
                }
            }
        }
    }
    String::new()
}

/// Max times the completion critic may block a single turn before it lets the
/// model finish — a guard that fires forever is not a guard.
const COMPLETION_CRITIC_MAX_BLOCKS: u32 = 2;

/// Build a bounded, chronological digest of the recent transcript — tool calls,
/// their outputs (with success/fail), and assistant messages — for the completion
/// critic (`reasoned_guidance::critique_completion`) to review. Focused on tool
/// OUTPUTS, where the real evidence lives (an API 404, a suspicious test pass).
/// Collected newest-first up to `budget` chars, then reversed to read in order.
fn recent_evidence(prompt: &Prompt, budget: usize) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut used = 0usize;
    for item in prompt.input.iter().rev() {
        if used >= budget {
            break;
        }
        let line = match item {
            ResponseItem::FunctionCall {
                name, arguments, ..
            } => format!("TOOL {name} {}", truncate_ev(arguments, 160)),
            ResponseItem::FunctionCallOutput { output, .. } => {
                let tag = if output.success == Some(false) {
                    "[FAILED] "
                } else {
                    ""
                };
                format!(
                    "  -> {tag}{}",
                    truncate_ev(&output.body.to_text().unwrap_or_default(), 400)
                )
            }
            ResponseItem::Message { role, content, .. } if role == "assistant" => {
                let text: String = content
                    .iter()
                    .filter_map(|c| match c {
                        ContentItem::OutputText { text } | ContentItem::InputText { text } => {
                            Some(text.as_str())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                if text.trim().is_empty() {
                    continue;
                }
                format!("ASSISTANT: {}", truncate_ev(&text, 300))
            }
            _ => continue,
        };
        used += line.len();
        lines.push(line);
    }
    lines.reverse();
    lines.join("\n")
}

/// Trim to `n` chars with an ellipsis, on a char boundary.
fn truncate_ev(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n {
        return s.to_string();
    }
    let end = s.char_indices().nth(n).map(|(i, _)| i).unwrap_or(s.len());
    format!("{}…", &s[..end])
}

/// Count recent tool calls and turns from conversation history.
fn count_recent_activity(prompt: &Prompt) -> (usize, usize) {
    let mut tool_calls = 0;
    let mut turns = 0;

    // Count from the last ~10 items
    for item in prompt.input.iter().rev().take(10) {
        match item {
            ResponseItem::Message { .. } => turns += 1,
            ResponseItem::FunctionCall { .. } | ResponseItem::LocalShellCall { .. } => {
                tool_calls += 1;
            }
            _ => {}
        }
    }

    (tool_calls, turns)
}

/// If the accumulated stream tool calls include a whole-file write that was still
/// being generated when the stream truncated, return the target path — used by the
/// output-truncation guard to name the file and steer to incremental writes. The
/// `path` field precedes `content` in the args, so it survives a mid-`content`
/// truncation even when the JSON as a whole is left unterminated; a plain scan
/// recovers it without needing valid JSON.
fn truncated_write_path(
    acc: &std::collections::BTreeMap<usize, StreamToolCallAcc>,
) -> Option<String> {
    for call in acc.values() {
        if !matches!(call.name.as_deref(), Some("write_file") | Some("create_file")) {
            continue;
        }
        let args = call.arguments.as_str();
        for key in ["\"path\"", "\"file_path\"", "\"filename\""] {
            let Some(i) = args.find(key) else { continue };
            let Some((_, after_key)) = args[i..].split_once(':') else {
                continue;
            };
            let Some(open) = after_key.find('"') else { continue };
            let tail = &after_key[open + 1..];
            if let Some(close) = tail.find('"') {
                let p = tail[..close].trim();
                if !p.is_empty() {
                    return Some(p.to_string());
                }
            }
        }
        return Some("the file".to_string());
    }
    None
}

/// Verified collapsed-Update → `write_file`. The reasoning-tuned Fabliq emits an
/// `apply_patch` Update whose hunk body is the whole file collapsed onto one `-`/`+`
/// line with LITERAL `\n` separators (double-escaped), so apply_patch's context never
/// matches and it fails every time. [`collapsed_update_patch_parts`] reconstructs
/// `(path, old, new)` from that shape; here we PROVE it is a complete replacement by
/// reading the file and checking the reconstructed OLD content equals what is on disk
/// (exactly, or modulo a single trailing newline — files conventionally end in `\n`
/// but the collapsed line often omits it). Only then do we rewrite to the robust
/// `write_file` path. Any mismatch → `None`, and the call falls through to normal
/// apply_patch normalization (failing as it does today, with no risk of truncation).
fn collapsed_update_to_write_file(
    args: &serde_json::Value,
) -> Option<codex_routing::tool_aliases::TranslatedCall> {
    let input = args
        .get("input")
        .or_else(|| args.get("patch"))
        .and_then(|v| v.as_str())?;
    let (path, old_content, new_content) =
        codex_routing::tool_aliases::collapsed_update_patch_parts(input)?;

    let candidate = match std::env::current_dir().ok() {
        Some(base) => base.join(&path),
        None => std::path::PathBuf::from(&path),
    };
    let disk = std::fs::read_to_string(&candidate).ok()?;

    // The reconstructed OLD block must account for the ENTIRE file — exactly, or with
    // exactly one trailing newline of slack on either side. That proves it is a full
    // replacement, so writing `new` cannot drop content the patch never mentioned.
    let full_match = disk == old_content
        || disk.strip_suffix('\n') == Some(old_content.as_str())
        || Some(disk.as_str()) == old_content.strip_suffix('\n');
    if !full_match {
        return None;
    }

    // Preserve the file's trailing-newline convention across the rewrite.
    let content = if disk.ends_with('\n') && !new_content.ends_with('\n') {
        format!("{new_content}\n")
    } else {
        new_content
    };

    info!(
        path = %path,
        bytes = content.len(),
        "apply_patch Update (double-escaped, verified full-file) -> write_file"
    );
    Some(codex_routing::tool_aliases::TranslatedCall {
        name: "write_file",
        args: serde_json::json!({ "path": path, "content": content }),
        command_line: format!("apply_patch Update (verified full-file) -> write_file ({path})"),
    })
}

/// Expand a COLLAPSED apply_patch — multi-line content crammed onto one `-`/`+` line with
/// literal `\n` — into a real multi-line hunk apply_patch can match. This is the DOMINANT
/// apply_patch failure (~2/3 of all failures observed: the `-` line can never match the
/// file's real newlines). Reads the target file, hands it to
/// [`codex_routing::tool_aliases::expand_collapsed_patch`] (which verifies the expanded
/// OLD block appears on disk before trusting the split), and returns a normal apply_patch
/// with the expanded body. `None` — fall through to normal normalization — when it isn't a
/// collapsed patch, the file can't be read, or the expansion doesn't verify.
fn expand_collapsed_update_patch(
    args: &serde_json::Value,
) -> Option<codex_routing::tool_aliases::TranslatedCall> {
    let input = args
        .get("input")
        .or_else(|| args.get("patch"))
        .and_then(|v| v.as_str())?;
    let path = input
        .lines()
        .find_map(|l| l.strip_prefix("*** Update File: "))?
        .trim()
        .to_string();
    if path.is_empty() || !input.contains("\\n") {
        return None; // no literal `\n` → nothing to expand; skip the disk read
    }
    let candidate = match std::env::current_dir().ok() {
        Some(base) => base.join(&path),
        None => std::path::PathBuf::from(&path),
    };
    let disk = std::fs::read_to_string(&candidate).ok()?;
    let expanded = codex_routing::tool_aliases::expand_collapsed_patch(input, &disk)?;
    let mut new_args = args.as_object()?.clone();
    new_args.insert("input".to_string(), serde_json::Value::String(expanded));
    new_args.remove("patch");
    info!(
        path = %path,
        "apply_patch: expanded a collapsed (literal-\\n) hunk into real lines, verified against disk"
    );
    Some(codex_routing::tool_aliases::TranslatedCall {
        name: "apply_patch",
        args: serde_json::Value::Object(new_args),
        command_line: format!("apply_patch (expanded collapsed hunk, verified) ({path})"),
    })
}

/// Assemble the fresh ground truth for a loop intervention — the single gatherer both
/// the excise rebuild and the loop redirect use: the repeated failing action (from the
/// trim layer), the files the model touched this turn re-read from disk, and the
/// dirty-only lint. The fs reads + lint probe run off-thread (they block).
async fn gather_loop_ground_truth(
    prompt: &Prompt,
    repeated: Option<codex_routing::ground_truth::RepeatedAction>,
) -> codex_routing::ground_truth::GroundTruth {
    let paths = files_touched_this_turn(prompt);
    let gathered = tokio::task::spawn_blocking(move || {
        std::env::current_dir().ok().map(|root| {
            let files = codex_routing::ground_truth::file_snapshot(
                &root,
                &paths,
                codex_routing::ground_truth::DEFAULT_FILE_CAP,
            );
            let lint = codex_routing::ground_truth::lint_digest(&root);
            (files, lint)
        })
    })
    .await
    .ok()
    .flatten();
    let mut gt = codex_routing::ground_truth::GroundTruth {
        repeated,
        ..Default::default()
    };
    if let Some((files, lint)) = gathered {
        gt.files = files;
        gt.lint_digest = lint;
    }
    gt
}

/// The files the model touched this turn — write/edit/patch/read targets — most-recent
/// first, deduped and capped. This is the set the ground-truth provider re-reads from
/// disk so a reasoned rebuild reflects what's ACTUALLY there, not the transcript's
/// echoes of what the model claimed it wrote.
fn files_touched_this_turn(prompt: &Prompt) -> Vec<String> {
    fn add(out: &mut Vec<String>, p: &str) {
        let p = p.trim();
        if !p.is_empty() && out.len() < 5 && !out.iter().any(|e| e == p) {
            out.push(p.to_string());
        }
    }
    let mut out: Vec<String> = Vec::new();
    for item in prompt.input.iter().rev() {
        if out.len() >= 5 {
            break;
        }
        let ResponseItem::FunctionCall {
            name, arguments, ..
        } = item
        else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(arguments) else {
            continue;
        };
        match name.as_str() {
            "write_file" | "create_file" | "edit_file" | "str_replace" | "read_file"
            | "cat_file" => {
                if let Some(p) = v.get("path").and_then(|p| p.as_str()) {
                    add(&mut out, p);
                }
            }
            "apply_patch" => {
                let input = v
                    .get("input")
                    .or_else(|| v.get("patch"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("");
                for line in input.lines() {
                    for marker in ["*** Update File: ", "*** Add File: "] {
                        if let Some(rest) = line.strip_prefix(marker) {
                            add(&mut out, rest);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// The real prior USER task for a compaction handoff's `# Current Request` — never
/// the compaction directive itself. When the harness fires compaction, the newest
/// user message is the directive (identified STRUCTURALLY by the sentinel the
/// harness injects, so this holds regardless of the directive's configured prose).
/// Walk user messages newest-first, skip the directive and the pinned project
/// instructions (AGENTS.md), and return the first genuine task. `None` when there
/// isn't one — the handoff then omits the section rather than surfacing the wrong
/// request.
fn extract_real_user_task(prompt: &Prompt) -> Option<String> {
    for item in prompt.input.iter().rev() {
        let ResponseItem::Message { role, content, .. } = item else {
            continue;
        };
        if role != "user" {
            continue;
        }
        let text: String = content
            .iter()
            .filter_map(|c| match c {
                ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        // The compaction directive carries the sentinel `is_compaction_request`
        // keys on — skip it structurally, so a reworded directive still can't
        // become the resume request.
        if text.contains("<<<LOCAL_COMPACT>>>") || text.contains("CONTEXT CHECKPOINT COMPACTION") {
            continue;
        }
        // The AGENTS.md/CLAUDE.md pin is a user message too, but it's project
        // context, not the task.
        if is_project_instructions_message(&text) {
            continue;
        }
        let cleaned = text.trim();
        if !cleaned.is_empty() {
            return Some(cleaned.to_string());
        }
    }
    None
}

/// Extract the AGENTS.md / CLAUDE.md content from the conversation, if any.
/// Codex injects these as a user message early in `prompt.input` with a
/// recognizable header. We pull the content (between the `<INSTRUCTIONS>`
/// markers when present, otherwise the full message) so it can be pinned to
/// the local model's persistent-context block — same content, more prominent
/// placement than just being one user message in a long history.
fn extract_project_instructions(prompt: &Prompt) -> Option<String> {
    for item in &prompt.input {
        let ResponseItem::Message { role, content, .. } = item else {
            continue;
        };
        if role != "user" {
            continue;
        }
        let text: String = content
            .iter()
            .filter_map(|c| match c {
                ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !is_project_instructions_message(&text) {
            continue;
        }
        // Strip the surrounding `<INSTRUCTIONS>...</INSTRUCTIONS>` if present
        // so the prelude doesn't carry the wrapper tags.
        let body = strip_instructions_wrapper(&text);
        if !body.trim().is_empty() {
            return Some(body);
        }
    }
    None
}

fn is_project_instructions_message(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("# AGENTS.md")
        || trimmed.starts_with("# CLAUDE.md")
        || trimmed.starts_with("AGENTS.md instructions")
        || trimmed.starts_with("CLAUDE.md instructions")
}

fn strip_instructions_wrapper(text: &str) -> String {
    let Some(start) = text.find("<INSTRUCTIONS>") else {
        return text.to_string();
    };
    let after_open = &text[start + "<INSTRUCTIONS>".len()..];
    let inner = match after_open.find("</INSTRUCTIONS>") {
        Some(end) => &after_open[..end],
        None => after_open,
    };
    inner.trim_matches(['\n', '\r']).to_string()
}

/// Trigger threshold, as a percent of the **effective (estimate-space) fit
/// budget**: compaction runs only when trim's estimate exceeds this much of the
/// budget it already targets. Set to 100 — compact ONLY when mechanical trim
/// genuinely couldn't get the prompt under budget (a long active turn whose
/// protected bulk exceeds the window), NOT preemptively. A lower value made
/// compaction fire every turn on prompts that already fit (estimate ~12.5k vs a
/// ~13.6k budget → real ~22.5k in a 32k window): a ~35s LLM-summarization tax per
/// turn that also shredded the model's record of its own recent steps, feeding
/// re-read loops. The overflow handler (which also learns the token ratio) is the
/// backstop for the rare case where the real prompt still overshoots.
const INLINE_COMPACT_TRIGGER_FRACTION: usize = 100;

// ---------------------------------------------------------------------------
// Real-token calibration. We don't have to guess: the server reports the ACTUAL
// prompt-token count on every response (`prompt_tokens`) and on overflow
// (`n_prompt_tokens`). The `chars/4` estimate runs anywhere from 1.8× to ~2.8×
// low on JSON/code-dense content, so instead of a fixed safety factor we learn
// the real ratio (real ÷ estimate) per model and budget against it. Seeds at the
// static default and self-corrects after the first real response.
// ---------------------------------------------------------------------------

/// Default chars→token multiplier before we've measured a model. MUST match
/// `codex_routing::trim`'s internal `ESTIMATE_SAFETY_FACTOR`, because
/// [`calibrated_trim_budget`] pre-scales the budget assuming trim divides by this.
const DEFAULT_SAFETY_FACTOR: f64 = 1.8;
/// Clamp ceiling — even pathologically dense content rarely exceeds this, and it
/// stops one outlier response from starving the budget.
const MAX_SAFETY_FACTOR: f64 = 3.5;

static TOKEN_RATIO: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, f64>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Learned real÷estimate ratio for `model`, or the static default until measured.
fn observed_token_ratio(model: &str) -> f64 {
    TOKEN_RATIO
        .lock()
        .ok()
        .and_then(|m| m.get(model).copied())
        .unwrap_or(DEFAULT_SAFETY_FACTOR)
}

/// Feed the server's REAL prompt-token count back. `ratio = real / estimate`,
/// where `estimate` is the FULL prompt estimate (messages + system + tool
/// schemas) so the value is pure tokenizer density, not conflated with the
/// tool-overhead fraction (which would swing the ratio as the message bulk grows
/// and shrinks). Updated with an ASYMMETRIC EWMA — rise fast, fall slow — because
/// the costs are asymmetric: under-estimating the ratio overflows the window (a
/// wasted re-trim round-trip on a slow box), while over-estimating only spends a
/// little less context. So a denser-than-seen turn pulls the ratio up hard; a
/// lighter turn barely lowers our guard. Clamped to `[DEFAULT, MAX]`. Returns
/// `Some(new_ratio)` only when the value shifted notably (so the caller surfaces
/// "learned X" once, not every turn after it converges).
fn record_token_ratio(model: &str, real_tokens: u64, estimate: usize) -> Option<f64> {
    if real_tokens == 0 || estimate == 0 {
        return None;
    }
    let observed =
        (real_tokens as f64 / estimate as f64).clamp(DEFAULT_SAFETY_FACTOR, MAX_SAFETY_FACTOR);
    let mut m = TOKEN_RATIO.lock().ok()?;
    let cur = m.entry(model.to_string()).or_insert(DEFAULT_SAFETY_FACTOR);
    let before = *cur;
    *cur = if observed > *cur {
        *cur * 0.3 + observed * 0.7 // rise fast toward a denser turn
    } else {
        *cur * 0.8 + observed * 0.2 // fall slow — one light turn shouldn't drop our guard
    };
    ((*cur - before).abs() > 0.15).then_some(*cur)
}

/// A real-token budget (the window already net of the tool-schema reserve — the
/// caller subtracts that first), pre-scaled by the learned ratio so trim's
/// internal (estimate-space, ÷`DEFAULT_SAFETY_FACTOR`) math lands on a prompt that
/// actually fits the server. With a learned ratio of 2.84 and a 1.8 default, this
/// shrinks the budget to ~63%, so the real prompt fits on the first attempt
/// instead of overflowing and being re-trimmed. Identity until the model is
/// measured.
fn calibrated_trim_budget(model: &str, trim_budget: usize) -> usize {
    let observed = observed_token_ratio(model);
    ((trim_budget as f64) * DEFAULT_SAFETY_FACTOR / observed) as usize
}

// ---------------------------------------------------------------------------
// Server context window: detect the REAL n_ctx instead of trusting trim_budget.
// The server's window (llama.cpp `--ctx-size`) is fixed at startup; the per-
// request `num_ctx` we send is ignored. We read it once from `/props` (and learn
// it from overflow errors), then size the prompt budget from it.
// ---------------------------------------------------------------------------

/// Reserve for the model's OWN output when `max_tokens` is unset (it generates
/// into the same window, so the prompt can't fill it).
const CTX_OUTPUT_RESERVE_DEFAULT: usize = 4096;
/// Slop for chat-template / BOS-EOS tokens not in our estimate.
const CTX_MARGIN: usize = 512;
/// Budget when the window is unknown AND no `trim_budget` is configured.
const CTX_FALLBACK: usize = 8192;

static SERVER_CTX: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<String, u64>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// The real context window (`n_ctx`) of the active local CODER endpoint, as
/// detected from `/props`. `0` = not yet known. Kept separate from [`SERVER_CTX`]
/// (which is keyed by base_url and also holds classifier windows) so the
/// harness can ask "what's the window of the model actually doing the work?"
/// without knowing any URLs. Codex's NATIVE auto-compaction reads
/// `model_info.context_window` (compact limit = 90% of it); a local model's
/// default metadata advertises a far larger window than llama.cpp actually loaded,
/// so native compaction never fires. Overriding it with this true value is what
/// lets Codex compact the current turn on its own. See [`detected_coder_context_window`].
static CODER_CONTEXT_WINDOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The real `n_ctx` of the local coder, if local routing has detected it yet.
/// Read by the harness (`codex.rs`) to override the model's advertised window so
/// native auto-compaction has a truthful threshold. `None` before the first coder
/// call (the window falls back to the model's default until then).
pub fn detected_coder_context_window() -> Option<i64> {
    match CODER_CONTEXT_WINDOW.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        n => i64::try_from(n).ok(),
    }
}

/// Probe the local coder endpoint's `/props` ONCE, up front, so
/// [`detected_coder_context_window`] is populated BEFORE the first turn builds its
/// `model_info`. Without this the very first (possibly long) turn would resolve
/// the model's advertised default window and never compact. Cheap: a no-op once
/// the window is known, when local routing isn't configured, or when the probe
/// fails (the per-call path re-detects and the model default applies until then).
pub async fn ensure_coder_context_window() {
    if detected_coder_context_window().is_some() {
        return;
    }
    let Some(state) = get_routing_state().await.as_ref() else {
        return;
    };
    if let Some(n) = resolve_server_ctx(&state.config.light_coder, state).await {
        CODER_CONTEXT_WINDOW.store(n, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Real context window for `endpoint`, cached per base URL. Probes `/props` once;
/// also fed by the overflow handler. `None` if the server doesn't report it.
async fn resolve_server_ctx(endpoint: &OllamaEndpoint, state: &RoutingState) -> Option<u64> {
    if let Some(c) = SERVER_CTX
        .lock()
        .ok()
        .and_then(|m| m.get(&endpoint.base_url).copied())
    {
        return Some(c);
    }
    let probed =
        codex_routing::ollama::probe_context_window(state.pool.client(), &endpoint.base_url).await;
    if let Some(n) = probed {
        record_server_ctx(&endpoint.base_url, n);
    }
    probed
}

fn record_server_ctx(base_url: &str, n_ctx: u64) {
    if n_ctx > 0
        && let Ok(mut m) = SERVER_CTX.lock()
    {
        m.insert(base_url.to_string(), n_ctx);
    }
}

/// The real-token budget for the whole prompt (system + messages + tool schemas),
/// derived from the server's window minus the output reserve and a margin.
/// `trim_budget` is now an OPTIONAL CAP: `0` = use the full detected window;
/// non-zero = cap there (but never above the real window). When the window is
/// unknown, fall back to the configured `trim_budget` (today's behavior).
fn effective_window(
    trim_budget: usize,
    output_reserve: Option<usize>,
    server_ctx: Option<u64>,
) -> usize {
    let reserve = output_reserve
        .filter(|n| *n > 0)
        .unwrap_or(CTX_OUTPUT_RESERVE_DEFAULT);
    match server_ctx {
        Some(ctx) => {
            let derived = (ctx as usize).saturating_sub(reserve + CTX_MARGIN);
            if trim_budget == 0 {
                derived
            } else {
                trim_budget.min(derived)
            }
        }
        None if trim_budget > 0 => trim_budget,
        None => CTX_FALLBACK,
    }
}

/// If the trimmed transcript still exceeds the local model's context budget,
/// run the compaction pipeline on the older-turn portion and replace it with
/// a single summary message. The active turn is left untouched.
///
/// Cached by hash of the older-turn message contents so repeated requests
/// within a session reuse the same summary instead of recompacting.
/// Build the per-response [`TokenUsage`] for a local model. Critically sets
/// `total_tokens` (= input + output). Codex's NATIVE auto-compaction triggers on
/// `total_tokens` (via [`TokenUsage::blended_total`]); leaving it 0 — which
/// `..Default::default()` does — makes Codex believe the context is empty, so it
/// never compacts no matter how full the real window is. One helper so the three
/// local completion paths can't drift apart (which is how the field got missed).
fn local_token_usage(input_tokens: i64, output_tokens: i64) -> TokenUsage {
    TokenUsage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
        ..Default::default()
    }
}

async fn maybe_inline_compact(
    mut trimmed: codex_routing::trim::TrimResult,
    fit_budget: usize,
    state: &RoutingState,
) -> codex_routing::trim::TrimResult {
    // Trigger in ESTIMATE space, relative to the budget that actually fits the
    // server (net of the tokenizer safety factor) — NOT a fraction of the raw,
    // real-token `trim_budget`. `fit_budget` is already calibrated to the model's
    // learned real÷estimate ratio. The estimate (≈ chars/4) runs ~1.8–2.8× low on
    // JSON/code-dense content, so a real-space threshold sat far above any estimate
    // the trimmer produces: the real prompt overflowed the window while the
    // estimate looked comfortably under budget, and compaction never fired.
    // Comparing the estimate against the effective (estimate-space) fit budget
    // closes that dead band. See docs/spec/local-coder-massaging §12.
    let trigger = codex_routing::trim::effective_budget(fit_budget)
        .saturating_mul(INLINE_COMPACT_TRIGGER_FRACTION)
        / 100;

    // Above the trigger, summarize FIRST (semantic, faithful). Compact whichever
    // region holds the BULK of the tokens. Crucially, trim has *already collapsed*
    // older turns into a small state prelude, so in a long agentic loop the active
    // turn is almost always the bulk and the older prelude is tiny — summarizing
    // that prelude (`older_count > 0 → compact older`) does nothing and leaves the
    // real weight untouched, which is why active-turn compaction never ran. So we
    // measure both and compact the heavier side: the active turn's own middle when
    // it dominates, the older prelude only when it genuinely outweighs the turn.
    if trimmed.summary.estimated_input_tokens > trigger {
        if state.config.compactor.enabled {
            let older_count = trimmed.summary.older_turn_message_count;
            let older_est = estimate_combined_tokens("", &trimmed.messages[..older_count]);
            let active_est = estimate_combined_tokens("", &trimmed.messages[older_count..]);
            trimmed = if active_est >= older_est {
                compact_active_turn(trimmed, older_count, fit_budget, state).await
            } else {
                compact_older_turns(trimmed, older_count, fit_budget, state).await
            };
        } else {
            warn!(
                estimated_tokens = trimmed.summary.estimated_input_tokens,
                fit_budget,
                "Trimmed transcript over budget but compactor endpoint is disabled — relying on last-resort drop"
            );
        }
    }

    // FLOOR (always). Mechanical trim no longer drops whole messages, so this is
    // the single place that guarantees a servable prompt: drop oldest messages
    // (always keeping the user request) until the estimate fits the real window.
    // A no-op when trim/compaction already brought it under budget.
    if last_resort_drop(&mut trimmed, fit_budget) > 0 {
        state.push_nudge(
            "Compaction wasn't enough — dropped the oldest messages to fit context (last resort)"
                .to_string(),
        );
    }
    trimmed
}

/// Summarize the older turns (everything before the active turn) into one
/// rolling-summary message, preserving the active turn verbatim. Reuses a cached
/// summary when the older history hasn't shifted. Returns `trimmed` unchanged on
/// failure — the caller's [`last_resort_drop`] then guarantees fit.
async fn compact_older_turns(
    mut trimmed: codex_routing::trim::TrimResult,
    older_count: usize,
    fit_budget: usize,
    state: &RoutingState,
) -> codex_routing::trim::TrimResult {
    // Hash the older messages so we can reuse the summary if the conversation
    // history hasn't shifted between requests.
    let older_messages: Vec<serde_json::Value> = trimmed.messages[..older_count].to_vec();
    let active_messages: Vec<serde_json::Value> = trimmed.messages[older_count..].to_vec();
    let content_hash = hash_messages(&older_messages);

    if let Some(cached) = state
        .inline_compact_cache
        .lock()
        .ok()
        .and_then(|guard| guard.clone())
        && cached.older_content_hash == content_hash
    {
        info!("Reusing cached inline-compaction summary");
        state.push_nudge("Reused cached compaction summary for older turns".to_string());
        let mut new_messages = vec![cached.summary_message];
        new_messages.extend(active_messages);
        let new_token_estimate = estimate_combined_tokens(&trimmed.system, &new_messages);
        trimmed.messages = new_messages;
        trimmed.summary.older_turn_message_count = 1;
        trimmed.summary.estimated_input_tokens = new_token_estimate;
        return trimmed;
    }

    info!(
        estimated_tokens = trimmed.summary.estimated_input_tokens,
        fit_budget, older_count, "Trimmed transcript over budget — running inline compaction"
    );

    let compaction_config = codex_routing::compaction::CompactionConfig::default();
    // Use the most recent older user message as the "current request" anchor
    // for the summary.
    let anchor = older_messages
        .iter()
        .rev()
        .find_map(|m| {
            if m.get("role").and_then(|r| r.as_str()) == Some("user") {
                m.get("content")
                    .and_then(|c| c.as_str())
                    .map(str::to_string)
            } else {
                None
            }
        })
        .unwrap_or_else(|| "(rolling summary)".to_string());

    let summary_text = match codex_routing::compaction::compact_transcript(
        &older_messages,
        &anchor,
        &state.pool,
        &state.config.compactor,
        &compaction_config,
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "Inline compaction failed — sending trimmed transcript as-is");
            return trimmed;
        }
    };

    let summary_message = serde_json::json!({
        "role": "user",
        "content": format!(
            "[earlier conversation summarized]\n\n{summary_text}"
        ),
    });

    if let Ok(mut guard) = state.inline_compact_cache.lock() {
        *guard = Some(InlineCompactCacheEntry {
            older_content_hash: content_hash,
            summary_message: summary_message.clone(),
        });
    }

    let mut new_messages = vec![summary_message];
    new_messages.extend(active_messages);
    let new_token_estimate = estimate_combined_tokens(&trimmed.system, &new_messages);
    info!(
        before_tokens = trimmed.summary.estimated_input_tokens,
        after_tokens = new_token_estimate,
        "Inline compaction complete"
    );
    state.push_nudge(format!(
        "Compacted older turns to fit context — ~{}k→{}k tokens (estimate)",
        trimmed.summary.estimated_input_tokens / 1000,
        new_token_estimate / 1000,
    ));
    trimmed.messages = new_messages;
    trimmed.summary.older_turn_message_count = 1;
    trimmed.summary.estimated_input_tokens = new_token_estimate;
    trimmed
}

/// Keep verbatim in a long active turn: the user request + the most recent
/// exchanges. Everything between is summarized.
const KEEP_RECENT_ACTIVE_MESSAGES: usize = 6;

/// Plan the split of an over-budget active turn. `active_start` is where the
/// active turn begins (= `older_turn_message_count`; everything before it is a
/// collapsed older-turn prelude that is kept verbatim). Returns
/// `(request_idx, summarize_start, summarize_end)` so that `messages[..=request_idx]`
/// (prelude + the request) and `messages[summarize_end..]` (recent steps) are kept
/// verbatim and `messages[summarize_start..summarize_end]` is summarized. `None`
/// when the middle is too small to be worth a compaction call.
fn plan_active_turn_split(
    messages: &[serde_json::Value],
    active_start: usize,
    keep_recent: usize,
) -> Option<(usize, usize, usize)> {
    let n = messages.len();
    // The request is the first user-role message at/after the active-turn start.
    let request_idx = messages
        .iter()
        .enumerate()
        .skip(active_start)
        .find(|(_, m)| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .map(|(i, _)| i)
        .unwrap_or(active_start.min(n.saturating_sub(1)));
    let recent = keep_recent.min(n.saturating_sub(request_idx + 1));
    let start = request_idx + 1;
    let end = n.saturating_sub(recent);
    if end <= start + 1 {
        return None; // need ≥2 middle messages to bother summarizing
    }
    Some((request_idx, start, end))
}

/// Summarize the *middle* of a single active turn that is over budget on its own
/// (a long agentic loop with no new user message). Keeps the user request and the
/// most recent steps verbatim; replaces the middle with one rolling-summary
/// message. Returns `trimmed` unchanged when there's too little to compact or the
/// compactor fails — the caller's [`last_resort_drop`] then guarantees fit. This
/// is the case that previously self-terminated the session (the active turn alone
/// blew the window and trim crudely dropped the user request).
async fn compact_active_turn(
    mut trimmed: codex_routing::trim::TrimResult,
    active_start: usize,
    fit_budget: usize,
    state: &RoutingState,
) -> codex_routing::trim::TrimResult {
    let Some((request_idx, start, end)) =
        plan_active_turn_split(&trimmed.messages, active_start, KEEP_RECENT_ACTIVE_MESSAGES)
    else {
        warn!(
            estimated_tokens = trimmed.summary.estimated_input_tokens,
            fit_budget,
            "Active turn over budget but too short to compact — deferring to last-resort drop"
        );
        return trimmed;
    };

    let middle: Vec<serde_json::Value> = trimmed.messages[start..end].to_vec();
    let anchor = trimmed.messages[request_idx]
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("(current task)")
        .to_string();

    // Reuse a rolling summary for the unchanged prefix; only LLM-compact the new
    // tail. Without this, every overflow re-summarizes the whole (growing) turn
    // from scratch — the compaction storm. With it, each re-compaction is O(delta).
    let cache = state
        .active_compact_cache
        .lock()
        .ok()
        .and_then(|g| g.clone());
    let plan = plan_active_compaction(&middle, cache.as_ref());
    let to_compact: Vec<serde_json::Value> = match &plan {
        ActiveCompactPlan::Incremental { summary, from } => {
            info!(
                middle = middle.len(),
                reused_prefix = *from,
                new_tail = middle.len() - *from,
                "Active-turn compaction reusing rolling summary (incremental)"
            );
            let mut v = vec![serde_json::json!({
                "role": "user",
                "content": format!("[summary of earlier steps in this task]\n\n{summary}"),
            })];
            v.extend_from_slice(&middle[*from..]);
            v
        }
        ActiveCompactPlan::Full => {
            info!(
                estimated_tokens = trimmed.summary.estimated_input_tokens,
                fit_budget,
                middle = middle.len(),
                "Active turn over budget — compacting its middle (full)"
            );
            middle.clone()
        }
    };

    let summary_text = match codex_routing::compaction::compact_transcript(
        &to_compact,
        &anchor,
        &state.pool,
        &state.config.compactor,
        &codex_routing::compaction::CompactionConfig::default(),
    )
    .await
    {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "Active-turn compaction failed — deferring to last-resort drop");
            return trimmed;
        }
    };

    // Store the rolling summary now covering the WHOLE current middle, so the next
    // overflow reuses it and only folds in whatever the model appended since.
    if let Ok(mut g) = state.active_compact_cache.lock() {
        *g = Some(ActiveCompactEntry {
            prefix_len: middle.len(),
            prefix_hash: hash_prefix(&middle),
            summary: summary_text.clone(),
        });
    }

    let summary_message = serde_json::json!({
        "role": "user",
        "content": format!(
            "[earlier steps in this task summarized]\n\n{summary_text}"
        ),
    });

    let mut new_messages: Vec<serde_json::Value> = trimmed.messages[..=request_idx].to_vec();
    new_messages.push(summary_message);
    new_messages.extend_from_slice(&trimmed.messages[end..]);
    let new_token_estimate = estimate_combined_tokens(&trimmed.system, &new_messages);
    info!(
        before_tokens = trimmed.summary.estimated_input_tokens,
        after_tokens = new_token_estimate,
        kept_recent = trimmed.messages.len() - end,
        "Active-turn compaction complete"
    );
    state.push_nudge(format!(
        "Compacted the active turn ({} steps summarized; kept the request + last {}) to fit context — ~{}k→{}k tokens",
        end - start,
        trimmed.messages.len() - end,
        trimmed.summary.estimated_input_tokens / 1000,
        new_token_estimate / 1000,
    ));
    trimmed.summary.older_turn_message_count = request_idx + 2; // request(s) + summary
    trimmed.messages = new_messages;
    trimmed.summary.estimated_input_tokens = new_token_estimate;
    trimmed
}

/// Floor that guarantees a servable prompt after compaction. Drops oldest
/// messages (always keeping the user request) until the estimate fits the real
/// window. A no-op when trim/compaction already brought it under budget. Returns
/// the number of chars dropped (0 = nothing dropped) so the caller can surface it.
fn last_resort_drop(trimmed: &mut codex_routing::trim::TrimResult, fit_budget: usize) -> usize {
    let dropped =
        codex_routing::trim::drop_to_fit(&trimmed.system, &mut trimmed.messages, fit_budget);
    if dropped > 0 {
        trimmed.summary.kept_items = trimmed.messages.len();
        trimmed.summary.estimated_input_tokens =
            estimate_combined_tokens(&trimmed.system, &trimmed.messages);
        warn!(
            dropped_chars = dropped,
            kept_items = trimmed.summary.kept_items,
            fit_budget,
            "Compaction insufficient — dropped oldest messages as last resort"
        );
    }
    dropped
}

/// Sum the token estimate of the system prompt and the text content of every
/// message — same shape `trim_for_local` uses internally.
fn estimate_combined_tokens(system: &str, messages: &[serde_json::Value]) -> usize {
    let messages_text: String = messages
        .iter()
        .filter_map(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .map(str::to_string)
        })
        .collect::<Vec<_>>()
        .join("\n");
    codex_routing::metrics::estimate_tokens(system)
        + codex_routing::metrics::estimate_tokens(&messages_text)
}

fn hash_messages(messages: &[serde_json::Value]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    for m in messages {
        if let Some(s) = m.get("content").and_then(|c| c.as_str()) {
            s.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Rough estimate of how many tokens the cloud model would have processed for
/// this prompt. Used as the savings metric when routing locally.
///
/// Walks every message item in the prompt, counts text length, and applies the
/// shared `estimate_tokens` heuristic. Tool calls and outputs are not counted
/// here — we underestimate slightly, but this is only a coarse savings number.
fn estimate_prompt_tokens(prompt: &Prompt) -> usize {
    let mut acc = String::new();
    if !prompt.base_instructions.text.is_empty() {
        acc.push_str(&prompt.base_instructions.text);
        acc.push('\n');
    }
    for item in &prompt.input {
        if let ResponseItem::Message { content, .. } = item {
            for c in content {
                match c {
                    ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                        acc.push_str(text);
                        acc.push('\n');
                    }
                    _ => {}
                }
            }
        }
    }
    codex_routing::metrics::estimate_tokens(&acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_bare_json_update_plan_leak() {
        // The EXACT leak from session 019f38a5: update_plan dumped as content, which the
        // model repeated until the turn died empty. Now recovered into a real call.
        let leak = r#"{"plan":[{"status":"completed","step":"Write Lambda handler code"},{"status":"in_progress","step":"Create integration test"}]}"#;
        assert!(recover_bare_update_plan(leak).is_some(), "the plan leak recovers");
        // A bare JSON that isn't a plan → not recovered (left to the strip/re-prompt).
        assert!(recover_bare_update_plan(r#"{"result": "done", "status": 200}"#).is_none());
        // Prose isn't a plan either.
        assert!(recover_bare_update_plan("I finished the task.").is_none());
    }

    #[test]
    fn bare_json_leak_detected_prose_not() {
        // The exact Fabliq leak: an update_plan call dumped as a JSON object.
        assert!(looks_like_bare_json_object(
            "\n\n{\n  \"plan\": \"1. List files.\\n2. Create handler.\",\n  \"update_plan\": {"
        ));
        assert!(looks_like_bare_json_object("{\"path\":\"a.py\",\"content\":\"x\"}"));
        // Ordinary prose (even mentioning braces) is NOT a leak.
        assert!(!looks_like_bare_json_object("I created the file. Done."));
        assert!(!looks_like_bare_json_object("Use `{}` for an empty dict."));
        assert!(!looks_like_bare_json_object("1. Do the thing\n2. Verify"));
    }

    #[test]
    fn extract_real_user_task_skips_compaction_directive() {
        let msg = |role: &str, text: &str| ResponseItem::Message {
            id: None,
            role: role.to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
            end_turn: None,
            phase: None,
        };
        let prompt = Prompt {
            input: vec![
                msg("user", "Write a Python Lambda handler that resolves an Ada Handle."),
                msg("assistant", "working on it"),
                msg(
                    "user",
                    "<<<LOCAL_COMPACT>>> Summarize the thread for continuation. Preserve edits, unfinished work, blockers, decisions, and the latest real user intent.",
                ),
            ],
            ..Default::default()
        };
        // The newest user message is the compaction directive; it must be skipped
        // so the resuming model gets the REAL task, not "summarize the thread".
        let task = extract_real_user_task(&prompt).expect("a real task is present");
        assert!(task.contains("Ada Handle"), "got: {task}");
        assert!(
            !task.contains("Summarize the thread"),
            "the compaction directive leaked into current_request: {task}"
        );
    }

    #[test]
    fn truncated_write_path_recovers_path_from_unterminated_args() {
        use std::collections::BTreeMap;
        // A write_file cut off mid-content: the JSON is unterminated, but `path`
        // precedes `content` and is intact.
        let mut acc: BTreeMap<usize, StreamToolCallAcc> = BTreeMap::new();
        acc.insert(
            0,
            StreamToolCallAcc {
                id: None,
                name: Some("write_file".to_string()),
                arguments: r##"{"path":"/home/x/test_lambda.py","content":"#!/usr/bin/env python3\ndef test_"##
                    .to_string(),
            },
        );
        assert_eq!(
            truncated_write_path(&acc).as_deref(),
            Some("/home/x/test_lambda.py")
        );

        // A non-write tool call is not a truncated-write footgun.
        let mut acc2: BTreeMap<usize, StreamToolCallAcc> = BTreeMap::new();
        acc2.insert(
            0,
            StreamToolCallAcc {
                id: None,
                name: Some("shell".to_string()),
                arguments: "{}".to_string(),
            },
        );
        assert_eq!(truncated_write_path(&acc2), None);
    }

    #[test]
    fn detected_coder_context_window_maps_zero_to_none() {
        use std::sync::atomic::Ordering;
        // 0 (never detected) → None so the harness keeps the model's default window.
        CODER_CONTEXT_WINDOW.store(0, Ordering::Relaxed);
        assert_eq!(detected_coder_context_window(), None);
        // A detected n_ctx is surfaced as the real window for native auto-compaction.
        CODER_CONTEXT_WINDOW.store(49152, Ordering::Relaxed);
        assert_eq!(detected_coder_context_window(), Some(49152));
        CODER_CONTEXT_WINDOW.store(0, Ordering::Relaxed); // reset for other tests
    }

    #[test]
    fn present_local_tools_renames_web_search_and_moves_shell_last() {
        let tool = |name: &str| {
            serde_json::json!({
                "type": "function",
                "function": { "name": name, "parameters": {"type": "object", "properties": {}} }
            })
        };
        let mut tools = vec![
            tool("shell"),
            tool("local_web_search"),
            tool("web_fetch"),
            tool("write_file"),
        ];
        present_local_tools(&mut tools);

        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(|v| v.as_str()))
            .collect();
        // The Brave search tool is presented to the model as plain `web_search`;
        // the internal name never reaches it.
        assert!(names.contains(&"web_search"), "names: {names:?}");
        assert!(!names.contains(&"local_web_search"), "names: {names:?}");
        // `shell` is last so the model reaches for the specific tools first.
        assert_eq!(names.last(), Some(&"shell"), "names: {names:?}");
    }

    #[test]
    fn effective_window_derives_from_server_ctx() {
        // 32768 window − 2048 output − 512 margin = 30208 when trim_budget=0 (auto).
        assert_eq!(effective_window(0, Some(2048), Some(32768)), 30208);
        // A trim_budget below the derived window caps there.
        assert_eq!(effective_window(24576, Some(2048), Some(32768)), 24576);
        // A trim_budget above the window can't exceed it.
        assert_eq!(effective_window(40000, Some(2048), Some(32768)), 30208);
        // Unset max_tokens uses the default output reserve.
        assert_eq!(
            effective_window(0, None, Some(32768)),
            32768 - CTX_OUTPUT_RESERVE_DEFAULT - CTX_MARGIN
        );
        // Window unknown → configured trim_budget, or the fallback when it's 0.
        assert_eq!(effective_window(24576, None, None), 24576);
        assert_eq!(effective_window(0, None, None), CTX_FALLBACK);
    }

    #[test]
    fn token_ratio_rises_fast_and_falls_slow() {
        // Unique model key isolates this test's slot in the global ratio map.
        let model = "test-ewma-asymmetric-9b";
        // From the 1.8 default, a dense turn (observed 3.0) must jump most of the way.
        record_token_ratio(model, 3000, 1000);
        let dense = observed_token_ratio(model);
        assert!(
            dense > 2.5,
            "a dense turn must pull the ratio up hard: {dense}"
        );
        // A following light turn (observed 1.8) should ease it down only slightly —
        // one cheap turn must not drop our guard and re-open the overflow door.
        record_token_ratio(model, 1800, 1000);
        let light = observed_token_ratio(model);
        assert!(
            light < dense,
            "a light turn eases the ratio down: {light} vs {dense}"
        );
        assert!(
            light > dense - 0.4,
            "but only slightly — fall slow: {light} vs {dense}"
        );
    }

    #[test]
    fn budget_reserves_tools_so_real_prompt_fits_window() {
        // Converge a unique model's ratio to ~2.5 (dense JSON/code territory).
        let model = "test-budget-fits-9b";
        for _ in 0..8 {
            record_token_ratio(model, 2500, 1000);
        }
        let ratio = observed_token_ratio(model);
        let window = 45056usize; // e.g. 49664 ctx − 4096 output − 512 margin
        let tool_est = 3000usize;
        // Mirror the production budget math: reserve tools in REAL tokens, then
        // calibrate the remainder to estimate space.
        let real_tools = (tool_est as f64 * ratio) as usize;
        let fit = calibrated_trim_budget(model, window - real_tools);
        let messages_est = (fit as f64 / DEFAULT_SAFETY_FACTOR) as usize;
        // What the server will actually see: (messages + tools) tokenized at `ratio`.
        let real_total = ((messages_est + tool_est) as f64 * ratio) as usize;
        assert!(
            real_total <= window + 64,
            "real prompt must fit the window: {real_total} > {window}"
        );
        assert!(
            real_total > window * 9 / 10,
            "and should use most of it, not starve context: {real_total}"
        );
    }

    #[test]
    fn represent_shell_writes_restores_write_file() {
        // A write_file the model emitted last turn was lowered to a base64 shell
        // call and RECORDED as shell. The inbound pass must turn it back into
        // write_file (same call_id, original path+content) so the model sees its
        // own tool and trim's state-extraction recognizes the write.
        let t = codex_routing::tool_aliases::write_file_to_base64_shell(
            &serde_json::json!({"path": "src/x.py", "content": "print('hi')\n"}),
        )
        .unwrap();
        let items = vec![
            ResponseItem::FunctionCall {
                id: Some("fc0".into()),
                name: "shell".into(),
                namespace: None,
                arguments: t.args.to_string(),
                call_id: "call0".into(),
            },
            // A real (non-massage) shell call must pass through untouched.
            ResponseItem::FunctionCall {
                id: Some("fc1".into()),
                name: "shell".into(),
                namespace: None,
                arguments: serde_json::json!({"command": ["bash", "-lc", "pytest -q"]}).to_string(),
                call_id: "call1".into(),
            },
        ];
        let out = represent_shell_writes(&items);
        match &out[0] {
            ResponseItem::FunctionCall {
                name,
                arguments,
                call_id,
                ..
            } => {
                assert_eq!(name, "write_file", "the base64 shell write is re-presented");
                assert_eq!(
                    call_id, "call0",
                    "call_id preserved so its output still matches"
                );
                let v: serde_json::Value = serde_json::from_str(arguments).unwrap();
                assert_eq!(v["path"], "src/x.py");
                assert_eq!(v["content"], "print('hi')\n");
            }
            other => panic!("expected write_file, got {other:?}"),
        }
        match &out[1] {
            ResponseItem::FunctionCall { name, .. } => {
                assert_eq!(name, "shell", "a genuine shell call is left alone")
            }
            other => panic!("expected shell, got {other:?}"),
        }
    }

    /// The actual transcript content from the hour-long stuck loop
    /// (session 019f15a2): 140 real messages — web_fetch/exec/shell/apply_patch.
    const LOOP_FIXTURE: &str = include_str!("compaction_loop_fixture.json");

    fn loop_messages() -> Vec<serde_json::Value> {
        serde_json::from_str(LOOP_FIXTURE).expect("fixture parses")
    }

    #[test]
    fn active_compaction_reuses_rolling_summary_on_real_loop_content() {
        let msgs = loop_messages();
        assert!(msgs.len() >= 64, "fixture has the real loop content");

        // Round 1 — cold cache: must compact the whole middle.
        let round1: Vec<_> = msgs[..60].to_vec();
        assert_eq!(
            plan_active_compaction(&round1, None),
            ActiveCompactPlan::Full
        );

        // Store the round-1 rolling summary (what compact_active_turn would cache).
        let entry = ActiveCompactEntry {
            prefix_len: round1.len(),
            prefix_hash: hash_prefix(&round1),
            summary: "ROLLING SUMMARY v1".to_string(),
        };

        // Round 2 — the append-only turn grew by 4 real steps. Must reuse the
        // 60-message prefix and only re-compact the 4 new ones.
        let round2: Vec<_> = msgs[..64].to_vec();
        match plan_active_compaction(&round2, Some(&entry)) {
            ActiveCompactPlan::Incremental { summary, from } => {
                assert_eq!(from, 60, "reuse the whole prior prefix");
                assert_eq!(summary, "ROLLING SUMMARY v1");
                assert_eq!(
                    round2.len() - from,
                    4,
                    "only the 4 new steps are re-compacted"
                );
            }
            ActiveCompactPlan::Full => panic!("append-only growth must be incremental"),
        }

        // Correctness: if an EARLIER step changes (e.g. trim dropped one and the
        // prefix shifted), the cache must invalidate and recompact fully.
        let mut mutated = round2.clone();
        mutated[10] = serde_json::json!({"role": "user", "content": "CHANGED"});
        assert_eq!(
            plan_active_compaction(&mutated, Some(&entry)),
            ActiveCompactPlan::Full,
            "a changed prefix must fall back to full compaction"
        );
    }

    #[test]
    fn incremental_compaction_collapses_the_storm_on_real_content() {
        // Simulate the real loop: the active turn grows a few steps between each
        // overflow, and compaction runs each time. Compare total work re-compacted
        // under the OLD (always-full) path vs the NEW (incremental) path.
        let msgs = loop_messages();
        let mut cache: Option<ActiveCompactEntry> = None;
        let (mut full_work, mut incr_work) = (0usize, 0usize);
        let mut size = 40;
        while size <= msgs.len() {
            let middle = &msgs[..size];
            full_work += middle.len(); // old: re-summarize the whole growing turn
            incr_work += match plan_active_compaction(middle, cache.as_ref()) {
                ActiveCompactPlan::Full => middle.len(),
                ActiveCompactPlan::Incremental { from, .. } => middle.len() - from,
            };
            cache = Some(ActiveCompactEntry {
                prefix_len: middle.len(),
                prefix_hash: hash_prefix(middle),
                summary: format!("s{size}"),
            });
            size += 4;
        }
        println!("re-compacted items — old(full)={full_work}  new(incremental)={incr_work}");
        assert!(
            incr_work * 3 < full_work,
            "incremental must do far less work on the real loop: {incr_work} vs {full_work}"
        );
    }

    #[test]
    fn plan_active_turn_split_keeps_request_and_recent() {
        let msg = |role: &str, body: &str| serde_json::json!({ "role": role, "content": body });
        // [request, step0 .. step7] — one user request + 8 agentic steps.
        let mut msgs = vec![msg("user", "do the task")];
        for i in 0..8 {
            msgs.push(msg("assistant", &format!("step {i}")));
        }
        let (req, start, end) =
            plan_active_turn_split(&msgs, 0, 3).expect("a 9-message turn should split");
        assert_eq!(req, 0, "the request is index 0 and is kept verbatim");
        assert_eq!(start, 1, "summarization starts right after the request");
        assert_eq!(end, msgs.len() - 3, "the last 3 messages are kept verbatim");
        assert!(end > start + 1, "there is a real middle to summarize");
    }

    #[test]
    fn plan_active_turn_split_skips_older_prelude() {
        // [older_prelude, request, step0..step6] — a collapsed older turn sits in
        // front. The split must find the ACTIVE request (index 1), not the prelude,
        // and keep the prelude verbatim.
        let msg = |role: &str, body: &str| serde_json::json!({ "role": role, "content": body });
        let mut msgs = vec![msg("user", "[state prelude] earlier turns summarized")];
        msgs.push(msg("user", "do the active task"));
        for i in 0..7 {
            msgs.push(msg("assistant", &format!("step {i}")));
        }
        let (req, start, _end) =
            plan_active_turn_split(&msgs, 1, 3).expect("should split the active turn");
        assert_eq!(
            req, 1,
            "request is the active turn's user message, after the prelude"
        );
        assert_eq!(
            start, 2,
            "summarization starts after that request, preserving the prelude"
        );
    }

    #[test]
    fn plan_active_turn_split_none_when_too_short() {
        let msg = |role: &str| serde_json::json!({ "role": role, "content": "x" });
        // request + 2 messages, keep_recent 6 → no middle left to summarize.
        let msgs = vec![msg("user"), msg("assistant"), msg("tool")];
        assert!(
            plan_active_turn_split(&msgs, 0, 6).is_none(),
            "too short → None, so the caller falls back to last-resort drop"
        );
    }

    #[test]
    fn role_maps_to_route_for_tool_attachment() {
        // The crux of the failover-tool fix: when a cloud-classified request
        // falls back to the local coder, the effective route must read as
        // LightCoder so tools are attached (use_tools keys on this).
        assert_eq!(
            route_target_for_role("light_coder"),
            Some(RouteTarget::LightCoder)
        );
        assert_eq!(
            route_target_for_role("light_reasoner"),
            Some(RouteTarget::LightReasoner)
        );
        assert_eq!(
            route_target_for_role("light_reasoner_backup"),
            Some(RouteTarget::LightReasoner)
        );
        // Non-request roles leave the caller's original route intact.
        assert_eq!(route_target_for_role("classifier"), None);
        assert_eq!(route_target_for_role("cloud_mini"), None);
    }

    #[test]
    fn text_is_product_is_the_only_role_difference() {
        // Every local role is a full coder (same tools, same edit ability, set in
        // `build_local_tools` with LIGHT_CODER_TOOL_NAMES). The ONLY behavioral
        // difference is whether a text-only turn completes the turn: a coder must
        // act; a reasoner's text can be the answer.
        assert!(!role_text_is_product("light_coder"));
        assert!(role_text_is_product("light_reasoner"));
        assert!(role_text_is_product("light_reasoner_backup"));
        // A cloud role served locally falls through to coder behavior (must act).
        assert!(!role_text_is_product("cloud_mini"));
    }

    #[test]
    fn surviving_synthetic_tool_is_detected() {
        // A real tool (apply_patch) or shell never trips it.
        let ok = vec![
            serde_json::json!({"function": {"name": "apply_patch", "arguments": "{}"}}),
            serde_json::json!({"name": "shell", "arguments": "{}"}),
        ];
        assert!(surviving_untranslated_synthetic(&ok).is_none());
        // write_file/create_file have real handlers and are NOT translated, so
        // they must NOT be flagged as "surviving" — listing them did, which made
        // every write_file call re-prompt as malformed. Their botched args are
        // repaired upstream in translate_one_native_call instead.
        let real_handler = vec![
            serde_json::json!({"function": {"name": "write_file", "arguments": "{malformed"}}),
            serde_json::json!({"name": "create_file", "arguments": "{malformed"}),
        ];
        assert!(surviving_untranslated_synthetic(&real_handler).is_none());
        // A TRANSLATED synthetic (no real handler) that survived = malformed args.
        let bad =
            vec![serde_json::json!({"function": {"name": "edit_file", "arguments": "{malformed"}})];
        assert_eq!(
            surviving_untranslated_synthetic(&bad).as_deref(),
            Some("edit_file")
        );
    }

    #[test]
    fn tool_call_is_enforced_only_on_actor_retries() {
        // First call (count 0): never forced — must allow a natural text
        // completion so the turn can terminate.
        assert!(!enforce_tool_call_on_retry(0, true, false, &None));
        // Actor retry: enforced.
        assert!(enforce_tool_call_on_retry(1, true, false, &None));
        // Reasoner retry: never forced — its text is the product.
        assert!(!enforce_tool_call_on_retry(2, true, true, &None));
        // No tool set: nothing to constrain to.
        assert!(!enforce_tool_call_on_retry(1, false, false, &None));
        // Operator set an explicit tool_choice → that wins, we don't override.
        assert!(!enforce_tool_call_on_retry(
            1,
            true,
            false,
            &Some("auto".into())
        ));
    }
}
