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

use crate::client_common::{Prompt, ResponseStream};
use codex_api::ResponseEvent;
use codex_protocol::models::{ContentItem, ResponseItem};
use codex_protocol::protocol::TokenUsage;
use codex_routing::OllamaClientPool;
use codex_routing::classifier::RouteTarget;
use codex_routing::config::{OllamaEndpoint, RoutingConfig};
use codex_routing::failover::{self, FailoverAction, FailureType};
use codex_routing::local_dispatch::OllamaTextResponse;
use std::sync::Arc;
use tokio::sync::{OnceCell, mpsc};
use tracing::{info, warn};

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
const LIGHT_CODER_TOOL_NAMES: &[&str] = &[
    "shell",
    "list_dir",
    "view_image",
    "update_plan",
    "local_web_search",
    "web_fetch",
    "request_permissions",
    "exec_command",
    "write_stdin",
];

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
                "description": "Create OR completely overwrite a file with its full content. This is the DEFAULT, most reliable way to write or change a file — you supply the entire file, so there is nothing to match and nothing to fail. Keep files small and focused (prefer several small modules over one big file) so a full rewrite stays cheap. Do NOT use apply_patch, diff, or patch syntax — that path is unavailable and will fail.",
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
    let args_value: serde_json::Value = raw_arguments
        .map(|v| {
            if let Some(s) = v.as_str() {
                serde_json::from_str(s).unwrap_or(serde_json::Value::Null)
            } else {
                v.clone()
            }
        })
        .unwrap_or(serde_json::Value::Null);

    // `write_file`/`create_file` have real handlers and are not translated — but a
    // small model often emits file content the JSON parser rejects (raw newlines,
    // bare quotes). Repair it here so the handler receives valid `{path, content}`
    // instead of erroring and triggering a re-prompt loop. Valid args fall through
    // untouched.
    if matches!(name.as_str(), "write_file" | "create_file") {
        if args_value.is_null()
            && let Some(raw) = raw_args_str.as_deref()
            && let Some(repaired) = codex_routing::tool_aliases::recover_write_file_args(raw)
        {
            info!(
                tool = %name,
                bytes = repaired.get("content").and_then(|c| c.as_str()).map(str::len).unwrap_or(0),
                "Recovered botched write_file JSON arguments (raw newlines / unescaped quotes)"
            );
            set_call_arguments(&mut call, &name, &repaired.to_string());
        }
        return call;
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
            // `write_file`/`create_file` are NOT translated — they have a real
            // handler (handlers/write_file.rs) so the model sees its own call,
            // not a `shell` printf it misreads as mangled. See the diagnosis in
            // that file.
            "read_file" | "cat_file" => {
                codex_routing::tool_aliases::normalize_read_file_call(&args_value)
            }
            // apply_patch is being phased out for local models (it chronically
            // fails — the 9B can't produce matching context). A pure Add File is
            // equivalent to writing the whole file, so route it to the robust
            // write_file handler (which overwrites, so it can't hit "Cannot add:
            // already exists"). Update/Delete patches still normalize as before;
            // failed Updates get steered to a write_file rewrite by the trim layer.
            "apply_patch" => codex_routing::tool_aliases::apply_patch_add_to_write_file(
                &args_value,
            )
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
            .filter(|t| names.contains(&t.name()))
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
    }
    ollama_tools
}

fn build_tool_hint(tool_names: &[&str]) -> String {
    let has = |name: &str| tool_names.contains(&name);
    let mut lines = vec!["You have ONLY the following tools. You MUST call them by these exact names with the exact argument shape shown in the examples — never invent tool names, never guess at argument shapes.".to_string()];

    if has("write_file") || has("edit_file") {
        lines.push(
            "CRITICAL — writing files: to create or change a file you MUST call `write_file` (new file) or `edit_file` (existing file). Code, diffs, or file contents placed in your REPLY TEXT are NOT saved to disk — only tool calls modify files. Never satisfy a \"write/create/fix this file\" request by pasting a code block in your message; always emit the corresponding tool call instead.".to_string(),
        );
    }

    for name in tool_names {
        let block = match *name {
            "shell" => {
                "- `shell`: Run any shell command. Use this for `ls`, `cat`, `rg`, `grep`, `find`, `mkdir`, `rm`, `cd`, `pwd`, build/test commands, package installs, writing files via heredoc — anything you would type at a terminal.\n  REQUIRED ARG SHAPE: `command` MUST be a JSON array of strings, ALWAYS prefixed with `[\"bash\", \"-lc\", \"<your command line>\"]`.\n  Correct example: `{\"command\": [\"bash\", \"-lc\", \"ls -la\"]}`.\n  WRONG: `{\"command\": \"ls -la\"}` (must be an array).\n  WRONG: `{\"command\": [\"bash\", \"-lc\", \"[bash, -lc, ls]\"]}` (do NOT nest the bash invocation; the third element is your literal shell command)."
            }
            "apply_patch" => {
                "- `apply_patch`: Create, modify, or delete files via a structured patch. Prefer this over `shell echo > file` for writing files.\n\n  TWO FORMATS ACCEPTED — pick whichever is most natural:\n\n  FORMAT A: standard unified diff (the format `git diff` produces). This works as-is — file headers `--- a/path` / `+++ b/path` and hunk headers `@@ -L,N +L,N @@` are fine. Example:\n  ```\n  --- a/handler.py\n  +++ b/handler.py\n  @@ -17,7 +17,7 @@\n   def lambda_handler(event, context):\n  -    url = \"https://api.handle.me/resolve/{handle}\"\n  +    url = \"https://api.handle.me/handles/{handle}\"\n       return requests.get(url)\n  ```\n  `/dev/null` for one side means create or delete: `--- /dev/null` + `+++ b/new.py` adds a new file; `--- a/old.py` + `+++ /dev/null` deletes one.\n\n  FORMAT B: Codex native format. Use this when you want explicit anchor-by-context matching:\n  ```\n  *** Begin Patch\n  *** Update File: handler.py\n  @@ def lambda_handler(event, context):\n  -    url = \"https://api.handle.me/resolve/{handle}\"\n  +    url = \"https://api.handle.me/handles/{handle}\"\n  *** End Patch\n  ```\n  Use `*** Add File: <path>` for new files (every body line prefixed `+`), `*** Update File: <path>` for edits, `*** Delete File: <path>` for deletes.\n\n  PREFIX RULE (both formats) — every non-empty line in a hunk body MUST start with EXACTLY ONE of:\n    `+` ... a line you are ADDING\n    `-` ... a line you are REMOVING (Update only)\n    ` ` (a single space) ... a line that is UNCHANGED, included only as context to anchor the change (Update only)\n  Bare code lines without one of these prefixes are INVALID."
            }
            "write_file" => {
                "- `write_file`: Create OR overwrite a file with its FULL contents. Args: `{\"path\": \"<file>\", \"content\": \"<entire file>\"}`. This is the DEFAULT and most reliable way to write or change a file — you supply the whole file, so there is nothing to match and nothing to fail. Keep files SMALL and focused (prefer several small modules over one big file) so rewriting a whole file stays cheap. Do NOT use apply_patch, diff, or patch syntax — that path is unavailable here and will fail."
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
            "local_web_search" => {
                "- `local_web_search`: Search the web via Brave; returns titles, URLs, and short descriptions. Args: `{\"query\": \"<search terms>\", \"count\": 10}` (count optional, 1-20). Pair this with `web_fetch` to read a specific result."
            }
            "web_fetch" => {
                "- `web_fetch`: Fetch a single http(s) URL and return the page body as text. Use this BEFORE writing code against an unfamiliar API or library — read the docs page rather than guessing the endpoint shape. Args: `{\"url\": \"https://...\"}`. Body is capped at 512KB; binary responses return a placeholder."
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
    /// Pending local-model "nudge" notices — each time a guard intervenes
    /// (repetition guard, rumination guard, quality gate, completion verifier)
    /// it queues a one-line message here. The TUI drains these via
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

/// Whether to force `tool_choice="required"` on a re-prompt retry.
///
/// The local-coder massaging escalation (completion verifier / rumination /
/// quality) re-prompts a model that gave us a no-tool-call response when we
/// needed an action. By the retry the prose nudge has already failed, so we
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
                    // Compaction summarizes history, so pinning fresh file
                    // content doesn't help the compactor and would inflate
                    // the input. Skip the file-state injection here.
                    current_files: None,
                    flavor: endpoint.flavor,
                    // Compaction summarizes history; leave its system prompt
                    // alone so the summary keeps full instruction fidelity.
                    system_budget_pct: 0,
                };
                let trimmed =
                    codex_routing::trim::trim_for_local(&trim_input, endpoint.trim_budget);
                info!(
                    trim_summary = %trimmed.summary.to_log_line(),
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

                let last_msg = extract_last_message(prompt);
                let current_request = last_msg
                    .replace("<<<LOCAL_COMPACT>>>", "")
                    .trim()
                    .to_string();

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

    // Re-read every file the active turn has edited so the trimmer can pin
    // fresh content into the prelude. Without this the model works from
    // its memory of the pre-patch state and writes patches with stale `-`
    // lines — the same failure mode that caused multi-turn patch loops in
    // early local-model sessions (see docs/spec/local-coder-massaging.md).
    let current_files = load_active_turn_files(&prompt.input);

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

    // System-prompt compression (Stage 2): if the base prompt exceeds its budget,
    // summarize it via the compaction track and CACHE by content hash, so a
    // harness sending the same prompt every turn pays the cost once. The
    // deterministic head/tail elision in trim stays the floor — we still pass
    // `system_budget_pct`, so trim hard-enforces the budget even if the summary
    // overshoots or the compactor is unreachable.
    let system_budget_tokens = if state.project_config.routing.system_budget_pct == 0 {
        0
    } else {
        endpoint
            .trim_budget
            .saturating_mul(state.project_config.routing.system_budget_pct as usize)
            / 100
    };
    let summarized_system =
        maybe_summarize_system(&prompt.base_instructions.text, system_budget_tokens, state).await;
    let system_prompt_ref: &str = summarized_system
        .as_deref()
        .unwrap_or(&prompt.base_instructions.text);

    let trim_input = codex_routing::trim::TrimInput {
        items: &prompt.input,
        system_prompt: system_prompt_ref,
        user_instructions: project_instructions.as_deref(),
        current_files: current_files.as_ref(),
        flavor: endpoint.flavor,
        system_budget_pct: state.project_config.routing.system_budget_pct,
    };
    // Budget against the model's LEARNED real÷estimate ratio, not the raw
    // trim_budget — the chars/4 estimate undercounts dense content ~1.8–2.8×, so
    // the configured budget overflowed the real window. Seeds at the default and
    // self-corrects from the server's reported prompt_tokens (see record below).
    let fit_budget = calibrated_trim_budget(&endpoint.model, endpoint.trim_budget);
    let trimmed = codex_routing::trim::trim_for_local(
        &trim_input,
        fit_budget.saturating_sub(tool_reserve_tokens),
    );
    info!(
        trim_summary = %trimmed.summary.to_log_line(),
        fit_budget,
        observed_token_ratio = observed_token_ratio(&endpoint.model),
        "Trimmed transcript for local model"
    );

    // If the trimmed transcript still exceeds the fit budget, summarize the bulk
    // (older prelude or the active turn's own middle) via the compaction pipeline
    // and replace it with a single summary message. Cached by hash so we don't
    // recompact identical history each turn.
    let trimmed = maybe_inline_compact(trimmed, fit_budget, endpoint, state).await;

    // Surface the loop guards so the user sees the coaching happen, in
    // escalation order:
    //  - context reset → loop excised from context + reframed (nudge+block ignored)
    //  - exact repeat → STOP directive
    //  - thrash (same goal, varying commands, still failing) → forced diagnosis
    if trimmed.system.contains("[HARNESS — STUCK; LOOP REMOVED") {
        state.push_nudge(
            "Context-reset guard fired — model ignored the nudges and the hard block; excised the loop from its context and reframed it to the unsolved step".to_string(),
        );
    } else if trimmed.system.contains("[STOP — REPETITION DETECTED]") {
        state.push_nudge(
            "Repetition guard fired — model was repeating an identical tool call; injected a stop directive".to_string(),
        );
    } else if trimmed.system.contains("[NO PROGRESS — DIAGNOSE") {
        state.push_nudge(
            "Forced-diagnosis guard fired — model was thrashing; requiring it to read the failure and state the root cause before acting".to_string(),
        );
    }
    // A failed apply_patch steers the model to a write_file rewrite (prelude
    // directive). When the target file is small enough to safely emit in full,
    // FORCE the write_file tool too, so the model can't keep re-patching — the
    // decisive fix for the failed-write → bad-edit cycle. Big files get the
    // directive only (a forced full rewrite of a huge file risks truncation).
    const MAX_FORCED_REWRITE_BYTES: usize = 24 * 1024; // ~6k tokens
    let force_write_file = trimmed.patch_rewrite_path.as_ref().is_some_and(|path| {
        current_files
            .as_ref()
            .and_then(|m| {
                m.get(path).or_else(|| {
                    let base = path.rsplit('/').next().unwrap_or(path);
                    m.iter()
                        .find(|(k, _)| k.rsplit('/').next().unwrap_or(k) == base)
                        .map(|(_, v)| v)
                })
            })
            .is_some_and(|content| content.len() <= MAX_FORCED_REWRITE_BYTES)
    });
    if let Some(path) = trimmed.patch_rewrite_path.as_ref() {
        state.push_nudge(format!(
            "Patch failed to apply — steering the model to rewrite {} with write_file (the code it was editing was never actually written){}",
            path.rsplit('/').next().unwrap_or(path),
            if force_write_file { "; forcing write_file" } else { "" },
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
        let mut system = Some(format!("{trimmed_system}\n\n{hint}"));

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
        // when the completion verifier flags a "bail" — see the body of the
        // loop for details. Set to 3 so a model that's making genuine
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
                    let _ = record_token_ratio(&endpoint.model, n_prompt_tokens, sent_estimate);
                    // The tokenized prompt overflowed the server's context window.
                    // The server hands us the real numbers, so re-trim to fit and
                    // retry the SAME coder — context maxing out must NOT crash the
                    // turn into a tools-less role. Token estimates undercount dense
                    // content (addresses, JSON, code), so scale the budget by how
                    // far we actually overshot rather than trusting the estimate.
                    const OVERFLOW_MARGIN: u64 = 2048;
                    const MIN_CODER_BUDGET: usize = 4096;
                    let reserve = endpoint.max_tokens.unwrap_or(0) as u64
                        + tool_reserve_tokens as u64
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
                    let re = codex_routing::trim::trim_for_local(
                        &trim_input,
                        coder_budget.saturating_sub(tool_reserve_tokens),
                    );
                    let re = maybe_inline_compact(re, coder_budget, endpoint, state).await;
                    system = Some(format!("{}\n\n{hint}", re.system));
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
                codex_routing::rumination_detector::RuminationDetector::from_endpoint_max_tokens(
                    endpoint.max_tokens,
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
                    } => {
                        input_tokens = it;
                        output_tokens = ot;
                        reasoning_tokens_seen = rt;
                        prompt_ms = pm;
                        gen_ms = gm;
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

            if !reasoning.is_empty() {
                tracing::debug!(
                    reasoning_len = reasoning.len(),
                    reasoning_tokens = reasoning_tokens_seen,
                    reasoning = %reasoning,
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
            if let Some(ratio) = record_token_ratio(&endpoint.model, input_tokens, sent_estimate) {
                state.push_nudge(format!(
                    "Calibrated context budget — this model packs ~{ratio:.1}× the tokens our estimate assumed ({input_tokens} real vs ~{sent_estimate} est)"
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

            // Deterministic quality gate: discard obviously-broken text-only
            // responses (empty, too short, prompt echo, model refusal, empty
            // code fence, degenerate repetition) and re-prompt. Cheap and runs
            // before the LLM-based completion verifier so we don't spend a
            // verifier call judging garbage. Only applies to text-only
            // responses — a tool call is the model making progress.
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
            // of acting". We combine what the model *did* (files actually
            // modified this task, witnessed in the transcript — not timestamps)
            // with what it *says* (the completion verifier):
            //
            //  - A coder that changed nothing is bailing/bluffing no matter how
            //    confident it sounds. Nudge it without spending a verifier call,
            //    and NEVER let it finish on an unbacked "done". This is the hard
            //    guard against "claimed done but wrote nothing".
            //  - Otherwise consult the verifier. A coder may only FINISH on an
            //    explicit `Complete` (and here its claim is ground-truth-backed);
            //    a reasoner (text-only by design) finishes unless it's an
            //    explicit `Bail`. Anything else re-prompts.
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

            if native_tool_calls.is_empty()
                && !content.trim().is_empty()
                && continuation_count < MAX_BAIL_RETRIES
            {
                let did_real_work =
                    !codex_routing::trim::files_modified_in_active_turn(&prompt.input).is_empty();

                let reprompt: Option<&str> = if !text_is_product && !did_real_work {
                    Some(
                        "Coder used no tools and changed nothing — nudging it to act via a tool call",
                    )
                } else if last_user_message.trim().is_empty() {
                    None
                } else {
                    let verifier_endpoint = if state.config.classifier.enabled {
                        &state.config.classifier
                    } else {
                        &state.config.light_coder
                    };
                    let verdict = codex_routing::completion_verifier::verify_completion(
                        &last_user_message,
                        &content,
                        verifier_endpoint,
                        &state.pool,
                    )
                    .await;
                    info!(
                        verdict = ?verdict,
                        did_real_work,
                        "Completion verifier judged a no-tool-call response"
                    );
                    use codex_routing::completion_verifier::CompletionVerdict;
                    let finish = if text_is_product {
                        // Reasoner: text is its product; finish unless it bailed.
                        !matches!(verdict, CompletionVerdict::Bail)
                    } else {
                        // Coder/actor: only a confident completion ends the turn,
                        // and its work is already ground-truth-backed here.
                        matches!(verdict, CompletionVerdict::Complete)
                    };
                    if finish {
                        None
                    } else {
                        Some(
                            "Completion check fired — task isn't shown as done; re-prompting to act",
                        )
                    }
                };

                if let Some(notice) = reprompt {
                    warn!("Re-prompting local model after a no-tool-call turn: {notice}");
                    state.push_nudge(notice.to_string());
                    let continuation = codex_routing::completion_verifier::continuation_prompt(
                        &content,
                        continuation_count,
                    );
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
            }

            return Ok(ollama_tool_response_to_stream(
                content,
                native_tool_calls,
                model_name.clone(),
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
    use std::hash::{Hash, Hasher};
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
                token_usage: Some(TokenUsage {
                    input_tokens: response.input_tokens as i64,
                    output_tokens: response.output_tokens as i64,
                    ..Default::default()
                }),
            }))
            .await;
    });

    ResponseStream { rx_event: rx }
}

/// Convert an Ollama response with native tool_calls to a ResponseStream.
/// Handles both native Ollama tool_calls and embedded JSON tool calls.
fn ollama_tool_response_to_stream(
    content: String,
    native_tool_calls: Vec<serde_json::Value>,
    model: String,
    input_tokens: u64,
    output_tokens: u64,
) -> ResponseStream {
    let (tx, rx) = mpsc::channel(16);

    tokio::spawn(async move {
        let _ = tx.send(Ok(ResponseEvent::Created)).await;

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
                token_usage: Some(TokenUsage {
                    input_tokens: input_tokens as i64,
                    output_tokens: output_tokens as i64,
                    ..Default::default()
                }),
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
                token_usage: Some(TokenUsage {
                    input_tokens: response.input_tokens as i64,
                    output_tokens: response.output_tokens as i64,
                    ..Default::default()
                }),
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

/// Read every file the active turn has edited (via `apply_patch` Add/Update)
/// and return a `path -> current_content` map. Missing files, unreadable
/// files, and non-UTF-8 files are silently skipped. Returns `None` when the
/// active turn hasn't modified any files, so the trimmer's file-state block
/// is omitted entirely in the common case.
///
/// Paths are resolved against the process `cwd` — matching how every other
/// local-coder tool handler in this crate resolves paths. The trimmer has
/// no IO of its own by design; this function is the only place the routing
/// layer reads from disk on behalf of the prelude builder.
fn load_active_turn_files(
    items: &[codex_protocol::models::ResponseItem],
) -> Option<std::collections::HashMap<String, String>> {
    let paths = codex_routing::trim::files_modified_in_active_turn(items);
    if paths.is_empty() {
        return None;
    }
    let cwd = std::env::current_dir().ok();
    let mut out = std::collections::HashMap::with_capacity(paths.len());
    for path in paths {
        let candidate = match &cwd {
            Some(base) => base.join(&path),
            None => std::path::PathBuf::from(&path),
        };
        if let Ok(content) = std::fs::read_to_string(&candidate) {
            out.insert(path, content);
        }
    }
    if out.is_empty() { None } else { Some(out) }
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
/// EWMA per model, clamped to `[DEFAULT, MAX]` so we never provision below the
/// safe default nor over-react to a single outlier. Returns `Some(new_ratio)`
/// only when the value shifted notably (so the caller can surface "learned X"
/// without spamming a nudge every turn once it has converged).
fn record_token_ratio(model: &str, real_tokens: u64, estimate: usize) -> Option<f64> {
    if real_tokens == 0 || estimate == 0 {
        return None;
    }
    let observed =
        (real_tokens as f64 / estimate as f64).clamp(DEFAULT_SAFETY_FACTOR, MAX_SAFETY_FACTOR);
    let mut m = TOKEN_RATIO.lock().ok()?;
    let cur = m.entry(model.to_string()).or_insert(DEFAULT_SAFETY_FACTOR);
    let before = *cur;
    *cur = *cur * 0.5 + observed * 0.5;
    ((*cur - before).abs() > 0.15).then_some(*cur)
}

/// The configured `trim_budget`, pre-scaled by the learned ratio so trim's
/// internal (estimate-space, ÷`DEFAULT_SAFETY_FACTOR`) math lands on a prompt that
/// actually fits the server. With a learned ratio of 2.84 and a 1.8 default, this
/// shrinks the budget to ~63%, so the real prompt fits on the first attempt
/// instead of overflowing and being re-trimmed. Identity until the model is
/// measured.
fn calibrated_trim_budget(model: &str, trim_budget: usize) -> usize {
    let observed = observed_token_ratio(model);
    ((trim_budget as f64) * DEFAULT_SAFETY_FACTOR / observed) as usize
}

/// Cache of compressed system prompts, keyed by a hash of (system, budget). The
/// base system prompt is stable per harness/session, so summarizing it once and
/// reusing it makes the (slow) compaction call a one-time cost — important once
/// this runs as a service, where the same prompt arrives on every request.
/// Global so it survives across sessions; cleared when it hits a small cap.
static SYSTEM_SUMMARY_CACHE: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, Option<String>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
const SYSTEM_SUMMARY_CACHE_CAP: usize = 64;

fn system_summary_key(system: &str, budget_tokens: usize) -> u64 {
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    system.hash(&mut h);
    budget_tokens.hash(&mut h);
    h.finish()
}

/// Compress an oversized system prompt via the compaction track, cached by
/// content hash. Returns `None` to mean "use the original" — when it already
/// fits, compression is disabled (`budget_tokens == 0`), or the compactor was
/// unreachable (the deterministic head/tail tier in `trim_for_local` then
/// enforces the budget). The summary is the higher-fidelity path; the
/// deterministic tier is the always-available floor.
async fn maybe_summarize_system(
    system: &str,
    budget_tokens: usize,
    state: &RoutingState,
) -> Option<String> {
    if budget_tokens == 0 || codex_routing::metrics::estimate_tokens(system) <= budget_tokens {
        return None;
    }
    let key = system_summary_key(system, budget_tokens);
    // Cache stores the OUTCOME, including a negative (`None`) result, so the
    // compactor runs at most once per unique (system, budget). The earlier bug:
    // only successes were cached, so a non-shrinking summary re-ran the full
    // (slow) compactor call EVERY turn and silently dominated the turn.
    if let Ok(cache) = SYSTEM_SUMMARY_CACHE.lock()
        && let Some(hit) = cache.get(&key)
    {
        return hit.clone();
    }
    let summary = summarize_system_via_compactor(system, budget_tokens, state).await;
    if let Ok(mut cache) = SYSTEM_SUMMARY_CACHE.lock() {
        if cache.len() >= SYSTEM_SUMMARY_CACHE_CAP {
            cache.clear();
        }
        cache.insert(key, summary.clone());
    }
    match &summary {
        Some(s) => info!(
            orig_tokens = codex_routing::metrics::estimate_tokens(system),
            summary_tokens = codex_routing::metrics::estimate_tokens(s),
            budget_tokens,
            "Compressed oversized system prompt via compaction track (cached once)"
        ),
        None => info!(
            budget_tokens,
            "Compactor did not shrink the system prompt; caching miss, using deterministic tier"
        ),
    }
    summary
}

/// One-shot system-prompt summarization through the `compactor` endpoint. Returns
/// `None` if the compactor is disabled/unreachable or didn't actually shrink it,
/// so the caller falls back to the deterministic tier.
async fn summarize_system_via_compactor(
    system: &str,
    budget_tokens: usize,
    state: &RoutingState,
) -> Option<String> {
    let compactor = &state.config.compactor;
    if !compactor.enabled {
        return None;
    }
    // Keep this single-goal and ACHIEVABLE. An earlier version asked the model
    // to "preserve VERBATIM every rule" AND "make it smaller" — a contradiction
    // (verbatim = don't change any words), especially on an already-tight prompt
    // with little prose to cut. The 9B burned 28k tokens reasoning about how to
    // satisfy both and never produced output. "Shorten the wording" (not the
    // content) is coherent and the model just does it.
    const SUMMARIZER_SYSTEM: &str = "You compress an AI coding agent's SYSTEM PROMPT to fit a small context window. \
Keep the rules and instructions; shorten the wording. Output ONLY the compressed system prompt — no commentary.";
    let user =
        format!("Reduce the following system prompt to about {budget_tokens} tokens:\n\n{system}");
    let mut ep = compactor.clone();
    ep.think = false; // summarization, not reasoning
    // Cap output length. Without this the compactor inherits "unbounded" and
    // generates until the context window is full (observed: 28,158 tokens =
    // n_ctx − prompt, `finish_reason: length`). A summary should never exceed the
    // budget anyway, so cap a little above it to leave room for the answer. Wall
    // time is governed by the compactor's configured `timeout_seconds`.
    ep.max_tokens = Some(budget_tokens.saturating_add(512));
    let body = state
        .pool
        .chat(
            &ep,
            vec![serde_json::json!({"role": "user", "content": user})],
            Some(SUMMARIZER_SYSTEM),
            None,
        )
        .await?;
    let content = body
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or_default();
    let content = codex_routing::classifier::strip_think_tags(content);
    let content = content.trim();
    // Only accept it if it genuinely shrank — otherwise fall back so we don't
    // swap the real prompt for a same-size (or larger) paraphrase.
    if content.is_empty()
        || codex_routing::metrics::estimate_tokens(content)
            >= codex_routing::metrics::estimate_tokens(system)
    {
        return None;
    }
    Some(content.to_string())
}

/// If the trimmed transcript still exceeds the local model's context budget,
/// run the compaction pipeline on the older-turn portion and replace it with
/// a single summary message. The active turn is left untouched.
///
/// Cached by hash of the older-turn message contents so repeated requests
/// within a session reuse the same summary instead of recompacting.
async fn maybe_inline_compact(
    mut trimmed: codex_routing::trim::TrimResult,
    fit_budget: usize,
    endpoint: &OllamaEndpoint,
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
                compact_active_turn(trimmed, older_count, endpoint, state).await
            } else {
                compact_older_turns(trimmed, older_count, endpoint, state).await
            };
        } else {
            warn!(
                estimated_tokens = trimmed.summary.estimated_input_tokens,
                target_ctx = endpoint.trim_budget,
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
    endpoint: &OllamaEndpoint,
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
        target_ctx = endpoint.trim_budget,
        older_count,
        "Trimmed transcript over budget — running inline compaction"
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
    endpoint: &OllamaEndpoint,
    state: &RoutingState,
) -> codex_routing::trim::TrimResult {
    let Some((request_idx, start, end)) =
        plan_active_turn_split(&trimmed.messages, active_start, KEEP_RECENT_ACTIVE_MESSAGES)
    else {
        warn!(
            estimated_tokens = trimmed.summary.estimated_input_tokens,
            target_ctx = endpoint.trim_budget,
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

    info!(
        estimated_tokens = trimmed.summary.estimated_input_tokens,
        target_ctx = endpoint.trim_budget,
        middle = middle.len(),
        "Active turn over budget — compacting its middle"
    );

    let summary_text = match codex_routing::compaction::compact_transcript(
        &middle,
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
    fn system_summary_key_is_stable_and_content_sensitive() {
        // Same inputs → same key (cache hits); different content or budget → miss.
        assert_eq!(
            system_summary_key("You are Codex.", 1000),
            system_summary_key("You are Codex.", 1000)
        );
        assert_ne!(
            system_summary_key("You are Codex.", 1000),
            system_summary_key("You are a different agent.", 1000)
        );
        assert_ne!(
            system_summary_key("You are Codex.", 1000),
            system_summary_key("You are Codex.", 2000)
        );
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
