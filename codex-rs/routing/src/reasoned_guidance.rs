//! Reasoned Guidance — assist #1: PLAN-FIRST.
//!
//! On a new user task, the light **reasoner** is engaged first to draft a short
//! plan of SMALL steps, tailored to the fact that the executor is a small,
//! low-context model that works best one small goal at a time. That plan is
//! pinned at the top of the coder's prompt for the whole turn, so the small model
//! never has to hold the task's whole shape in its head — it just follows the next
//! small step and verifies before moving on.
//!
//! The planner GATHERS before it plans: it is given a **read-only** tool subset
//! (inspect the working dir, read files, fetch docs, search the web) and runs a
//! bounded loop — inspect → see results → inspect more — until it stops calling
//! tools and emits the plan, now GROUNDED in what it actually found rather than
//! assumptions. Read-only by construction (a write/mutate command is refused;
//! building is the coder's job), and the leaked-tool-call parser means quirky
//! dialects (Gemma's `<|tool_call>…`) are recovered like any other.
//!
//! This is the first member of the Reasoned Guidance family (see
//! docs/spec/heuristic-assists.md): the deterministic guards say *when* to
//! intervene; this spends a read-only gather-and-plan pass up front to shape the
//! *whole* turn.
//!
//! Cost is bounded: the plan is content-addressed by the task text, so the whole
//! gather-and-plan pass runs ONCE per user task and every coder step in that turn
//! reuses the cached plan (never regenerated per step). The reasoner shares the
//! local GPU with the coder, so "once per turn" is the deliberate ceiling.

use crate::config::ClientFlavor;
use crate::config::OllamaEndpoint;
use crate::ollama::OllamaClientPool;
use serde_json::Value as JsonValue;
use serde_json::json;
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::LazyLock;
use std::sync::Mutex;
use tracing::info;
use tracing::warn;

/// What the reasoner is told when drafting the plan. It plans FOR a small,
/// low-context executor — small steps, small goals, one thing at a time.
const PLANNER_SYSTEM_PROMPT: &str = "\
You are the PLANNER for a SMALL local coding model (~9B parameters, a limited context window, and a \
short memory — it forgets earlier steps easily). The user has given a coding task. Your job has TWO \
phases: first INVESTIGATE using your read-only tools, then output the plan.\n\
PHASE 1 — INVESTIGATE (use your tools, as many calls as you need): You have READ-ONLY tools — \
`exec_command` (read-only shell: ls, cat, grep, find, head, git status/log/diff, curl a doc URL), \
`read_file`, `web_fetch`, and `web_search`. Before planning, actually LOOK: inspect the working \
directory to see what already exists (files, code, config, conventions to match); read the relevant \
existing files; and fetch/search any external sources the task implies (documentation, schemas, \
API endpoints, reference material). You CANNOT write or modify anything — that is the coder's job, \
not yours. Keep investigating until you genuinely understand the task's context.\n\
PHASE 2 — PLAN: When you have gathered enough, STOP calling tools and output ONLY the plan. Break \
the task into a SHORT, ordered list of SMALL steps the coder will execute ONE AT A TIME, GROUNDED in \
what you actually found (name the real files, endpoints, and conventions you saw — not guesses).\n\
Rules for the plan:\n\
- Each step is ONE concrete, self-contained action with a SMALL goal the model can finish and verify \
before moving on (e.g. \"Create lambda_handler.py with a resolve() that calls GET /handles/{name}\", \
\"Run pytest test_handler.py\").\n\
- Keep steps and files small: prefer many tiny steps over a few big ones. Never a step that writes a \
huge file or does several unrelated things at once.\n\
- Use 4 to 9 steps, ending with a run/build/test step wherever progress should be checked. Order them \
so each builds on the last.\n\
- Do NOT write code in the plan. Do NOT explain your reasoning. Once investigating is done, output \
ONLY a numbered list — one short imperative sentence per step, naming the specific file, command, \
source, or change.";

/// Header on the injected plan block — tells the small model how to use it.
const PLAN_HEADER: &str = "[PLAN — a reasoner broke your task into small steps because you are a \
small model with limited context. Do ONE step at a time - in order - and verify before the next. \
This plan was created by another model - not the user - and may contain mistakes]";

/// Content-addressed cache: `hash(task) -> plan block` (or `None` when planning
/// was skipped/failed for that task). Guarantees a single reasoner call per user
/// task; a new task evicts the whole map when it is full (simple, bounded).
static PLAN_CACHE: LazyLock<Mutex<HashMap<u64, Option<String>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
const PLAN_CACHE_CAP: usize = 32;

/// Below this the task is too trivial to be worth a planning round-trip.
const MIN_TASK_CHARS: usize = 24;

fn task_key(task: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    task.trim().hash(&mut h);
    h.finish()
}

/// Draft (or return the cached) small-step plan block for `task`, ready to pin at
/// the top of the coder's system prompt. Returns `None` when the reasoner is
/// disabled/unreachable, the task is trivial, or the plan came back empty.
///
/// The returned bool is `true` only when the plan was **freshly drafted** this
/// call (a cache MISS) — the caller uses it to surface the plan in the TUI once
/// per task instead of on every reused step. The result (including a `None`) is
/// cached per task, so this is at most one reasoner call per user turn.
pub async fn plan_for_task(
    pool: &OllamaClientPool,
    reasoner: &OllamaEndpoint,
    task: &str,
    search_api_key: &str,
) -> Option<(String, bool)> {
    let task = task.trim();
    if !reasoner.enabled || task.chars().count() < MIN_TASK_CHARS {
        return None;
    }
    let key = task_key(task);
    if let Ok(cache) = PLAN_CACHE.lock() {
        if let Some(cached) = cache.get(&key) {
            // Already planned this task (Some block, or cached-negative None);
            // `false` = reused, so the caller won't re-announce it.
            return cached.clone().map(|b| (b, false));
        }
    }

    // A plain agentic tool-use loop — the same shape as any turn. The planner is
    // given READ-ONLY tools (inspect the dir, read files, fetch docs, search the
    // web); each round it either calls tools (we run them and append the results
    // as `tool` messages, so the loop continues) or answers with no tool call —
    // and THAT answer is the plan, grounded in what it actually found. Read-only
    // BY CONSTRUCTION: a write/mutate command is refused (building is the coder's
    // job). No fixed round cap (by design) — the model self-terminates by
    // answering; a REPEATED call signature is the only forced stop, meaning
    // "gathered enough / stuck" rather than more signal.
    //
    // The tool call and its result are fed back as PROTOCOL — a structured
    // `assistant.tool_calls` turn plus a `role:"tool"` result — exactly like a
    // normal turn, NOT flattened into assistant prose. That distinction matters:
    // prose in the transcript gets parroted back by a small model (an earlier
    // version injected `(called web_fetch …)` as content and the model echoed it
    // as its "plan"); a structured tool_calls field never leaks into content.
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let mut ep = reasoner.clone();
    ep.timeout_seconds = ep.timeout_seconds.max(60); // per gather call; the loop is unbounded
    let tools = planner_tools();

    let mut messages = vec![json!({ "role": "user", "content": task })];
    let mut seen_sigs: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut forced = false; // set when the stuck guard disables tools for a final plan

    let plan: Option<String> = loop {
        let tools_arg = if forced { None } else { Some(tools.clone()) };
        let Some(body) = pool
            .chat_with_tools(&ep, messages.clone(), Some(PLANNER_SYSTEM_PROMPT), None, tools_arg)
            .await
        else {
            break None;
        };
        let content = body
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();
        let calls = extract_planner_tool_calls(&body);

        // No tool call (or tools were disabled after a stuck loop) → the content IS
        // the plan. Strip `<think>` and any leaked reasoner-format tokens.
        if forced || calls.is_empty() {
            let cleaned = clean_plan(&content);
            break (!cleaned.is_empty()).then_some(cleaned);
        }

        // Repeated a round we already ran (full seen-set, so A→B→A loops count too)
        // → stop gathering and ask once more, tools off, for just the plan.
        let sig = calls_signature(&calls);
        if !seen_sigs.insert(sig) {
            messages.push(json!({
                "role": "user",
                "content": "You've gathered enough. Stop investigating and output ONLY the numbered plan now."
            }));
            forced = true;
            continue;
        }

        // Feed the round back as protocol: a structured assistant tool-call turn,
        // then one `tool` result per call. execute_planner_tool enforces read-only.
        let (assistant_turn, ids) = build_gather_turn(&body, &calls, ep.flavor);
        messages.push(assistant_turn);
        for ((name, args), id) in calls.iter().zip(&ids) {
            let result = execute_planner_tool(name, args, &cwd, search_api_key).await;
            info!(tool = %name, "Reasoned guidance: planner gather call (read-only)");
            messages.push(json!({ "role": "tool", "tool_call_id": id, "content": result }));
        }
    };

    let block = match plan {
        Some(p) if !p.is_empty() => {
            let steps = p.lines().filter(|l| !l.trim().is_empty()).count();
            info!(steps, plan = %p, "Reasoned guidance: reasoner drafted a plan for the turn");
            Some(format!("{PLAN_HEADER}\n{p}"))
        }
        _ => {
            warn!("Reasoned guidance: reasoner returned no usable plan — proceeding without one");
            None
        }
    };

    if let Ok(mut cache) = PLAN_CACHE.lock() {
        if cache.len() >= PLAN_CACHE_CAP {
            cache.clear();
        }
        cache.insert(key, block.clone());
    }
    block.map(|b| (b, true)) // freshly drafted this call
}

/// The READ-ONLY tool subset offered to the planner for its gathering phase.
/// Deliberately minimal, inline schemas (local models are lenient): inspect the
/// project (`exec_command`, read-only-enforced), read a file, fetch a URL, search
/// the web. No write/patch/exec-mutate tools — planning is not building.
fn planner_tools() -> Vec<JsonValue> {
    vec![
        json!({"type":"function","function":{"name":"exec_command","description":"Run a READ-ONLY shell command to inspect the project (ls, cat, head, tail, grep, find, wc, git status/log/diff, curl a doc URL, …). Writes/mutations are refused — you are planning, not building.","parameters":{"type":"object","properties":{"cmd":{"type":"string","description":"the command line"}},"required":["cmd"]}}}),
        json!({"type":"function","function":{"name":"read_file","description":"Read a file's full contents to understand existing code/config/conventions.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}),
        json!({"type":"function","function":{"name":"web_fetch","description":"Fetch a URL (docs, an OpenAPI/JSON schema, a reference page) and return its text.","parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}}),
        json!({"type":"function","function":{"name":"web_search","description":"Search the web for documentation, APIs, or references the task implies.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}),
    ]
}

/// Pull tool calls out of a planner response — both the server-parsed structured
/// `message.tool_calls` and any LEAKED calls in the content (Hermes/XML/Gemma via
/// the shared `tool_aliases` parser). Returns `(name, args-object)` pairs.
fn extract_planner_tool_calls(body: &JsonValue) -> Vec<(String, JsonValue)> {
    let mut out = Vec::new();
    if let Some(arr) = body
        .get("message")
        .and_then(|m| m.get("tool_calls"))
        .and_then(|t| t.as_array())
    {
        for tc in arr {
            let f = tc.get("function").unwrap_or(tc);
            if let Some(name) = f.get("name").and_then(|n| n.as_str()) {
                let args = match f.get("arguments") {
                    Some(JsonValue::String(s)) => serde_json::from_str(s).unwrap_or(json!({})),
                    Some(other) => other.clone(),
                    None => json!({}),
                };
                out.push((name.to_string(), args));
            }
        }
    }
    let content = body
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    for wire in crate::tool_aliases::parse_leaked_tool_calls(content) {
        if let Some(name) = wire
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
        {
            let args = wire
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .and_then(|s| serde_json::from_str::<JsonValue>(s).ok())
                .unwrap_or(json!({}));
            out.push((name.to_string(), args));
        }
    }
    out
}

/// A stable signature of a round's calls, for the repeated-call stuck guard.
fn calls_signature(calls: &[(String, JsonValue)]) -> String {
    calls
        .iter()
        .map(|(n, a)| format!("{n}:{a}"))
        .collect::<Vec<_>>()
        .join("|")
}

/// Strip `<think>` blocks and leaked tool-call tokens, normalize exotic Unicode
/// whitespace to plain ASCII (see `normalize_whitespace`), then trim.
fn clean_plan(content: &str) -> String {
    let stripped =
        crate::tool_aliases::strip_leaked_tool_calls(&crate::classifier::strip_think_tags(content));
    normalize_whitespace(&stripped).trim().to_string()
}

/// Replace exotic Unicode whitespace with a plain ASCII space and drop zero-width
/// characters. Some local reasoners (Fabliq) sprinkle NARROW NO-BREAK SPACE (U+202F)
/// and word-joiners around code identifiers, which makes a pinned plan/redirect read
/// as mangled and slips invisible bytes into the coder's prompt. Real `\n`/`\t` are
/// preserved so the plan's numbered-list structure survives.
fn normalize_whitespace(s: &str) -> String {
    s.chars()
        .filter_map(|c| match c {
            '\n' | '\t' => Some(c),
            // zero-width space / joiners / BOM → drop entirely
            '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{2060}' | '\u{FEFF}' => None,
            // any other Unicode whitespace (U+00A0, U+202F, U+2009, U+2007, …) → ASCII space
            c if c.is_whitespace() => Some(' '),
            c => Some(c),
        })
        .collect()
}

/// Build the assistant turn that records a gather round, plus the ids used to pair
/// each `tool` result to its call — the PROTOCOL representation of a tool call,
/// not prose. Two cases:
///   - the server returned structured `tool_calls`: re-feed its message verbatim
///     (already in the server's native wire shape, so it round-trips), reusing the
///     server's ids (synthesizing one only where absent, e.g. Ollama).
///   - the calls were LEAKED as content text (Gemma `<|tool_call>…`, Hermes, XML):
///     reconstruct a clean structured `tool_calls` turn from the parsed calls, with
///     flavor-correct argument encoding (OpenAI wants a JSON string, Ollama an
///     object) so the raw dialect tokens never re-enter the transcript.
fn build_gather_turn(
    body: &JsonValue,
    calls: &[(String, JsonValue)],
    flavor: ClientFlavor,
) -> (JsonValue, Vec<String>) {
    let structured = body
        .get("message")
        .and_then(|m| m.get("tool_calls"))
        .and_then(|t| t.as_array())
        .filter(|a| !a.is_empty());
    if let Some(arr) = structured {
        let ids = arr
            .iter()
            .enumerate()
            .map(|(i, tc)| {
                tc.get("id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("call_{i}"))
            })
            .collect();
        let mut msg = body
            .get("message")
            .cloned()
            .unwrap_or_else(|| json!({ "role": "assistant", "content": "" }));
        // Guarantee a role even if the server omitted it (some servers reject a
        // re-fed assistant message with no role).
        if let Some(obj) = msg.as_object_mut() {
            obj.entry("role").or_insert_with(|| json!("assistant"));
        }
        return (msg, ids);
    }
    // Leaked calls → reconstruct a clean structured turn.
    let ids: Vec<String> = (0..calls.len()).map(|i| format!("call_{i}")).collect();
    let tool_calls: Vec<JsonValue> = calls
        .iter()
        .zip(&ids)
        .map(|((name, args), id)| {
            let arguments = match flavor {
                ClientFlavor::OpenAICompat => json!(args.to_string()),
                ClientFlavor::Ollama => args.clone(),
            };
            json!({ "id": id, "type": "function", "function": { "name": name, "arguments": arguments } })
        })
        .collect();
    (
        json!({ "role": "assistant", "content": "", "tool_calls": tool_calls }),
        ids,
    )
}

/// Execute ONE planner tool call, READ-ONLY. Mutating shell commands are refused;
/// file reads / fetches / searches run for real. Returns human-readable output for
/// the loop to feed back to the planner.
async fn execute_planner_tool(
    name: &str,
    args: &JsonValue,
    cwd: &std::path::Path,
    search_api_key: &str,
) -> String {
    match name {
        "exec_command" | "shell" | "bash" | "local_shell" => {
            let cmd = args
                .get("cmd")
                .or_else(|| args.get("command"))
                .map(|c| match c {
                    JsonValue::String(s) => s.clone(),
                    JsonValue::Array(a) => a
                        .iter()
                        .filter_map(|x| x.as_str())
                        .collect::<Vec<_>>()
                        .join(" "),
                    other => other.to_string(),
                })
                .unwrap_or_default();
            if cmd.trim().is_empty() {
                return "[no command given]".to_string();
            }
            if !is_read_only_command(&cmd) {
                return format!(
                    "[refused: planning is READ-ONLY — `{}` would write or mutate. Don't run it; \
                     just plan for the coder to do it.]",
                    cmd.chars().take(100).collect::<String>()
                );
            }
            let (c, d) = (cmd.clone(), cwd.to_path_buf());
            tokio::task::spawn_blocking(move || {
                run_readonly_command(&c, &d, std::time::Duration::from_secs(20))
            })
            .await
            .unwrap_or_else(|_| "[exec task panicked]".to_string())
        }
        "read_file" | "cat_file" => {
            let path = args
                .get("path")
                .or_else(|| args.get("file_path"))
                .and_then(|p| p.as_str())
                .unwrap_or("");
            let full = if std::path::Path::new(path).is_absolute() {
                std::path::PathBuf::from(path)
            } else {
                cwd.join(path)
            };
            match std::fs::read_to_string(&full) {
                Ok(s) => truncate_output(&s, 8000),
                Err(e) => format!("[read_file error: {e}]"),
            }
        }
        "web_fetch" => {
            let url = args.get("url").and_then(|u| u.as_str()).unwrap_or("");
            match crate::web_fetch::fetch(url, None).await {
                Ok(r) => format!("HTTP {} · {}\n{}", r.status, r.final_url, truncate_output(&r.body, 6000)),
                Err(e) => format!("[web_fetch error: {e}]"),
            }
        }
        "web_search" | "local_web_search" => {
            let q = args
                .get("query")
                .or_else(|| args.get("q"))
                .and_then(|x| x.as_str())
                .unwrap_or("");
            match crate::local_web_search::search(search_api_key, q, 5, None).await {
                Ok(results) => crate::local_web_search::format_results(q, &results),
                Err(e) => format!("[web_search error: {e}]"),
            }
        }
        other => format!("[planner has no `{other}` tool — you are read-only: exec_command (read), read_file, web_fetch, web_search]"),
    }
}

/// Conservative read-only gate for the planner's `exec_command`. An ALLOW-LIST of
/// known read commands (reject anything else), no output redirects, no in-place
/// edits, and `git`/`curl`/`wget` restricted to their read-only uses. Erring toward
/// refusal is correct — a refused inspect just costs the planner a retry, a slipped
/// mutation corrupts the workspace.
fn is_read_only_command(cmd: &str) -> bool {
    if cmd.contains('>') {
        return false; // any redirect writes a file
    }
    const READ_ONLY: &[&str] = &[
        "ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "find", "fd", "tree",
        "file", "stat", "du", "pwd", "echo", "printf", "which", "type", "env", "date", "whoami",
        "uname", "basename", "dirname", "realpath", "readlink", "cut", "sort", "uniq", "tr", "nl",
        "tac", "rev", "column", "diff", "cmp", "comm", "od", "xxd", "strings", "jq", "yq", "cd",
        "true", "test", "[",
    ];
    for seg in cmd.split(['|', ';', '&', '\n']) {
        let seg = seg.trim();
        if seg.is_empty() {
            continue;
        }
        let head = seg.split_whitespace().next().unwrap_or("");
        let base = head.rsplit('/').next().unwrap_or(head);
        let sub = seg.split_whitespace().nth(1).unwrap_or("");
        match base {
            "git" => {
                if !matches!(
                    sub,
                    "status" | "log" | "diff" | "show" | "ls-files" | "branch" | "rev-parse"
                        | "cat-file" | "blame" | "describe" | "remote" | "config" | "grep"
                ) {
                    return false;
                }
            }
            "sed" => {
                if seg.contains("-i") {
                    return false; // in-place edit
                }
            }
            "curl" | "wget" => {
                if seg.contains("-o") || seg.contains("-O") {
                    return false; // saves to a file
                }
            }
            _ if READ_ONLY.contains(&base) => {}
            _ => return false,
        }
    }
    true
}

/// Run a (pre-vetted read-only) shell command with a hard timeout, returning its
/// combined output. Mirrors the probe runner's subprocess pattern.
fn run_readonly_command(cmd: &str, cwd: &std::path::Path, timeout: std::time::Duration) -> String {
    use std::io::Read;
    use std::process::{Command, Stdio};
    let mut child = match Command::new("bash")
        .arg("-lc")
        .arg(cmd)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return format!("[exec failed to launch: {e}]"),
    };
    let mut so = child.stdout.take();
    let mut se = child.stderr.take();
    let ho = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(p) = so.as_mut() {
            let _ = p.read_to_string(&mut s);
        }
        s
    });
    let he = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(p) = se.as_mut() {
            let _ = p.read_to_string(&mut s);
        }
        s
    });
    let start = std::time::Instant::now();
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(40));
            }
            Err(_) => break,
        }
    }
    let out = ho.join().unwrap_or_default();
    let err = he.join().unwrap_or_default();
    let mut combined = out;
    if !err.trim().is_empty() {
        combined.push_str("\n[stderr] ");
        combined.push_str(&err);
    }
    if timed_out {
        combined.push_str("\n[timed out]");
    }
    let combined = combined.trim();
    if combined.is_empty() {
        "(no output)".to_string()
    } else {
        truncate_output(combined, 6000)
    }
}

/// Byte-bounded, char-boundary-safe truncation for gathered tool output.
fn truncate_output(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut end = n;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[… {} more chars truncated …]", &s[..end], s.len() - end)
}

/// What the reasoner is told when critiquing a "done" claim (assist #2). It hunts
/// for exactly what the deterministic gates cannot see — semantic shortcuts.
const COMPLETION_CRITIC_PROMPT: &str = "\
You are the COMPLETION CRITIC. A small local coding model just declared a task DONE. You are given the \
original TASK, the EVIDENCE of what the model did (recent tool calls, their outputs, and its final \
claim), and the FRESH PROBE RESULTS of lint/tests the harness JUST RAN on the workspace. Decide \
whether the task is genuinely and fully complete.\n\
Treat the PROBE RESULTS as ground truth, and read them precisely — passing is NOT the only \
alternative to failing:\n\
- Tests that PASSED (ran, exit 0) support completion.\n\
- Tests that FAILED (ran, non-zero, real failures) mean NOT done — fix them.\n\
- Tests that DID NOT RUN — an import error, nothing collected, a timeout, or a missing runner — do \
NOT count as passing. If the task asked for working tests, code whose tests cannot even execute is \
NOT complete; the model must MAKE them run (install the missing dependency, fix the import, provide \
the runner). Do NOT wave a non-running suite away as \"probably environmental\" — you cannot confirm \
that, so treat it as the model's problem to fix.\n\
Then catch the semantic shortcuts the probes can't:\n\
- INVALID ASSUMPTIONS: accepting an error/empty/404 as success; a test that passes only because the \
real path was never exercised; an endpoint/URL never verified against reality.\n\
- SHORTCUTS: stubbed or mocked-away real behavior, hardcoded values, TODOs, or a \"tests pass\" that \
does not actually prove the requirement.\n\
- MISSING REQUIREMENTS: did it do EVERY part the task asked for?\n\
- UNVERIFIED CLAIMS: did it confirm against reality (a real request/output), or just assert it is done?\n\
Output JSON ONLY: {\"complete\": true|false, \"issues\": [\"...\", ...]}. When complete is false, each \
issue must be concrete and actionable — name the exact requirement, wrong assumption, or probe result \
and what to check or fix. When genuinely done, output {\"complete\": true, \"issues\": []}. Do not \
invent problems; if TASK + EVIDENCE + PROBES support completion, say so.";

#[derive(serde::Deserialize)]
struct CritiqueResponse {
    #[serde(default)]
    complete: bool,
    #[serde(default)]
    issues: Vec<CritiqueIssue>,
}

/// One critic issue — tolerant of BOTH the requested `["string", …]` shape and the
/// object shape a small reasoner often emits instead (`[{"description":…,"root_cause":…,
/// "severity":…}]`). Demanding `Vec<String>` discarded a CORRECT critique that had
/// diagnosed a real ImportError (session 019f38a5), letting a false completion through —
/// so we extract a human string from whichever shape arrives.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum CritiqueIssue {
    Text(String),
    Structured {
        #[serde(default)]
        description: String,
        #[serde(default)]
        issue: String,
        #[serde(default)]
        message: String,
        #[serde(default)]
        problem: String,
        #[serde(default)]
        detail: String,
        #[serde(default)]
        root_cause: String,
    },
    /// Anything else (a number, a nested array) → ignored rather than failing the whole
    /// parse, so one odd element can't discard the entire critique.
    Other(serde_json::Value),
}

impl CritiqueIssue {
    fn text(&self) -> String {
        match self {
            CritiqueIssue::Text(s) => s.trim().to_string(),
            CritiqueIssue::Structured {
                description,
                issue,
                message,
                problem,
                detail,
                root_cause,
            } => {
                let main = [description, issue, message, problem, detail]
                    .into_iter()
                    .map(|s| s.trim())
                    .find(|s| !s.is_empty())
                    .unwrap_or("");
                let cause = root_cause.trim();
                if !main.is_empty() && !cause.is_empty() && main != cause {
                    format!("{main} (root cause: {cause})")
                } else if !main.is_empty() {
                    main.to_string()
                } else {
                    cause.to_string()
                }
            }
            CritiqueIssue::Other(_) => String::new(),
        }
    }
}

/// Reasoned Guidance assist #2 — COMPLETION CRITIC. On a "done" claim that already
/// passed the deterministic gates (repo diagnostics), spend ONE reasoner call to
/// catch what they can't: shortcuts, invalid assumptions (an error accepted as
/// success), unmet requirements — and, crucially, tests that DID NOT RUN. The
/// deterministic gate is silent unless a probe emits `file:line` findings, so a
/// suite that fails to import or collects nothing looks identical to a green one.
/// This critic is handed the FRESH probe results (`probe_results` — from
/// [`crate::probe_run::completion_probe_digest`]) so it can weigh that difference as
/// ground truth rather than judge blind. Returns `Some(re-prompt)` with the concrete
/// issues when the reasoner judges it NOT complete; `None` when it's satisfied — or
/// when the reasoner is unavailable/unparseable, since we must never block completion
/// on a critic we couldn't actually run.
pub async fn critique_completion(
    pool: &OllamaClientPool,
    reasoner: &OllamaEndpoint,
    task: &str,
    evidence: &str,
    probe_results: &str,
) -> Option<String> {
    if !reasoner.enabled || task.trim().is_empty() {
        return None;
    }
    let probe_block = if probe_results.trim().is_empty() {
        "(no lint/test probe results were available this turn)".to_string()
    } else {
        probe_results.trim().to_string()
    };
    let payload = format!(
        "TASK:\n{task}\n\n\
         FRESH PROBE RESULTS (lint/tests the harness just ran on the workspace — ground truth):\n{probe_block}\n\n\
         EVIDENCE (recent actions, tool outputs, and the model's final claim):\n{evidence}"
    );
    let mut ep = reasoner.clone();
    ep.timeout_seconds = ep.timeout_seconds.clamp(20, 90);
    let body = pool
        .chat(
            &ep,
            vec![json!({ "role": "user", "content": payload })],
            Some(COMPLETION_CRITIC_PROMPT),
            Some("json"),
        )
        .await;
    // TRUTH CAPTURE. The critic's verdict is otherwise recorded NOWHERE, so a silent
    // failure — the reasoner call timed out, or it produced JSON we couldn't parse —
    // is indistinguishable from "approved completion": both just let the "done" pass.
    // (This is exactly how a false completion sailed through: no critic log at all.)
    // Log the raw output + parse outcome, and warn loudly on each non-blocking exit.
    let Some(body) = body else {
        warn!("Completion critic: reasoner call returned NOTHING (timeout/error) — cannot run the critic, NOT blocking");
        return None;
    };
    let content = body
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let stripped = crate::classifier::strip_think_tags(&content);
    let json = crate::classifier::extract_json_object(stripped.trim());
    let parsed = serde_json::from_str::<CritiqueResponse>(json);
    info!(
        raw_output = %content,
        parsed_ok = parsed.is_ok(),
        "Completion critic: reasoner I/O (truth capture)"
    );
    let Ok(parsed) = parsed else {
        warn!("Completion critic: reasoner output did NOT parse as JSON — cannot run the critic, NOT blocking");
        return None;
    };
    if parsed.complete || parsed.issues.is_empty() {
        info!("Completion critic: reasoner confirmed the task is complete");
        return None;
    }
    let issues = parsed
        .issues
        .iter()
        .map(CritiqueIssue::text)
        .filter(|s| !s.is_empty())
        .take(6)
        .map(|s| format!("- {s}"))
        .collect::<Vec<_>>()
        .join("\n");
    if issues.is_empty() {
        return None;
    }
    warn!(
        issues = parsed.issues.len(),
        "Completion critic: reasoner found the task INCOMPLETE — blocking completion"
    );
    Some(format!(
        "[COMPLETION REVIEW — not done yet] A reviewer checked your work against the task and found \
         problems. Fix these, then finish:\n{issues}"
    ))
}

/// What the reasoner is told when REBUILDING a flailing coder's context (the excise).
const REBUILD_CONTEXT_PROMPT: &str = "\
A small local coding model is STUCK IN A LOOP and has ignored every warning, so its context has been \
cleared of the looping turns. You are its senior guide. You are given the TASK and fresh GROUND TRUTH: \
the exact action it keeps repeating with the result it keeps getting, the workspace lint, and the \
ACTUAL current files on disk. Write a SHORT, clean working context to replace the mess — at most ~150 \
words, in these four labeled parts and nothing else:\n\
TASK: one line.\n\
STATE: what actually exists on disk now (from the files given), 1-2 lines.\n\
WHY STUCK: the concrete reason the repeated action fails — quote the real error — 1 line.\n\
NEXT: the ONE specific next action, DIFFERENT from the loop (a concrete tool call or edit). If the \
repeated action is nonsensical (e.g. `cat` on a directory, or re-running a search that already \
returned the same result), say plainly what to do instead.\n\
Be terse. Do NOT restate the transcript, do NOT ramble, do NOT invent facts not in the ground truth.";

/// Reasoned Guidance — CONTEXT REBUILD ON FLAIL. When the loop guard escalates to
/// context surgery (the excise), the model has ignored every softer nudge. Instead of
/// excising the loop and pasting a canned reframe that points back at now-STALE
/// context, hand the reasoner the FRESH ground truth ([`crate::ground_truth::GroundTruth`]
/// — the repeated failing action + the actual current files + any dirty lint) and have
/// it author a SMALL clean working context to replace the flailing history. Returns
/// `None` when the reasoner is unavailable / produced nothing / there is no signal — the
/// caller then keeps the canned excise (never nothing).
pub async fn rebuild_context_from_loop(
    pool: &OllamaClientPool,
    reasoner: &OllamaEndpoint,
    task: &str,
    ground: &crate::ground_truth::GroundTruth,
) -> Option<String> {
    if !reasoner.enabled || task.trim().is_empty() || !ground.has_signal() {
        return None;
    }
    let payload = format!(
        "TASK:\n{task}\n\n\
         GROUND TRUTH (fresh — the loop's repeated failure, the workspace lint, and the files as they \
         are on disk NOW):\n{}",
        ground.render()
    );
    let mut ep = reasoner.clone();
    ep.timeout_seconds = ep.timeout_seconds.clamp(20, 90);
    let body = pool
        .chat(
            &ep,
            vec![json!({ "role": "user", "content": payload })],
            Some(REBUILD_CONTEXT_PROMPT),
            None,
        )
        .await;
    let Some(body) = body else {
        warn!("Context rebuild: reasoner returned NOTHING (timeout/error) — keeping the canned excise");
        return None;
    };
    let content = body
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    // Reuse the plan cleaner: strips <think>, leaked tool-call sentinels, and exotic
    // whitespace so the rebuilt context can't smuggle garbage back into the coder.
    let cleaned = clean_plan(&content);
    info!(
        raw_output = %content,
        cleaned_len = cleaned.len(),
        "Context rebuild: reasoner I/O (truth capture)"
    );
    if cleaned.trim().is_empty() {
        warn!("Context rebuild: reasoner produced empty/unusable output — keeping the canned excise");
        return None;
    }
    if redirect_is_degenerate(&cleaned, ground.lint_digest.is_some(), 1400) {
        warn!("Context rebuild: reasoner output is degenerate (code dump / contradictory 'stop editing' / over-long) — keeping the canned excise");
        return None;
    }
    Some(format!(
        "[HARNESS — STUCK; CONTEXT REBUILT] You were looping and ignored every warning, so the repeated \
         calls were removed and your working context was rebuilt from the ACTUAL files on disk. Here is \
         where things really stand and the single next step — follow it, do not reproduce the loop:\n\n{cleaned}"
    ))
}

/// What the reasoner is told when redirecting a looping coder (assist #3). Its
/// job is to WRITE THE CODER'S NEXT INSTRUCTION — grounded in a fresh LINTER/SYNTAX
/// PROBE of the actual workspace (ground truth), not in the model's own claims.
const REDIRECT_SYSTEM_PROMPT: &str = "\
A small local coding model is STUCK IN A LOOP — repeating an action or thrashing on the same goal \
without progress, and ignoring generic \"stop\" nudges. You are its senior guide. You are given the \
TASK, the LOOP it is stuck in, fresh GROUND TRUTH (the exact action it keeps repeating WITH the result \
it keeps getting, a lint/syntax probe of its workspace, and the actual files on disk), and its RECENT \
ATTEMPTS. WRITE THE CODER'S NEXT INSTRUCTION: a short, direct, imperative prompt (addressed to \"you\") \
telling it exactly what to do next — grounded in the GROUND TRUTH, not in what it claims.\n\
- If the REPEATED ACTION keeps failing the same way, read its result and say plainly what to do \
instead — e.g. `cat` on a directory fails, use `ls`; a search that returns the same result is a dead \
end, so stop searching and write the code with what you already have.\n\
- If the lint reports ERRORS: name the exact file and line, and give the ONE targeted fix — go to that \
line, do not rewrite the whole file blind.\n\
- If lint is CLEAN and the action isn't obviously wrong: the defect is NOT a syntax error. Reason \
about the real cause — a wrong/missing import, a stub shadowing real code, a stale assertion — or, if \
the task looks satisfied, tell it to run the test to verify and then STOP.\n\
- First name the trap in one clause (\"You keep <doing X> and it is not working because <Y>\").\n\
- Do NOT repeat what it is already doing, and do NOT write the code for it. Keep it to 1-3 sentences.\n\
Output ONLY that instruction — it will be shown to the coder verbatim as its next directive.";

/// Per-task redirect count, so a persistent loop can't drive endless reasoner
/// calls. Past the cap the harness falls back to its canned escalation.
static REDIRECT_COUNT: LazyLock<Mutex<HashMap<u64, u32>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
const MAX_REDIRECTS_PER_TASK: u32 = 6;

/// Reject a reasoner "redirect" that a weak reasoner mangled instead of honoring the
/// "1-3 sentences of INSTRUCTION, do not write the code" contract. Observed live from
/// the reasoning-Fabliq: (1) a full `cat <<'EOF' … EOF` code dump of the whole file,
/// and (2) self-contradictory "stop editing live_test.py" advice when live_test.py had
/// a syntax error — which can ONLY be fixed by editing it. A degenerate redirect is
/// discarded (the caller falls back to the RAW lint `file:line`, already a correct,
/// non-contradictory directive). `max_chars` differs by caller (a terse redirect vs a
/// slightly longer context rebuild).
fn redirect_is_degenerate(text: &str, has_dirty_lint: bool, max_chars: usize) -> bool {
    let t = text.trim();
    // Code dump: the reasoner pasted a file/heredoc instead of an instruction.
    if t.contains("cat <<") || t.contains("<<'EOF'") || t.contains("<<EOF") || t.contains("<<\"EOF\"")
    {
        return true;
    }
    // Pasted code: several lines carrying code markers (real OR literal-escaped `\n`).
    let newlines = t.matches('\n').count() + t.matches("\\n").count();
    let code_markers = t.contains("def ")
        || t.contains("import ")
        || t.contains("return {")
        || t.contains("):");
    if newlines > 5 && code_markers {
        return true;
    }
    // Over-long: a redirect is a short instruction, not a wall of text.
    if t.chars().count() > max_chars {
        return true;
    }
    // Contradictory: telling the coder to STOP editing / making changes to a file when
    // a file has a syntax error to FIX (you cannot fix a syntax error without editing).
    if has_dirty_lint {
        let l = t.to_lowercase();
        let says_stop =
            l.contains("stop ") || l.contains("don't ") || l.contains("do not ") || l.contains("avoid ");
        let about_editing = l.contains("edit")
            || l.contains("making change")
            || l.contains("changes to")
            || l.contains("changes there")
            || l.contains("touch")
            || l.contains("modif");
        if says_stop && about_editing {
            return true;
        }
    }
    false
}

/// Reasoned Guidance assist #3 — THRASH → PROBE → REASONED GUIDANCE. When a loop
/// guard fires, a canned nudge is soft prompt text a 9B routinely ignores. Instead:
/// the caller runs a read-only LINTER/SYNTAX probe of the workspace for ground
/// truth, and this hands the reasoner {task, loop, PROBE, recent attempts} to author
/// the coder's NEXT INSTRUCTION grounded in the real errors (or, on a clean probe,
/// to reason past syntax toward the actual cause). Returns `None` when the reasoner
/// is disabled/unreachable, the per-task cap is hit, or the redirect came back empty
/// — the caller then falls back to the raw probe grounding / canned directive.
pub async fn redirect_from_loop(
    pool: &OllamaClientPool,
    reasoner: &OllamaEndpoint,
    task: &str,
    loop_summary: &str,
    ground: &crate::ground_truth::GroundTruth,
    evidence: &str,
) -> Option<String> {
    // No reasoner, or NO fresh signal to reason over → don't call it (a groundless
    // reasoner hallucinates a cause). `has_signal()` is true whenever there's a
    // repeated action, dirty lint, or a live file — which covers action-loops whose
    // lint is clean, the exact case the old dirty-only gate dropped.
    if !reasoner.enabled || !ground.has_signal() {
        return None;
    }
    let key = task_key(task);
    {
        let cache = REDIRECT_COUNT.lock().ok()?;
        if cache.get(&key).copied().unwrap_or(0) >= MAX_REDIRECTS_PER_TASK {
            return None;
        }
    }
    let payload = format!(
        "TASK:\n{task}\n\nLOOP — the coder is {loop_summary}.\n\n\
         GROUND TRUTH (fresh — the repeated failing action, the workspace lint, and the files on disk NOW):\n{}\n\n\
         RECENT ATTEMPTS:\n{evidence}",
        ground.render()
    );
    let mut ep = reasoner.clone();
    ep.timeout_seconds = ep.timeout_seconds.clamp(20, 90);
    let body = pool
        .chat(
            &ep,
            vec![json!({ "role": "user", "content": payload })],
            Some(REDIRECT_SYSTEM_PROMPT),
            None,
        )
        .await;
    // Capture the RAW reasoner output before any cleaning.
    let raw = body
        .as_ref()
        .and_then(|b| b.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    // Strip `<think>` AND leaked model-format tokens so a Gemma-fable reasoner's
    // `<|tool_call>…` can't pollute the redirect pinned into the coder's context.
    // Same cleaning as the plan: strip think/leaked tokens AND normalize exotic
    // Unicode whitespace (Fabliq emits U+202F around identifiers) before it's pinned.
    let redirect = clean_plan(&raw);

    // TRUTH CAPTURE. The redirect the reasoner produces is otherwise recorded
    // NOWHERE — it goes to the TUI (push_nudge, unlogged) and the local model's
    // per-request prompt (not persisted to the rollout), so a degenerate redirect
    // (an echo, a self-tag, an empty) vanishes and leaves only conjecture. Log the
    // exact input + raw + cleaned output so the NEXT occurrence is ground truth on
    // disk. Fires only on loops (not per turn), so it isn't spam. Do NOT gate the
    // model's behavior on a heuristic here — observe first, decide from real data.
    info!(
        loop_summary,
        payload = %payload,
        raw_output = %raw,
        cleaned_output = %redirect,
        "Reasoned guidance: redirect reasoner I/O (truth capture)"
    );

    if redirect.trim().is_empty() {
        warn!("Reasoned guidance: reasoner returned an empty redirect — keeping the canned nudge");
        return None;
    }
    if redirect_is_degenerate(&redirect, ground.lint_digest.is_some(), 600) {
        warn!(
            redirect = %redirect,
            "Reasoned guidance: redirect is degenerate (code dump / contradictory 'stop editing' a file with a lint error / over-long) — discarding, falling back to raw grounding"
        );
        return None;
    }
    if let Ok(mut cache) = REDIRECT_COUNT.lock() {
        if cache.len() >= PLAN_CACHE_CAP {
            cache.clear();
        }
        *cache.entry(key).or_insert(0) += 1;
    }
    info!(loop_summary, "Reasoned guidance: reasoner authored a redirect for a loop");
    Some(format!(
        "[REDIRECT — you are stuck in a loop and the generic nudge is not working. A guide has \
         chosen your next step. Follow this exactly:]\n{redirect}"
    ))
}

/// What the reasoner is told when deciding whether a looping coder has GENUINELY
/// changed course (assist #4). A confirmed pivot buys the coder a CLEAN SLATE —
/// the loop-detection state is reset — so this must be strict: a reworded version
/// of the SAME failing approach is NOT a change of course.
const COURSE_CHANGE_PROMPT: &str = "\
A small local coding model has been STUCK IN A LOOP (repeating an action or thrashing on one goal \
without progress), and the harness has been firing \"you're looping\" guards at it. It just produced a \
NEW response. Decide ONE thing: is this new response a GENUINE CHANGE OF COURSE that breaks the loop, \
or the SAME approach reworded?\n\
A genuine change of course is a materially DIFFERENT strategy: a different tool, a different target \
(file / URL / command / endpoint), or an explicit decision to ABANDON the failing approach for a new \
one it has not already tried. Rewording the same query, retrying the same failing call with cosmetic \
tweaks, or merely RESTATING an intention to change without a concrete different action is NOT a change \
of course.\n\
Restarting from scratch — rewriting everything, \"starting over\", a \"fresh implementation\" — is \
ALSO NOT a change of course: it throws the existing work away and re-enters the SAME loop from the \
top. Answer false for a restart.\n\
Be strict and refuse by default: if you are not clearly convinced the new response is a genuinely \
different approach, answer false. Output JSON ONLY: {\"changed\": true|false, \"reason\": \"<one clause>\"}.";

#[derive(serde::Deserialize)]
struct CourseChangeResponse {
    #[serde(default)]
    changed: bool,
    #[serde(default)]
    reason: String,
}

/// True when `text` describes a RESTART — throwing the work away and starting over —
/// rather than a genuine pivot. A restart re-enters the same loop from the top, so it
/// must never be granted a loop-state reset. Deterministic backstop for the weak
/// course-change reasoner (see [`course_change`]).
fn looks_like_restart(text: &str) -> bool {
    let t = text.to_lowercase();
    const MARKERS: &[&str] = &[
        "from scratch",
        "start over",
        "starting over",
        "start fresh",
        "fresh implementation",
        "fresh start",
        "rewrite everything",
        "rewrite the whole",
        "rewrite it all",
        "reimplement",
        "re-implement",
        "begin again",
        "start again",
        "starting again",
        "scrap everything",
        "delete everything",
        "start from the beginning",
        "start from scratch",
    ];
    MARKERS.iter().any(|m| t.contains(m))
}

/// Per-task count of GRANTED course-change resets, so a coder that keeps
/// "pivoting" (genuinely or by feint) cannot farm an unlimited number of clean
/// slates and thereby make the loop guards permanently toothless.
static COURSE_CHANGE_COUNT: LazyLock<Mutex<HashMap<u64, u32>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
const MAX_COURSE_CHANGES_PER_TASK: u32 = 6;

/// Reasoned Guidance assist #4 — COURSE-CHANGE RESET. When the coder returns a new
/// response WHILE a loop guard is active, ask the reasoner whether it is genuinely
/// changing course. A `true` verdict tells the caller to reset the loop-detection
/// state (a clean slate) so a real pivot is not instantly re-buried by the same
/// guards re-firing on stale history — the exact failure where the coder figured
/// out "drop the /v1" but never got an unobstructed turn to try it.
///
/// Strict and bounded: the cap is checked FIRST (past budget we neither call the
/// reasoner nor reset), the reasoner refuses by default, and unparseable output is
/// treated as "no change". Returns `false` (no reset) whenever we cannot justify a
/// reset — a disabled/unreachable reasoner, a hit cap, or a non-affirmative verdict.
pub async fn course_change(
    pool: &OllamaClientPool,
    reasoner: &OllamaEndpoint,
    task: &str,
    prior_loop: &str,
    new_action: &str,
) -> bool {
    if !reasoner.enabled {
        return false;
    }
    let key = task_key(task);
    if let Ok(cache) = COURSE_CHANGE_COUNT.lock()
        && cache.get(&key).copied().unwrap_or(0) >= MAX_COURSE_CHANGES_PER_TASK
    {
        return false;
    }
    let payload = format!(
        "TASK:\n{task}\n\nLOOP — the coder is {prior_loop}.\n\nITS NEW RESPONSE (reasoning + the tool \
         call it is about to make):\n{new_action}"
    );
    let mut ep = reasoner.clone();
    ep.timeout_seconds = ep.timeout_seconds.clamp(20, 90);
    let body = pool
        .chat(
            &ep,
            vec![json!({ "role": "user", "content": payload })],
            Some(COURSE_CHANGE_PROMPT),
            Some("json"),
        )
        .await;
    let Some(body) = body else { return false };
    let Some(content) = body
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
    else {
        return false;
    };
    let stripped = crate::classifier::strip_think_tags(content);
    let json = crate::classifier::extract_json_object(stripped.trim());
    let parsed: CourseChangeResponse = match serde_json::from_str(json) {
        Ok(p) => p,
        Err(_) => return false, // refuse by default on unparseable output
    };
    if !parsed.changed {
        return false;
    }
    // BACKSTOP: a weak reasoner can mistake a RESTART for a pivot — it superficially
    // reads as a "new strategy". But restarting throws the work away and re-enters the
    // SAME loop from the top; it must NOT reset loop state (in session 019f35d3
    // "beginning a fresh implementation from scratch" was granted as a genuine pivot,
    // which just let the looping continue). Veto it deterministically, whatever the
    // reasoner said, checking both the new action and the reasoner's own stated reason.
    if looks_like_restart(new_action) || looks_like_restart(&parsed.reason) {
        info!(
            reason = %parsed.reason,
            "Reasoned guidance: course-change REFUSED — the 'new' approach is a RESTART (throws work away, re-enters the loop), not a pivot"
        );
        return false;
    }
    // Granted — charge it against the per-task budget so pivots stay finite.
    if let Ok(mut cache) = COURSE_CHANGE_COUNT.lock() {
        if cache.len() >= PLAN_CACHE_CAP {
            cache.clear();
        }
        *cache.entry(key).or_insert(0) += 1;
    }
    info!(
        reason = %parsed.reason,
        "Reasoned guidance: reasoner confirmed a genuine course change — resetting loop state"
    );
    true
}

/// Clear the per-task redirect budget. Called alongside a confirmed course-change
/// reset so a genuine pivot also restores the coder's redirect allowance for any
/// *new* loop it might hit later — the redirect count is loop state, so it resets
/// with the rest of the loop-detection subsystem.
pub fn reset_redirect_budget(task: &str) {
    let key = task_key(task);
    if let Ok(mut cache) = REDIRECT_COUNT.lock() {
        cache.remove(&key);
    }
}

/// After a confirmed course change, suppress the (transcript-derived) loop nudges
/// for this many consecutive coder model-returns. The point: the family-A guards
/// re-derive from the SAME still-loopy history every turn, so a pivot that adds a
/// few new reads/curls keeps them firing and gets re-buried before it can finish.
/// A short grace gives the new approach unobstructed turns to EXECUTE. Small on
/// purpose — enough to act, not so many that a genuinely new loop goes unnoticed.
static LOOP_GRACE: LazyLock<Mutex<HashMap<u64, u32>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
const COURSE_CHANGE_GRACE_TURNS: u32 = 3;

/// Grant `task` its post-course-change grace window. Called when [`course_change`]
/// confirms a genuine pivot.
pub fn grant_loop_grace(task: &str) {
    let key = task_key(task);
    if let Ok(mut cache) = LOOP_GRACE.lock() {
        if cache.len() >= PLAN_CACHE_CAP {
            cache.clear();
        }
        cache.insert(key, COURSE_CHANGE_GRACE_TURNS);
    }
}

/// Consume one turn of grace for `task`: returns `true` while grace remains (the
/// caller then suppresses this turn's loop nudges), decrementing as it goes, and
/// `false` once the window is spent. A missing entry is simply "no grace".
pub fn consume_loop_grace(task: &str) -> bool {
    let key = task_key(task);
    let Ok(mut cache) = LOOP_GRACE.lock() else {
        return false;
    };
    match cache.get_mut(&key) {
        Some(n) if *n > 0 => {
            *n -= 1;
            if *n == 0 {
                cache.remove(&key);
            }
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_gate_allows_inspection_blocks_mutation() {
        // Allowed: pure reads + read-only git/curl + safe chains.
        for ok in [
            "ls -la",
            "cat src/lambda.py",
            "grep -rn handle .",
            "find . -name '*.py'",
            "head -50 README.md | grep API",
            "git status",
            "git log --oneline -20",
            "git diff HEAD~1",
            "curl -s https://api.handle.me/openapi.json",
            "wc -l *.py",
        ] {
            assert!(is_read_only_command(ok), "should ALLOW: {ok}");
        }
        // Refused: writes, redirects, in-place edits, installs, mutating git/curl.
        for bad in [
            "rm -rf build",
            "echo x > file.txt",
            "sed -i 's/a/b/' f.py",
            "git commit -m x",
            "git checkout .",
            "pip install requests",
            "curl -o out.json https://x/y",
            "mv a b",
            "cp a b",
            "python3 -c 'open(\"f\",\"w\")'",
            "cat a >> b",
        ] {
            assert!(!is_read_only_command(bad), "should REFUSE: {bad}");
        }
    }

    #[test]
    fn planner_extracts_structured_and_leaked_gemma_calls() {
        // Structured tool_calls (server-parsed).
        let structured = json!({"message":{"content":"","tool_calls":[
            {"function":{"name":"read_file","arguments":{"path":"foo.py"}}}
        ]}});
        let calls = extract_planner_tool_calls(&structured);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "read_file");
        assert_eq!(calls[0].1["path"], "foo.py");

        // Leaked Gemma dialect in the content — recovered via the shared parser.
        let leaked = json!({"message":{"content":"<|tool_call>call:exec_command{cmd:<|\"|>ls -la<|\"|>}<tool_call|>"}});
        let calls = extract_planner_tool_calls(&leaked);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "exec_command");
        assert_eq!(calls[0].1["cmd"], "ls -la");
    }

    #[test]
    fn clean_plan_strips_think_and_leaked_tokens() {
        let raw = "<think>let me plan</think>1. Do the thing\n2. Verify it";
        let cleaned = clean_plan(raw);
        assert!(!cleaned.contains("<think>"));
        assert!(cleaned.contains("1. Do the thing"));
    }

    #[test]
    fn rejects_degenerate_reasoner_redirects() {
        // The two real bad redirects observed live (session 019f387d):
        // (1) a `cat <<EOF` code dump instead of a 1-3 sentence instruction.
        let dump = "`cat <<'EOF'\\nimport json, requests\\ndef lambda_handler(event, context):\\n    return {}\\nEOF\\n`";
        assert!(redirect_is_degenerate(dump, true, 600), "code dump rejected");
        // (2) contradictory 'stop editing' a file that HAS a lint error to fix.
        let contradiction =
            "You keep editing live_test.py and introducing syntax errors; stop making changes there.";
        assert!(
            redirect_is_degenerate(contradiction, true, 600),
            "stop-editing-a-broken-file rejected when lint is dirty"
        );
        // Same words but NO dirty lint → allowed (could legitimately mean 'bug is elsewhere').
        assert!(!redirect_is_degenerate(contradiction, false, 600));
        // A normal terse instruction passes.
        assert!(!redirect_is_degenerate("It's a directory — use `ls` instead of `cat`.", true, 600));
    }

    #[test]
    fn critique_parses_object_shaped_issues() {
        // The EXACT shape the reasoner emitted in session 019f38a5 — issues as OBJECTS,
        // not strings — which the old `Vec<String>` parser rejected, discarding a correct
        // critique (a real ImportError) and letting a false completion through.
        let raw = r#"{"complete": false, "issues": [{"description": "test_lambda.py failed: resolve_ada_handle not defined", "root_cause": "Missing implementation in lambda_handler.py", "severity": "high"}]}"#;
        let parsed: CritiqueResponse =
            serde_json::from_str(raw).expect("object-shaped issues must parse");
        assert!(!parsed.complete);
        assert_eq!(parsed.issues.len(), 1);
        let t = parsed.issues[0].text();
        assert!(t.contains("resolve_ada_handle not defined"), "extracts a human string: {t}");
        assert!(t.contains("root cause"), "includes the root cause: {t}");
        // The originally-requested string shape still works.
        let raw2 = r#"{"complete": false, "issues": ["missing tests", "wrong endpoint"]}"#;
        let p2: CritiqueResponse = serde_json::from_str(raw2).unwrap();
        assert_eq!(
            p2.issues.iter().map(CritiqueIssue::text).collect::<Vec<_>>(),
            vec!["missing tests", "wrong endpoint"]
        );
    }

    #[test]
    fn restart_is_not_a_course_change() {
        // The exact phrase that fooled the reasoner in session 019f35d3, plus siblings.
        assert!(looks_like_restart("Beginning a fresh implementation of the files from scratch"));
        assert!(looks_like_restart("Let me start over and rewrite everything"));
        assert!(looks_like_restart("I'll REIMPLEMENT the handler"));
        assert!(looks_like_restart("scrap everything and start again"));
        // A genuine pivot (different concrete approach) is NOT flagged as a restart.
        assert!(!looks_like_restart("Switch from web_search to web_fetch on the API URL"));
        assert!(!looks_like_restart("Add the missing import and re-run the test"));
        assert!(!looks_like_restart("Use ls instead of cat since it's a directory"));
    }

    #[test]
    fn clean_plan_normalizes_exotic_unicode_whitespace() {
        // The exact Fabliq defect: NARROW NO-BREAK SPACE (U+202F) around identifiers,
        // plus a zero-width space — real newlines (list structure) must survive.
        // U+202F / U+00A0 (visible spaces) → ASCII space; U+200B (zero-width) dropped.
        let raw = "1. Add a docstring to\u{202f}lambda_handler.py\u{202f}here\n2. Use\u{00a0}unittest.mock now\u{200b}";
        let cleaned = clean_plan(raw);
        assert!(!cleaned.chars().any(|c| c == '\u{202f}' || c == '\u{00a0}' || c == '\u{200b}'),
            "no exotic whitespace remains: {cleaned:?}");
        assert_eq!(cleaned, "1. Add a docstring to lambda_handler.py here\n2. Use unittest.mock now");
        assert!(cleaned.contains('\n'), "the numbered-list newline is preserved");
    }

    #[test]
    fn gather_turn_round_trips_structured_calls_as_protocol() {
        // A server-structured tool call is re-fed VERBATIM (native shape) with its
        // id preserved for pairing — never flattened into echoable prose.
        let body = json!({"message":{"content":"","tool_calls":[
            {"id":"abc","function":{"name":"web_fetch","arguments":{"url":"https://x/y"}}}
        ]}});
        let calls = extract_planner_tool_calls(&body);
        let (turn, ids) = build_gather_turn(&body, &calls, ClientFlavor::OpenAICompat);
        assert_eq!(ids, vec!["abc".to_string()]);
        assert_eq!(turn["role"], "assistant");
        assert_eq!(turn["tool_calls"][0]["function"]["name"], "web_fetch");
        // The `(called …)` prose shape must NOT appear anywhere in the turn.
        assert!(!turn.to_string().contains("(called "));
    }

    #[test]
    fn gather_turn_reconstructs_leaked_calls_cleanly() {
        // A LEAKED Gemma call (in content, no structured tool_calls) is rebuilt as a
        // clean structured turn — the raw dialect tokens don't re-enter the transcript.
        let body = json!({"message":{"content":"<|tool_call>call:read_file{path:<|\"|>foo.py<|\"|>}<tool_call|>"}});
        let calls = extract_planner_tool_calls(&body);
        assert_eq!(calls.len(), 1);
        // OpenAI-compat encodes arguments as a JSON string.
        let (turn, ids) = build_gather_turn(&body, &calls, ClientFlavor::OpenAICompat);
        assert_eq!(ids, vec!["call_0".to_string()]);
        assert_eq!(turn["tool_calls"][0]["function"]["name"], "read_file");
        assert!(turn["tool_calls"][0]["function"]["arguments"].is_string());
        assert!(!turn.to_string().contains("<|tool_call>"));
        // Ollama encodes arguments as an object.
        let (turn_o, _) = build_gather_turn(&body, &calls, ClientFlavor::Ollama);
        assert!(turn_o["tool_calls"][0]["function"]["arguments"].is_object());
    }

    #[test]
    fn trivial_or_disabled_tasks_are_skipped() {
        // A disabled reasoner never plans (endpoint construction is via config).
        let key_a = task_key("build a thing");
        let key_b = task_key("  build a thing  ");
        assert_eq!(key_a, key_b, "task key must be whitespace-insensitive");
    }

    #[test]
    fn min_task_length_gate() {
        assert!("tiny".chars().count() < MIN_TASK_CHARS);
        assert!(
            "add a comprehensive test suite for the resolver"
                .chars()
                .count()
                >= MIN_TASK_CHARS
        );
    }
}
