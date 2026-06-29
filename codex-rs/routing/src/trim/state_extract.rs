//! Deterministic state extractor.
//!
//! Walks the parsed transcript and synthesizes the four state blocks the
//! local model sees in its prelude:
//!   - World state: files seen / modified, branch (if known), test results
//!   - Actions taken: an audit log derived from successful tool calls
//!   - Unresolved errors: any tool output where `success = false`
//!   - In-flight work: orchestration calls (spawn/wait) that don't have a
//!     terminal status yet
//!
//! Pure data extraction — no LLM calls, no judgment.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use codex_protocol::models::ResponseItem;

use super::items;
use super::items::ParsedTranscript;
use super::items::TrimItem;
use super::signatures::path_from_signature;

#[derive(Debug, Clone, Default)]
pub struct ExtractedState {
    pub files_seen: BTreeSet<String>,
    pub files_modified: BTreeMap<String, ModifiedFile>,
    pub actions: Vec<ActionReceipt>,
    pub unresolved_errors: Vec<UnresolvedError>,
    pub in_flight: Vec<InFlight>,
    pub test_runs: Vec<TestRun>,
    /// When the model is stuck calling the same tool with identical args
    /// repeatedly. Surfaced prominently in the prelude so the model is
    /// nudged to try a different approach.
    pub repetition: Option<RepetitionAlert>,
    /// The most recent `apply_patch` attempt FAILED. Drives a directive steering
    /// the model to rewrite the whole file with `write_file` instead of
    /// re-patching against a file that doesn't match its mental model.
    pub patch_failure: Option<PatchFailure>,
}

#[derive(Debug, Clone)]
pub struct PatchFailure {
    /// File the failed patch targeted (from the `*** Update/Add File:` header).
    pub path: String,
    /// Short excerpt of the apply_patch error (for the directive's "why").
    pub excerpt: String,
}

#[derive(Debug, Clone)]
pub struct RepetitionAlert {
    pub tool_name: String,
    pub command_summary: String,
    pub count: usize,
    pub last_output_excerpt: String,
    /// `call_id` of the most-recent repeated tool call. The renderer uses this
    /// to replace that call's tool-output with a synthesized "stop repeating"
    /// result, so the model sees the intervention in the slot it's trained to
    /// read instead of only as prelude prose it can ignore.
    pub call_id: String,
    /// When `true`, the model isn't repeating a byte-identical call — it's
    /// *thrashing* toward the same goal with varying commands (e.g. editing the
    /// same file and re-running the same failing test). The renderer turns this
    /// into a forced-diagnosis directive ("read the failure and explain the root
    /// cause before acting") instead of a plain STOP, because the model needs to
    /// engage with the error, not just stop. See [`detect_unproductive_recurrence`].
    pub force_diagnosis: bool,
    /// The `(tool_name, signature)`-derived signature of the looping call, when
    /// the loop is a single repeated signature (empty for same-target thrash
    /// where args vary). Lets the renderer prune exactly the loop's calls.
    pub signature: String,
    /// The loop has persisted past the point where advisory nudges + the hard
    /// block have demonstrably been ignored (`count >= ESCALATION_THRESHOLD`).
    /// Triggers context surgery: the renderer excises the loop's own turns and
    /// replaces them with a single reframe, so the model stops copying the
    /// pattern out of its own saturated context. Only set for signature-based
    /// loops (we can prune those precisely).
    pub escalate: bool,
}

/// Count at which a repeated loop escalates from advisory nudge + hard block to
/// context surgery (excise the loop + reframe). Past this, the model has ignored
/// the gentler interventions, so we change what it *sees* rather than what we
/// *tell* it.
const ESCALATION_THRESHOLD: usize = 6;

#[derive(Debug, Clone)]
pub struct ModifiedFile {
    pub op: ModifyOp,
    pub turn_id: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum ModifyOp {
    Created,
    Edited,
    Deleted,
}

#[derive(Debug, Clone)]
pub struct ActionReceipt {
    pub turn_id: u32,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub struct UnresolvedError {
    pub turn_id: u32,
    pub tool_name: String,
    pub call_id: String,
    pub excerpt: String,
}

#[derive(Debug, Clone)]
pub struct InFlight {
    pub turn_id: u32,
    pub tool_name: String,
    pub call_id: String,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct TestRun {
    pub turn_id: u32,
    pub command: String,
    pub passed: bool,
    pub summary: String,
}

/// Extract state from the entire transcript. We extract from active turn too —
/// the model benefits from seeing materialized state for very-recent actions
/// it took itself, especially after a long active turn.
pub fn extract(parsed: &ParsedTranscript, _active_turn: u32) -> ExtractedState {
    let mut state = ExtractedState::default();

    // Map call_id -> (tool_name, turn_id) so we can resolve outputs efficiently
    // when extracting actions and errors.
    let mut call_index: BTreeMap<String, (String, u32, String)> = BTreeMap::new();
    for item in &parsed.items {
        if let TrimItem::ToolCall {
            call_id,
            tool_name,
            turn_id,
            args,
            ..
        } = item
        {
            call_index.insert(call_id.clone(), (tool_name.clone(), *turn_id, args.clone()));
        }
    }

    // Track outputs that had a terminal status, so anything left over is
    // considered in-flight.
    let mut completed_calls: BTreeSet<String> = BTreeSet::new();

    for item in &parsed.items {
        match item {
            TrimItem::ToolCall {
                tool_name,
                args,
                signature,
                turn_id,
                ..
            } => {
                if let Some(path) = path_from_signature(signature) {
                    if matches!(
                        tool_name.as_str(),
                        "text_editor" | "view_image" | "list_dir"
                    ) && !path.is_empty()
                        && path != "?"
                    {
                        state.files_seen.insert(path.to_string());
                    }
                }
                if let Some((path, op)) = derive_modification(tool_name, args) {
                    state.files_modified.insert(
                        path,
                        ModifiedFile {
                            op,
                            turn_id: *turn_id,
                        },
                    );
                }
            }
            TrimItem::ToolOutput {
                tool_name,
                call_id,
                success,
                content,
                turn_id,
                ..
            } => {
                completed_calls.insert(call_id.clone());

                if !*success {
                    state.unresolved_errors.push(UnresolvedError {
                        turn_id: *turn_id,
                        tool_name: tool_name.clone(),
                        call_id: call_id.clone(),
                        excerpt: excerpt(content, 240),
                    });
                    continue;
                }

                // Look up the originating call for full context (args).
                let Some((_call_tool, _call_turn, args)) = call_index.get(call_id) else {
                    continue;
                };

                if let Some(receipt) = derive_action_receipt(tool_name, args, content, *turn_id) {
                    state.actions.push(receipt);
                }
                if let Some(test) = derive_test_run(tool_name, args, content, *turn_id) {
                    state.test_runs.push(test);
                }
            }
            _ => {}
        }
    }

    // Anything called but not completed is in-flight.
    for (call_id, (tool_name, turn_id, args)) in &call_index {
        if completed_calls.contains(call_id) {
            continue;
        }
        if !is_orchestration_or_async(tool_name) {
            continue;
        }
        state.in_flight.push(InFlight {
            turn_id: *turn_id,
            tool_name: tool_name.clone(),
            call_id: call_id.clone(),
            note: short_args(args, 80),
        });
    }

    // Two kinds of repetition: exact-signature (byte-identical args) and
    // same-target failure streak (different args, same file, consecutive
    // failures). The exact-signature detector catches "stuck calling X with
    // the same args" loops; the failure-streak detector catches "writing
    // syntactically-different-but-semantically-wrong patches on the same
    // file" loops.
    state.patch_failure = detect_patch_failure(parsed);
    state.repetition = detect_repetition(parsed)
        .or_else(|| detect_same_target_failure_repetition(parsed))
        .or_else(|| detect_unproductive_recurrence(parsed));
    if let Some(alert) = state.repetition.as_ref() {
        tracing::info!(
            tool_name = %alert.tool_name,
            count = alert.count,
            summary = %alert.command_summary,
            "Repetition alert fired — STOP block will be added to next prelude"
        );
    }

    state
}

/// Detect when the model is stuck calling the same tool with the same args
/// repeatedly. Walk the most recent ToolCall items in order; if the last 3+
/// share the same `(tool_name, signature)`, that's a stuck loop.
///
/// Threshold: 3 consecutive identical calls. Two could be a legitimate retry
/// after a transient error; three means the model isn't learning from the
/// outputs.
fn detect_repetition(parsed: &ParsedTranscript) -> Option<RepetitionAlert> {
    const THRESHOLD: usize = 3;

    // Walk from the end, collecting consecutive ToolCall signatures until we
    // hit a different signature or a non-ToolCall, non-ToolOutput item.
    let mut last_signature: Option<(String, String)> = None;
    let mut count = 0usize;
    let mut last_call_args: Option<String> = None;
    let mut last_call_id: Option<String> = None;

    for item in parsed.items.iter().rev() {
        match item {
            TrimItem::ToolCall {
                tool_name,
                signature,
                args,
                call_id,
                ..
            } => {
                let key = (tool_name.clone(), signature.clone());
                match &last_signature {
                    None => {
                        last_signature = Some(key);
                        last_call_args = Some(args.clone());
                        last_call_id = Some(call_id.clone());
                        count = 1;
                    }
                    Some(prev) if *prev == key => {
                        count += 1;
                    }
                    Some(_) => break,
                }
            }
            // Tool outputs interleave with calls; skip them.
            TrimItem::ToolOutput { .. } => continue,
            // Short assistant narration between identical calls is common
            // (many local models emit 1-2 chars of content like "." or " "
            // alongside every tool_call). Treat it as transparent so the
            // streak keeps counting — same reasoning the same-target
            // detector below already applies.
            TrimItem::AssistantText { .. } => continue,
            // Reasoning channel text is also transparent — it's private
            // scratchpad, not a meaningful break in the stuck-loop pattern.
            TrimItem::Reasoning { .. } => continue,
            // User messages or anything else (Other, etc.) legitimately
            // break the streak — a new user turn is a new task.
            _ => break,
        }
    }

    if count < THRESHOLD {
        return None;
    }

    let (tool_name, signature) = last_signature?;
    let command_summary = short_args(last_call_args.as_deref().unwrap_or(""), 100);

    // Pull the most recent matching output's excerpt for context.
    let last_output_excerpt = last_call_id
        .as_ref()
        .and_then(|id| {
            parsed.items.iter().rev().find_map(|item| {
                if let TrimItem::ToolOutput {
                    call_id, content, ..
                } = item
                    && call_id == id
                {
                    Some(excerpt(content, 200))
                } else {
                    None
                }
            })
        })
        .unwrap_or_default();

    Some(RepetitionAlert {
        tool_name,
        command_summary,
        count,
        last_output_excerpt,
        call_id: last_call_id.unwrap_or_default(),
        force_diagnosis: false,
        escalate: count >= ESCALATION_THRESHOLD,
        signature,
    })
}

/// Detect interleaved no-progress "thrash": the model working toward the same
/// end with *varying* commands, so the consecutive-identical detector misses it
/// — e.g. running the same test 3+ times with edits between, never passing. We
/// count tool-call signatures across a recent window; a signature whose *failing*
/// occurrences recur `THRESHOLD`+ times means the model keeps trying the same
/// thing without progress. Rendered as a forced-diagnosis directive (read the
/// failure, state the root cause, then make ONE change) rather than a plain STOP.
///
/// Crucially this is **productivity-gated**: only occurrences whose output
/// failed count toward the threshold. A signature that recurs but *succeeds*
/// (a test that now passes, a routine `ls`/`grep`/`git status` re-run) is
/// healthy and must not trip the nudge — and "diagnose the failure" is
/// incoherent when there is no failure to read.
fn detect_unproductive_recurrence(parsed: &ParsedTranscript) -> Option<RepetitionAlert> {
    const WINDOW: usize = 24;
    const THRESHOLD: usize = 3;

    // Outcome lookup, so a recurrence only counts as thrash when it is failing.
    let mut success_by_call: BTreeMap<String, bool> = BTreeMap::new();
    for item in &parsed.items {
        if let TrimItem::ToolOutput {
            call_id, success, ..
        } = item
        {
            success_by_call.insert(call_id.clone(), *success);
        }
    }

    let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
    // Walking from the end, the first time we see a key is its most-recent
    // occurrence — capture its args + call_id for the diagnosis prompt.
    let mut most_recent: BTreeMap<(String, String), (String, String)> = BTreeMap::new();
    let mut window_calls = 0usize;

    for item in parsed.items.iter().rev() {
        match item {
            TrimItem::ToolCall {
                tool_name,
                signature,
                args,
                call_id,
                ..
            } => {
                // The window is the last WINDOW tool calls regardless of outcome,
                // but only FAILED occurrences accrue toward the threshold. An
                // unknown outcome (no output recorded yet — an in-flight call) is
                // treated as not-failed, so we never fire on a pending call.
                window_calls += 1;
                if success_by_call.get(call_id) == Some(&false) {
                    let key = (tool_name.clone(), signature.clone());
                    *counts.entry(key.clone()).or_insert(0) += 1;
                    most_recent
                        .entry(key)
                        .or_insert_with(|| (args.clone(), call_id.clone()));
                }
                if window_calls >= WINDOW {
                    break;
                }
            }
            // Outputs / narration / reasoning are transparent within the window.
            TrimItem::ToolOutput { .. }
            | TrimItem::AssistantText { .. }
            | TrimItem::Reasoning { .. } => continue,
            // A user message is a new task boundary — stop the window there.
            _ => break,
        }
    }

    let (key, count) = counts
        .into_iter()
        .filter(|(_, c)| *c >= THRESHOLD)
        .max_by_key(|(_, c)| *c)?;
    let (tool_name, signature) = key.clone();
    let (args, call_id) = most_recent.get(&key)?.clone();

    let last_output_excerpt = parsed
        .items
        .iter()
        .rev()
        .find_map(|item| match item {
            TrimItem::ToolOutput {
                call_id: cid,
                content,
                ..
            } if *cid == call_id => Some(excerpt(content, 300)),
            _ => None,
        })
        .unwrap_or_default();

    Some(RepetitionAlert {
        tool_name,
        command_summary: short_args(&args, 100),
        count,
        last_output_excerpt,
        call_id,
        force_diagnosis: true,
        escalate: count >= ESCALATION_THRESHOLD,
        signature,
    })
}

/// Did the MOST RECENT `apply_patch` attempt fail? Returns the target file and a
/// short error excerpt so the prelude can steer the model to a full `write_file`
/// rewrite. Fires on the *latest* patch outcome (not a streak): a small model
/// often patches against code it believes it wrote but never landed (an earlier
/// failed write), so its `Update` context never matches and re-patching fails the
/// same way — a whole-file rewrite (which overwrites, no context to match) breaks
/// the cycle on the first failure. If the latest patch SUCCEEDED, returns `None`
/// (the model recovered on its own), so the directive only persists while stuck.
fn detect_patch_failure(parsed: &ParsedTranscript) -> Option<PatchFailure> {
    let mut outputs: BTreeMap<String, (bool, String)> = BTreeMap::new();
    for item in &parsed.items {
        if let TrimItem::ToolOutput {
            call_id,
            success,
            content,
            ..
        } = item
        {
            outputs.insert(call_id.clone(), (*success, content.clone()));
        }
    }
    // Walk reverse and keep only each file's MOST-RECENT edit outcome (keyed by
    // basename so an absolute apply_patch path and a relative write_file path to
    // the same file reconcile). Fire only if that latest outcome is a failed
    // apply_patch — so a later successful `write_file` rewrite (the remedy this
    // directive asks for) clears it instead of looping forever.
    let mut settled: BTreeSet<String> = BTreeSet::new();
    for item in parsed.items.iter().rev() {
        let TrimItem::ToolCall {
            tool_name,
            args,
            call_id,
            ..
        } = item
        else {
            continue;
        };
        let Some(path) = edit_target_path(tool_name, args) else {
            continue;
        };
        let base = path.rsplit('/').next().unwrap_or(&path).to_string();
        if !settled.insert(base) {
            continue; // a more-recent edit to this file already decided its state
        }
        match outputs.get(call_id) {
            // Pending or succeeded → this file is not in a failed-patch state.
            None => {}
            Some((true, _)) => {}
            Some((false, content)) => {
                if tool_name == "apply_patch" {
                    return Some(PatchFailure {
                        path,
                        excerpt: excerpt(content, 200),
                    });
                }
            }
        }
    }
    None
}

/// The file a content-editing tool call targets, across the tools a local model
/// uses (`apply_patch` Update/Add header, the synthetic `write_file`/`edit_file`
/// family, and `text_editor`). Used to reconcile edit outcomes per file.
fn edit_target_path(tool_name: &str, args_raw: &str) -> Option<String> {
    match tool_name {
        "apply_patch" | "text_editor" => derive_modification(tool_name, args_raw).map(|(p, _)| p),
        "write_file" | "create_file" | "edit_file" | "str_replace" => {
            let v: serde_json::Value = serde_json::from_str(args_raw).ok()?;
            ["path", "file_path", "file", "filename"]
                .iter()
                .find_map(|k| v.get(*k).and_then(|x| x.as_str()))
                .map(str::to_string)
        }
        _ => None,
    }
}

/// Walk the parsed transcript looking for 3+ consecutive tool-call failures
/// targeting the same file. Catches the pattern where a model is writing
/// syntactically-valid-but-semantically-wrong patches on the same file over
/// and over — the exact-signature detector misses this because the args
/// differ on each attempt. Called as a fallback after
/// [`detect_repetition_by_signature`] finds nothing.
fn detect_same_target_failure_repetition(parsed: &ParsedTranscript) -> Option<RepetitionAlert> {
    const THRESHOLD: usize = 3;

    // call_id -> output content + success flag, so we can look up outcomes.
    let mut outputs: BTreeMap<String, (bool, String)> = BTreeMap::new();
    for item in &parsed.items {
        if let TrimItem::ToolOutput {
            call_id,
            success,
            content,
            ..
        } = item
        {
            outputs.insert(call_id.clone(), (*success, content.clone()));
        }
    }

    // Walk calls in reverse. Collect the streak of most-recent failed calls
    // that target the same file via the same tool.
    let mut streak_tool: Option<String> = None;
    let mut streak_path: Option<String> = None;
    let mut streak_call_id: Option<String> = None;
    let mut count = 0usize;
    let mut last_output_excerpt = String::new();

    for item in parsed.items.iter().rev() {
        match item {
            TrimItem::ToolCall {
                tool_name,
                signature,
                args,
                call_id,
                ..
            } => {
                // Try both path extraction strategies: the signature-based
                // one (works for grep/list_dir/text_editor where the path
                // is in the normalized signature) and the args-based one
                // (works for apply_patch where the path is buried inside
                // the `input` field).
                let path = path_from_signature(signature)
                    .filter(|p| !p.is_empty() && *p != "?")
                    .map(str::to_string)
                    .or_else(|| derive_modification(tool_name, args).map(|(p, _)| p));
                let Some(path) = path else {
                    break;
                };
                let Some((success, content)) = outputs.get(call_id) else {
                    break;
                };
                if *success {
                    break;
                }
                let key = (tool_name.clone(), path);
                match (&streak_tool, &streak_path) {
                    (None, None) => {
                        streak_tool = Some(key.0);
                        streak_path = Some(key.1);
                        streak_call_id = Some(call_id.clone());
                        last_output_excerpt = excerpt(content, 200);
                        count = 1;
                    }
                    (Some(t), Some(p)) if *t == key.0 && *p == key.1 => {
                        count += 1;
                    }
                    _ => break,
                }
            }
            // Tool outputs interleave with calls; skip them.
            TrimItem::ToolOutput { .. } => continue,
            // Assistant narration between failed attempts is common — don't
            // break the streak on it, since this repetition mode is about
            // repeated FAILURES on the same target regardless of whether
            // the model narrates its confusion between attempts.
            TrimItem::AssistantText { .. } => continue,
            // Anything else (user message, reasoning, other) breaks the streak.
            _ => break,
        }
    }

    if count < THRESHOLD {
        return None;
    }

    let tool_name = streak_tool?;
    let path = streak_path?;
    Some(RepetitionAlert {
        tool_name,
        command_summary: format!("{count} consecutive failures on {path}"),
        count,
        last_output_excerpt,
        call_id: streak_call_id.unwrap_or_default(),
        // Same file, varying attempts, still failing — a thrash, not a literal
        // repeat. The model needs to read the failure, not just "stop".
        force_diagnosis: true,
        // Args vary each attempt, so there's no single signature to prune; and
        // the model IS editing (making progress attempts), so don't nuke its
        // context. Forced-diagnosis is the right intervention here, not surgery.
        escalate: false,
        signature: String::new(),
    })
}

/// Public helper: return the list of file paths that were created or edited
/// during the active turn. Used by callers (e.g. `local_routing`) to decide
/// which files to read fresh from disk and pass back via
/// [`super::TrimInput::current_files`]. Deleted files are not returned (they
/// don't exist to read).
pub fn files_modified_in_active_turn(items: &[ResponseItem]) -> Vec<String> {
    let parsed = items::parse(items);
    let active = parsed.active_turn_id();
    let mut out: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for item in &parsed.items {
        if let TrimItem::ToolCall {
            tool_name,
            args,
            turn_id,
            ..
        } = item
        {
            if *turn_id != active {
                continue;
            }
            if let Some((path, op)) = derive_modification(tool_name, args) {
                if matches!(op, ModifyOp::Deleted) {
                    continue;
                }
                if seen.insert(path.clone()) {
                    out.push(path);
                }
            }
        }
    }
    out
}

fn derive_modification(tool_name: &str, args_raw: &str) -> Option<(String, ModifyOp)> {
    match tool_name {
        "apply_patch" => {
            let parsed: serde_json::Value = serde_json::from_str(args_raw).ok()?;
            let input = parsed
                .get("input")
                .or_else(|| parsed.get("patch"))
                .and_then(|p| p.as_str())?;
            for line in input.lines() {
                if let Some(rest) = line.strip_prefix("*** Add File: ") {
                    return Some((rest.trim().to_string(), ModifyOp::Created));
                }
                if let Some(rest) = line.strip_prefix("*** Update File: ") {
                    return Some((rest.trim().to_string(), ModifyOp::Edited));
                }
                if let Some(rest) = line.strip_prefix("*** Delete File: ") {
                    return Some((rest.trim().to_string(), ModifyOp::Deleted));
                }
            }
            None
        }
        "text_editor" => {
            let parsed: serde_json::Value = serde_json::from_str(args_raw).ok()?;
            let cmd = parsed.get("command").and_then(|c| c.as_str()).unwrap_or("");
            let path = parsed.get("path").and_then(|p| p.as_str())?.to_string();
            let op = match cmd {
                "create" => ModifyOp::Created,
                "str_replace" | "insert" | "edit" | "write" => ModifyOp::Edited,
                "delete" => ModifyOp::Deleted,
                _ => return None,
            };
            Some((path, op))
        }
        _ => None,
    }
}

fn derive_action_receipt(
    tool_name: &str,
    args_raw: &str,
    output: &str,
    turn_id: u32,
) -> Option<ActionReceipt> {
    let summary = match tool_name {
        "apply_patch" => {
            let (op, path) = match derive_modification("apply_patch", args_raw) {
                Some((p, ModifyOp::Created)) => ("Created", p),
                Some((p, ModifyOp::Edited)) => ("Modified", p),
                Some((p, ModifyOp::Deleted)) => ("Deleted", p),
                None => return None,
            };
            format!("{op} {path}")
        }
        "text_editor" => {
            let (op, path) = match derive_modification("text_editor", args_raw) {
                Some((p, ModifyOp::Created)) => ("Created", p),
                Some((p, ModifyOp::Edited)) => ("Modified", p),
                Some((p, ModifyOp::Deleted)) => ("Deleted", p),
                None => return None,
            };
            format!("{op} {path}")
        }
        "shell" | "shell_command" | "exec_command" | "local_shell" => {
            let parsed: serde_json::Value = serde_json::from_str(args_raw).unwrap_or_default();
            let cmd = parsed
                .get("command")
                .or_else(|| parsed.get("cmd"))
                .or_else(|| parsed.get("argv"))
                .map(stringify_command)
                .unwrap_or_default();
            if cmd.is_empty() {
                return None;
            }
            let exit = shell_exit_code(output).unwrap_or(0);
            format!("Ran `{}` → exit {exit}", short_str(&cmd, 80))
        }
        _ => return None,
    };
    Some(ActionReceipt { turn_id, summary })
}

fn derive_test_run(tool_name: &str, args_raw: &str, output: &str, turn_id: u32) -> Option<TestRun> {
    if !matches!(
        tool_name,
        "shell" | "shell_command" | "exec_command" | "local_shell"
    ) {
        return None;
    }
    let parsed: serde_json::Value = serde_json::from_str(args_raw).ok()?;
    let cmd = parsed
        .get("command")
        .or_else(|| parsed.get("cmd"))
        .or_else(|| parsed.get("argv"))
        .map(stringify_command)
        .unwrap_or_default();
    if !looks_like_test_command(&cmd) {
        return None;
    }
    let exit = shell_exit_code(output).unwrap_or(0);
    let passed = exit == 0;
    let summary = summarize_test_output(output);
    Some(TestRun {
        turn_id,
        command: cmd,
        passed,
        summary,
    })
}

fn looks_like_test_command(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    lower.contains("cargo test")
        || lower.contains("pytest")
        || lower.contains("npm test")
        || lower.contains("npm run test")
        || lower.contains("yarn test")
        || lower.contains("pnpm test")
        || lower.contains("jest")
        || lower.contains("go test")
        || lower.contains("mvn test")
        || lower.contains("gradle test")
}

fn summarize_test_output(output: &str) -> String {
    // Best-effort: find the last "passed/failed" summary line.
    for line in output.lines().rev() {
        let trimmed = line.trim();
        let lower = trimmed.to_lowercase();
        if lower.contains("passed") || lower.contains("failed") {
            return trimmed.to_string();
        }
    }
    "(no summary line found)".to_string()
}

fn is_orchestration_or_async(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "spawn_agent"
            | "spawn_subagent_v2"
            | "send_input"
            | "send_message_v2"
            | "wait"
            | "wait_agent"
            | "exec_command"
            | "write_stdin"
            | "supervisor"
            | "agent_jobs"
    )
}

fn shell_exit_code(output: &str) -> Option<i32> {
    // Codex's structured shell output places exit_code in JSON.
    let parsed: serde_json::Value = serde_json::from_str(output).ok()?;
    parsed
        .get("metadata")
        .and_then(|m| m.get("exit_code"))
        .and_then(|c| c.as_i64())
        .map(|c| c as i32)
        .or_else(|| {
            parsed
                .get("exit_code")
                .and_then(|c| c.as_i64())
                .map(|c| c as i32)
        })
}

fn excerpt(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

fn short_str(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

fn short_args(s: &str, n: usize) -> String {
    let cleaned = s.replace(['\n', '\r'], " ");
    short_str(&cleaned, n)
}

fn stringify_command(v: &serde_json::Value) -> String {
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
