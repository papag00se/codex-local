//! Block renderer.
//!
//! Composes the trimmer's two outputs:
//!   - The synthesized prelude (persistent context + world state + open issues
//!     + in-flight + tests). Concatenated onto the system prompt by `mod.rs`.
//!   - The Ollama-format chat messages: per-turn collapsed summaries for older
//!     turns, then verbatim items for the active turn.
//!
//! Pure formatting — no decisions about what to keep.

use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::collections::HashMap;

use super::items::ParsedTranscript;
use super::items::TrimItem;
use super::rules::CompressedOlder;
use super::state_extract::ExtractedState;
use super::state_extract::ModifyOp;
use super::state_extract::RepetitionAlert;

/// Cap applied to each injected current-file block. Large files get
/// truncated with a notice so a single edit doesn't blow the whole context
/// budget. 10 KB is enough for any normal source file.
const CURRENT_FILE_MAX_BYTES: usize = 10_240;

/// Render the full prelude block. Empty if there's nothing to say (e.g. an
/// empty transcript with no user instructions). The `active_turn` is used to
/// compute "turns since last modification" hints in the world-state block.
/// `current_files` (if provided) supplies fresh disk content for files that
/// were modified in the active turn so the model can't work from a stale
/// mental model after its own edits land.
pub fn render_prelude(
    user_instructions: Option<&str>,
    state: &ExtractedState,
    active_turn: u32,
    current_files: Option<&HashMap<String, String>>,
) -> String {
    let mut sections: Vec<String> = Vec::new();

    // Repetition alert goes FIRST, before everything else, so the model can't
    // miss it. Local models otherwise get stuck calling the same tool with
    // the same args 5+ times in a row, ignoring identical outputs.
    if let Some(alert) = render_repetition_alert(state) {
        sections.push(alert);
    }

    if let Some(inst) = user_instructions
        && !inst.trim().is_empty()
    {
        sections.push(format!("[Persistent project context]\n{}", inst.trim()));
    }

    // Pin current on-disk contents of any files the active turn has edited.
    // This block is right under the repetition alert because when a model
    // is looping on stale patches, the authoritative remedy is "here is
    // what the file ACTUALLY says right now."
    if let Some(block) = render_current_files(state, current_files, active_turn) {
        sections.push(block);
    }

    // If the model's last apply_patch failed, steer it to a full write_file
    // rewrite right after the pinned file contents it should rewrite from.
    if let Some(directive) = render_patch_rewrite_directive(state) {
        sections.push(directive);
    }

    let world = render_world_state(state, active_turn);
    if !world.is_empty() {
        sections.push(world);
    }

    let actions = render_actions(state);
    if !actions.is_empty() {
        sections.push(actions);
    }

    let errors = render_errors(state);
    if !errors.is_empty() {
        sections.push(errors);
    }

    let in_flight = render_in_flight(state);
    if !in_flight.is_empty() {
        sections.push(in_flight);
    }

    let tests = render_tests(state);
    if !tests.is_empty() {
        sections.push(tests);
    }

    sections.join("\n\n")
}

/// When the model's most recent `apply_patch` failed, steer it to a whole-file
/// `write_file` rewrite instead of re-patching a file whose contents don't match
/// its mental model. The authoritative file is pinned just above this directive.
fn render_patch_rewrite_directive(state: &ExtractedState) -> Option<String> {
    let pf = state.patch_failure.as_ref()?;
    Some(format!(
        "[PATCH DID NOT APPLY — REWRITE THE FILE]\n\
         Your last `apply_patch` to `{path}` FAILED: its target lines are not in the file as \
         written (commonly because an earlier edit never actually landed, so the code you tried to \
         change was never created). Re-running apply_patch will keep failing the same way.\n\
         The file's CURRENT on-disk contents are pinned above — rewrite from those. Output the \
         COMPLETE intended file and save it with `write_file` (path=\"{path}\", content=<the entire \
         file>). A full rewrite overwrites the file, so there is nothing to match. Do NOT call \
         apply_patch on `{path}` again.",
        path = pf.path,
    ))
}

/// Render a `[Current file state]` block listing the verbatim on-disk
/// contents of every file the active turn has modified (Add File / Update
/// File, but not Delete). Returns `None` when there's nothing to inject.
fn render_current_files(
    state: &ExtractedState,
    current_files: Option<&HashMap<String, String>>,
    active_turn: u32,
) -> Option<String> {
    let current_files = current_files?;
    let mut entries: Vec<String> = Vec::new();
    for (path, modified) in &state.files_modified {
        if modified.turn_id != active_turn {
            continue;
        }
        if matches!(modified.op, ModifyOp::Deleted) {
            continue;
        }
        let Some(content) = current_files.get(path) else {
            continue;
        };
        let total = content.len();
        let (body, truncated) = if total > CURRENT_FILE_MAX_BYTES {
            let mut cut = CURRENT_FILE_MAX_BYTES;
            while cut < total && !content.is_char_boundary(cut) {
                cut += 1;
            }
            (&content[..cut], true)
        } else {
            (content.as_str(), false)
        };
        let hash = short_hash(content);
        let line_count = content.lines().count();
        let mut header = format!(
            "--- Current content of {path} (hash {hash}, {line_count} lines, {total} bytes)"
        );
        if truncated {
            header.push_str(" — TRUNCATED to first ");
            header.push_str(&CURRENT_FILE_MAX_BYTES.to_string());
            header.push_str(" bytes below");
        }
        entries.push(format!("{header}\n{body}\n--- End of {path}"));
    }
    if entries.is_empty() {
        return None;
    }
    Some(format!(
        "[Current file state — authoritative. Work from this content, not from memory of earlier patches.]\n{}",
        entries.join("\n\n")
    ))
}

/// Tiny non-cryptographic hash for displaying content identity. Only needs
/// to be stable within one render; collision resistance is not required.
fn short_hash(s: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    let raw = hasher.finish();
    // Take the first 8 hex chars for a compact, readable ID.
    format!("{raw:016x}")[..8].to_string()
}

fn render_repetition_alert(state: &ExtractedState) -> Option<String> {
    let alert = state.repetition.as_ref()?;
    if alert.escalate {
        // Top tier: the advisory nudge and the hard block were both ignored.
        // The loop's own turns have been excised from the messages (see
        // `render_messages`); restate its single, unchanging result once and
        // point the model at the live state (current file + last test result,
        // rendered elsewhere in this prelude) so it acts instead of copying the
        // loop it can no longer see.
        return Some(format!(
            "[HARNESS — STUCK; LOOP REMOVED FROM CONTEXT]\n\
             You ran `{}` {} times and the result never changed, so those repeated calls were REMOVED from this conversation. Here is that result once — it will NOT change, do not run it again:\n{}\n\n\
             The current file contents and latest test results are in your context above. Use them: make EXACTLY ONE edit to fix the failing test, then run the test once. Do not repeat any command you have already run.",
            alert.command_summary, alert.count, alert.last_output_excerpt
        ));
    }
    if alert.force_diagnosis {
        // Thrash: same goal, varying commands, still failing. The model is
        // guessing edits without reading the failure. Force it to diagnose
        // before it's allowed to act again — this is the agent coaching the
        // model through a problem it keeps bouncing off.
        return Some(format!(
            "[NO PROGRESS — DIAGNOSE BEFORE YOUR NEXT ACTION]\n\
             You have attempted `{}` {} times now and the result keeps coming back the same. \
             Your changes are NOT fixing it — you are guessing instead of reading the failure.\n\
             Before you call ANY tool, reply in PLAIN TEXT with exactly these three things:\n\
             1. Quote the exact error or failing line from the output below.\n\
             2. State the single root cause of that specific error.\n\
             3. State the ONE concrete change that fixes that root cause — and how it differs from what you already tried.\n\n\
             Most recent result:\n{}\n\n\
             Do not repeat a variation of an edit you have already made. Diagnose first, then make exactly one targeted change.",
            alert.command_summary, alert.count, alert.last_output_excerpt
        ));
    }
    Some(format!(
        "[STOP — REPETITION DETECTED]\n\
         You have called `{}` with identical arguments {} times in a row. The result will not change. STOP making this call.\n\
         Last call: {}\n\
         Last output excerpt: {}\n\
         You MUST try a different approach now: change the arguments, use a different tool, or report what you've learned to the user. Repeating the same call is a wasted turn.",
        alert.tool_name, alert.count, alert.command_summary, alert.last_output_excerpt
    ))
}

/// Synthesized tool-output for the most-recent call in a detected repetition
/// loop. Replaces the real (identical) result so the model reads, in the
/// tool-output slot it actually attends to, an unambiguous "stop repeating"
/// signal. The prelude alert ([`render_repetition_alert`]) stays as a
/// reinforcing nudge, but this is the part a looping local model can't talk
/// past — it's the result of its own last call.
fn render_repetition_override(alert: &RepetitionAlert, content: &str) -> String {
    if alert.force_diagnosis {
        // Thrash override: planted in the tool-output slot the model actually
        // attends to, demanding a diagnosis before its next move. This is the
        // part a guessing local model can't talk past — it's the result of its
        // own last call.
        return format!(
            "[HARNESS: NO PROGRESS — DIAGNOSE BEFORE ACTING]\n\
             You have attempted this {} times and the outcome has not changed. More guesses will not help.\n\
             Your NEXT reply must be PLAIN TEXT (no tool call) containing:\n\
             1. The exact error/failing line, quoted from the result below.\n\
             2. The single root cause.\n\
             3. The one specific change that fixes it, and why it's different from what you already tried.\n\n\
             The result you keep getting:\n{}",
            alert.count, content
        );
    }
    format!(
        "[REPEATED CALL BLOCKED BY HARNESS]\n\
         You have now called `{}` {} times in a row in the exact same way and gotten the exact same result every time ({}). \
         Calling it again will produce the identical result — this is a no-op loop and a wasted turn.\n\n\
         You MUST change approach now: use different arguments, switch to a different tool, or stop and report to the user what you've learned and what is blocking you. \
         Do NOT issue this same `{}` call again.\n\n\
         The identical result you already received (unchanged):\n{}",
        alert.tool_name, alert.count, alert.command_summary, alert.tool_name, content
    )
}

fn render_world_state(state: &ExtractedState, active_turn: u32) -> String {
    if state.files_seen.is_empty() && state.files_modified.is_empty() {
        return String::new();
    }
    let mut lines = vec!["[World state]".to_string()];
    if !state.files_seen.is_empty() {
        lines.push(format!(
            "Files seen: {}",
            state
                .files_seen
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let mut any_stale = false;
    if !state.files_modified.is_empty() {
        let mut by_op: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        for (path, m) in &state.files_modified {
            let label = match m.op {
                ModifyOp::Created => "Created",
                ModifyOp::Edited => "Edited",
                ModifyOp::Deleted => "Deleted",
            };
            // Show modification turn so the model can judge freshness.
            // Anything older than 2 turns from the active turn is likely
            // stale in the model's working memory and should be re-read
            // before further edits.
            let turns_since = active_turn.saturating_sub(m.turn_id);
            let entry = if turns_since >= 2 {
                any_stale = true;
                format!(
                    "{path} (turn {}, {} turns ago — content likely stale)",
                    m.turn_id, turns_since
                )
            } else {
                format!("{path} (turn {})", m.turn_id)
            };
            by_op.entry(label).or_default().push(entry);
        }
        for (label, paths) in by_op {
            lines.push(format!("{label}: {}", paths.join(", ")));
        }
    }
    if any_stale {
        lines.push(
            "NOTE: Some files were edited multiple turns ago. Before patching them again, re-read with `cat <path>` (or `apply_patch` will likely fail with 'Failed to find context')."
                .to_string(),
        );
    }
    lines.join("\n")
}

fn render_actions(state: &ExtractedState) -> String {
    if state.actions.is_empty() {
        return String::new();
    }
    let mut lines = vec!["[Actions taken]".to_string()];
    for a in &state.actions {
        lines.push(format!("- (turn {}) {}", a.turn_id, a.summary));
    }
    lines.join("\n")
}

fn render_errors(state: &ExtractedState) -> String {
    if state.unresolved_errors.is_empty() {
        return String::new();
    }
    let mut lines = vec!["[UNRESOLVED ERRORS]".to_string()];
    for e in &state.unresolved_errors {
        lines.push(format!(
            "- (turn {}) {}: {}",
            e.turn_id, e.tool_name, e.excerpt
        ));
    }
    lines.join("\n")
}

fn render_in_flight(state: &ExtractedState) -> String {
    if state.in_flight.is_empty() {
        return String::new();
    }
    let mut lines = vec!["[In-flight]".to_string()];
    for f in &state.in_flight {
        lines.push(format!(
            "- (turn {}) {} call_id={} args={}",
            f.turn_id, f.tool_name, f.call_id, f.note
        ));
    }
    lines.join("\n")
}

fn render_tests(state: &ExtractedState) -> String {
    if state.test_runs.is_empty() {
        return String::new();
    }
    let mut lines = vec!["[Tests]".to_string()];
    for t in &state.test_runs {
        let verdict = if t.passed { "PASS" } else { "FAIL" };
        lines.push(format!(
            "- (turn {}) {} `{}` → {}",
            t.turn_id, verdict, t.command, t.summary
        ));
    }
    lines.join("\n")
}

/// Build the chat messages and report how many of the leading messages
/// represent older (collapsed) turns. Active-turn messages occupy
/// `messages[older_turn_message_count..]`. Callers that need to summarize
/// just the older portion (e.g. when even the trimmed transcript exceeds
/// the local model's context budget) can use the count to slice cleanly.
pub fn render_messages(
    older: &CompressedOlder,
    parsed: &ParsedTranscript,
    active_turn: u32,
    flavor: crate::config::ClientFlavor,
    repetition: Option<&RepetitionAlert>,
) -> (Vec<JsonValue>, usize) {
    let mut messages: Vec<JsonValue> = Vec::new();

    // Older turns: render a single user-message-shaped item per turn that
    // contains the verbatim user message + a one-line action summary (already
    // handled by the prelude's [Actions taken] block, so the per-turn message
    // here just preserves the user's words and the call signatures from that
    // turn that survived compression).
    let mut older_by_turn: BTreeMap<u32, Vec<&TrimItem>> = BTreeMap::new();
    for item in &older.items {
        older_by_turn.entry(item.turn_id()).or_default().push(item);
    }
    for (turn, turn_items) in older_by_turn {
        let mut user_text = String::new();
        let mut tool_lines: Vec<String> = Vec::new();
        for item in turn_items {
            match item {
                TrimItem::User { text, .. } => {
                    if !user_text.is_empty() {
                        user_text.push('\n');
                    }
                    user_text.push_str(text);
                }
                TrimItem::ToolCall {
                    tool_name,
                    args,
                    signature,
                    ..
                } => {
                    let _ = signature;
                    tool_lines.push(format!("  - called {tool_name}({})", short(args, 80)));
                }
                TrimItem::ToolOutput {
                    tool_name,
                    success,
                    content,
                    ..
                } => {
                    if !*success {
                        tool_lines.push(format!("  - {tool_name} ERROR: {}", short(content, 200)));
                    } else {
                        // Read-shaped tools (grep, list_dir, text_editor view)
                        // survived older-turn compression — include the data
                        // for the model to reference. Action-only tools were
                        // already dropped by `rules::compress_older_turns`.
                        tool_lines.push(format!("  - {tool_name} output:"));
                        for line in content.lines() {
                            tool_lines.push(format!("    {line}"));
                        }
                    }
                }
                _ => {}
            }
        }
        let mut content = format!("[turn {turn} — user]\n{user_text}");
        if !tool_lines.is_empty() {
            content.push_str("\n[turn ");
            content.push_str(&turn.to_string());
            content.push_str(" — surviving tool activity]\n");
            content.push_str(&tool_lines.join("\n"));
        }
        messages.push(serde_json::json!({
            "role": "user",
            "content": content,
        }));
    }

    let older_turn_message_count = messages.len();

    // Context reset: once a repeat-loop has escalated (advisory nudge + hard
    // block both ignored), excise the loop's own calls + outputs from the
    // active turn so the model stops copying the pattern out of its own
    // saturated context. The prelude reframe restates the loop's single result
    // once. Only signature-based loops are pruned (we can match those exactly).
    let pruned_loop: std::collections::HashSet<&str> = match repetition {
        Some(alert) if alert.escalate && !alert.signature.is_empty() => parsed
            .items
            .iter()
            .filter_map(|item| match item {
                TrimItem::ToolCall {
                    signature,
                    call_id,
                    turn_id,
                    ..
                } if *turn_id == active_turn && *signature == alert.signature => {
                    Some(call_id.as_str())
                }
                _ => None,
            })
            .collect(),
        _ => std::collections::HashSet::new(),
    };

    // Active turn: pass through verbatim, preserving the original
    // role/structure as best Ollama can represent it.
    for item in &parsed.items {
        if item.turn_id() != active_turn {
            continue;
        }
        match item {
            TrimItem::User { text, .. } => {
                messages.push(serde_json::json!({"role": "user", "content": text}));
            }
            TrimItem::AssistantText { text, .. } => {
                messages.push(serde_json::json!({"role": "assistant", "content": text}));
            }
            TrimItem::Reasoning { text, .. } => {
                // Ollama doesn't have a dedicated reasoning role; tag it inline
                // so the model knows it's its own prior thinking.
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": format!("<reasoning>{text}</reasoning>"),
                }));
            }
            TrimItem::ToolCall {
                tool_name,
                args,
                call_id,
                ..
            } => {
                if pruned_loop.contains(call_id.as_str()) {
                    continue; // loop call excised by context reset
                }
                // Wire format for `arguments` differs by flavor:
                //   - Ollama: JSON OBJECT (sending a string triggers parser
                //     errors). We parse our stored args as JSON and embed
                //     verbatim, defaulting to `{}` on parse failure.
                //   - OpenAI-compat: JSON STRING (the OpenAI spec requires
                //     this; LM Studio rejects object-form with
                //     "Invalid 'messages' in payload"). Pass the raw args
                //     string through; if the model emitted invalid JSON
                //     for some reason, send "{}" so the wire payload still
                //     parses.
                let args_value = match flavor {
                    crate::config::ClientFlavor::Ollama => {
                        serde_json::from_str(args).unwrap_or_else(|_| serde_json::json!({}))
                    }
                    crate::config::ClientFlavor::OpenAICompat => {
                        let validated = if serde_json::from_str::<serde_json::Value>(args).is_ok() {
                            args.to_string()
                        } else {
                            "{}".to_string()
                        };
                        serde_json::Value::String(validated)
                    }
                };
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": args_value,
                        }
                    }]
                }));
            }
            TrimItem::ToolOutput {
                content,
                call_id,
                tool_name,
                success,
                ..
            } => {
                if pruned_loop.contains(call_id.as_str()) {
                    continue; // output of an excised loop call
                }
                // Render tool output as a `user`-role message wrapped in a
                // `<tool_result>` block (or `<tool_error>` if the call failed)
                // instead of the OpenAI-native `{role: "tool", ...}` form.
                // The `role: "tool"` shape relies on the model's chat template
                // rendering it back into prompt context — many local model
                // templates either skip it or render it with a marker the
                // model wasn't trained to attend to. A user-role wrapper is
                // universally rendered.
                //
                // Distinguishing `<tool_error>` from `<tool_result>` gives
                // the model an obvious visual signal that the previous call
                // failed and needs to be retried with a different approach.
                //
                // For specific error patterns we recognize, append a hint that
                // points the model toward the right next action. This is
                // important for local models that don't always parse error
                // messages closely enough to figure out the recovery on their
                // own.
                // Hard repetition stop: if this is the most-recent call in a
                // detected repeat-loop, replace its output with a synthesized
                // result that explicitly tells the model it has repeated this
                // exact call and gotten the identical result. Lands in the
                // tool-output slot the model is trained to read — far harder to
                // ignore than the prelude alert (which a stuck local model
                // routinely talks right past).
                let repetition_override = repetition
                    .filter(|alert| !alert.call_id.is_empty() && alert.call_id == *call_id)
                    .map(|alert| render_repetition_override(alert, content));
                let success = *success && repetition_override.is_none();
                let body: &str = repetition_override.as_deref().unwrap_or(content);
                let tag = if success { "tool_result" } else { "tool_error" };
                let hint = if !success && repetition_override.is_none() {
                    tool_failure_hint(tool_name, body)
                } else {
                    String::new()
                };
                match flavor {
                    crate::config::ClientFlavor::OpenAICompat => {
                        // OpenAI-compat servers (LM Studio, vLLM, llama.cpp)
                        // strictly require an `assistant{tool_calls}` to be
                        // followed by `role: tool` with a matching
                        // `tool_call_id`. Anything else (e.g. our `<tool_result>`
                        // user-message wrapper) is rejected as
                        // "Invalid 'messages' in payload."
                        let mut content_str = if success {
                            body.to_string()
                        } else {
                            format!("<{tag} tool=\"{tool_name}\">\n{body}\n</{tag}>")
                        };
                        if !hint.is_empty() {
                            content_str.push_str(&hint);
                        }
                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": content_str,
                        }));
                    }
                    crate::config::ClientFlavor::Ollama => {
                        // Ollama is lenient — wrap in user-role <tool_result>
                        // so the chat template renders it as visible context.
                        // The role:tool form often gets stripped by local
                        // model templates that weren't trained on it.
                        messages.push(serde_json::json!({
                            "role": "user",
                            "content": format!(
                                "<{tag} tool=\"{tool_name}\" call_id=\"{call_id}\">\n{body}\n</{tag}>{hint}"
                            ),
                        }));
                    }
                }
            }
            TrimItem::Other { .. } => {
                // Skip unknown types in the active turn rather than risk
                // sending Ollama-incompatible JSON.
            }
        }
    }

    (messages, older_turn_message_count)
}

fn short(s: &str, n: usize) -> String {
    let cleaned = s.replace(['\n', '\r'], " ");
    if cleaned.len() <= n {
        cleaned
    } else {
        format!("{}…", &cleaned[..n])
    }
}

/// Produce a follow-up hint for a failed tool output, matched on tool name
/// and error content. Returned hint is appended after the `<tool_error>`
/// block so the model sees a clear next step. Empty string means no hint.
fn tool_failure_hint(tool_name: &str, content: &str) -> String {
    let prefix = "\n\n→ Hint: ";
    match tool_name {
        "apply_patch" => {
            if content.contains("Failed to find context")
                || content.contains("Failed to find expected lines")
            {
                format!(
                    "{prefix}The patch's target lines aren't in the file as written — often because an earlier edit never actually landed. Re-patching will keep failing. The file's current contents are pinned above; rewrite the WHOLE file with `write_file` instead."
                )
            } else if content.contains("first line of the patch must be '*** Begin Patch'") {
                format!(
                    "{prefix}Add `*** Begin Patch` as the very first line of the `input` string."
                )
            } else if content.contains("last line of the patch must be '*** End Patch'") {
                format!("{prefix}Add `*** End Patch` as the very last line of the `input` string.")
            } else if content.contains("not a valid hunk header") {
                format!(
                    "{prefix}Hunk content lines must be prefixed with `+` (additions), `-` (deletions), or ` ` (context). Headers are `*** Add File: <path>`, `*** Update File: <path>`, `*** Delete File: <path>`, or `@@ ... @@`."
                )
            } else {
                String::new()
            }
        }
        "shell" | "exec_command" | "shell_command" | "local_shell" => {
            if content.contains("regex parse error")
                || content.contains("repetition operator missing expression")
            {
                format!(
                    "{prefix}`rg` interpreted your argument as a regex with invalid syntax. For file globbing use `rg --files -g '<glob>'` (e.g. `-g '*.ts'`), repeated for each glob."
                )
            } else if content.contains("command not found") {
                format!(
                    "{prefix}The command isn't installed or isn't on PATH in this sandbox. Try `which <command>` first, or use a different tool that's available."
                )
            } else if content.contains("Permission denied") || content.contains("EACCES") {
                format!(
                    "{prefix}The sandbox blocked this. Use `request_permissions` first to escalate, then retry the command."
                )
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}
