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

/// A single tool OUTPUT is "too large to show whole" once it exceeds this
/// fraction of the trim budget — so the ceiling SCALES with the detected window
/// (a 32K-ctx model tolerates a bigger dump than an 8K one) instead of a
/// hardcoded byte count. The bulk of runaway prompts is one giant dump kept
/// verbatim in the active turn — a `curl … openapi.json | json.tool` (61 KB), a
/// whole-file `cat`, a verbose test log. The estimate counts these, but they also
/// tokenize 2–3× and tip the real prompt over the window. Bounding them at the
/// source is the lever the conversational compactor can't reach.
const MAX_OUTPUT_FRACTION_PCT: usize = 25;
/// Floor for the dynamic ceiling, so a tiny budget still shows a usable output.
const MAX_OUTPUT_FLOOR_CHARS: usize = 6_000;

/// The per-output char ceiling for a given trim budget (`target_ctx`, in chars/4
/// estimate-tokens). Outputs above it are reduced or omitted by `bound_tool_output`.
pub(super) fn max_output_chars(target_ctx: usize) -> usize {
    // target_ctx is estimate-tokens (chars/4); ×4 back to chars.
    (target_ctx * MAX_OUTPUT_FRACTION_PCT / 100 * 4).max(MAX_OUTPUT_FLOOR_CHARS)
}

/// Bound an oversized tool output **without ever handing the model a broken or
/// info-stripped document.** `max_chars` is the dynamic per-output ceiling (a
/// fraction of the detected window, via [`max_output_chars`]). Two outcomes only:
///   1. **Lossless reduce** — JSON minify (valid→valid, nothing removed) or
///      HTML→text / guarded stopword-strip on text. If that gets it under the cap,
///      the model sees the COMPLETE content, just compact.
///   2. **Omit with a pointer** — if it's *still* too big to include in full, we do
///      NOT minify-then-strip-prose (which drops `description`-type fields) and we
///      do NOT keep a head+tail (which is a syntactically broken fragment). We
///      replace it with a short, honest stub telling the model how to fetch the
///      specific part it needs. Dropping > eliding: a clear "it's omitted, here's
///      how to get it" never confuses, a broken partial document can.
/// Returns `None` for outputs already small enough to keep verbatim.
fn bound_tool_output(tool_name: &str, content: &str, max_chars: usize) -> Option<String> {
    if content.len() <= max_chars {
        return None;
    }
    let t = content.trim_start();
    let ct = if t.starts_with('{') || t.starts_with('[') {
        Some("application/json")
    } else if t.starts_with('<') {
        Some("text/html")
    } else {
        None
    };
    // JSON: minify ONLY (reduce_lossless never strips value/description fields).
    // Everything else: content_reduce (HTML→text, guarded stopword-strip) — both
    // keep the actual information; neither leaves a broken structure.
    let reduced = if ct == Some("application/json") {
        crate::content_reduce::reduce_lossless(content, ct)
    } else {
        crate::content_reduce::content_reduce(content, ct, max_chars / 4)
    };
    if reduced.chars().count() <= max_chars {
        return Some(reduced); // complete, just compact
    }
    // Still too big to include without cutting content → omit, don't mangle.
    let kb = content.len() / 1024;
    let how = match tool_name {
        "web_fetch" => {
            "call web_fetch again with find=\"<keyword>\" to jump straight to the part you need"
        }
        "read_file" | "cat_file" => "read_file again with a start_line/end_line range",
        _ => {
            "re-run it more narrowly — pipe through `grep <pattern>`, `sed -n 'A,Bp'`, `head`, or `tail` — to see only the part you need"
        }
    };
    Some(format!(
        "[output omitted: ~{kb} KB, too large to include in full without breaking it. {how}.]"
    ))
}

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

/// When the model's most recent edit failed to apply (an `edit_file` snippet that
/// didn't match — executed via `apply_patch` underneath, which the model never
/// sees), steer it to a whole-file `write_file` rewrite instead of re-trying a
/// snippet against a file whose contents don't match its mental model. Phrased in
/// the model's OWN vocabulary — it called `edit_file`/`write_file`, not
/// `apply_patch`, so the directive never mentions the executor underneath. The
/// authoritative file is pinned just above this directive.
fn render_patch_rewrite_directive(state: &ExtractedState) -> Option<String> {
    let pf = state.patch_failure.as_ref()?;
    Some(format!(
        "[EDIT DID NOT APPLY — REWRITE THE WHOLE FILE]\n\
         Your last edit to `{path}` could not be applied: the text you tried to change is not in the \
         file as written (commonly because an earlier edit never actually landed, so the code you \
         tried to change was never there). Editing the same snippet again will keep failing.\n\
         The file's CURRENT on-disk contents are pinned above — rewrite from those. Output the \
         COMPLETE intended file and save it with `write_file` (path=\"{path}\", content=<the entire \
         file>). A full rewrite overwrites the file, so there is nothing to match.",
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

/// The SINGLE source of the "you're looping" directive. The TRIGGER and the
/// escalation TIER differ — footprint stalled → CIRCLING; a literal repeat → STOP;
/// same-goal thrash → DIAGNOSE; the loop ignored past threshold → EXCISE + reframe
/// — but the message is one thing, not five. `result` is the unchanging output to
/// show once (an excerpt in the prelude, the full last output in the tool-output
/// slot). Both delivery mechanisms ([`render_repetition_alert`] = prelude,
/// [`render_repetition_override`] = the tool-output slot a looping model can't talk
/// past) call this, so they always say the same thing.
fn render_loop_directive(alert: &RepetitionAlert, result: &str) -> String {
    let what = if alert.command_summary.is_empty() {
        alert.tool_name.as_str()
    } else {
        alert.command_summary.as_str()
    };
    let n = alert.count;
    // Footprint stalled — circling a fixed set of targets without expanding (see
    // `detect_tunnel_vision`). Checked FIRST because tunnel-vision also sets
    // `force_diagnosis`; this gives it its own specific, target-naming framing.
    if alert.tunnel_vision {
        let targets = if what.is_empty() {
            "the same places"
        } else {
            what
        };
        return format!(
            "[STUCK — CIRCLING THE SAME PLACES]\n\
             Your last {n} tool calls kept returning to {targets} without touching anything new — you are circling, not progressing. Another edit to the same place will not break the loop.\n\
             Before your NEXT tool call, reply in PLAIN TEXT:\n\
             1. What you are trying to make happen, and the EXACT evidence it isn't (quote the real output — do not assume).\n\
             2. Why it isn't working — and look where you have NOT: the spot you keep returning to is apparently not the cause.\n\
             3. ONE next step that is genuinely DIFFERENT — a new file, or a check that proves the actual state (print the real value / run the failing thing and read it) — NOT another pass over {targets}.\n\n\
             Most recent result:\n{result}"
        );
    }
    if alert.escalate {
        // Tier 3 — advisory + STOP both ignored. The loop's calls have been excised
        // from the messages (see `render_messages`); restate the one result and
        // point at the live state so the model acts instead of copying a loop it
        // can no longer see.
        return format!(
            "[HARNESS — STUCK; LOOP REMOVED FROM CONTEXT]\n\
             You ran `{what}` {n} times and the result never changed, so those repeated calls were removed from this conversation. Here it is once — it will NOT change, do not run it again:\n\n{result}\n\n\
             The current file and latest results are in your context above. Make EXACTLY ONE different change now; do not reproduce the loop you can no longer see."
        );
    }
    if alert.force_diagnosis {
        // Tier 2 — same goal, varying attempts, still failing: guessing without
        // reading the failure. Force a diagnosis, aimed where it helps — the edits
        // aren't moving the result, so the cause is elsewhere; observe, don't deduce.
        return format!(
            "[NO PROGRESS — DIAGNOSE BEFORE YOUR NEXT ACTION]\n\
             You've tried `{what}` {n} times and the result keeps coming back the same — your changes are NOT moving it, which means the cause is NOT where you keep editing.\n\
             Before you call ANY tool, reply in PLAIN TEXT with:\n\
             1. The exact error or failing line, quoted from the result below.\n\
             2. Its single root cause — and look where you HAVEN'T: the error names the symptom, not the cause.\n\
             3. The ONE change that fixes that cause, and how it differs from what you already tried.\n\
             If you can't tell why it fails, stop guessing — print the actual value and type right before the point that fails, run it once, and read what you really get.\n\n\
             The result you keep getting:\n{result}"
        );
    }
    // Tier 1 — a literal repeat.
    format!(
        "[STOP — REPETITION DETECTED]\n\
         You've called `{what}` {n} times in a row the same way and gotten the same result every time. Calling it again is a no-op — a wasted turn.\n\
         Change approach now: different arguments, a different tool, or stop and tell the user what you've learned and what's blocking you.\n\n\
         The unchanged result:\n{result}"
    )
}

fn render_repetition_alert(state: &ExtractedState) -> Option<String> {
    let alert = state.repetition.as_ref()?;
    Some(render_loop_directive(alert, &alert.last_output_excerpt))
}

/// Synthesized tool-output for the most-recent call in a detected repetition loop,
/// planted in the slot the model actually attends to — a looping local model talks
/// past the prelude but not the result of its own last call. Same directive as the
/// prelude (via [`render_loop_directive`]); the difference is placement and that it
/// shows the FULL last output, not the excerpt. (Escalated loops are excised before
/// this runs, so only the STOP / DIAGNOSE tiers reach here.)
fn render_repetition_override(alert: &RepetitionAlert, content: &str) -> String {
    render_loop_directive(alert, content)
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
    max_chars: usize,
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
    // When a loop is excised (context reset), collapse it into ONE inline marker
    // the first time we hit it rather than deleting into a gap — see the prune below.
    let mut loop_collapsed = false;
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
                    // Don't delete into a gap. Collapse the whole excised loop into
                    // ONE coherent inline marker the first time we hit it, so the
                    // transcript still reads "I tried this N times, it didn't work"
                    // — instead of leaving the surviving interleaved calls (e.g. the
                    // test runs) with no record of the edits between them, which a
                    // small model reads as "I haven't acted yet" and repeats. The
                    // rest of the loop's calls + outputs are dropped silently.
                    if !loop_collapsed {
                        loop_collapsed = true;
                        if let Some(alert) = repetition {
                            messages.push(serde_json::json!({
                                "role": "user",
                                "content": format!(
                                    "[loop collapsed — {} repeated `{}` attempts here were removed to break a loop; each returned the same result, shown once. Do NOT repeat them; make a different change.]\n{}",
                                    alert.count, alert.tool_name, alert.last_output_excerpt
                                ),
                            }));
                        }
                    }
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
                // Bound an oversized output so one 60 KB dump can't dominate the
                // prompt and tip it over the window (the dominant overflow cause).
                let bounded = if repetition_override.is_none() {
                    bound_tool_output(tool_name, content, max_chars)
                } else {
                    None
                };
                let content: &str = bounded.as_deref().unwrap_or(content);
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
        return cleaned;
    }
    // Char-boundary-safe: a raw `&cleaned[..n]` panics if `n` splits a multibyte
    // char (tool outputs are arbitrary UTF-8 — e.g. a `·` in web-search results).
    let mut end = n;
    while end > 0 && !cleaned.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &cleaned[..end])
}

/// Produce a follow-up hint for a failed tool output, matched on tool name
/// and error content. Returned hint is appended after the `<tool_error>`
/// block so the model sees a clear next step. Empty string means no hint.
fn tool_failure_hint(tool_name: &str, content: &str) -> String {
    let prefix = "\n\n→ Hint: ";
    match tool_name {
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
