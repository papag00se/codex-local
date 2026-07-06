//! Deterministic loop / patch-failure detector.
//!
//! Walks the parsed transcript and derives the two prelude signals the local
//! model still sees:
//!   - A repetition/loop alert (exact-repeat, same-target failure streak,
//!     unproductive recurrence, tunnel-vision, or read-without-write).
//!   - Whether the most recent `apply_patch` failed (drives the whole-file
//!     rewrite directive).
//!
//! (World-state / actions / errors / in-flight / test blocks were removed — the
//! LLM compaction summary and the verbatim transcript carry that context now.)
//!
//! Pure data extraction — no LLM calls, no judgment.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use codex_protocol::models::ResponseItem;

use super::items;
use super::items::ParsedTranscript;
use super::items::TrimItem;

#[derive(Debug, Clone, Default)]
pub struct ExtractedState {
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
}

/// Which specialised loop a [`RepetitionAlert`] represents, so the renderer can
/// emit a *targeted* directive instead of the generic STOP. `Generic` covers the
/// exact-repeat / thrash / tunnel-vision cases whose framing is already selected
/// by the `force_diagnosis` / `escalate` / `tunnel_vision` flags; the read-mode
/// variants carry their own message. (Same-prefix-search and failing-fetch are
/// handled at the tool layer — see `local_web_search::format_results` and
/// `web_fetch::append_guess_hint` — so they are deliberately NOT variants here.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopKind {
    #[default]
    Generic,
    /// N read-type ops (search / fetch / read / list / grep) with ZERO writes
    /// among them — researching without ever acting. The one loop that is
    /// inherently cross-tool, so it can't live in any single tool handler.
    /// See [`detect_read_without_write`].
    ReadWithoutWrite,
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
    /// The model's *footprint stopped expanding*: several tool calls in a row that
    /// only revisit already-seen well-defined targets (files/urls/queries) without
    /// touching anything new — a structural, content-blind stuck signal, distinct
    /// from an exact repeat or a same-target failure streak. The renderer turns it
    /// into a forceful "you're circling these — do ONE genuinely different thing"
    /// directive. See [`detect_tunnel_vision`].
    pub tunnel_vision: bool,
    /// Which specialised loop this is, selecting a targeted directive in the
    /// renderer. `Generic` for the exact-repeat / thrash / tunnel-vision alerts.
    pub kind: LoopKind,
}

/// Count at which a repeated loop escalates from advisory nudge + hard block to
/// context surgery (excise the loop + reframe). Past this, the model has ignored
/// the gentler interventions, so we change what it *sees* rather than what we
/// *tell* it. Kept low (one advisory round at the 3× detection, then excise): a
/// weak local model routinely ignores the "stop" nudge, and leaving the repeated
/// calls in its context lets it copy the pattern back out — so we clean the
/// context sooner rather than letting the loop bloat to 6+ identical copies. The
/// excision is footgun-safe: the loop collapses to ONE marker that keeps the last
/// result and any error, so the model still knows "I tried this, it didn't work".
const ESCALATION_THRESHOLD: usize = 4;

#[derive(Debug, Clone, Copy)]
pub enum ModifyOp {
    Created,
    Edited,
    Deleted,
}

/// Derive the two prelude signals (loop alert + patch-failure) from the whole
/// transcript. Every detector re-walks `parsed` directly, so there is no shared
/// materialized state to build up first.
pub fn extract(parsed: &ParsedTranscript, _active_turn: u32) -> ExtractedState {
    let mut state = ExtractedState::default();

    // Repetition kinds, tried in order: exact-signature (byte-identical args),
    // then unproductive recurrence (a signature whose FAILING occurrences recur
    // within a recent window — the productivity-gated thrash detector), then the
    // structural stuck signals (tunnel-vision, read-without-write).
    state.patch_failure = detect_patch_failure(parsed);
    state.repetition = detect_repetition(parsed)
        .or_else(|| detect_unproductive_recurrence(parsed))
        // Broadest, last: structural "footprint stopped expanding" — catches
        // circling a fixed set of files/urls even when individual calls neither
        // repeat byte-for-byte nor record an explicit failure.
        .or_else(|| detect_tunnel_vision(parsed))
        // Backstop after tunnel-vision: the model keeps READING (varied targets,
        // so tunnel-vision's footprint never stalls) but never WRITES. Catches
        // "research forever, act never"; resets only on a write.
        .or_else(|| detect_read_without_write(parsed));
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
        tunnel_vision: false,
        kind: LoopKind::Generic,
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
        tunnel_vision: false,
        kind: LoopKind::Generic,
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
            Some((false, _content)) => {
                if tool_name == "apply_patch" {
                    return Some(PatchFailure { path });
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

/// The well-defined target a tool call acts on, for the footprint detector — a
/// file path, a url, or a search query. Returns `None` for tools whose target we
/// cannot classify without guessing (notably `shell`/`exec_command`): those calls
/// are ABSTAINED — counted as neither progress nor non-progress — because reading
/// an ad-hoc command line for "what it touched" is exactly the flaky guess we
/// refuse to make. Only clean, unambiguous targets move the tunnel-vision counter.
fn well_defined_target(tool_name: &str, args_raw: &str) -> Option<String> {
    if let Some(path) = edit_target_path(tool_name, args_raw) {
        return Some(format!("file:{path}"));
    }
    let v: serde_json::Value = serde_json::from_str(args_raw).ok()?;
    match tool_name {
        "read_file" | "cat_file" => v
            .get("path")
            .and_then(|x| x.as_str())
            .map(|p| format!("file:{p}")),
        "web_fetch" => v
            .get("url")
            .and_then(|x| x.as_str())
            .map(|u| format!("url:{u}")),
        "web_search" | "local_web_search" => v
            .get("query")
            .or_else(|| v.get("q"))
            .and_then(|x| x.as_str())
            .map(|q| format!("search:{q}")),
        _ => None,
    }
}

/// Tunnel-vision / "footprint stopped expanding" detector — the broadest stuck
/// signal, run only after the exact-repeat and same-target-failure detectors find
/// nothing. Structural and content-blind: it watches whether the model keeps
/// reaching NEW well-defined targets (healthy) or churns within a fixed set
/// (stuck), without reading the code or the outcomes.
///
/// The counter is "verified revisits since the last new target": walking the
/// active turn forward, a NEW well-defined target resets it to 0 (the footprint
/// expanded = progress), an already-seen one increments it (circling), and a call
/// with no classifiable target (`shell`, …) is ABSTAINED. Fires at `THRESHOLD`.
///
/// A low threshold is safe BECAUSE of the reset: progressing work — even tightly
/// focused work — keeps touching new targets and never accumulates; only
/// non-expanding churn climbs. And firing is cheap and self-pacing: the signal is
/// recomputed from the transcript each turn, so the moment the model touches
/// anything new it clears (no persistent state to leak, no treadmill). `escalate`
/// stays false — context surgery / "is it hopeless" is a separate backstop's job.
fn detect_tunnel_vision(parsed: &ParsedTranscript) -> Option<RepetitionAlert> {
    const THRESHOLD: usize = 8;

    // Active turn only: collect tool calls back to the last user message (a new
    // task is a fresh footprint), then process them in order.
    let mut calls: Vec<(&str, &str, &str)> = Vec::new();
    for item in parsed.items.iter().rev() {
        match item {
            TrimItem::ToolCall {
                tool_name,
                args,
                call_id,
                ..
            } => calls.push((tool_name, args, call_id)),
            TrimItem::ToolOutput { .. }
            | TrimItem::AssistantText { .. }
            | TrimItem::Reasoning { .. } => continue,
            _ => break, // user message → task boundary
        }
    }
    calls.reverse();

    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut streak = 0usize;
    let mut cycled: Vec<String> = Vec::new();
    let mut last_call_id = String::new();
    for (tool_name, args, call_id) in &calls {
        let Some(target) = well_defined_target(tool_name, args) else {
            continue; // abstain on shell / unclassifiable
        };
        if seen.insert(target.clone()) {
            // Footprint expanded → progress → reset the streak.
            streak = 0;
            cycled.clear();
        } else {
            streak += 1;
            if !cycled.contains(&target) {
                cycled.push(target.clone());
            }
            last_call_id = (*call_id).to_string();
        }
    }

    if streak < THRESHOLD {
        return None;
    }

    // Display targets without the internal `file:`/`url:`/`search:` tag.
    let targets = cycled
        .iter()
        .map(|t| t.splitn(2, ':').nth(1).unwrap_or(t.as_str()))
        .collect::<Vec<_>>()
        .join(", ");

    let last_output_excerpt = parsed
        .items
        .iter()
        .rev()
        .find_map(|item| match item {
            TrimItem::ToolOutput {
                call_id: cid,
                content,
                ..
            } if *cid == last_call_id => Some(excerpt(content, 300)),
            _ => None,
        })
        .unwrap_or_default();

    Some(RepetitionAlert {
        tool_name: String::new(),
        command_summary: targets,
        count: streak,
        last_output_excerpt,
        call_id: last_call_id,
        force_diagnosis: true,
        escalate: false,
        signature: String::new(),
        tunnel_vision: true,
        kind: LoopKind::Generic,
    })
}

/// True for read-type tools — ops that only GATHER information and never change
/// the workspace. The complement (writes) is anything [`edit_target_path`]
/// classifies; everything else (`shell`, `exec_command`, `update_plan`, …) is
/// neither and is ignored by the read-without-write counter.
fn is_read_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file"
            | "cat_file"
            | "list_dir"
            | "grep_files"
            | "web_fetch"
            | "web_search"
            | "local_web_search"
    )
}

/// Read-without-write loop: the model spins in research mode — searching,
/// fetching, reading — gathering information without ever acting. Distinct from
/// tunnel-vision (which needs the SAME target revisited): here the targets may
/// ALL differ, so the footprint keeps "expanding" and tunnel-vision resets every
/// call — yet no write ever lands. This is the approach-not-target reset made
/// concrete: the counter is "reads since the last write", and ONLY a write
/// (real progress) clears it — a new query / url / file does not.
///
/// Runs late in the chain (after the exact-repeat and tunnel-vision detectors),
/// so a tighter, more specific loop is claimed first. The threshold is high
/// because legitimate exploration reads a lot before the first edit; this is the
/// broad backstop for "researching forever", not a nag on healthy exploration.
fn detect_read_without_write(parsed: &ParsedTranscript) -> Option<RepetitionAlert> {
    const THRESHOLD: usize = 12;

    // Active turn only: walk back to the last user message (a new task is a fresh
    // count), collect calls, then process forward.
    let mut calls: Vec<(&str, &str)> = Vec::new();
    for item in parsed.items.iter().rev() {
        match item {
            TrimItem::ToolCall {
                tool_name, call_id, ..
            } => calls.push((tool_name, call_id)),
            TrimItem::ToolOutput { .. }
            | TrimItem::AssistantText { .. }
            | TrimItem::Reasoning { .. } => continue,
            _ => break, // user message → task boundary
        }
    }
    calls.reverse();

    let mut reads = 0usize;
    let mut last_read_call_id = String::new();
    for (tool_name, call_id) in &calls {
        // A write is real progress — reset. We only have the tool name here, but
        // the write tools are name-identifiable, so re-derive from the item's args
        // is unnecessary: any edit-classified tool name counts as a write.
        if matches!(
            *tool_name,
            "write_file"
                | "create_file"
                | "edit_file"
                | "str_replace"
                | "apply_patch"
                | "text_editor"
        ) {
            reads = 0;
            last_read_call_id.clear();
        } else if is_read_tool(tool_name) {
            reads += 1;
            last_read_call_id = (*call_id).to_string();
        }
        // else (shell / exec_command / update_plan / …): neither read nor write —
        // ignore, so an interleaved shell probe neither counts nor resets.
    }

    if reads < THRESHOLD {
        return None;
    }

    let last_output_excerpt = parsed
        .items
        .iter()
        .rev()
        .find_map(|item| match item {
            TrimItem::ToolOutput {
                call_id: cid,
                content,
                ..
            } if *cid == last_read_call_id => Some(excerpt(content, 300)),
            _ => None,
        })
        .unwrap_or_default();

    Some(RepetitionAlert {
        tool_name: String::new(),
        command_summary: String::new(),
        count: reads,
        last_output_excerpt,
        call_id: last_read_call_id,
        force_diagnosis: false,
        escalate: false,
        signature: String::new(),
        tunnel_vision: false,
        kind: LoopKind::ReadWithoutWrite,
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

/// Truncate `s` to at most `n` bytes, ending on a CHAR BOUNDARY, with an ellipsis.
/// A raw byte slice `&s[..n]` PANICS when `n` lands inside a multibyte char (e.g. a
/// `·` at bytes 199..201 in a web-search result) — the crash this guards against.
/// Tool outputs are arbitrary UTF-8, so every truncation here must be boundary-safe.
fn truncate_ellipsis(s: &str, n: usize) -> String {
    if s.len() <= n {
        return s.to_string();
    }
    let mut end = n;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

fn excerpt(s: &str, n: usize) -> String {
    truncate_ellipsis(s, n)
}

fn short_str(s: &str, n: usize) -> String {
    truncate_ellipsis(s, n)
}

fn short_args(s: &str, n: usize) -> String {
    let cleaned = s.replace(['\n', '\r'], " ");
    short_str(&cleaned, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_ellipsis_does_not_split_a_multibyte_char() {
        // Reproduces the live crash: a `·` (U+00B7, bytes 199..201) straddling the
        // cut at byte 200. A raw `&s[..200]` panicked ("not a char boundary") and
        // took down the whole session when a web-search result was excerpted.
        let s = format!("{}·{}", "a".repeat(199), "b".repeat(100));
        let out = truncate_ellipsis(&s, 200); // must not panic
        assert!(out.ends_with('…'));
        // Backed off to the boundary before the `·` (byte 199).
        assert_eq!(out, format!("{}…", "a".repeat(199)));
    }

    #[test]
    fn truncate_ellipsis_keeps_short_or_exact_strings_whole() {
        assert_eq!(truncate_ellipsis("héllo", 100), "héllo");
        assert_eq!(truncate_ellipsis("abc", 3), "abc");
    }
}
