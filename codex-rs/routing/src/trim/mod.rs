//! Deterministic role-aware transcript trimming for local models.
//!
//! Used by both per-request context preparation (replacing the older
//! `context_strip` module) and as the first stage of compaction.
//! Same rules everywhere; mode (regular vs local-only) only changes routing.
//!
//! Design principles (see docs/spec/trim-design.md when written):
//! - Local models always get the same treatment regardless of mode.
//! - Older history is replaced with synthesized state, not just chopped.
//! - The active turn (everything from the most recent user message onward)
//!   is preserved verbatim with no compression.
//! - Per-tool semantic rules — never blanket character truncation.
//! - Errors are sticky: any failed tool output is preserved regardless of age.
//! - System prompt is never stubbed.

mod items;
mod render;
mod signatures;
mod state_extract;

#[cfg(test)]
mod tests;

use codex_protocol::models::ResponseItem;
use serde::Serialize;
use serde_json::Value as JsonValue;

pub use items::TrimItem;
pub use signatures::signature_for_call;
pub use state_extract::files_modified_in_active_turn;

/// Input handed to the trimmer. Decoupled from the codex-core `Prompt` type so
/// the routing crate stays independent of `codex-core`.
#[derive(Debug, Clone, Default)]
pub struct TrimInput<'a> {
    /// Conversation items, oldest first, exactly as they appear in `Prompt::input`.
    pub items: &'a [ResponseItem],
    /// The full Codex system prompt (base instructions). Passed through verbatim.
    pub system_prompt: &'a str,
    /// Project-level user instructions (AGENTS.md / CLAUDE.md content), if any.
    /// These are pinned into the persistent context block.
    pub user_instructions: Option<&'a str>,
    /// Wire-format flavor for the model the trimmed transcript will be sent
    /// to. Affects per-message rendering for tool calls / tool outputs:
    /// Ollama is lenient about message shapes (we wrap tool outputs in
    /// `<tool_result>` user messages); OpenAI-compat servers reject
    /// anything that doesn't match the strict
    /// `assistant{tool_calls}` → `tool{tool_call_id, content}` pairing.
    /// Defaults to Ollama for backwards compatibility.
    pub flavor: crate::config::ClientFlavor,
    /// Max share of `target_ctx` the **system prompt** may occupy, as a percent.
    /// When `system_prompt` exceeds it, the trimmer deterministically compresses
    /// it (head+tail kept, middle elided) so a large incoming system prompt can't
    /// crowd out the conversation. `0` (the `Default`) disables compression —
    /// preserving the legacy "system is never stubbed" behavior for callers that
    /// don't opt in. The generated state prelude is always preserved on top.
    pub system_budget_pct: u8,
    /// When set, drop the transcript-derived loop/repetition alert for this
    /// request. The caller sets it during a post-course-change **grace window**
    /// (see `reasoned_guidance::consume_loop_grace`): the reasoner has confirmed
    /// the coder genuinely changed approach, so the family-A guards — which
    /// re-derive from the still-loopy history and would otherwise re-fire on the
    /// pivot's own new reads/curls — are paused for a few turns so the new
    /// approach gets an unobstructed chance to execute.
    pub suppress_loop_alerts: bool,
}

/// Result of trimming, ready to send to a local model via the Ollama chat API.
#[derive(Debug, Clone)]
pub struct TrimResult {
    /// The base system prompt (frame/role), bounded to its budget. Sent as the
    /// chat `system` field, so it stays a STABLE prefix (good for prompt caching).
    pub system: String,
    /// The synthesized per-turn state prelude — pinned files, world state, and any
    /// loop-guard directive. Delivered by the caller as a FINAL message (end of the
    /// prompt) rather than in `system`, because a small model attends to the END of
    /// the context far more than the beginning (recency). Empty when there is none.
    pub guidance: String,
    /// Chat messages in Ollama format. Older turns are collapsed into single
    /// summary messages; the active turn is preserved verbatim including any
    /// tool calls and tool outputs.
    pub messages: Vec<JsonValue>,
    /// Diagnostics about what was kept, dropped, or collapsed.
    pub summary: TrimSummary,
    /// Set when the model's most recent `apply_patch` failed: the target file the
    /// prelude is steering it to rewrite with `write_file`. Lets the caller
    /// surface a nudge and force the `write_file` tool on the next call.
    pub patch_rewrite_path: Option<String>,
    /// The repetition/thrash count behind the loop-guard directive in `system`
    /// this turn (if any). Surfaced so the caller can LOG which guard fired and how
    /// deep the loop is — the guard notices themselves only reach the TUI, so
    /// without this the logs make a firing guard look like it never fired.
    pub repetition_count: Option<usize>,
    /// The STRUCTURED repeated call + its actual output behind the loop directive
    /// (when one fired). This is the loop's ground truth — the exact command the
    /// model keeps running and the result it keeps getting — surfaced so the caller
    /// can hand it to the reasoner as fresh grounding (the repeat proves the output
    /// is current) instead of only pasting canned prose the model ignores.
    pub repeated_action: Option<crate::ground_truth::RepeatedAction>,
}

/// Diagnostics emitted by `trim_for_local`. Logged by the caller; not seen by
/// the model.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TrimSummary {
    pub original_items: usize,
    pub kept_items: usize,
    pub older_turns_collapsed: u32,
    pub stale_reads_dropped: u32,
    pub superseded_outputs_dropped: u32,
    pub elided_output_chars: usize,
    pub estimated_input_tokens: usize,
    /// Count of older-turn messages at the head of `messages`. Active-turn
    /// messages start at this offset. Callers can use this to slice older
    /// content out for further processing (e.g. summarization when even
    /// the trimmed transcript exceeds the local model's context budget).
    pub older_turn_message_count: usize,
}

impl TrimSummary {
    /// Render a one-line summary suitable for tracing/info logs.
    pub fn to_log_line(&self) -> String {
        format!(
            "kept {}/{} items; collapsed {} older turns; dropped {} stale reads, {} superseded outputs; elided {} chars; ~{} input tokens",
            self.kept_items,
            self.original_items,
            self.older_turns_collapsed,
            self.stale_reads_dropped,
            self.superseded_outputs_dropped,
            self.elided_output_chars,
            self.estimated_input_tokens,
        )
    }
}

/// `estimate_tokens` (≈ chars/4) systematically undercounts real BPE tokens on
/// code/JSON-heavy transcripts. We budget against the estimate inflated by this
/// factor so the *real* tokenized prompt stays under the model's window — the
/// gap is what let a "16k" prompt actually tokenize to ~25k and overflow.
///
/// Measured against a live llama.cpp server (Ornith): a prompt this estimator
/// put at ~21k tokenized to 37k — a **1.75×** undercount on dense agentic
/// content (JSON tool args, code, API specs), well past the old 1.35. With 1.35
/// the trimmer handed back prompts that overflowed the 32k window, and on a slow
/// CPU box the oversized re-prefill then timed out → "chain exhausted". Budget
/// at 1.8 so the real prompt fits with margin on the first try.
const ESTIMATE_SAFETY_FACTOR: f64 = 1.8;

/// Trim a transcript for a local model.
///
/// Mechanical only: after collapsing older turns, [`enforce_token_budget`]
/// truncates the bulkiest tool outputs until the rendered prompt fits. The active
/// user request and the model's own messages are never truncated — only tool data.
/// This usually leaves the result inside `target_ctx`, but a long single active
/// turn (lots of protected assistant bulk) can still exceed it; the caller closes
/// that gap with inline compaction and, as a floor, [`drop_to_fit`]. Callers should
/// set `target_ctx` (`trim_budget`) below the server's context size so the
/// remaining headroom covers tool schemas and the model's output.
pub fn trim_for_local(input: &TrimInput, target_ctx: usize) -> TrimResult {
    let parsed = items::parse(input.items);
    let active_turn = parsed.active_turn_id();

    let mut extracted = state_extract::extract(&parsed, active_turn);
    // Post-course-change grace: a reasoner confirmed the coder genuinely pivoted,
    // so pause the transcript-derived loop nudges — they would re-fire on the new
    // approach's own reads/curls and re-bury it before it can finish.
    if input.suppress_loop_alerts {
        extracted.repetition = None;
    }
    // No older-turn collapse: the transcript renders VERBATIM (pass-through). The
    // stale/superseded/elided/collapsed stats go with it.
    let prelude = render::render_prelude(input.user_instructions, &extracted);
    let (mut messages, older_turn_message_count) = render::render_messages(
        &parsed,
        active_turn,
        input.flavor,
        extracted.repetition.as_ref(),
        render::max_output_chars(target_ctx),
    );

    let mut summary = TrimSummary {
        original_items: input.items.len(),
        kept_items: messages.len(),
        older_turns_collapsed: 0,
        stale_reads_dropped: 0,
        superseded_outputs_dropped: 0,
        elided_output_chars: 0,
        estimated_input_tokens: 0,
        older_turn_message_count,
    };

    // Bound the incoming system prompt to its configured share of the budget.
    // A large base prompt (Codex's is ~9k real tokens; an arbitrary harness's
    // could be more) is otherwise an immovable tax that crowds out the
    // conversation and, on a small window, forces overflow. We compress only the
    // *incoming* system prompt and always keep our freshly-generated state
    // prelude on top. Deterministic here (pure); a caller may pre-summarize.
    let system_budget = if input.system_budget_pct == 0 {
        usize::MAX
    } else {
        target_ctx.saturating_mul(input.system_budget_pct as usize) / 100
    };
    let system_prompt = compress_system_prompt(input.system_prompt, system_budget);
    // Kept only for the budget math below; `system_prompt` and `prelude` are
    // returned SEPARATELY (frame vs end-guidance), so don't consume them here.
    let combined_system = if prelude.is_empty() {
        system_prompt.clone()
    } else {
        format!("{system_prompt}\n\n{prelude}")
    };

    // Hard budget enforcement: after compaction, GUARANTEE the rendered prompt
    // fits `target_ctx` by truncating the bulkiest tool outputs — never the
    // user's request or the model's own messages. Without this the trimmer can
    // hand back an oversized prompt that fills the model's context with no room
    // left to generate, which surfaces to the user as "model unavailable".
    //
    // Two corrections make `target_ctx` a TRUE ceiling rather than an
    // optimistic one:
    //   1. `estimate_tokens` (≈ chars/4) undercounts real BPE on code-heavy
    //      content by ~1.3×, so we budget against the inflated figure.
    //   2. The caller already subtracted a reserve for the tool schemas it adds
    //      *after* trimming (they're not in `messages`), so `target_ctx` here is
    //      net of those.
    // Together: real_prompt ≈ estimate × SAFETY_FACTOR + tool_schemas ≤ trim_budget.
    let effective_budget = effective_budget(target_ctx);
    summary.elided_output_chars = summary
        .elided_output_chars
        .saturating_add(enforce_token_budget(
            &combined_system,
            &mut messages,
            effective_budget,
        ));

    // NOTE: trim is now purely *mechanical* — it collapses older turns and
    // truncates tool data, but it deliberately does NOT drop whole messages.
    // A long single active turn (an agentic loop of many model iterations with
    // no new user message) can still exceed the window on protected assistant
    // bulk alone. That case is handled one layer up, in order: (1) inline
    // *compaction* summarizes the turn's middle (semantic, lossy-but-faithful),
    // and only if that's unavailable/insufficient does (2) [`drop_to_fit`] do a
    // bounded last-resort drop. Doing the drop here would pre-empt compaction —
    // the prompt would already fit, so the compactor never runs. See
    // `maybe_inline_compact` in codex-core and docs/spec/content-reduce.md.
    summary.kept_items = messages.len();

    summary.estimated_input_tokens =
        crate::metrics::estimate_tokens(&combined_system) + estimate_messages_tokens(&messages);

    TrimResult {
        // The base frame stays as `system` (stable prefix); the per-turn state
        // prelude rides out as `guidance` for the caller to place at the END.
        // `combined_system` above is kept only for the budget math.
        system: system_prompt,
        guidance: prelude,
        messages,
        summary,
        patch_rewrite_path: extracted.patch_failure.as_ref().map(|pf| pf.path.clone()),
        repetition_count: extracted.repetition.as_ref().map(|a| a.count),
        repeated_action: extracted.repetition.as_ref().map(|a| crate::ground_truth::RepeatedAction {
            command: a.command_summary.clone(),
            output: a.last_output_excerpt.clone(),
            count: a.count,
        }),
    }
}

/// Compress a system prompt to fit `budget_tokens` (estimate-space). Returns it
/// unchanged if it already fits or the budget is `usize::MAX` (disabled).
///
/// Deterministic, content-agnostic: keep the **head** (where prompts establish
/// role and high-level rules) and the **tail** (where they often put the most
/// recent / most specific directives and output-format rules), eliding the
/// middle with a visible marker. Generic by design — bounds any incoming system
/// prompt without knowing its structure. A caller that can afford an LLM call
/// may pre-summarize for a higher-fidelity result; this is the always-available
/// floor.
fn compress_system_prompt(text: &str, budget_tokens: usize) -> String {
    if budget_tokens == usize::MAX {
        return text.to_string();
    }
    if crate::metrics::estimate_tokens(text) <= budget_tokens {
        return text.to_string();
    }
    const MARKER: &str =
        "\n\n[… system prompt trimmed to fit the local model's context; middle elided …]\n\n";
    let budget_chars = budget_tokens.saturating_mul(4);
    let keep = budget_chars.saturating_sub(MARKER.chars().count());
    let chars: Vec<char> = text.chars().collect();
    if keep == 0 || chars.len() <= keep {
        return text.to_string();
    }
    // Favor the head (role/framing) slightly over the tail.
    let head_len = keep * 6 / 10;
    let tail_len = keep.saturating_sub(head_len);
    let head: String = chars[..head_len].iter().collect();
    let tail: String = chars[chars.len() - tail_len..].iter().collect();
    format!("{head}{MARKER}{tail}")
}

/// Last-resort fit when even fully-truncated tool data leaves the prompt over
/// budget — i.e. the protected content (the model's own assistant turns) is
/// itself too large. Removes the OLDEST messages one at a time until the prompt
/// fits or only the final message remains, then strips any leading orphan `tool`
/// message (its originating assistant tool_call may have been removed) so the
/// wire payload stays well-formed.
///
/// CRITICAL: the user's request is the *oldest* item in a turn, so naive
/// front-first dropping removes it first. Many chat templates then raise — Ornith
/// literally `raise_exception('No user query found in messages')` (a 500 that
/// dead-ended whole turns). So we always preserve the most recent user message
/// and re-insert it if dropping removed every user role. Returns chars dropped.
/// Last-resort bounded drop, applied by the caller AFTER inline compaction has
/// had its chance. Converts `trim_budget` to the same estimate-space
/// `effective_budget` [`trim_for_local`] targets (net of [`ESTIMATE_SAFETY_FACTOR`])
/// and drops oldest messages until it fits, always keeping the user request.
/// Returns chars dropped (0 if it already fit). This is the floor that guarantees
/// a servable prompt when compaction is disabled or insufficient.
pub fn drop_to_fit(system: &str, messages: &mut Vec<JsonValue>, trim_budget: usize) -> usize {
    drop_oldest_until_fit(system, messages, effective_budget(trim_budget))
}

/// The **estimate-space** budget that fits `trim_budget` real tokens, net of the
/// tokenizer undercount ([`ESTIMATE_SAFETY_FACTOR`]). Trim reduces the estimate
/// down to this; the inline compactor should trigger as the estimate *approaches*
/// it (not at a fraction of the raw, real-token `trim_budget`, which left a dead
/// band where the real prompt overflowed but the estimate never crossed the
/// threshold). One definition, used by trim, `drop_to_fit`, and the compactor.
pub fn effective_budget(trim_budget: usize) -> usize {
    (trim_budget as f64 / ESTIMATE_SAFETY_FACTOR) as usize
}

fn drop_oldest_until_fit(system: &str, messages: &mut Vec<JsonValue>, target_ctx: usize) -> usize {
    if target_ctx == 0 {
        return 0;
    }
    let is_role = |m: &JsonValue, role: &str| m.get("role").and_then(|r| r.as_str()) == Some(role);
    let last_user = messages.iter().rev().find(|m| is_role(m, "user")).cloned();

    let sys_tokens = crate::metrics::estimate_tokens(system);
    let mut dropped = 0usize;
    while messages.len() > 1 && sys_tokens + estimate_messages_tokens(messages) > target_ctx {
        let removed = messages.remove(0);
        dropped = dropped.saturating_add(message_content_len(&removed));
    }
    // Strip leading orphan tool results so a drop that removed the matching
    // assistant tool_call doesn't leave a result with no call. Catch BOTH the
    // OpenAI `role:"tool"` form and our Ollama `role:"user"` `<tool_result>` wrapper
    // (the latter was previously missed — a silent orphan-result swiss-cheese).
    let is_orphan_result = |m: &JsonValue| {
        is_role(m, "tool")
            || (is_role(m, "user")
                && m.get("content").and_then(|c| c.as_str()).is_some_and(|s| {
                    let t = s.trim_start();
                    t.starts_with("<tool_result") || t.starts_with("<tool_error")
                }))
    };
    while messages.len() > 1 && messages.first().is_some_and(|m| is_orphan_result(m)) {
        let removed = messages.remove(0);
        dropped = dropped.saturating_add(message_content_len(&removed));
    }
    // Guarantee a user message survives so the chat template can locate "the
    // query". After the orphan-tool strip the front is a user/assistant message,
    // so prepending the request keeps a valid `user → assistant/tool` order.
    if let Some(user) = last_user
        && !messages.iter().any(|m| is_role(m, "user"))
    {
        messages.insert(0, user);
    }
    dropped
}

fn estimate_messages_tokens(messages: &[JsonValue]) -> usize {
    let joined: String = messages
        .iter()
        .filter_map(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .map(str::to_string)
        })
        .collect::<Vec<_>>()
        .join("\n");
    crate::metrics::estimate_tokens(&joined)
}

/// Character count of a chat message's `content` field.
fn message_content_len(message: &JsonValue) -> usize {
    message
        .get("content")
        .and_then(|c| c.as_str())
        .map(|s| s.chars().count())
        .unwrap_or(0)
}

/// A message is truncatable only if it carries bulk tool data — a `role: tool`
/// output, an Ollama-wrapped `<tool_result>`/`<tool_error>`, or a collapsed
/// older-turn summary (`[turn N — …]`). The active user request and the
/// model's own assistant messages are never touched.
fn is_truncatable(message: &JsonValue) -> bool {
    if message.get("role").and_then(|r| r.as_str()) == Some("tool") {
        return true;
    }
    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or_default();
    content.contains("<tool_result")
        || content.contains("<tool_error")
        || content.starts_with("[turn ")
}

/// Replace a message's `content` with its first `keep_chars` characters and a
/// visible truncation marker (char-boundary safe).
fn truncate_message_content(message: &mut JsonValue, keep_chars: usize) {
    let Some(content) = message.get("content").and_then(|c| c.as_str()) else {
        return;
    };
    let total = content.chars().count();
    if total <= keep_chars {
        return;
    }
    let head: String = content.chars().take(keep_chars).collect();
    let dropped = total - keep_chars;
    let new_content = format!(
        "{head}\n[... {dropped} chars truncated to fit the local model's context window — re-read with an explicit line range if you need the rest ...]"
    );
    if let Some(obj) = message.as_object_mut() {
        obj.insert("content".to_string(), JsonValue::String(new_content));
    }
}

/// Guarantee the prompt fits the model's context: while
/// `estimate(system) + estimate(messages)` exceeds `target_ctx`, halve the
/// largest truncatable message. Returns the number of characters truncated.
///
/// Termination: each iteration strictly shrinks the single largest truncatable
/// message (halving toward `MIN_KEEP_CHARS`), so truncatable bulk converges to
/// zero; an iteration cap is a backstop. If the only remaining content is
/// protected (the user's request / assistant turns) and still over budget,
/// it stops and returns what it cut — that pathological case (a single
/// request larger than the whole context) is the caller's to avoid.
fn enforce_token_budget(system: &str, messages: &mut [JsonValue], target_ctx: usize) -> usize {
    const MIN_KEEP_CHARS: usize = 200;
    if target_ctx == 0 {
        return 0;
    }
    let sys_tokens = crate::metrics::estimate_tokens(system);
    let mut truncated_chars = 0usize;
    for _ in 0..512 {
        if sys_tokens + estimate_messages_tokens(messages) <= target_ctx {
            break;
        }
        let largest = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| is_truncatable(m))
            .map(|(i, m)| (i, message_content_len(m)))
            .max_by_key(|(_, len)| *len);
        let Some((idx, len)) = largest else {
            break; // nothing left we're willing to truncate
        };
        if len <= MIN_KEEP_CHARS {
            break; // remaining bulk is already minimal
        }
        let keep = (len / 2).max(MIN_KEEP_CHARS);
        truncate_message_content(&mut messages[idx], keep);
        truncated_chars = truncated_chars.saturating_add(len - keep);
    }
    truncated_chars
}
