//! Streaming-time "reasoning loop" detector for local reasoning models.
//!
//! Problem: models like Qwopus 3.5 emit unbounded `<think>` content and can
//! spiral into self-interrupting loops ("Actually, wait. Let me reconsider.
//! Hmm, on second thought..."), burning max_tokens without ever producing
//! output or a tool call. When that happens, the response arrives with
//! `content=""` + `tool_calls=[]` after 2–10 minutes of wall time.
//!
//! This module provides a lightweight phrase-count heuristic: count how
//! many times the reasoning stream uses "rumination markers" (second-guessing
//! phrases like `actually`, `wait`, `let me reconsider`), gate by a
//! reasoning-token budget so we don't false-positive on a model that naturally
//! self-critiques once or twice, and if the count exceeds a threshold, the
//! caller aborts the in-flight request and re-prompts with a directive to
//! stop ruminating.
//!
//! Design constraints:
//! - Pure function, no I/O. Suitable for calling on every chunk in a hot
//!   streaming loop.
//! - Case-insensitive word-boundary matching (avoid matching "waiting",
//!   "factually", "reconsideration" as false positives for "wait",
//!   "actually", "reconsider").
//! - Conservative: markers chosen to be characteristic of LLM thinking drift
//!   rather than ordinary prose. We'd rather miss some loops than abort a
//!   legitimate reasoning chain.

use std::sync::OnceLock;

use regex::Regex;

/// Phrases that signal the model is second-guessing itself. Each marker is
/// matched with word boundaries and case-insensitively. The list is
/// deliberately focused on drift/backtracking cues; neutral reasoning
/// language like "therefore", "because", "so" is excluded.
const RUMINATION_MARKERS: &[&str] = &[
    "actually",
    "wait",
    "but wait",
    "hold on",
    "hmm",
    "let me reconsider",
    "on second thought",
    "let me think again",
    "or maybe",
    "or perhaps",
    "rethinking",
    "reconsider",
    "going back",
    "scratch that",
    "nope",
    "let me re-examine",
    "let me reexamine",
    "let me revisit",
    "i'm overthinking",
    "am i overthinking",
    "let me start over",
    "wait no",
    "actually no",
];

/// The default threshold for how many rumination markers a reasoning
/// stream must contain before we call it a loop. Tuned to a starting
/// value; expect to revisit once we have data from real sessions.
pub const DEFAULT_MARKER_THRESHOLD: usize = 6;

/// Default reasoning-token budget when the endpoint hasn't set `max_tokens`.
/// Models that can't control thinking will blow through this quickly; the
/// gate ensures we only flag a loop once the model has had a fair shot at
/// normal reasoning.
pub const DEFAULT_REASONING_BUDGET: usize = 4096;

/// Verdict from a single rumination check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuminationVerdict {
    /// Reasoning looks normal — either under budget, or markers are below
    /// threshold.
    Ok,
    /// Marker count crossed threshold after the budget gate. Caller should
    /// abort the in-flight request and re-prompt with a directive to stop
    /// ruminating.
    Ruminating {
        /// How many rumination markers the detector counted.
        hits: usize,
        /// Approximate reasoning tokens consumed when the check fired.
        reasoning_tokens: usize,
    },
}

/// Stateful detector that caches the compiled regex so that repeated calls
/// during streaming don't re-compile it. Not `Clone` — callers construct
/// one per stream.
pub struct RuminationDetector {
    /// Half of this is the budget gate; below half we never flag, regardless
    /// of marker count. Pulled from `endpoint.max_tokens` when set, or
    /// `DEFAULT_REASONING_BUDGET` otherwise.
    budget: usize,
    /// Minimum marker count to consider it a loop, once the budget gate
    /// opens.
    threshold: usize,
}

impl RuminationDetector {
    pub fn new(budget: usize, threshold: usize) -> Self {
        let budget = if budget == 0 {
            DEFAULT_REASONING_BUDGET
        } else {
            budget
        };
        let threshold = threshold.max(1);
        Self { budget, threshold }
    }

    /// Build a detector from the endpoint's `max_tokens` config. `None`
    /// means "unlimited" upstream, which for our purposes means "use the
    /// default budget."
    pub fn from_endpoint_max_tokens(max_tokens: Option<usize>) -> Self {
        Self::new(
            max_tokens.unwrap_or(DEFAULT_REASONING_BUDGET),
            DEFAULT_MARKER_THRESHOLD,
        )
    }

    /// Half-of-budget gate. Below this, never flag.
    pub fn budget_gate(&self) -> usize {
        self.budget / 2
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Run the rumination check on the accumulated reasoning text. The
    /// `reasoning_tokens` argument should be the server's reported token
    /// count when available; if the stream hasn't reported one yet, the
    /// caller can estimate with `estimate_reasoning_tokens`.
    pub fn check(&self, reasoning_so_far: &str, reasoning_tokens: usize) -> RuminationVerdict {
        if reasoning_tokens < self.budget_gate() {
            return RuminationVerdict::Ok;
        }
        let hits = count_rumination_markers(reasoning_so_far);
        if hits >= self.threshold {
            RuminationVerdict::Ruminating {
                hits,
                reasoning_tokens,
            }
        } else {
            RuminationVerdict::Ok
        }
    }
}

/// Rough char-to-token heuristic for English. Real tokenizers vary a lot
/// but ~4 chars/token is the long-run average for GPT-ish BPE.
pub fn estimate_reasoning_tokens(text: &str) -> usize {
    text.chars().count() / 4
}

/// Directive injected as a new `user`-role message when the detector
/// aborts an in-flight reasoning pass. Mirrors the shape of the bail
/// continuation prompt but for a different failure mode: the model was
/// generating tokens, just tokens that signalled self-doubt rather than
/// progress.
pub fn continuation_prompt(hits: usize, reasoning_tokens: usize) -> String {
    format!(
        "[RUMINATION GUARD] Your last reasoning pass hit {hits} second-guessing phrases \
         (\"actually\", \"wait\", \"let me reconsider\", etc.) after ~{reasoning_tokens} \
         reasoning tokens and was aborted before producing output.\n\n\
         Stop re-examining. Pick the simplest next concrete step and take it \
         via a tool call. If you are choosing between options, just pick one \
         and proceed — do not revisit the decision. Do NOT restart from scratch; \
         continue from what you already know."
    )
}

/// Count how many rumination markers appear in `text`. Matching is
/// case-insensitive and word-bounded so `waiting` doesn't fire `wait`.
pub fn count_rumination_markers(text: &str) -> usize {
    let re = compiled_regex();
    re.find_iter(text).count()
}

fn compiled_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let alternation = RUMINATION_MARKERS
            .iter()
            .map(|m| regex::escape(m))
            .collect::<Vec<_>>()
            .join("|");
        // (?i) = case-insensitive. \b doesn't behave well for phrases with
        // spaces inside them, so we use lookarounds-free boundary via
        // explicit non-alphanumeric before/after. The `regex` crate doesn't
        // support look-arounds, but `\b` actually DOES work for alternations
        // where each branch starts and ends with a word character — which
        // is true of every marker we list.
        Regex::new(&format!(r"(?i)\b(?:{alternation})\b")).expect("rumination regex must compile")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_budget_gate_never_flags() {
        let det = RuminationDetector::new(10_000, 3);
        // Deliberately pack markers but stay below the 5000-token gate.
        let text = "actually wait hmm let me reconsider actually wait";
        let v = det.check(text, 100);
        assert_eq!(v, RuminationVerdict::Ok);
    }

    #[test]
    fn above_budget_with_threshold_hits_flags() {
        let det = RuminationDetector::new(1000, 3);
        let text = "Actually, wait. Hmm, let me reconsider.";
        let v = det.check(text, 600);
        assert!(matches!(v, RuminationVerdict::Ruminating { .. }));
    }

    #[test]
    fn above_budget_below_threshold_stays_ok() {
        let det = RuminationDetector::new(1000, 5);
        let text = "I'll check the file. Actually, that's fine. Proceeding.";
        let v = det.check(text, 600);
        assert_eq!(v, RuminationVerdict::Ok);
    }

    #[test]
    fn word_boundary_prevents_false_positives() {
        // `waiting`, `factually`, `reconsideration`, `whatever` should NOT
        // match `wait`, `actually`, `reconsider`, `hmm`/`wait`.
        let text = "waiting factually reconsideration whatever";
        assert_eq!(count_rumination_markers(text), 0);
    }

    #[test]
    fn case_insensitive_matching() {
        assert_eq!(count_rumination_markers("Actually"), 1);
        assert_eq!(count_rumination_markers("ACTUALLY"), 1);
        assert_eq!(count_rumination_markers("actually"), 1);
    }

    #[test]
    fn multi_word_markers_match() {
        assert_eq!(count_rumination_markers("But wait, let me think again."), 2);
        assert_eq!(
            count_rumination_markers("on second thought, scratch that"),
            2
        );
    }

    #[test]
    fn zero_budget_falls_back_to_default() {
        let det = RuminationDetector::new(0, 3);
        assert_eq!(det.budget_gate(), DEFAULT_REASONING_BUDGET / 2);
    }

    #[test]
    fn from_endpoint_max_tokens_none_uses_default() {
        let det = RuminationDetector::from_endpoint_max_tokens(None);
        assert_eq!(det.budget_gate(), DEFAULT_REASONING_BUDGET / 2);
        assert_eq!(det.threshold(), DEFAULT_MARKER_THRESHOLD);
    }

    #[test]
    fn from_endpoint_max_tokens_some_uses_value() {
        let det = RuminationDetector::from_endpoint_max_tokens(Some(8000));
        assert_eq!(det.budget_gate(), 4000);
    }

    #[test]
    fn estimate_reasoning_tokens_approximates() {
        // 16 chars / 4 = 4 tokens
        assert_eq!(estimate_reasoning_tokens("abcd abcd abcd a"), 4);
    }

    #[test]
    fn realistic_rumination_passage_flags() {
        // A passage modeled on actual qwopus3.5 drift.
        let text = "\
            Let me think about this step by step. Actually, wait. \
            I should check the file first. But hmm, maybe I need to \
            reconsider the approach. On second thought, let me \
            re-examine what we're trying to do. Actually no, the \
            original plan was fine. Wait, or perhaps not — let me \
            reconsider once more.";
        assert!(count_rumination_markers(text) >= 6);
    }
}
