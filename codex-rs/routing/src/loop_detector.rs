//! Agentic-loop detection that the consecutive-identical tool guard misses.
//!
//! The hard repetition guard (`ToolRepetitionGuard` in core) only counts
//! *consecutive byte-identical* tool calls and resets on any change. The Ada-
//! handle session showed two loops that slipped through it:
//!   - a **cycle** of distinct calls repeated over and over
//!     (apply_patch → run tests → cat file → apply_patch …), and
//!   - **the same file edited** ~20 times with near-identical content,
//! while the model emitted the **same assistant preamble** ("I see the issue —
//! HTTPError handling causes recursion. Let me fix…") ~18 times.
//!
//! This module detects all three, productivity-gated so a *healthy* iterative
//! loop (edits whose content actually changes, varied actions) never trips:
//!   - (b) [`LoopDetector::note_tool_call`] — short cyclic tool-call patterns.
//!   - (c) [`LoopDetector::note_tool_call`] — repeated near-identical edits to
//!     one file.
//!   - (a) [`LoopDetector::note_assistant_text`] — repeated near-identical
//!     assistant text.
//!
//! Pure logic with no I/O so it's unit-testable; the caller (core) owns the
//! session-scoped state and turns a [`LoopVerdict`] into a block/nudge.

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;

/// How many recent tool-call signatures to retain for cycle detection (b).
const SIG_RING: usize = 12;
/// A cycle (period 2–4) must repeat at least this many times to trip (b).
const CYCLE_REPEATS: usize = 3;
/// Recent content fingerprints retained per edited path (c).
const EDIT_RING_PER_PATH: usize = 8;
/// Identical-content edits to one path that trip the same-target guard (c).
const SAME_EDIT_THRESHOLD: usize = 3;
/// Recent assistant-preamble fingerprints retained (a).
const TEXT_RING: usize = 8;
/// Identical preambles that trip the repeated-text guard (a).
const SAME_TEXT_THRESHOLD: usize = 3;
/// Tokens of the assistant message used as its "preamble" fingerprint (a).
const PREAMBLE_TOKENS: usize = 40;

/// The kind of loop detected, with a model-facing directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopVerdict {
    pub message: String,
}

/// Session-scoped loop detector. Cheap; holds only small ring buffers.
#[derive(Default)]
pub struct LoopDetector {
    /// (b) recent tool-call signatures, oldest→newest.
    sigs: VecDeque<String>,
    /// (c) per edited path → recent content fingerprints.
    edits: Vec<(String, VecDeque<u64>)>,
    /// (a) recent normalized assistant preambles.
    texts: VecDeque<String>,
}

impl LoopDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a dispatched tool call and report a loop if one is detected:
    /// (b) a short cyclic pattern, or (c) the same file re-edited with
    /// near-identical content. `args_raw` is the raw JSON arguments string.
    pub fn note_tool_call(&mut self, tool_name: &str, args_raw: &str) -> Option<LoopVerdict> {
        // (c) same-target near-identical edits.
        if let Some((path, content_fp)) = extract_edit_target(tool_name, args_raw) {
            let ring = match self.edits.iter_mut().find(|(p, _)| *p == path) {
                Some((_, r)) => r,
                None => {
                    self.edits.push((path.clone(), VecDeque::new()));
                    &mut self.edits.last_mut().unwrap().1
                }
            };
            let identical = ring.iter().filter(|fp| **fp == content_fp).count() + 1;
            ring.push_back(content_fp);
            while ring.len() > EDIT_RING_PER_PATH {
                ring.pop_front();
            }
            if identical >= SAME_EDIT_THRESHOLD {
                return Some(LoopVerdict {
                    message: format!(
                        "BLOCKED (loop guard): you have written essentially the same content to \
                         `{path}` {identical} times and the result has not changed. This edit was \
                         NOT applied. Stop re-applying the same edit — read the CURRENT file and \
                         the FULL error/test output, then make a DIFFERENT change, or explain what \
                         is blocking you."
                    ),
                });
            }
        }

        // (b) cyclic tool-call pattern.
        let sig = format!("{tool_name}|{}", short_fp(args_raw));
        self.sigs.push_back(sig);
        while self.sigs.len() > SIG_RING {
            self.sigs.pop_front();
        }
        if let Some(period) = detect_cycle(&self.sigs, CYCLE_REPEATS) {
            return Some(LoopVerdict {
                message: format!(
                    "BLOCKED (loop guard): your last {} tool calls repeat the same {period}-step \
                     cycle with no progress. This call was NOT executed. Break the loop — the \
                     repeated steps aren't changing anything. Re-read the current state, then take \
                     a genuinely different action or explain what is blocking you.",
                    period * CYCLE_REPEATS
                ),
            });
        }
        None
    }

    /// Record an assistant message that ended a turn and report a loop if the
    /// model keeps emitting the same preamble (a).
    pub fn note_assistant_text(&mut self, text: &str) -> Option<LoopVerdict> {
        let norm = normalize_preamble(text);
        if norm.is_empty() {
            return None;
        }
        let same = self.texts.iter().filter(|t| **t == norm).count() + 1;
        self.texts.push_back(norm);
        while self.texts.len() > TEXT_RING {
            self.texts.pop_front();
        }
        if same >= SAME_TEXT_THRESHOLD {
            return Some(LoopVerdict {
                message: format!(
                    "You have now said essentially the same thing {same} times without resolving \
                     it. Repeating the diagnosis is not making progress. STOP restating it: read \
                     the current file and the full error output carefully, then either make ONE \
                     concrete different change via a tool call, or tell the user exactly what is \
                     blocking you and what you need."
                ),
            });
        }
        None
    }
}

/// Detect whether `ring` ends with a period-`p` cycle repeated `>= repeats`
/// times, for p in 2..=4. Returns the period if so. (Period 1 — consecutive
/// identical calls — is left to the core hard guard.)
fn detect_cycle(ring: &VecDeque<String>, repeats: usize) -> Option<usize> {
    let n = ring.len();
    for p in 2..=4usize {
        let needed = p * repeats;
        if n < needed {
            continue;
        }
        let tail: Vec<&String> = ring.iter().skip(n - needed).collect();
        // tail must be the same p-gram repeated `repeats` times.
        let is_cycle = (0..needed).all(|i| tail[i] == tail[i % p]);
        if is_cycle {
            return Some(p);
        }
    }
    None
}

/// Extract `(path, content_fingerprint)` from an edit-tool call, or `None` if
/// this isn't an edit. Handles `apply_patch` (parses the `*** … File:` header)
/// and JSON-shaped `write_file`/`edit_file`/`create_file`.
fn extract_edit_target(tool_name: &str, args_raw: &str) -> Option<(String, u64)> {
    match tool_name {
        "apply_patch" => {
            // args is JSON like {"input":"*** Begin Patch\n*** Update File: p\n..."}.
            let body = serde_json::from_str::<serde_json::Value>(args_raw)
                .ok()
                .and_then(|v| {
                    v.get("input")
                        .or_else(|| v.get("patch"))
                        .and_then(|s| s.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_else(|| args_raw.to_string());
            let path = patch_target_path(&body)?;
            Some((path, short_fp(&body)))
        }
        "write_file" | "edit_file" | "create_file" => {
            let v = serde_json::from_str::<serde_json::Value>(args_raw).ok()?;
            let path = ["path", "file_path", "filename", "file"]
                .iter()
                .find_map(|k| v.get(*k).and_then(|s| s.as_str()))?
                .to_string();
            Some((path, short_fp(args_raw)))
        }
        _ => None,
    }
}

/// Pull the target path out of an apply_patch body's `*** … File:` header.
fn patch_target_path(body: &str) -> Option<String> {
    for line in body.lines() {
        let l = line.trim();
        for marker in [
            "*** Update File:",
            "*** Add File:",
            "*** Delete File:",
            "*** Move to:",
        ] {
            if let Some(rest) = l.strip_prefix(marker) {
                return Some(rest.trim().to_string());
            }
        }
    }
    None
}

/// Stable fingerprint of a string (within a process run).
fn short_fp(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Common English function words dropped from preamble fingerprints (a) so the
/// similarity reflects *content*, not the function-word floor every sentence
/// shares. See docs/spec/local-coder-massaging.md §26.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "is", "are",
    "was", "were", "be", "been", "i", "ill", "im", "let", "me", "now", "next", "then", "this",
    "that", "it", "its", "so", "will", "would", "can", "need", "needs", "going", "go", "lets",
    "have", "has", "do", "does", "my", "we", "us",
];

pub(crate) fn is_stopword(t: &str) -> bool {
    STOPWORDS.contains(&t)
}

/// Naive suffix stem so tense/plural variants collapse (fixing/fixed → fix).
pub(crate) fn stem(t: &str) -> &str {
    for suf in ["ing", "ed", "es", "s"] {
        if t.len() > suf.len() + 2 && t.ends_with(suf) {
            return &t[..t.len() - suf.len()];
        }
    }
    t
}

/// Normalize an assistant message to a content-token preamble: lowercase, keep
/// alphanumeric tokens, drop stopwords, light-stem, and take the first
/// [`PREAMBLE_TOKENS`]. Empty if the message has no content words.
fn normalize_preamble(text: &str) -> String {
    // Strip apostrophes first so contractions collapse to a single token
    // ("I'll" → "ill", "don't" → "dont") instead of leaking a fragment ("ll").
    text.to_lowercase()
        .replace(['\'', '\u{2019}'], "")
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty() && !is_stopword(t))
        .map(stem)
        .take(PREAMBLE_TOKENS)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_file_same_content_edits_trip_after_threshold() {
        let mut d = LoopDetector::new();
        let patch = r#"{"input":"*** Begin Patch\n*** Update File: lambda_handler.py\n-a\n+b\n*** End Patch"}"#;
        assert!(d.note_tool_call("apply_patch", patch).is_none()); // 1
        assert!(d.note_tool_call("apply_patch", patch).is_none()); // 2
        let v = d.note_tool_call("apply_patch", patch); // 3 → trip
        assert!(v.unwrap().message.contains("lambda_handler.py"));
    }

    #[test]
    fn differing_edits_to_same_file_do_not_trip() {
        // Healthy iteration: content changes each time → progress → no trip.
        let mut d = LoopDetector::new();
        for i in 0..6 {
            let patch = format!(
                r#"{{"input":"*** Begin Patch\n*** Update File: h.py\n-a\n+change{i}\n*** End Patch"}}"#
            );
            assert!(
                d.note_tool_call("apply_patch", &patch).is_none(),
                "distinct edit {i} must not trip"
            );
        }
    }

    #[test]
    fn cyclic_tool_pattern_trips() {
        // A 3-step cycle of NON-edit tools (run tests → cat → grep) repeated —
        // the shape that reset the consecutive-identical guard in the session.
        let mut d = LoopDetector::new();
        let cycle = [
            ("exec_command", r#"{"cmd":"pytest"}"#),
            ("shell", r#"{"command":["cat","x"]}"#),
            ("grep_files", r#"{"pattern":"foo"}"#),
        ];
        let mut tripped = None;
        for _ in 0..CYCLE_REPEATS {
            for (name, args) in &cycle {
                if let Some(v) = d.note_tool_call(name, args) {
                    tripped = Some(v);
                }
            }
        }
        assert!(
            tripped.is_some_and(|v| v.message.contains("cycle")),
            "a 3-step cycle repeated 3x must trip"
        );
    }

    #[test]
    fn varied_tool_calls_do_not_trip_cycle() {
        let mut d = LoopDetector::new();
        for i in 0..10 {
            assert!(
                d.note_tool_call("exec_command", &format!(r#"{{"cmd":"step{i}"}}"#))
                    .is_none()
            );
        }
    }

    #[test]
    fn repeated_preamble_trips_despite_wording_variation() {
        let mut d = LoopDetector::new();
        // Same content words, different function words / tense → must collapse.
        assert!(
            d.note_assistant_text(
                "I see the issue — the HTTPError handling causes recursion. Let me fix the handler."
            )
            .is_none()
        );
        assert!(
            d.note_assistant_text(
                "I see an issue: HTTPError handling is causing recursion. I'll fix the handler now."
            )
            .is_none()
        );
        let v = d.note_assistant_text(
            "Now I see the issue, HTTPError handling caused recursion — let me fix the handler.",
        );
        assert!(v.is_some(), "third near-identical preamble must trip");
    }

    #[test]
    fn distinct_assistant_messages_do_not_trip() {
        let mut d = LoopDetector::new();
        assert!(d.note_assistant_text("Reading the handler file.").is_none());
        assert!(
            d.note_assistant_text("Running the unit tests now.")
                .is_none()
        );
        assert!(
            d.note_assistant_text("Tests pass; writing the README.")
                .is_none()
        );
    }
}
