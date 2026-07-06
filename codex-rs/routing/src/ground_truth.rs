//! Ground-truth provider — the single place the reasoned-guidance layer gets FRESH
//! facts to reason over.
//!
//! The hard lesson from live sessions: a weak reasoner handed the model's own claims,
//! or a *clean* probe dressed up as a diagnosis, hallucinates — it once told the coder
//! to "add an X-API-Key header" for what was actually a runtime `TypeError`. So every
//! reasoned intervention pulls its grounding from here and nowhere else: the actual
//! files re-read from disk, the actual lint/syntax probe, and the actual repeated call
//! plus the output it keeps producing — never the transcript's stale echoes, and never
//! a clean probe (which is the *absence* of signal, not signal).
//!
//! PURE + bounded: file reads are capped per file so a huge file can't blow the
//! reasoner's context, and the whole bundle renders to one compact block.

use std::path::{Path, PathBuf};

/// Default per-file cap for a snapshot read. Enough to show a small handler/test file
/// whole; large files are truncated (the reasoner is told so).
pub const DEFAULT_FILE_CAP: usize = 8 * 1024;

/// A fresh, bounded read of one file as it exists on disk RIGHT NOW.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileSnapshot {
    pub path: String,
    /// Contents, truncated to the caller's byte cap when `truncated` is set.
    pub content: String,
    /// True when the file exists and was read (even if empty). False → missing/unreadable.
    pub exists: bool,
    /// True when `content` was cut to the byte cap.
    pub truncated: bool,
}

/// The loop's dominant repeated call and the ACTUAL output it keeps producing. For an
/// ACTIVE repetition this output is current (the model just produced it again), so it
/// is legitimate grounding — the stale-signal caveat only applies to a one-off past
/// failure a later action may already have fixed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepeatedAction {
    pub command: String,
    pub output: String,
    pub count: usize,
}

/// The assembled fresh-truth bundle a reasoned intervention reasons over.
#[derive(Debug, Clone, Default)]
pub struct GroundTruth {
    pub files: Vec<FileSnapshot>,
    /// Dirty-only lint/syntax digest (`file:line`). `None` when clean or no signal —
    /// a clean probe is NOT signal and must never be handed to the reasoner as one.
    pub lint_digest: Option<String>,
    pub repeated: Option<RepeatedAction>,
}

impl GroundTruth {
    /// Is there ANY real signal to reason over? When false the caller MUST NOT invoke
    /// the reasoner — that is exactly when it hallucinates a cause from nothing.
    pub fn has_signal(&self) -> bool {
        self.lint_digest.is_some()
            || self.repeated.is_some()
            || self.files.iter().any(|f| f.exists)
    }

    /// Compact, reasoner-ready rendering. Bounded by construction (files were capped at
    /// read time). Order: the repeated failure first (the loop's core fact), then the
    /// fresh lint, then the live file contents.
    pub fn render(&self) -> String {
        let mut out = Vec::new();
        if let Some(r) = &self.repeated {
            out.push(format!(
                "REPEATED ACTION (ran {}× with the SAME result — doing it again is a no-op):\n$ {}\n{}",
                r.count,
                r.command.trim(),
                r.output.trim()
            ));
        }
        if let Some(l) = &self.lint_digest {
            out.push(format!("LINT/SYNTAX (fresh probe of the workspace):\n{}", l.trim()));
        }
        for f in &self.files {
            if f.exists {
                let note = if f.truncated { " (truncated)" } else { "" };
                out.push(format!("FILE {}{} — as it is on disk NOW:\n{}", f.path, note, f.content));
            } else {
                out.push(format!("FILE {} — does NOT exist on disk", f.path));
            }
        }
        out.join("\n\n")
    }
}

/// Resolve `path` against `root` (cwd), accepting absolute or relative. `Path::join`
/// returns `path` unchanged when it is absolute, so this matches the existing
/// apply_patch / file-length resolution exactly.
fn resolve(root: &Path, path: &str) -> PathBuf {
    root.join(path)
}

/// Re-read the given files from disk RIGHT NOW, each capped at `max_bytes`. A missing or
/// unreadable file comes back with `exists=false` (not an error — the reasoner should
/// know the file isn't there). Never reads the transcript; always the live file.
pub fn file_snapshot(root: &Path, paths: &[String], max_bytes: usize) -> Vec<FileSnapshot> {
    paths
        .iter()
        .map(|p| match std::fs::read(resolve(root, p)) {
            Ok(bytes) => {
                let truncated = bytes.len() > max_bytes;
                let end = if truncated { max_bytes } else { bytes.len() };
                FileSnapshot {
                    path: p.clone(),
                    content: String::from_utf8_lossy(&bytes[..end]).into_owned(),
                    exists: true,
                    truncated,
                }
            }
            Err(_) => FileSnapshot {
                path: p.clone(),
                content: String::new(),
                exists: false,
                truncated: false,
            },
        })
        .collect()
}

/// Byte length of a file resolved against `root`. A metadata stat — no read — and
/// `None` when it can't be stat'd. (Replaces the ad-hoc `active_turn_file_len`.)
pub fn file_len(root: &Path, path: &str) -> Option<usize> {
    std::fs::metadata(resolve(root, path)).ok().map(|m| m.len() as usize)
}

/// Fresh lint/syntax probe of the workspace, as a DIRTY-ONLY digest. Returns `Some`
/// only when the probe actually found a `file:line` problem; a clean workspace returns
/// `None` — encoding the "clean probe is not signal" rule in ONE place so callers can't
/// accidentally hand a clean digest to the reasoner.
pub fn lint_digest(root: &Path) -> Option<String> {
    let report = crate::linter_probe::run_linter_probe(root);
    if report.is_clean() {
        None
    } else {
        Some(report.probe_digest())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!("ground_truth_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn snapshot_reads_live_file_and_reports_missing() {
        let root = tmp();
        std::fs::write(root.join("a.py"), "print('hi')\n").unwrap();
        let snaps = file_snapshot(
            &root,
            &["a.py".into(), "gone.py".into()],
            DEFAULT_FILE_CAP,
        );
        assert_eq!(snaps[0].path, "a.py");
        assert!(snaps[0].exists);
        assert_eq!(snaps[0].content, "print('hi')\n");
        assert!(!snaps[0].truncated);
        assert!(!snaps[1].exists, "missing file → exists=false, not an error");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn snapshot_is_bounded() {
        let root = tmp();
        std::fs::write(root.join("big.txt"), "x".repeat(10_000)).unwrap();
        let snaps = file_snapshot(&root, &["big.txt".into()], 100);
        assert!(snaps[0].truncated);
        assert_eq!(snaps[0].content.len(), 100);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn absolute_path_resolves_regardless_of_root() {
        let root = tmp();
        let abs = root.join("abs.py");
        std::fs::write(&abs, "ok").unwrap();
        // A different (bogus) root must not matter for an absolute path.
        let snaps = file_snapshot(
            Path::new("/nonexistent-root"),
            &[abs.to_string_lossy().into_owned()],
            DEFAULT_FILE_CAP,
        );
        assert!(snaps[0].exists);
        assert_eq!(snaps[0].content, "ok");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn has_signal_is_false_on_empty_and_true_with_any_fact() {
        assert!(!GroundTruth::default().has_signal(), "nothing gathered → no signal");
        let g = GroundTruth {
            repeated: Some(RepeatedAction {
                command: "cat /dir".into(),
                output: "Is a directory".into(),
                count: 9,
            }),
            ..Default::default()
        };
        assert!(g.has_signal());
    }

    #[test]
    fn render_leads_with_repeated_action_then_lint_then_files() {
        let g = GroundTruth {
            files: vec![FileSnapshot {
                path: "h.py".into(),
                content: "def h(): pass".into(),
                exists: true,
                truncated: false,
            }],
            lint_digest: Some("h.py:1: syntax error".into()),
            repeated: Some(RepeatedAction {
                command: "cat '/work'".into(),
                output: "Is a directory".into(),
                count: 9,
            }),
        };
        let r = g.render();
        let ri = r.find("REPEATED ACTION").unwrap();
        let li = r.find("LINT/SYNTAX").unwrap();
        let fi = r.find("FILE h.py").unwrap();
        assert!(ri < li && li < fi, "order must be repeated → lint → files:\n{r}");
        assert!(r.contains("9×") && r.contains("Is a directory"));
    }
}
