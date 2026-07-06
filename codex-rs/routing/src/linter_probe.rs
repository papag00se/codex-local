//! Language-aware linter probe — the first concrete "Probe".
//!
//! Shephard runs a project's own deterministic checker (a read-only subprocess)
//! and feeds the exact `file:line: message` back to the model as ground truth.
//! This is what the eval showed the weaker local models could not generate for
//! themselves: they'd rewrite a whole file many times chasing a syntax error they
//! couldn't localize (Ornith's `IndentationError line 95`, rewritten 9×), because
//! `pytest`'s "1 collection error" doesn't point at the line. `py_compile` does.
//!
//! Design constraints (from the eval + review):
//! - **Detection is DISK-based, never prompt-based.** We lint whatever the model
//!   actually wrote. We do NOT infer an "intended language" from the task text —
//!   prompts often omit the language or state it negatively ("don't use Python"),
//!   so keyword-matching the prompt is wrong. Whether the chosen language honors
//!   the task is a *reasoning* judgment left to the LLM completion-verifier; this
//!   probe only answers the deterministic question "is the code valid?".
//! - **Graceful degradation to always-available floors** (`py_compile`, `node
//!   --check` ship with the runtime). Escalate to `pyflakes`/`ruff` only if present;
//!   never require an install.
//! - **Errors, not style.** Syntax + real-bug lints (undefined names), never
//!   line-length nags — style noise would just feed the rewrite churn.
//! - **Read-only / idempotent.** A checker never mutates the workspace.

use std::path::{Path, PathBuf};
use std::process::Command;

/// One checker's result for one language present in the project.
#[derive(Debug, Clone)]
pub struct LinterFinding {
    /// e.g. "python", "javascript".
    pub language: String,
    /// The checker that produced this, e.g. "py_compile", "pyflakes", "node --check".
    pub tool: String,
    pub passed: bool,
    /// `file:line: message` text (empty when `passed`). Fed to the model verbatim.
    pub errors: String,
}

/// The probe's verdict across every language it found + could check.
#[derive(Debug, Clone, Default)]
pub struct LinterReport {
    pub findings: Vec<LinterFinding>,
    /// Languages found on disk but skipped because their checker binary was absent
    /// (so a green report can't be mistaken for "everything checked").
    pub skipped: Vec<String>,
}

impl LinterReport {
    /// True only if every checker that ran passed. An empty report (no source /
    /// no checker available) is `true` — the probe has no deterministic objection,
    /// so it must not block completion on its own.
    pub fn is_clean(&self) -> bool {
        self.findings.iter().all(|f| f.passed)
    }

    pub fn failing(&self) -> impl Iterator<Item = &LinterFinding> {
        self.findings.iter().filter(|f| !f.passed)
    }

    /// A ground-truth digest for the REASONER. Unlike `nudge_text` (which is `None`
    /// when clean, because it's a coder-facing directive), this ALWAYS returns text
    /// describing the probe verdict, so the reasoner can route on it:
    ///   - errors  → the exact `file:line` messages to fix;
    ///   - clean   → an explicit "not a syntax error" so it looks at logic/imports/
    ///               a stub shadowing real code, or whether the task is done;
    ///   - nothing → says nothing ran, so a green result isn't over-trusted.
    pub fn probe_digest(&self) -> String {
        let failing: Vec<&LinterFinding> = self.failing().collect();
        if !failing.is_empty() {
            let mut out =
                String::from("Errors — the workspace does NOT pass its syntax/linter check:");
            for f in failing {
                out.push_str(&format!("\n• {} ({}):\n{}", f.language, f.tool, f.errors.trim()));
            }
            out
        } else if self.findings.is_empty() {
            "No checker ran (no recognized source files on disk, or no checker binary installed) \
             — treat this as no signal."
                .to_string()
        } else {
            "Clean — every source file passes its syntax/linter check. The defect is NOT a syntax \
             error; look at logic, a wrong or missing import, a stub/placeholder shadowing real \
             code, or whether the task is already satisfied."
                .to_string()
        }
    }

    /// The grounding message to inject when the probe found broken code, or `None`
    /// when clean. Deliberately imperative and localized — the whole point is to
    /// hand the model the exact line so it stops rewriting the whole file.
    pub fn nudge_text(&self) -> Option<String> {
        let failing: Vec<&LinterFinding> = self.failing().collect();
        if failing.is_empty() {
            return None;
        }
        let mut out = String::from(
            "[GROUND TRUTH — your code does not pass its own checker] \
             Fix these exact errors before continuing. Do NOT rewrite the whole file — \
             go to the reported line and fix only what it points to:\n",
        );
        for f in failing {
            out.push_str(&format!(
                "\n• {} ({}):\n{}\n",
                f.language,
                f.tool,
                f.errors.trim()
            ));
        }
        Some(out)
    }
}

/// Directories whose contents are never the model's own source under test.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".codex-multi",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".mypy_cache",
    ".ruff_cache",
    "target",
];

/// Run the probe over `project_dir` and return a per-language report.
///
/// Blocking (spawns checker subprocesses); call from `spawn_blocking` in async
/// contexts. Fast in practice — `py_compile` over a handful of files is sub-second.
pub fn run_linter_probe(project_dir: &Path) -> LinterReport {
    let mut report = LinterReport::default();

    let py_files = collect_files(project_dir, &["py"]);
    if !py_files.is_empty() {
        check_python(&py_files, &mut report);
    }

    let js_files = collect_files(project_dir, &["js", "mjs", "cjs"]);
    if !js_files.is_empty() {
        check_javascript(&js_files, &mut report);
    }

    report
}

/// Recursively collect files with any of `exts`, skipping vendor/build dirs and
/// hidden directories. Returns paths relative-friendly (absolute, but the checker
/// output echoes whatever we pass, so we pass paths as-is).
fn collect_files(root: &Path, exts: &[&str]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_files_inner(root, exts, &mut out);
    out.sort();
    out
}

fn collect_files_inner(dir: &Path, exts: &[&str], out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name.starts_with('.') || SKIP_DIRS.contains(&name.as_ref()) {
                continue;
            }
            collect_files_inner(&path, exts, out);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if exts.contains(&ext) {
                out.push(path);
            }
        }
    }
}

/// Whether a command exists / is runnable (probe by attempting `--version` or the
/// real call and treating a NotFound error as "absent").
fn command_missing(err: &std::io::Error) -> bool {
    err.kind() == std::io::ErrorKind::NotFound
}

fn check_python(files: &[PathBuf], report: &mut LinterReport) {
    // Floor: syntax via py_compile (stdlib, always present with python3).
    let mut cmd = Command::new("python3");
    cmd.arg("-m").arg("py_compile");
    for f in files {
        cmd.arg(f);
    }
    match cmd.output() {
        Ok(out) if out.status.success() => {
            report.findings.push(LinterFinding {
                language: "python".into(),
                tool: "py_compile".into(),
                passed: true,
                errors: String::new(),
            });
            // Escalate to pyflakes (undefined names, bad imports) only if installed.
            escalate_pyflakes(files, report);
        }
        Ok(out) => {
            // Syntax broken — this is the localized error the model needs. Don't
            // bother escalating; parse must pass first.
            report.findings.push(LinterFinding {
                language: "python".into(),
                tool: "py_compile".into(),
                passed: false,
                errors: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        Err(e) if command_missing(&e) => report.skipped.push("python".into()),
        Err(_) => report.skipped.push("python".into()),
    }
}

/// pyflakes catches the real-bug class beyond syntax (undefined names, unused
/// imports that signal a typo). Skipped silently if not installed.
fn escalate_pyflakes(files: &[PathBuf], report: &mut LinterReport) {
    let mut cmd = Command::new("python3");
    cmd.arg("-m").arg("pyflakes");
    for f in files {
        cmd.arg(f);
    }
    match cmd.output() {
        Ok(out) => {
            let combined = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stderr),
                String::from_utf8_lossy(&out.stdout)
            );
            // "No module named pyflakes" => not installed; treat as absent, not a fail.
            if combined.contains("No module named pyflakes") {
                return;
            }
            let passed = out.status.success();
            report.findings.push(LinterFinding {
                language: "python".into(),
                tool: "pyflakes".into(),
                passed,
                errors: if passed {
                    String::new()
                } else {
                    combined.trim().to_owned()
                },
            });
        }
        Err(_) => {}
    }
}

fn check_javascript(files: &[PathBuf], report: &mut LinterReport) {
    // Floor: `node --check` (syntax). It only accepts one file at a time.
    let mut any_ran = false;
    let mut errors = String::new();
    let mut failed = false;
    for f in files {
        match Command::new("node").arg("--check").arg(f).output() {
            Ok(out) => {
                any_ran = true;
                if !out.status.success() {
                    failed = true;
                    errors.push_str(&String::from_utf8_lossy(&out.stderr));
                    errors.push('\n');
                }
            }
            Err(e) if command_missing(&e) => {
                report.skipped.push("javascript".into());
                return;
            }
            Err(_) => {}
        }
    }
    if any_ran {
        report.findings.push(LinterFinding {
            language: "javascript".into(),
            tool: "node --check".into(),
            passed: !failed,
            errors: errors.trim().to_owned(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "linter_probe_test_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn clean_python_passes() {
        let d = tmp();
        std::fs::write(d.join("ok.py"), "def f():\n    return 1\n").unwrap();
        let r = run_linter_probe(&d);
        assert!(r.is_clean(), "clean python should pass: {:?}", r.findings);
        assert!(r.nudge_text().is_none());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn broken_python_syntax_is_caught_and_localized() {
        let d = tmp();
        // IndentationError — the exact failure that trapped Ornith-ON.
        std::fs::write(
            d.join("bad.py"),
            "def f():\n    x = 1\n   try:\n        pass\n",
        )
        .unwrap();
        let r = run_linter_probe(&d);
        assert!(!r.is_clean(), "broken python must fail");
        let nudge = r.nudge_text().expect("should produce a nudge");
        assert!(
            nudge.contains("bad.py"),
            "nudge must name the file: {nudge}"
        );
        assert!(
            nudge.to_lowercase().contains("line"),
            "nudge must localize: {nudge}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn vendor_dirs_are_skipped() {
        let d = tmp();
        std::fs::write(d.join("ok.py"), "x = 1\n").unwrap();
        std::fs::create_dir_all(d.join("node_modules")).unwrap();
        std::fs::write(d.join("node_modules").join("broken.py"), "def (:\n").unwrap();
        let r = run_linter_probe(&d);
        assert!(r.is_clean(), "must not lint node_modules: {:?}", r.findings);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn empty_project_is_clean_and_silent() {
        let d = tmp();
        let r = run_linter_probe(&d);
        assert!(r.is_clean());
        assert!(r.nudge_text().is_none());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn probe_digest_routes_the_reasoner_on_the_verdict() {
        // Dirty → names the file so the reasoner can target it.
        let d = tmp();
        std::fs::write(d.join("bad.py"), "def f(:\n").unwrap();
        let dirty = run_linter_probe(&d).probe_digest();
        assert!(dirty.contains("bad.py"), "dirty digest names the file: {dirty}");
        assert!(dirty.to_lowercase().contains("error"));
        std::fs::remove_dir_all(&d).ok();

        // Clean-with-source → explicit "not a syntax error" so the reasoner looks
        // at logic/imports (the Qwythos placeholder-shadowing case).
        let d = tmp();
        std::fs::write(d.join("ok.py"), "x = 1\n").unwrap();
        let clean = run_linter_probe(&d).probe_digest();
        assert!(clean.to_lowercase().contains("not a syntax error"), "clean digest: {clean}");
        std::fs::remove_dir_all(&d).ok();

        // Nothing to check → "no signal", never mistaken for a green pass.
        let empty = LinterReport::default().probe_digest();
        assert!(empty.to_lowercase().contains("no signal"), "empty digest: {empty}");
    }
}
