//! Bounded probe execution + orchestration — ties discovery → run → parse together.
//!
//! Runs a small number of the highest-ranked SAFE probes with a hard timeout,
//! captures output without blocking on full pipes, and hands each result to
//! [`crate::probe_parse`]. The end product is a compact report the small model can
//! act on: which probes ran, what failed, and the `file:line` to fix.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::probe_discovery::{ProbeCandidate, discover, project_types};
use crate::probe_parse::{ProbeResult, parse_output};

#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub project_type: Vec<String>,
    pub selected: Vec<ProbeCandidate>,
    pub results: Vec<ProbeResult>,
}

/// Default: run at most 3 probes, 120s each. Callers can override.
pub fn run_probes(root: &std::path::Path) -> ProbeReport {
    run_probes_with(root, 3, Duration::from_secs(120))
}

pub fn run_probes_with(root: &std::path::Path, max: usize, timeout: Duration) -> ProbeReport {
    let candidates = discover(root); // already safe + ranked best-first
    let selected: Vec<ProbeCandidate> = candidates.into_iter().take(max).collect();
    let results = selected.iter().map(|c| run_candidate(c, timeout)).collect();
    ProbeReport {
        project_type: project_types(root).into_iter().map(String::from).collect(),
        selected,
        results,
    }
}

/// Probe selection for the COMPLETION gate specifically. `run_probes_with(_, 1, _)`
/// runs only the single best-ranked probe — but discovery ranks typecheck/lint/build
/// ABOVE tests, so a workspace whose TESTS fail gets green-lit by a passing lint (a
/// false completion sailed through exactly this way). At a "done" claim we must
/// actually run the tests: run the top-ranked probe AND the top-ranked Test probe
/// (deduped by command). Each bounded by `timeout`.
pub fn run_completion_probes(root: &std::path::Path, timeout: Duration) -> ProbeReport {
    let candidates = discover(root); // ranked best-first, already safe
    let mut selected: Vec<ProbeCandidate> = Vec::new();
    if let Some(top) = candidates.first() {
        selected.push(top.clone());
    }
    if let Some(test) = candidates
        .iter()
        .find(|c| c.kind == crate::probe_discovery::ProbeKind::Test)
    {
        if !selected.iter().any(|c| c.command == test.command) {
            selected.push(test.clone());
        }
    }
    let results = selected.iter().map(|c| run_candidate(c, timeout)).collect();
    ProbeReport {
        project_type: project_types(root).into_iter().map(String::from).collect(),
        selected,
        results,
    }
}

/// Build the completion-gate re-prompt from a probe run + the always-available
/// syntax floor ([`crate::linter_probe`]). Returns `None` when everything the
/// probes could check is clean (so completion is allowed).
///
/// Precedence: the syntax floor first (it localizes parse errors to the exact
/// line — the single most repairable signal), then any ecosystem probe that
/// produced structured `file:line` findings. A probe that couldn't launch (tool
/// absent) or merely timed out never blocks — we only block on a real diagnosis.
pub fn completion_block_nudge(
    report: &ProbeReport,
    floor: &crate::linter_probe::LinterReport,
) -> Option<String> {
    if !floor.is_clean() {
        return floor.nudge_text();
    }
    let mut lines = Vec::new();
    for r in &report.results {
        if r.findings.is_empty() {
            continue;
        }
        lines.push(format!("$ {} — {}", r.command, r.summary));
        for f in r.findings.iter().take(5) {
            let loc = match f.line {
                Some(l) => format!("{}:{}", f.file, l),
                None => f.file.clone(),
            };
            lines.push(format!("  • {loc}: {}", f.message));
        }
    }
    if lines.is_empty() {
        return None;
    }
    Some(format!(
        "[GROUND TRUTH — the repo's own checks fail] You are not done yet. Fix these exact \
         problems; go to the reported line, do not rewrite whole files:\n{}",
        lines.join("\n")
    ))
}

/// Human-readable digest of a completion probe run, for the COMPLETION CRITIC.
///
/// Unlike [`completion_block_nudge`] — which surfaces ONLY structured `file:line`
/// findings and is silent otherwise — this reports WHAT RAN and its raw outcome.
/// That exposes the cases the deterministic gate cannot act on and which look
/// identical to "clean": a test suite that FAILED TO RUN (import error, nothing
/// collected), TIMED OUT, or couldn't launch (runner not installed). The critic
/// needs this to tell "tests pass" apart from "tests never ran" — the exact hole a
/// task whose tests don't execute slips through today.
pub fn completion_probe_digest(
    report: &ProbeReport,
    floor: &crate::linter_probe::LinterReport,
) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "SYNTAX FLOOR: {}",
        if floor.is_clean() {
            "clean (files parse)".to_string()
        } else {
            floor
                .nudge_text()
                .unwrap_or_else(|| "parse/syntax issues found".to_string())
        }
    ));
    if report.results.is_empty() {
        lines.push(
            "PROBES: none ran — no lint/test command was discovered for this project.".to_string(),
        );
    } else {
        for r in &report.results {
            let exit = match r.exit_code {
                Some(0) => "exit 0 (ran clean)".to_string(),
                Some(c) => format!("exit {c}"),
                None => "did NOT launch (tool missing?)".to_string(),
            };
            lines.push(format!(
                "$ {} — {} — {} — {} structured finding(s)",
                r.command,
                exit,
                r.summary,
                r.findings.len()
            ));
        }
    }
    lines.join("\n")
}

/// Run one candidate with a hard timeout; capture stdout/stderr via drain threads so
/// a chatty tool can't deadlock on a full pipe. Never mutates the workspace (the
/// candidate was vetted safe upstream).
pub fn run_candidate(c: &ProbeCandidate, timeout: Duration) -> ProbeResult {
    let joined = c.command.join(" ");
    if c.command.is_empty() {
        return err_result(&joined, "empty command");
    }
    let mut cmd = Command::new(&c.command[0]);
    cmd.args(&c.command[1..])
        .current_dir(&c.working_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(ch) => ch,
        Err(e) => {
            return err_result(
                &joined,
                &format!("failed to launch ({e}) — tool not installed?"),
            );
        }
    };

    let mut so = child.stdout.take();
    let mut se = child.stderr.take();
    let th_o = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(p) = so.as_mut() {
            let _ = p.read_to_string(&mut s);
        }
        s
    });
    let th_e = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(p) = se.as_mut() {
            let _ = p.read_to_string(&mut s);
        }
        s
    });

    let start = Instant::now();
    let mut timed_out = false;
    let exit: Option<i32> = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status.code(),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break None;
                }
                std::thread::sleep(Duration::from_millis(40));
            }
            Err(_) => break None,
        }
    };

    let stdout = th_o.join().unwrap_or_default();
    let stderr = th_e.join().unwrap_or_default();
    let family = family_of(&c.command);
    let mut r = parse_output(&joined, family, exit, &stdout, &stderr);
    if timed_out {
        r.summary = format!(
            "TIMEOUT after {}s — probe did not finish (consider a narrower target)",
            timeout.as_secs()
        );
    }
    r
}

fn err_result(cmd: &str, msg: &str) -> ProbeResult {
    ProbeResult {
        command: cmd.to_string(),
        exit_code: None,
        summary: msg.to_string(),
        findings: Vec::new(),
    }
}

/// Pick the parse family from the command tokens (the tool actually invoked).
fn family_of(command: &[String]) -> &'static str {
    let has = |t: &str| command.iter().any(|x| x == t);
    if has("cargo") {
        "cargo"
    } else if has("tsc") || has("vue-tsc") {
        "tsc"
    } else if has("eslint") {
        "eslint"
    } else if has("pytest") || (has("python") && has("pytest")) || (has("python3") && has("pytest"))
    {
        "pytest"
    } else if has("mypy") {
        "mypy"
    } else {
        "" // generic scanners (ruff/flake8/go/gcc/etc.)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe_discovery::{ProbeCost, ProbeKind};
    use std::path::PathBuf;

    fn synth(cmd: &[&str]) -> ProbeCandidate {
        ProbeCandidate {
            kind: ProbeKind::BuildCheck,
            command: cmd.iter().map(|s| s.to_string()).collect(),
            working_dir: std::env::temp_dir(),
            confidence: 90,
            expected_value: 80,
            cost: ProbeCost::Cheap,
            mutates_code: false,
            may_hang: false,
            may_need_services: false,
            reason: "test".into(),
        }
    }

    #[test]
    fn runs_and_captures_exit_zero() {
        let c = synth(&["python3", "-c", "print('ok')"]);
        let r = run_candidate(&c, Duration::from_secs(10));
        assert_eq!(r.exit_code, Some(0));
        assert_eq!(r.summary, "no problems reported");
    }

    #[test]
    fn captures_stderr_and_nonzero_exit() {
        // emit a generic file:line diagnostic then fail
        let c = synth(&[
            "python3",
            "-c",
            "import sys; sys.stderr.write('src/x.py:9: error: boom\\n'); sys.exit(1)",
        ]);
        let r = run_candidate(&c, Duration::from_secs(10));
        assert_eq!(r.exit_code, Some(1));
        assert_eq!(r.findings.len(), 1, "{:?}", r.findings);
        assert_eq!(r.findings[0].file, "src/x.py");
        assert_eq!(r.findings[0].line, Some(9));
    }

    #[test]
    fn enforces_timeout() {
        let c = synth(&["python3", "-c", "import time; time.sleep(30)"]);
        let r = run_candidate(&c, Duration::from_millis(400));
        assert!(r.summary.starts_with("TIMEOUT"), "{}", r.summary);
    }

    #[test]
    fn missing_tool_is_reported_not_panicked() {
        let c = synth(&["definitely-not-a-real-binary-xyz", "check"]);
        let r = run_candidate(&c, Duration::from_secs(5));
        assert_eq!(r.exit_code, None);
        assert!(r.summary.contains("failed to launch"), "{}", r.summary);
    }

    #[test]
    fn family_selection() {
        assert_eq!(family_of(&["cargo".into(), "check".into()]), "cargo");
        assert_eq!(
            family_of(&[
                "pnpm".into(),
                "exec".into(),
                "tsc".into(),
                "--noEmit".into()
            ]),
            "tsc"
        );
        assert_eq!(family_of(&["ruff".into(), "check".into(), ".".into()]), "");
    }

    #[test]
    fn end_to_end_on_a_python_repo() {
        // a repo whose "test" is a trivial passing python invocation
        let mut dir = std::env::temp_dir();
        dir.push(format!("probe_run_e2e_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("go.mod"), "module x\n").unwrap();
        let report = run_probes_with(&dir, 1, Duration::from_secs(10));
        assert_eq!(report.project_type, vec!["go".to_string()]);
        assert_eq!(report.selected.len(), 1);
        // (go may or may not be installed; we only assert the orchestration shape)
        assert_eq!(report.results.len(), 1);
        let _ = PathBuf::from(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn digest_distinguishes_ran_clean_from_did_not_run() {
        use crate::linter_probe::LinterReport;
        let clean_floor = LinterReport {
            findings: vec![],
            skipped: vec![],
        };
        // A suite that RAN GREEN vs. one that FAILED TO RUN (import error) — both carry
        // ZERO structured findings, so the block gate can't tell them apart; the digest
        // must.
        let report = ProbeReport {
            project_type: vec!["python".into()],
            selected: vec![],
            results: vec![
                ProbeResult {
                    command: "pytest -q".into(),
                    exit_code: Some(0),
                    summary: "3 passed".into(),
                    findings: vec![],
                },
                ProbeResult {
                    command: "python -c import".into(),
                    exit_code: Some(1),
                    summary: "ModuleNotFoundError: No module named 'requests'".into(),
                    findings: vec![],
                },
            ],
        };
        let digest = completion_probe_digest(&report, &clean_floor);
        assert!(digest.contains("SYNTAX FLOOR: clean"));
        assert!(digest.contains("exit 0 (ran clean)"));
        assert!(digest.contains("ModuleNotFoundError"));
        assert!(digest.contains("exit 1"));

        // No probe discovered at all is stated explicitly, not silently omitted.
        let empty = ProbeReport {
            project_type: vec![],
            selected: vec![],
            results: vec![],
        };
        assert!(completion_probe_digest(&empty, &clean_floor).contains("none ran"));
    }
}
