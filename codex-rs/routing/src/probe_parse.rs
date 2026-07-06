//! Turn raw probe output into targeted repair hints for the small model.
//!
//! The last mile of the probe system: after cria-shepherd runs a diagnostic
//! command, this parses stdout/stderr into `file:line: message` [`Finding`]s and a
//! one-line [`ProbeResult::summary`], so the model gets "fix src/probes.rs:42:
//! missing field" instead of a wall of tool output.
//!
//! Parsers are conservative line-scanners (no regex dep) covering the formats the
//! eval + common ecosystems actually emit: rustc/cargo (`--> file:line:col`), tsc
//! (`file(line,col): error`), ESLint stylish (file header + indented `line:col`),
//! pytest, and the generic `file:line[:col]: message` shared by ruff / flake8 /
//! mypy / go vet / py_compile-style output.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub file: String,
    pub line: Option<u32>,
    pub col: Option<u32>,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub command: String,
    pub exit_code: Option<i32>,
    pub summary: String,
    pub findings: Vec<Finding>,
}

/// Parse a probe's combined output into findings + a summary. `family` is a hint
/// ("cargo", "tsc", "eslint", "pytest", …); when unknown, the generic scanners run.
pub fn parse_output(
    command: &str,
    family: &str,
    exit_code: Option<i32>,
    stdout: &str,
    stderr: &str,
) -> ProbeResult {
    let combined = format!("{stdout}\n{stderr}");
    let mut findings = match family {
        "cargo" => parse_rustc(&combined),
        "tsc" => parse_tsc(&combined),
        "eslint" => parse_eslint(&combined),
        "pytest" | "unittest" => parse_pytest(&combined),
        _ => Vec::new(),
    };
    // Always also run the generic scanners and merge (many tools share the format,
    // and cargo output also contains `-->` handled above). Dedup by (file,line,msg).
    if findings.is_empty() {
        findings = parse_rustc(&combined);
    }
    if findings.is_empty() {
        findings = parse_generic(&combined);
    }
    dedup(&mut findings);

    let summary = summarize(&findings, exit_code, &combined);
    ProbeResult {
        command: command.to_string(),
        exit_code,
        summary,
        findings,
    }
}

fn dedup(v: &mut Vec<Finding>) {
    let mut seen = std::collections::HashSet::new();
    v.retain(|f| seen.insert((f.file.clone(), f.line, f.message.clone())));
}

fn summarize(findings: &[Finding], exit: Option<i32>, combined: &str) -> String {
    if let Some(f) = findings.first() {
        let loc = match (f.line, &f.file) {
            (Some(l), file) if !file.is_empty() => format!("{file}:{l}"),
            (_, file) if !file.is_empty() => file.clone(),
            _ => "?".into(),
        };
        let more = if findings.len() > 1 {
            format!(" (+{} more)", findings.len() - 1)
        } else {
            String::new()
        };
        return format!("{loc}: {}{more}", truncate(&f.message, 100));
    }
    match exit {
        Some(0) | None => "no problems reported".into(),
        Some(code) => {
            // no structured findings, but non-zero exit — surface the last error-ish line
            let line = combined
                .lines()
                .rev()
                .find(|l| {
                    let ll = l.to_ascii_lowercase();
                    (ll.contains("error") || ll.contains("failed") || ll.contains("fatal"))
                        && !l.trim().is_empty()
                })
                .unwrap_or("")
                .trim();
            if line.is_empty() {
                format!("exited {code} with no parseable diagnostics")
            } else {
                format!("exited {code}: {}", truncate(line, 120))
            }
        }
    }
}

// ---- rustc / cargo: `error[E..]: msg` then `  --> file:line:col` ----
fn parse_rustc(s: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    let lines: Vec<&str> = s.lines().collect();
    let mut last_msg: Option<String> = None;
    for l in &lines {
        let t = l.trim_start();
        if let Some(rest) = t
            .strip_prefix("error")
            .or_else(|| t.strip_prefix("warning"))
        {
            // `error[E0433]: message` or `error: message`
            let msg =
                rest.trim_start_matches(|c: char| c == '[' || c.is_alphanumeric() || c == ']');
            let msg = msg.trim_start_matches(':').trim();
            if !msg.is_empty() {
                last_msg = Some(msg.to_string());
            }
        } else if let Some(loc) = t.strip_prefix("--> ") {
            if let Some((file, line, col)) = split_loc(loc.trim()) {
                out.push(Finding {
                    file,
                    line,
                    col,
                    message: last_msg.clone().unwrap_or_else(|| "compile error".into()),
                });
                last_msg = None;
            }
        }
    }
    out
}

// ---- tsc: `src/foo.ts(42,5): error TS2322: message` ----
fn parse_tsc(s: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    for l in s.lines() {
        if let Some(paren) = l.find('(') {
            let file = l[..paren].trim();
            if file.is_empty() || !l[paren..].contains("): ") {
                continue;
            }
            let rest = &l[paren + 1..];
            if let Some(close) = rest.find(')') {
                let nums = &rest[..close];
                let msg_part = rest[close + 1..].trim_start_matches(':').trim();
                let (line, col) = split_line_col(nums);
                if line.is_some() {
                    out.push(Finding {
                        file: file.to_string(),
                        line,
                        col,
                        message: msg_part.to_string(),
                    });
                }
            }
        }
    }
    out
}

// ---- ESLint stylish: a file path line, then `  42:5  error  msg  rule` ----
fn parse_eslint(s: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    let mut cur_file: Option<String> = None;
    for l in s.lines() {
        let t = l.trim();
        // file header: an absolute/relative path, not indented, not a count line
        if !l.starts_with(char::is_whitespace)
            && (l.contains('/') || l.contains('\\') || l.ends_with(".js") || l.ends_with(".ts"))
            && !t.contains("problem")
            && !t.is_empty()
        {
            cur_file = Some(t.to_string());
            continue;
        }
        // diagnostic row: starts with line:col
        if let Some(file) = &cur_file {
            let mut it = t.split_whitespace();
            if let Some(loc) = it.next() {
                let (line, col) = split_line_col(loc);
                if line.is_some() {
                    // skip the level token (error/warning), take the rest as message
                    let level = it.next().unwrap_or("");
                    let rest: Vec<&str> = it.collect();
                    let mut msg = rest.join(" ");
                    if !level.eq_ignore_ascii_case("error")
                        && !level.eq_ignore_ascii_case("warning")
                    {
                        msg = format!("{level} {msg}").trim().to_string();
                    }
                    out.push(Finding {
                        file: file.clone(),
                        line,
                        col,
                        message: msg.trim().to_string(),
                    });
                }
            }
        }
    }
    out
}

// ---- pytest: short-test-summary `FAILED path::test - Error` + `file:line:` ----
fn parse_pytest(s: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    for l in s.lines() {
        let t = l.trim();
        // The short-summary lines are the reliable, message-bearing signal:
        // `FAILED path/test_x.py::test_name - AssertionError: msg`
        if let Some(rest) = t
            .strip_prefix("FAILED ")
            .or_else(|| t.strip_prefix("ERROR "))
        {
            let (nodeid, msg) = rest.split_once(" - ").unwrap_or((rest, ""));
            let file = nodeid.split("::").next().unwrap_or(nodeid).to_string();
            out.push(Finding {
                file,
                line: None,
                col: None,
                message: if msg.is_empty() {
                    "test failed".into()
                } else {
                    msg.to_string()
                },
            });
        }
    }
    out
}

// ---- generic `file:line[:col]: message` (ruff / flake8 / mypy / go / gcc) ----
fn parse_generic(s: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    for l in s.lines() {
        let t = l.trim();
        if let Some((file, line, col, msg)) = split_diag(t) {
            if looks_like_path(&file) {
                out.push(Finding {
                    file,
                    line,
                    col,
                    message: msg,
                });
            }
        }
    }
    out
}

// ---- shared location parsing ----

/// `file:line:col` (optionally without col) → (file, line, col).
fn split_loc(s: &str) -> Option<(String, Option<u32>, Option<u32>)> {
    let parts: Vec<&str> = s.rsplitn(3, ':').collect(); // [col?, line?, file] reversed
    // try file:line:col
    if parts.len() == 3 {
        if let (Ok(col), Ok(line)) = (
            parts[0].trim().parse::<u32>(),
            parts[1].trim().parse::<u32>(),
        ) {
            return Some((parts[2].to_string(), Some(line), Some(col)));
        }
    }
    // try file:line
    if let Some((file, line)) = s.rsplit_once(':') {
        if let Ok(line) = line.trim().parse::<u32>() {
            return Some((file.to_string(), Some(line), None));
        }
    }
    None
}

/// Split `file:line[:col]: message` into parts. Handles both the with-column
/// (ruff/go/flake8) and no-column (mypy) shapes by parsing the location prefix
/// before the first `": "`.
fn split_diag(s: &str) -> Option<(String, Option<u32>, Option<u32>, String)> {
    let (loc, msg) = s.split_once(": ")?; // "file:line[:col]" , "message"
    // file:line:col
    let parts: Vec<&str> = loc.rsplitn(3, ':').collect(); // [col, line, file] reversed
    if parts.len() == 3 {
        if let (Ok(col), Ok(line)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
            return Some((
                parts[2].to_string(),
                Some(line),
                Some(col),
                msg.trim().to_string(),
            ));
        }
    }
    // file:line
    if let Some((file, line)) = loc.rsplit_once(':') {
        if let Ok(line) = line.parse::<u32>() {
            return Some((file.to_string(), Some(line), None, msg.trim().to_string()));
        }
    }
    None
}

fn split_line_col(s: &str) -> (Option<u32>, Option<u32>) {
    let mut it = s.split([':', ',']);
    let line = it.next().and_then(|x| x.trim().parse().ok());
    let col = it.next().and_then(|x| x.trim().parse().ok());
    (line, col)
}

fn looks_like_path(f: &str) -> bool {
    !f.is_empty()
        && !f.contains(' ')
        && (f.contains('/') || f.contains('\\') || f.contains('.'))
        && !f.starts_with("http")
}

fn truncate(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rustc_cargo() {
        let out = "error[E0063]: missing field `expected_value` in initializer of `ProbeCandidate`\n  --> src/probes.rs:42:5\n   |\n42 |     ProbeCandidate {\n";
        let r = parse_output("cargo check", "cargo", Some(101), out, "");
        assert_eq!(r.findings.len(), 1);
        let f = &r.findings[0];
        assert_eq!(f.file, "src/probes.rs");
        assert_eq!(f.line, Some(42));
        assert!(f.message.contains("missing field"));
        assert!(r.summary.starts_with("src/probes.rs:42:"), "{}", r.summary);
    }

    #[test]
    fn parses_tsc() {
        let out =
            "src/app.ts(12,7): error TS2322: Type 'string' is not assignable to type 'number'.";
        let r = parse_output("tsc --noEmit", "tsc", Some(2), out, "");
        assert_eq!(r.findings.len(), 1);
        assert_eq!(r.findings[0].file, "src/app.ts");
        assert_eq!(r.findings[0].line, Some(12));
        assert_eq!(r.findings[0].col, Some(7));
        assert!(r.findings[0].message.contains("not assignable"));
    }

    #[test]
    fn parses_eslint_stylish() {
        let out = "/repo/src/index.js\n  10:5  error  'x' is assigned a value but never used  no-unused-vars\n  12:1  error  Unexpected console statement  no-console\n\n✖ 2 problems";
        let r = parse_output("eslint .", "eslint", Some(1), out, "");
        assert_eq!(r.findings.len(), 2, "{:?}", r.findings);
        assert_eq!(r.findings[0].file, "/repo/src/index.js");
        assert_eq!(r.findings[0].line, Some(10));
        assert!(r.findings[0].message.contains("never used"));
    }

    #[test]
    fn parses_generic_ruff_mypy_go() {
        // ruff
        let ruff = parse_output(
            "ruff check .",
            "ruff",
            Some(1),
            "app/main.py:3:1: F401 `os` imported but unused",
            "",
        );
        assert_eq!(ruff.findings[0].file, "app/main.py");
        assert_eq!(ruff.findings[0].line, Some(3));
        assert_eq!(ruff.findings[0].col, Some(1));
        assert!(
            ruff.findings[0].message.contains("imported but unused"),
            "{:?}",
            ruff.findings
        );
        // mypy: `file:line: error: message`
        let mypy = parse_output(
            "mypy .",
            "mypy",
            Some(1),
            "src/x.py:7: error: Incompatible return value type",
            "",
        );
        assert_eq!(mypy.findings[0].file, "src/x.py");
        assert_eq!(mypy.findings[0].line, Some(7));
        assert!(mypy.findings[0].message.contains("Incompatible"));
        // go vet: `./file.go:9:2: message`
        let go = parse_output(
            "go vet ./...",
            "go",
            Some(1),
            "./server.go:9:2: unreachable code",
            "",
        );
        assert_eq!(go.findings[0].file, "./server.go");
        assert_eq!(go.findings[0].line, Some(9));
    }

    #[test]
    fn parses_pytest_summary() {
        let out = "=== FAILURES ===\nFAILED tests/test_api.py::test_resolve - AssertionError: expected 200\nFAILED tests/test_api.py::test_holder - KeyError: 'holder'\n=== short test summary ===";
        let r = parse_output("pytest -q", "pytest", Some(1), out, "");
        assert!(r.findings.len() >= 2, "{:?}", r.findings);
        assert_eq!(r.findings[0].file, "tests/test_api.py");
        assert!(r.findings[0].message.contains("AssertionError"));
    }

    #[test]
    fn clean_run_has_no_findings() {
        let r = parse_output(
            "cargo check",
            "cargo",
            Some(0),
            "    Finished dev [unoptimized]\n",
            "",
        );
        assert!(r.findings.is_empty());
        assert_eq!(r.summary, "no problems reported");
    }

    #[test]
    fn nonzero_exit_without_structured_output_summarizes_last_error() {
        let r = parse_output(
            "make check",
            "",
            Some(2),
            "building...\nfatal: something broke\n",
            "",
        );
        assert!(r.findings.is_empty());
        assert!(r.summary.contains("something broke"), "{}", r.summary);
    }
}
