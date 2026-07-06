//! Probe command classifier — "is THIS shell command a good probe, and what kind?"
//!
//! A sibling to [`crate::linter_probe`] (which *runs* a fixed set of checkers over
//! files on disk). This module instead *classifies a command string* so cria-shepherd
//! can recognize the model's own test/lint/typecheck/build/format commands — or pick
//! a safe one to run itself — without being fooled by shell syntax, installers,
//! searches, filenames, or branch names.
//!
//! Two independent scores:
//! - `intent_confidence`: is this command trying to test/lint/check code?
//! - `probe_quality`: should we actually run/recommend it as a probe? (fast, local,
//!   bounded, read-only, parseable — vs. watch-mode, mutating, service-heavy).
//!
//! Design principle: **direct executable/subcommand evidence is proof; names like
//! `test`/`lint`/`check`, paths, and branch names are supporting evidence only.**
//! We match on the resolved `argv[0]` (+ subcommand/flags), never on substrings, so
//! `grep "pytest"`, `echo "npm test"`, and `git checkout test` are not probes.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeKind {
    Test,
    Lint,
    Typecheck,
    FormatCheck,
    BuildCheck,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ProbeDetection {
    pub kind: ProbeKind,
    pub intent_confidence: u8,
    pub probe_quality: u8,
    pub family: Option<&'static str>,
    pub normalized: String,
    pub mutates_code: bool,
    pub may_hang: bool,
    pub may_need_services: bool,
    pub reasons: Vec<String>,
}

impl ProbeDetection {
    fn unknown(normalized: impl Into<String>) -> Self {
        ProbeDetection {
            kind: ProbeKind::Unknown,
            intent_confidence: 0,
            probe_quality: 0,
            family: None,
            normalized: normalized.into(),
            mutates_code: false,
            may_hang: false,
            may_need_services: false,
            reasons: Vec::new(),
        }
    }
    fn is_probe(&self) -> bool {
        !matches!(self.kind, ProbeKind::Unknown) && self.intent_confidence > 0
    }
    fn better_than(&self, other: &ProbeDetection) -> bool {
        (self.intent_confidence, self.probe_quality)
            > (other.intent_confidence, other.probe_quality)
    }
}

/// Convenience entry point without alias resolution.
pub fn classify_command(cmd: &str) -> ProbeDetection {
    classify(cmd, None)
}

/// Classify a (possibly chained) command. When `project_dir` is provided, package /
/// task aliases (`npm test`, `make lint`, …) are resolved against the manifest and the
/// resolved body is re-classified.
pub fn classify(cmd: &str, project_dir: Option<&Path>) -> ProbeDetection {
    let segments = split_chain(cmd);
    let mut best = ProbeDetection::unknown(cmd.trim());
    let mut probe_count = 0usize;
    for seg in &segments {
        let argv = tokenize(seg);
        if argv.is_empty() {
            continue;
        }
        let d = classify_segment(&argv, project_dir, 0);
        if d.is_probe() {
            probe_count += 1;
            if d.better_than(&best) {
                best = d;
            }
        }
    }
    if best.is_probe() && probe_count > 1 {
        best.reasons.push(format!(
            "chosen from {probe_count} probe-like commands in the chain"
        ));
    }
    best
}

// ---------------------------------------------------------------------------
// Shell parsing (conservative — no interpreter, just enough structure)
// ---------------------------------------------------------------------------

/// Split a command line into segments on `&&`, `||`, `;`, `|`, and newlines,
/// respecting single/double quotes. Background `&` and pipes are treated as breaks.
fn split_chain(cmd: &str) -> Vec<String> {
    let mut segs = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = cmd.chars().collect();
    let (mut in_s, mut in_d) = (false, false);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_s {
            cur.push(c);
            if c == '\'' {
                in_s = false;
            }
            i += 1;
            continue;
        }
        if in_d {
            cur.push(c);
            if c == '"' {
                in_d = false;
            }
            i += 1;
            continue;
        }
        match c {
            '\'' => {
                in_s = true;
                cur.push(c);
            }
            '"' => {
                in_d = true;
                cur.push(c);
            }
            '\n' | ';' => {
                segs.push(std::mem::take(&mut cur));
            }
            '|' => {
                // consume `||` too
                if i + 1 < chars.len() && chars[i + 1] == '|' {
                    i += 1;
                }
                segs.push(std::mem::take(&mut cur));
            }
            '&' => {
                if i + 1 < chars.len() && chars[i + 1] == '&' {
                    i += 1;
                }
                segs.push(std::mem::take(&mut cur));
            }
            _ => cur.push(c),
        }
        i += 1;
    }
    segs.push(cur);
    segs.into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Tokenize one segment into argv, honoring quotes and simple backslash escapes.
/// Quote characters are stripped, so `bash -lc "pytest -q"` yields the inner command
/// as a single token `pytest -q`.
fn tokenize(seg: &str) -> Vec<String> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    let mut has = false;
    let mut chars = seg.chars().peekable();
    let (mut in_s, mut in_d) = (false, false);
    while let Some(c) = chars.next() {
        if in_s {
            if c == '\'' {
                in_s = false;
            } else {
                cur.push(c);
            }
            continue;
        }
        if in_d {
            if c == '"' {
                in_d = false;
            } else if c == '\\' {
                if let Some(&n) = chars.peek() {
                    if n == '"' || n == '\\' {
                        cur.push(n);
                        chars.next();
                        continue;
                    }
                }
                cur.push(c);
            } else {
                cur.push(c);
            }
            continue;
        }
        match c {
            '\'' => {
                in_s = true;
                has = true;
            }
            '"' => {
                in_d = true;
                has = true;
            }
            '\\' => {
                if let Some(n) = chars.next() {
                    cur.push(n);
                    has = true;
                }
            }
            c if c.is_whitespace() => {
                if has {
                    toks.push(std::mem::take(&mut cur));
                    has = false;
                }
            }
            _ => {
                cur.push(c);
                has = true;
            }
        }
    }
    if has {
        toks.push(cur);
    }
    toks
}

// ---------------------------------------------------------------------------
// Per-segment classification
// ---------------------------------------------------------------------------

const MAX_DEPTH: usize = 4;

/// Commands that only print or search text — never a probe, even if their args
/// contain a runner name.
const PRINT_SEARCH: &[&str] = &[
    "echo", "printf", "cat", "grep", "rg", "sed", "awk", "find", "ag", "head", "tail", "less",
    "more", "tee", "true", "false", "test", "[", "[[", ":",
];

fn classify_segment(argv: &[String], project_dir: Option<&Path>, depth: usize) -> ProbeDetection {
    if depth > MAX_DEPTH || argv.is_empty() {
        return ProbeDetection::unknown(argv.join(" "));
    }

    // 1. Strip wrappers (sudo/env/npx/poetry run/bash -lc/…), possibly recursing.
    if let Some(inner) = strip_wrapper(argv, project_dir, depth) {
        return inner;
    }

    let head = argv[0].as_str();
    let base = basename(head);

    // 2. Reject print/search/no-op commands outright.
    if PRINT_SEARCH.contains(&base) {
        return ProbeDetection::unknown(argv.join(" "));
    }

    // 3. Docker/Podman: unwrap the inner command, downgrade for services.
    if matches!(base, "docker" | "podman" | "nerdctl") {
        return classify_container(argv, project_dir, depth);
    }

    // 4. Package-manager / task-runner aliases (npm/pnpm/yarn/bun/make/just/…).
    if let Some(a) = classify_alias(base, argv, project_dir, depth) {
        return a;
    }

    // 5. Direct seed match on argv[0] (+ subcommand/flags).
    if let Some(mut d) = match_seed(base, argv) {
        apply_modifiers(&mut d, argv);
        return d;
    }

    ProbeDetection::unknown(argv.join(" "))
}

fn basename(s: &str) -> &str {
    s.rsplit(['/', '\\']).next().unwrap_or(s)
}

// ---- wrappers --------------------------------------------------------------

/// Strip a leading wrapper and re-classify the remainder. Returns None if `argv`
/// doesn't start with a recognized wrapper.
fn strip_wrapper(
    argv: &[String],
    project_dir: Option<&Path>,
    depth: usize,
) -> Option<ProbeDetection> {
    let base = basename(argv[0].as_str());

    // shells: `bash -lc "<cmd>"` — the command is the next quoted token.
    if matches!(base, "bash" | "sh" | "zsh" | "dash")
        && argv.len() >= 3
        && matches!(argv[1].as_str(), "-c" | "-lc" | "-lic" | "-ic")
    {
        let inner = tokenize(&argv[2]);
        let mut d = classify_segment(&inner, project_dir, depth + 1);
        if d.is_probe() {
            d.reasons
                .insert(0, format!("unwrapped `{} {}`", argv[0], argv[1]));
        }
        return Some(d);
    }

    // `env [NAME=VAL ...] <cmd>`
    if base == "env" {
        let mut rest = &argv[1..];
        while !rest.is_empty() && is_env_assignment(&rest[0]) {
            rest = &rest[1..];
        }
        if rest.is_empty() {
            return Some(ProbeDetection::unknown(argv.join(" ")));
        }
        return Some(wrap_reason(
            classify_segment(rest, project_dir, depth + 1),
            "env",
        ));
    }

    // single-token wrappers
    const W1: &[&str] = &[
        "sudo", "time", "command", "nice", "ionice", "stdbuf", "npx", "bunx", "chronic",
    ];
    if W1.contains(&base) {
        // skip the wrapper and any of ITS flags (e.g. `time -v`, `npx --yes`)
        let mut j = 1;
        while j < argv.len() && argv[j].starts_with('-') {
            j += 1;
        }
        if j >= argv.len() {
            return Some(ProbeDetection::unknown(argv.join(" ")));
        }
        return Some(wrap_reason(
            classify_segment(&argv[j..], project_dir, depth + 1),
            base,
        ));
    }

    // two-token wrappers: `poetry run <cmd>`, `pnpm exec <cmd>`, `bundle exec <cmd>`…
    const W2: &[(&str, &str)] = &[
        ("poetry", "run"),
        ("uv", "run"),
        ("pipenv", "run"),
        ("bundle", "exec"),
        ("mise", "exec"),
        ("asdf", "exec"),
        ("direnv", "exec"),
        ("rye", "run"),
        ("pdm", "run"),
        ("pnpm", "exec"),
        ("yarn", "exec"),
        ("npm", "exec"),
        ("bun", "x"),
    ];
    if argv.len() >= 3 {
        for (a, b) in W2 {
            if base == *a && argv[1] == *b {
                return Some(wrap_reason(
                    classify_segment(&argv[2..], project_dir, depth + 1),
                    a,
                ));
            }
        }
    }

    // `nix develop -c <cmd>` / `nix develop .#dev -c <cmd>`
    if base == "nix" && argv.len() >= 2 && argv[1] == "develop" {
        if let Some(pos) = argv.iter().position(|t| t == "-c") {
            if pos + 1 < argv.len() {
                return Some(wrap_reason(
                    classify_segment(&argv[pos + 1..], project_dir, depth + 1),
                    "nix",
                ));
            }
        }
    }

    None
}

fn wrap_reason(mut d: ProbeDetection, wrapper: &str) -> ProbeDetection {
    if d.is_probe() {
        d.reasons.push(format!("unwrapped `{wrapper}`"));
    }
    d
}

fn is_env_assignment(t: &str) -> bool {
    if let Some(eq) = t.find('=') {
        let name = &t[..eq];
        !name.is_empty()
            && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && name
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
    } else {
        false
    }
}

// ---- container unwrapping --------------------------------------------------

fn classify_container(argv: &[String], project_dir: Option<&Path>, depth: usize) -> ProbeDetection {
    // Only `run`/`compose run`/`exec` can carry an inner command; `build`/`up` cannot.
    let sub = argv.get(1).map(|s| s.as_str()).unwrap_or("");
    let start = match sub {
        "run" | "exec" => 2,
        "compose" if argv.get(2).map(|s| s.as_str()) == Some("run") => 3,
        _ => return ProbeDetection::unknown(argv.join(" ")),
    };
    // Find the first suffix that is itself a probe (skips flags, image, service name).
    for i in start..argv.len() {
        let d = classify_segment(&argv[i..], project_dir, depth + 1);
        if d.is_probe() {
            let mut d = d;
            d.may_need_services = true;
            d.probe_quality = d.probe_quality.saturating_sub(50).max(20).min(45);
            d.reasons
                .push("inside a container run — needs image/services".into());
            return d;
        }
    }
    ProbeDetection::unknown(argv.join(" "))
}

// ---- package / task aliases ------------------------------------------------

/// Infer a probe kind from a free-form script/target name (supporting evidence only).
fn kind_from_name(name: &str) -> Option<ProbeKind> {
    let n = name.to_ascii_lowercase();
    // order matters: typecheck before test (contains "type"), format before lint, etc.
    if n.contains("typecheck") || n == "tsc" || n.contains("types") {
        Some(ProbeKind::Typecheck)
    } else if n.contains("format") || n == "fmt" || n.contains("prettier") {
        Some(ProbeKind::FormatCheck)
    } else if n.contains("lint") {
        Some(ProbeKind::Lint)
    } else if n.contains("test") || n == "t" || n.contains("spec") {
        Some(ProbeKind::Test)
    } else if n.contains("build") || n.contains("compile") || n == "check" {
        Some(ProbeKind::BuildCheck)
    } else {
        None
    }
}

fn classify_alias(
    base: &str,
    argv: &[String],
    project_dir: Option<&Path>,
    depth: usize,
) -> Option<ProbeDetection> {
    // JS package managers: npm/pnpm/yarn/bun. "<pm> test", "<pm> run <script>",
    // "<pm> <script>" (pnpm/yarn/bun shorthand).
    if matches!(base, "npm" | "pnpm" | "yarn" | "bun") {
        let sub = argv.get(1).map(|s| s.as_str())?;
        // reject non-script subcommands (install/add/ci/exec/dlx/create…)
        const NOT_SCRIPT: &[&str] = &[
            "install", "i", "add", "ci", "remove", "rm", "update", "up", "create", "init", "dlx",
            "audit", "publish",
        ];
        if NOT_SCRIPT.contains(&sub) {
            return Some(ProbeDetection::unknown(argv.join(" ")));
        }
        let (script, script_idx) = if sub == "run" || sub == "run-script" {
            (argv.get(2).map(|s| s.as_str())?, 2)
        } else if base == "npm" {
            // npm requires `test` (or `start`/`stop`) as bare; others need `run`.
            if sub == "test" || sub == "t" {
                (sub, 1)
            } else {
                return None;
            }
        } else {
            (sub, 1) // pnpm/yarn/bun shorthand: `pnpm lint`
        };
        let _ = script_idx;
        return Some(resolve_or_alias_js(base, script, argv, project_dir, depth));
    }

    // make / just / task / rake / composer
    let runner = match base {
        "make" | "gmake" => Some(("make", "Makefile")),
        "just" => Some(("just", "Justfile")),
        "task" | "go-task" => Some(("task", "Taskfile")),
        "rake" => Some(("rake", "Rakefile")),
        "composer" => Some(("composer", "composer.json")),
        _ => None,
    };
    if let Some((fam, manifest)) = runner {
        // first non-flag token is the target/script name
        let target = argv[1..]
            .iter()
            .find(|t| !t.starts_with('-'))
            .map(|s| s.as_str());
        let Some(target) = target else {
            return Some(ProbeDetection::unknown(argv.join(" ")));
        };
        let kind = kind_from_name(target)?; // only treat known-kind targets as probes
        let mut d = ProbeDetection::unknown(argv.join(" "));
        d.kind = kind;
        d.intent_confidence = 85;
        d.probe_quality = 50;
        d.family = Some("task-alias");
        d.reasons.push(format!(
            "{fam} target `{target}`; unresolved {manifest} recipe"
        ));
        // resolution for make/composer left as follow-up; detection stands.
        let _ = (manifest, project_dir);
        return Some(d);
    }

    // tox / nox — python task runners, direct
    if base == "tox" {
        let mut d = ProbeDetection::unknown(argv.join(" "));
        d.kind = ProbeKind::Test;
        d.intent_confidence = 85;
        d.probe_quality = 55;
        d.family = Some("tox");
        d.reasons.push("tox runs configured test envs".into());
        return Some(d);
    }
    if base == "nox" {
        let mut d = ProbeDetection::unknown(argv.join(" "));
        d.kind = ProbeKind::Test;
        d.intent_confidence = 80;
        d.probe_quality = 55;
        d.family = Some("nox");
        d.reasons.push("nox session runner".into());
        return Some(d);
    }

    None
}

fn resolve_or_alias_js(
    pm: &str,
    script: &str,
    argv: &[String],
    project_dir: Option<&Path>,
    depth: usize,
) -> ProbeDetection {
    // Try to resolve the script body from package.json.
    if let Some(dir) = project_dir {
        if let Some(body) = read_package_script(dir, script) {
            let sub = classify(&body, project_dir);
            if sub.is_probe() && depth < MAX_DEPTH {
                let mut d = sub;
                d.reasons.insert(
                    0,
                    format!(
                        "resolved package script `{script}` to `{}`",
                        truncate(&body, 80)
                    ),
                );
                // resolution earns confidence; keep the resolved quality.
                d.intent_confidence = d.intent_confidence.max(90);
                return d;
            }
        }
    }
    // Unresolved alias: infer kind from the script name, cap quality.
    let kind = kind_from_name(script).unwrap_or(ProbeKind::Unknown);
    if matches!(kind, ProbeKind::Unknown) {
        return ProbeDetection::unknown(argv.join(" "));
    }
    let mut d = ProbeDetection::unknown(argv.join(" "));
    d.kind = kind;
    d.intent_confidence = 90;
    d.probe_quality = 50;
    d.family = Some("package-script");
    d.reasons.push(format!(
        "{pm} `{script}` alias; unresolved package.json script"
    ));
    d
}

fn read_package_script(dir: &Path, script: &str) -> Option<String> {
    let raw = std::fs::read_to_string(dir.join("package.json")).ok()?;
    let json: serde_json::Value = serde_json::from_str(&raw).ok()?;
    json.get("scripts")?
        .get(script)?
        .as_str()
        .map(|s| s.to_string())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

// ---- direct seed matching --------------------------------------------------

struct Seed {
    kind: ProbeKind,
    family: &'static str,
    intent: u8,
    quality: u8,
}

fn seed(kind: ProbeKind, family: &'static str, intent: u8, quality: u8) -> Option<Seed> {
    Some(Seed {
        kind,
        family,
        intent,
        quality,
    })
}

/// Match a resolved command against the high-value probe seeds. Handles subcommand
/// and mode-flag logic per runner. Returns None if not a recognized probe.
fn match_seed(base: &str, argv: &[String]) -> Option<ProbeDetection> {
    let sub = argv.get(1).map(|s| s.as_str()).unwrap_or("");
    let has = |flag: &str| argv.iter().any(|a| a == flag);
    let has_any = |flags: &[&str]| argv.iter().any(|a| flags.contains(&a.as_str()));

    let s: Option<Seed> = match base {
        // ---- Rust: cargo ----
        "cargo" => match sub {
            "test" if !has("--no-run") => seed(ProbeKind::Test, "cargo", 90, 88),
            "test" => seed(ProbeKind::BuildCheck, "cargo", 80, 60), // --no-run: compiles, doesn't run
            "nextest" => seed(ProbeKind::Test, "cargo", 90, 88),
            "check" => seed(ProbeKind::BuildCheck, "cargo", 90, 95),
            "clippy" => seed(ProbeKind::Lint, "cargo", 95, 90),
            "fmt" if has_any(&["--check", "--", "-l"]) || has("--check") => {
                seed(ProbeKind::FormatCheck, "cargo", 95, 90)
            }
            "fmt" => seed(ProbeKind::FormatCheck, "cargo", 90, 15), // mutates without --check
            "build" | "b" => seed(ProbeKind::BuildCheck, "cargo", 75, 80),
            _ => None,
        },
        // ---- Go ----
        "go" => match sub {
            "test" => seed(ProbeKind::Test, "go", 92, 88),
            "vet" => seed(ProbeKind::Lint, "go", 90, 90),
            "build" => seed(ProbeKind::BuildCheck, "go", 75, 80),
            _ => None,
        },
        "golangci-lint" => seed(ProbeKind::Lint, "golangci-lint", 95, 90),
        "staticcheck" => seed(ProbeKind::Lint, "staticcheck", 92, 88),
        // ---- Python ----
        "pytest" => seed(ProbeKind::Test, "pytest", 95, 90),
        "py.test" => seed(ProbeKind::Test, "pytest", 95, 90),
        "python" | "python3" => match (sub, argv.get(2).map(|s| s.as_str())) {
            ("-m", Some("pytest")) => seed(ProbeKind::Test, "pytest", 95, 90),
            ("-m", Some("unittest")) => seed(ProbeKind::Test, "unittest", 90, 82),
            ("-m", Some("mypy")) => seed(ProbeKind::Typecheck, "mypy", 92, 90),
            ("-m", Some("ruff")) => seed(ProbeKind::Lint, "ruff", 92, 90),
            ("-m", Some("flake8")) => seed(ProbeKind::Lint, "flake8", 90, 88),
            ("-m", Some("pyflakes")) => seed(ProbeKind::Lint, "pyflakes", 90, 88),
            ("-m", Some("pylint")) => seed(ProbeKind::Lint, "pylint", 90, 82),
            ("-m", Some("pyright")) => seed(ProbeKind::Typecheck, "pyright", 92, 90),
            _ => None,
        },
        "mypy" => seed(ProbeKind::Typecheck, "mypy", 92, 90),
        "pyright" | "pyright-python" => seed(ProbeKind::Typecheck, "pyright", 92, 92),
        "pyre" => seed(ProbeKind::Typecheck, "pyre", 88, 82),
        "pylint" => seed(ProbeKind::Lint, "pylint", 90, 82),
        "flake8" => seed(ProbeKind::Lint, "flake8", 90, 88),
        "ruff" => match sub {
            "format" if has("--check") => seed(ProbeKind::FormatCheck, "ruff", 92, 90),
            "format" => seed(ProbeKind::FormatCheck, "ruff", 90, 15),
            _ => seed(ProbeKind::Lint, "ruff", 92, 92), // `ruff check .` or bare
        },
        "sqlfluff" => seed(ProbeKind::Lint, "sqlfluff", 88, 80),
        // ---- JS/TS direct tools ----
        "tsc" => seed(ProbeKind::Typecheck, "tsc", 95, 95),
        "vue-tsc" => seed(ProbeKind::Typecheck, "tsc", 92, 90),
        "eslint" => seed(ProbeKind::Lint, "eslint", 95, 90),
        "biome" => match sub {
            "lint" => seed(ProbeKind::Lint, "biome", 92, 90),
            "format" => seed(ProbeKind::FormatCheck, "biome", 90, 60),
            _ => seed(ProbeKind::Lint, "biome", 90, 85), // `biome check`
        },
        "prettier" => seed(ProbeKind::FormatCheck, "prettier", 90, 60),
        "stylelint" => seed(ProbeKind::Lint, "stylelint", 90, 82),
        "jest" => seed(ProbeKind::Test, "jest", 90, 82),
        "vitest" => match sub {
            "run" => seed(ProbeKind::Test, "vitest", 95, 88),
            _ => seed(ProbeKind::Test, "vitest", 90, 55), // bare vitest defaults to watch in TTY
        },
        "mocha" => seed(ProbeKind::Test, "mocha", 88, 78),
        "ava" => seed(ProbeKind::Test, "ava", 85, 78),
        "tap" => seed(ProbeKind::Test, "tap", 82, 75),
        "uvu" => seed(ProbeKind::Test, "uvu", 80, 75),
        "playwright" if sub == "test" => seed(ProbeKind::Test, "playwright", 90, 45),
        "cypress" if sub == "run" => seed(ProbeKind::Test, "cypress", 88, 45),
        "deno" if sub == "test" => seed(ProbeKind::Test, "deno", 90, 85),
        // ---- .NET ----
        "dotnet" => match sub {
            "test" => seed(ProbeKind::Test, "dotnet", 90, 82),
            "build" => seed(ProbeKind::BuildCheck, "dotnet", 80, 82),
            "format" => seed(ProbeKind::FormatCheck, "dotnet", 88, 60),
            _ => None,
        },
        // ---- JVM ----
        "mvn" | "mvnw" => match sub {
            "test" => seed(ProbeKind::Test, "maven", 88, 78),
            "verify" => seed(ProbeKind::BuildCheck, "maven", 82, 72),
            s if s.contains(':') && (s.contains("checkstyle") || s.contains("pmd")) => {
                seed(ProbeKind::Lint, "maven", 85, 80)
            }
            _ => None,
        },
        "gradle" | "gradlew" => match sub {
            "test" => seed(ProbeKind::Test, "gradle", 88, 78),
            "check" => seed(ProbeKind::BuildCheck, "gradle", 82, 78),
            "lint" | "ktlintCheck" | "detekt" => seed(ProbeKind::Lint, "gradle", 88, 82),
            "build" => seed(ProbeKind::BuildCheck, "gradle", 75, 72),
            _ => None,
        },
        "ktlint" => seed(ProbeKind::Lint, "ktlint", 90, 85),
        "sbt" if sub == "test" => seed(ProbeKind::Test, "sbt", 85, 70),
        // ---- Ruby ----
        "rspec" => seed(ProbeKind::Test, "rspec", 90, 82),
        "rubocop" => seed(ProbeKind::Lint, "rubocop", 92, 85),
        // ---- PHP ----
        "phpunit" => seed(ProbeKind::Test, "phpunit", 90, 82),
        "pest" => seed(ProbeKind::Test, "pest", 88, 80),
        "phpstan" => seed(ProbeKind::Lint, "phpstan", 90, 85),
        "psalm" => seed(ProbeKind::Typecheck, "psalm", 88, 85),
        "phpcs" => seed(ProbeKind::Lint, "phpcs", 88, 82),
        // ---- shells / misc linters ----
        "shellcheck" => seed(ProbeKind::Lint, "shellcheck", 92, 90),
        "yamllint" => seed(ProbeKind::Lint, "yamllint", 88, 85),
        "hadolint" => seed(ProbeKind::Lint, "hadolint", 88, 85),
        "actionlint" => seed(ProbeKind::Lint, "actionlint", 88, 88),
        "tflint" => seed(ProbeKind::Lint, "tflint", 85, 82),
        "ansible-lint" => seed(ProbeKind::Lint, "ansible-lint", 85, 80),
        "swiftlint" => seed(ProbeKind::Lint, "swiftlint", 88, 82),
        "clang-tidy" => seed(ProbeKind::Lint, "clang-tidy", 85, 78),
        "cppcheck" => seed(ProbeKind::Lint, "cppcheck", 85, 80),
        "hlint" => seed(ProbeKind::Lint, "hlint", 85, 82),
        "clj-kondo" => seed(ProbeKind::Lint, "clj-kondo", 85, 82),
        // ---- terraform (only validate is a probe) ----
        "terraform" if sub == "validate" => seed(ProbeKind::BuildCheck, "terraform", 85, 82),
        // ---- other test runners ----
        "ctest" => seed(ProbeKind::Test, "ctest", 82, 72),
        "bazel" if sub == "test" => seed(ProbeKind::Test, "bazel", 85, 65),
        "swift" if sub == "test" => seed(ProbeKind::Test, "swift", 85, 75),
        "flutter" if sub == "test" => seed(ProbeKind::Test, "flutter", 85, 72),
        "dart" if sub == "test" => seed(ProbeKind::Test, "dart", 85, 78),
        "zig" => match sub {
            "test" => seed(ProbeKind::Test, "zig", 85, 80),
            "build" if argv.get(2).map(|s| s.as_str()) == Some("test") => {
                seed(ProbeKind::Test, "zig", 85, 78)
            }
            _ => None,
        },
        "meson" if sub == "test" => seed(ProbeKind::Test, "meson", 82, 72),
        "lein" if sub == "test" => seed(ProbeKind::Test, "lein", 82, 72),
        "stack" if sub == "test" => seed(ProbeKind::Test, "stack", 82, 70),
        "cabal" if sub == "test" => seed(ProbeKind::Test, "cabal", 82, 70),
        "mix" if sub == "test" => seed(ProbeKind::Test, "mix", 85, 78),
        "nimble" if sub == "test" => seed(ProbeKind::Test, "nimble", 82, 75),
        "xcodebuild" if argv.iter().any(|a| a == "test") => {
            seed(ProbeKind::Test, "xcodebuild", 82, 55)
        }
        _ => None,
    };

    let s = s?;
    let mut d = ProbeDetection::unknown(argv.join(" "));
    d.kind = s.kind;
    d.intent_confidence = s.intent;
    d.probe_quality = s.quality;
    d.family = Some(s.family);
    d.reasons
        .push(format!("direct {} command", describe(s.kind)));
    Some(d)
}

fn describe(k: ProbeKind) -> &'static str {
    match k {
        ProbeKind::Test => "test",
        ProbeKind::Lint => "lint",
        ProbeKind::Typecheck => "typecheck",
        ProbeKind::FormatCheck => "format-check",
        ProbeKind::BuildCheck => "build/check",
        ProbeKind::Unknown => "unknown",
    }
}

// ---- modifier pass (flags that change safety/usefulness) -------------------

fn apply_modifiers(d: &mut ProbeDetection, argv: &[String]) {
    let flags: Vec<&str> = argv.iter().map(|s| s.as_str()).collect();
    let has = |f: &str| flags.contains(&f);

    // Mutating flags — classify intent but sharply cut quality.
    const MUTATE: &[&str] = &[
        "--fix",
        "--write",
        "-w",
        "--apply",
        "-a",
        "-A",
        "--updateSnapshot",
        "-u",
        "--update-snapshots",
        "--fix-dry-run",
    ];
    if flags.iter().any(|f| MUTATE.contains(f)) {
        d.mutates_code = true;
        d.probe_quality = d.probe_quality.min(10);
        d.reasons
            .push("mutates code (--fix/--write) — not a read-only probe".into());
    }

    // Watch / interactive — may hang forever.
    const WATCH: &[&str] = &["--watch", "-w", "--watchAll", "--interactive", "-i", "--ui"];
    if flags.iter().any(|f| WATCH.contains(f)) || has("watch") {
        d.may_hang = true;
        d.probe_quality = d.probe_quality.min(20);
        d.reasons
            .push("watch/interactive mode — may not terminate".into());
    }

    // "no real run" / skipped variants — cap quality.
    if has("--collect-only") || has("--co") {
        d.probe_quality = d.probe_quality.min(50);
        d.reasons
            .push("collect-only — does not actually run tests".into());
    }
    if has("-DskipTests") || flags.iter().any(|f| f.starts_with("-DskipTests")) {
        d.probe_quality = d.probe_quality.min(35);
        d.reasons.push("tests skipped (-DskipTests)".into());
    }
    // `gradle test -x test` (exclude)
    if let Some(pos) = flags.iter().position(|f| *f == "-x") {
        if flags.get(pos + 1).is_some_and(|t| t.contains("test")) {
            d.probe_quality = d.probe_quality.min(35);
            d.reasons.push("task excluded (-x test)".into());
        }
    }

    // Service-heavy signals in-line.
    if has("--browser") || flags.iter().any(|f| f.contains("selenium")) {
        d.may_need_services = true;
        d.probe_quality = d.probe_quality.min(45);
    }

    // Auto-writing formatters mutate the tree unless told to only check.
    if matches!(d.kind, ProbeKind::FormatCheck) {
        let base = basename(argv[0].as_str());
        let check = flags.iter().any(|f| {
            matches!(
                *f,
                "--check" | "--verify-no-changes" | "--check-format" | "-l" | "--diff"
            )
        });
        if matches!(
            base,
            "cargo" | "ruff" | "gofmt" | "dotnet" | "black" | "isort"
        ) && !check
        {
            d.mutates_code = true;
            d.probe_quality = d.probe_quality.min(12);
            d.reasons
                .push("formatter without a check flag writes files".into());
        }
    }
}

/// True if ANY segment of `cmd` does something a probe must never do: install
/// dependencies, mutate code (`--fix`/`--write`), watch, or bring up services.
///
/// Used to vet a discovered script BODY before offering it as a probe (design:
/// "avoid scripts that contain watch/fix/write/install/docker/service commands").
/// Unlike [`classify_command`], which finds the best probe in a chain, this taints
/// the WHOLE command if any part is unsafe — so `npm install && jest` is rejected.
pub fn has_unsafe_segment(cmd: &str) -> bool {
    for seg in split_chain(cmd) {
        let argv = tokenize(&seg);
        if argv.is_empty() {
            continue;
        }
        let base = basename(argv[0].as_str());
        let sub = argv.get(1).map(|s| s.as_str()).unwrap_or("");

        // dependency installation
        let pm_install = matches!(sub, "install" | "add" | "ci")
            && matches!(
                base,
                "npm"
                    | "pnpm"
                    | "yarn"
                    | "bun"
                    | "pip"
                    | "pip3"
                    | "apt"
                    | "apt-get"
                    | "brew"
                    | "gem"
                    | "bundle"
                    | "cargo"
                    | "composer"
                    | "poetry"
                    | "uv"
            );
        if pm_install
            || (matches!(base, "pip" | "pip3") && sub == "install")
            || (base == "go" && sub == "get")
        {
            return true;
        }

        // mutating / watch flags anywhere in the segment
        const BAD_FLAGS: &[&str] = &[
            "--fix",
            "--write",
            "--updateSnapshot",
            "-u",
            "--update-snapshots",
            "--watch",
            "--watchAll",
            "--interactive",
        ];
        if argv
            .iter()
            .any(|a| BAD_FLAGS.contains(&a.as_str()) || a == "watch")
        {
            return true;
        }

        // bringing up services / applying infra
        if matches!(base, "docker" | "podman" | "nerdctl") && matches!(sub, "up" | "run") {
            return true;
        }
        if base == "kubectl" && sub == "apply" {
            return true;
        }
        if base == "terraform" && sub == "apply" {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn c(cmd: &str) -> ProbeDetection {
        classify_command(cmd)
    }

    #[test]
    fn direct_test_commands() {
        let d = c("pytest -q");
        assert_eq!(d.kind, ProbeKind::Test);
        assert!(d.intent_confidence >= 90 && d.probe_quality >= 85, "{d:?}");

        assert_eq!(c("cargo test").kind, ProbeKind::Test);
        assert_eq!(c("go test ./...").kind, ProbeKind::Test);
        assert_eq!(c("cargo nextest run").kind, ProbeKind::Test);
    }

    #[test]
    fn direct_lint_typecheck_build() {
        assert_eq!(c("cargo check").kind, ProbeKind::BuildCheck);
        assert!(c("cargo check").probe_quality >= 90);
        let clippy = c("cargo clippy --all-targets");
        assert_eq!(clippy.kind, ProbeKind::Lint);
        assert!(clippy.intent_confidence >= 90);
        let tsc = c("tsc --noEmit");
        assert_eq!(tsc.kind, ProbeKind::Typecheck);
        assert_eq!(tsc.family, Some("tsc"));
        assert!(tsc.probe_quality >= 90);
        assert_eq!(c("ruff check .").kind, ProbeKind::Lint);
        assert_eq!(c("mypy .").kind, ProbeKind::Typecheck);
        assert_eq!(c("go vet ./...").kind, ProbeKind::Lint);
    }

    #[test]
    fn wrappers_are_unwrapped() {
        assert_eq!(c("sudo pytest").kind, ProbeKind::Test);
        assert_eq!(c("time cargo test").kind, ProbeKind::Test);
        assert_eq!(c("npx tsc --noEmit").kind, ProbeKind::Typecheck);
        assert_eq!(c("poetry run pytest").kind, ProbeKind::Test);
        assert_eq!(c("bundle exec rspec").kind, ProbeKind::Test);
        let d = c("bash -lc \"tsc --noEmit\"");
        assert_eq!(d.kind, ProbeKind::Typecheck);
        assert!(d.reasons.iter().any(|r| r.contains("unwrapped")));
    }

    #[test]
    fn package_script_alias_unresolved() {
        let d = c("npm test");
        assert_eq!(d.kind, ProbeKind::Test);
        assert_eq!(d.intent_confidence, 90);
        assert_eq!(d.probe_quality, 50);
        assert_eq!(d.family, Some("package-script"));
        assert_eq!(c("pnpm lint").kind, ProbeKind::Lint);
        assert_eq!(c("yarn run typecheck").kind, ProbeKind::Typecheck);
    }

    #[test]
    fn package_script_resolves_from_manifest() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "probe_pkg_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("package.json"),
            r#"{"scripts":{"lint":"eslint . && tsc --noEmit","test":"jest"}}"#,
        )
        .unwrap();
        let d = classify("npm run lint", Some(&dir));
        assert!(
            matches!(d.kind, ProbeKind::Lint | ProbeKind::Typecheck),
            "{d:?}"
        );
        assert!(
            d.reasons
                .iter()
                .any(|r| r.contains("resolved package script")),
            "{d:?}"
        );
        assert!(
            d.probe_quality > 50,
            "resolution should beat the capped alias: {d:?}"
        );
        let t = classify("npm test", Some(&dir));
        assert_eq!(t.family, Some("jest"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn false_positives_are_rejected() {
        for cmd in [
            "test -f package.json",
            "[ -f package.json ]",
            "[[ -n \"$CI\" ]]",
            "echo \"npm test\"",
            "grep -R \"pytest\" .",
            "rg \"go test\"",
            "cat package.json",
            "npm install jest",
            "pip install pytest",
            "cargo install clippy",
            "git checkout test",
            "git branch lint-fix",
            "terraform workspace select test",
            "kubectl config use-context test",
            "curl https://example.com/test",
            "mkdir test",
            "touch test.txt",
            "docker build --target test .",
        ] {
            let d = c(cmd);
            assert_eq!(d.kind, ProbeKind::Unknown, "should reject: {cmd} => {d:?}");
        }
    }

    #[test]
    fn mutating_commands_downgraded() {
        let d = c("eslint --fix .");
        assert_eq!(d.kind, ProbeKind::Lint);
        assert!(d.mutates_code);
        assert!(d.probe_quality <= 15, "{d:?}");
        assert!(c("prettier --write").mutates_code);
        assert!(c("cargo fmt").mutates_code); // no --check
        assert!(!c("cargo fmt --check").mutates_code);
        assert!(c("ruff check --fix .").mutates_code);
    }

    #[test]
    fn watch_mode_downgraded() {
        let d = c("vitest --watch");
        assert_eq!(d.kind, ProbeKind::Test);
        assert!(d.may_hang);
        assert!(d.probe_quality <= 20, "{d:?}");
        assert!(c("jest --watch").may_hang);
    }

    #[test]
    fn container_and_service_downgraded() {
        let d = c("docker compose run app pytest");
        assert_eq!(d.kind, ProbeKind::Test);
        assert!(d.may_need_services);
        assert!(d.probe_quality <= 45, "{d:?}");
        let p = c("playwright test");
        assert_eq!(p.kind, ProbeKind::Test);
        assert!(p.probe_quality <= 45);
    }

    #[test]
    fn special_cases() {
        let co = c("pytest --collect-only");
        assert_eq!(co.kind, ProbeKind::Test);
        assert!(co.probe_quality <= 50, "{co:?}");
        let nr = c("cargo test --no-run");
        assert!(nr.probe_quality <= 65, "{nr:?}");
        assert!(matches!(nr.kind, ProbeKind::BuildCheck | ProbeKind::Test));
    }

    #[test]
    fn chain_picks_the_probe() {
        // echo is ignored; the real probe wins.
        let d = c("echo running && pytest -q");
        assert_eq!(d.kind, ProbeKind::Test);
        // two probes: highest intent/quality wins, noted as chosen-from-chain.
        let d2 = c("eslint . && tsc --noEmit");
        assert!(d2.is_probe());
        assert!(d2.reasons.iter().any(|r| r.contains("chain")));
    }
}
