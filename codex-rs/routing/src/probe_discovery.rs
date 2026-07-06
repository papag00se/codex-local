//! Repository probe discovery — cria-shepherd's reconnaissance.
//!
//! This is NOT about classifying an incoming command. cria-shepherd is the one
//! *performing* probes: it inspects the repo on disk, infers what ecosystem/tooling
//! exists, chooses SAFE read-only diagnostic commands, ranks them, and (elsewhere)
//! runs a few and turns the output into targeted repair hints for the small model.
//!
//! This module covers steps 1–4: inventory → detect ecosystems/package-managers →
//! discover scripts → build a RANKED list of [`ProbeCandidate`]s. Execution +
//! output-parsing live in [`crate::probe_parse`] / the runner.
//!
//! Selection principles (from the design):
//! - read-only, bounded, no watch/mutation/install/services;
//! - prefer fast localized checks (typecheck/build/lint) before full test suites;
//! - script names + config files are *evidence for choosing commands*, never the goal;
//! - vet discovered `package.json` script BODIES before trusting them
//!   (via [`crate::probe_classifier`]).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::probe_classifier::{self, ProbeKind as CmdKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeKind {
    Test,
    Lint,
    Typecheck,
    FormatCheck,
    BuildCheck,
    StaticAnalysis,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeCost {
    Cheap,
    Moderate,
    Expensive,
    Risky,
}

impl ProbeCost {
    fn ord(self) -> u8 {
        match self {
            ProbeCost::Cheap => 0,
            ProbeCost::Moderate => 1,
            ProbeCost::Expensive => 2,
            ProbeCost::Risky => 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProbeCandidate {
    pub kind: ProbeKind,
    pub command: Vec<String>,
    pub working_dir: PathBuf,
    /// How likely this command is valid for THIS repo (evidence strength).
    pub confidence: u8,
    /// How useful the diagnostics are likely to be (localized file/line > pass/fail).
    pub expected_value: u8,
    pub cost: ProbeCost,
    pub mutates_code: bool,
    pub may_hang: bool,
    pub may_need_services: bool,
    pub reason: String,
}

impl ProbeCandidate {
    /// Safe to run unattended: read-only, terminates, no external services.
    pub fn is_safe(&self) -> bool {
        !self.mutates_code && !self.may_hang && !matches!(self.cost, ProbeCost::Risky)
    }

    /// Run-first priority tier (lower runs earlier): fast localized checks before
    /// heavy suites. Implements the design's initial-probe priority list.
    fn tier(&self) -> u8 {
        if self.may_need_services || matches!(self.cost, ProbeCost::Risky) {
            return 5;
        }
        match self.kind {
            ProbeKind::Typecheck | ProbeKind::BuildCheck => 1,
            ProbeKind::Lint | ProbeKind::StaticAnalysis | ProbeKind::FormatCheck => 2,
            ProbeKind::Test if matches!(self.cost, ProbeCost::Cheap | ProbeCost::Moderate) => 3,
            ProbeKind::Test => 4,
            ProbeKind::Unknown => 6,
        }
    }

    /// Sort key: confidence ↓, tier ↑, expected_value ↓, cost ↑.
    fn sort_key(&self) -> (i32, u8, i32, u8) {
        (
            -(self.confidence as i32),
            self.tier(),
            -(self.expected_value as i32),
            self.cost.ord(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ecosystem {
    JsTs,
    Python,
    Rust,
    Go,
    Jvm,
    DotNet,
    Php,
    Ruby,
    Elixir,
}

/// A discovered project root (has a primary manifest) and the marker files in it.
#[derive(Debug, Clone)]
pub struct ProjectDir {
    pub dir: PathBuf,
    pub files: BTreeSet<String>, // basenames of relevant files in THIS dir
}

impl ProjectDir {
    fn has(&self, name: &str) -> bool {
        self.files.contains(name)
    }
    fn has_glob(&self, ext: &str) -> bool {
        self.files.iter().any(|f| f.ends_with(ext))
    }
}

/// Entry point: inventory the repo and return probe candidates, ranked best-first,
/// already filtered to safe ones. Unsafe/mutating/watch commands are dropped here
/// (they never reach the caller as "run me").
pub fn discover(root: &Path) -> Vec<ProbeCandidate> {
    let mut all = discover_all(root);
    all.retain(|c| c.is_safe() && !matches!(c.kind, ProbeKind::Unknown));
    all.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
    all
}

/// Like [`discover`] but keeps unsafe candidates too (marked), for inspection/tests.
pub fn discover_all(root: &Path) -> Vec<ProbeCandidate> {
    let projects = inventory(root);
    let mut out = Vec::new();
    for p in &projects {
        for eco in detect_ecosystems(p) {
            match eco {
                Ecosystem::JsTs => build_js(root, p, &mut out),
                Ecosystem::Python => build_python(p, &mut out),
                Ecosystem::Rust => build_rust(p, &mut out),
                Ecosystem::Go => build_go(p, &mut out),
                Ecosystem::Jvm => build_jvm(p, &mut out),
                Ecosystem::DotNet => build_dotnet(p, &mut out),
                Ecosystem::Php => build_php(p, &mut out),
                Ecosystem::Ruby => build_ruby(p, &mut out),
                Ecosystem::Elixir => build_elixir(p, &mut out),
            }
        }
    }
    // repo-wide glue probes (config/CI), anchored at the root project if present
    if let Some(rootp) = projects.iter().find(|p| p.dir == root) {
        build_glue(rootp, &mut out);
    }
    out
}

/// The project types detected across a repo (for the summary `project_type`).
pub fn project_types(root: &Path) -> Vec<&'static str> {
    let mut set = BTreeSet::new();
    for p in inventory(root) {
        for eco in detect_ecosystems(&p) {
            set.insert(eco_name(eco));
        }
    }
    set.into_iter().collect()
}

fn eco_name(e: Ecosystem) -> &'static str {
    match e {
        Ecosystem::JsTs => "javascript",
        Ecosystem::Python => "python",
        Ecosystem::Rust => "rust",
        Ecosystem::Go => "go",
        Ecosystem::Jvm => "jvm",
        Ecosystem::DotNet => "dotnet",
        Ecosystem::Php => "php",
        Ecosystem::Ruby => "ruby",
        Ecosystem::Elixir => "elixir",
    }
}

// ---------------------------------------------------------------------------
// Inventory — find project dirs (dirs with a primary manifest), incl. monorepo
// ---------------------------------------------------------------------------

const PRIMARY_MANIFESTS: &[&str] = &[
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "composer.json",
    "Gemfile",
    "mix.exs",
];

const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    "vendor",
    ".gradle",
    "bin",
    "obj",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".tox",
    ".idea",
    ".vscode",
    "coverage",
];

const MAX_DEPTH: usize = 3;

/// Walk the repo (bounded depth, skipping vendor dirs) and group relevant files by
/// the directory that contains a primary manifest. The root is always a project dir.
fn inventory(root: &Path) -> Vec<ProjectDir> {
    let mut dirs: std::collections::BTreeMap<PathBuf, BTreeSet<String>> = Default::default();
    dirs.entry(root.to_path_buf()).or_default();
    walk(root, root, 0, &mut dirs);

    // Keep the root plus any dir that holds a primary manifest.
    dirs.into_iter()
        .filter(|(dir, files)| {
            dir == root
                || files
                    .iter()
                    .any(|f| PRIMARY_MANIFESTS.contains(&f.as_str()))
        })
        .map(|(dir, files)| ProjectDir { dir, files })
        .collect()
}

fn walk(
    root: &Path,
    dir: &Path,
    depth: usize,
    dirs: &mut std::collections::BTreeMap<PathBuf, BTreeSet<String>>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if path.is_dir() {
            if depth >= MAX_DEPTH
                || name.starts_with('.') && name != ".github"
                || SKIP_DIRS.contains(&name.as_str())
            {
                // still record .github one level for workflow detection
                if name == ".github" {
                    record_github(root, &path, dirs);
                }
                continue;
            }
            walk(root, &path, depth + 1, dirs);
        } else if is_relevant_file(&name) {
            dirs.entry(dir.to_path_buf()).or_default().insert(name);
        }
    }
}

fn record_github(
    root: &Path,
    gh: &Path,
    dirs: &mut std::collections::BTreeMap<PathBuf, BTreeSet<String>>,
) {
    let wf = gh.join("workflows");
    if let Ok(entries) = std::fs::read_dir(&wf) {
        if entries.flatten().any(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.ends_with(".yml") || n.ends_with(".yaml")
        }) {
            dirs.entry(root.to_path_buf())
                .or_default()
                .insert(".github-workflows".into());
        }
    }
}

fn is_relevant_file(name: &str) -> bool {
    const EXACT: &[&str] = &[
        // JS/TS
        "package.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "bun.lock",
        "bun.lockb",
        "package-lock.json",
        "tsconfig.json",
        "biome.json",
        "biome.jsonc",
        // Python
        "pyproject.toml",
        "requirements.txt",
        "uv.lock",
        "poetry.lock",
        "pytest.ini",
        "tox.ini",
        "noxfile.py",
        "ruff.toml",
        "mypy.ini",
        "pyrightconfig.json",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "Pipfile.lock",
        // Rust
        "Cargo.toml",
        "Cargo.lock",
        // Go
        "go.mod",
        "go.sum",
        // JVM
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
        "gradlew",
        // .NET / PHP / Ruby / Elixir
        "composer.json",
        "phpunit.xml",
        "phpunit.xml.dist",
        "phpstan.neon",
        "psalm.xml",
        "Gemfile",
        "Rakefile",
        "mix.exs",
        "rebar.config",
        // task runners / glue
        "Makefile",
        "Justfile",
        "justfile",
        "Taskfile.yml",
        "Taskfile.yaml",
        "Dockerfile",
    ];
    if EXACT.contains(&name) {
        return true;
    }
    // config globs + language hints
    name.starts_with(".eslintrc")
        || name.starts_with("eslint.config.")
        || name.starts_with("vitest.config.")
        || name.starts_with("jest.config.")
        || name.starts_with("playwright.config.")
        || name.ends_with(".csproj")
        || name.ends_with(".sln")
        || name.ends_with(".tf")
        || name.ends_with(".sh")
}

fn detect_ecosystems(p: &ProjectDir) -> Vec<Ecosystem> {
    let mut v = Vec::new();
    if p.has("package.json") {
        v.push(Ecosystem::JsTs);
    }
    if p.has("Cargo.toml") {
        v.push(Ecosystem::Rust);
    }
    if p.has("go.mod") {
        v.push(Ecosystem::Go);
    }
    if p.has("pyproject.toml")
        || p.has("requirements.txt")
        || p.has("setup.py")
        || p.has("setup.cfg")
        || p.has("tox.ini")
        || p.has("noxfile.py")
        || p.has("Pipfile")
    {
        v.push(Ecosystem::Python);
    }
    if p.has("pom.xml") || p.has("build.gradle") || p.has("build.gradle.kts") {
        v.push(Ecosystem::Jvm);
    }
    if p.has_glob(".csproj") || p.has_glob(".sln") {
        v.push(Ecosystem::DotNet);
    }
    if p.has("composer.json") {
        v.push(Ecosystem::Php);
    }
    if p.has("Gemfile") {
        v.push(Ecosystem::Ruby);
    }
    if p.has("mix.exs") {
        v.push(Ecosystem::Elixir);
    }
    v
}

// ---------------------------------------------------------------------------
// Candidate builders
// ---------------------------------------------------------------------------

fn cand(
    kind: ProbeKind,
    cmd: &[&str],
    dir: &Path,
    confidence: u8,
    expected_value: u8,
    cost: ProbeCost,
    reason: impl Into<String>,
) -> ProbeCandidate {
    ProbeCandidate {
        kind,
        command: cmd.iter().map(|s| s.to_string()).collect(),
        working_dir: dir.to_path_buf(),
        confidence,
        expected_value,
        cost,
        mutates_code: false,
        may_hang: false,
        may_need_services: false,
        reason: reason.into(),
    }
}

// ---- Rust ----
fn build_rust(p: &ProjectDir, out: &mut Vec<ProbeCandidate>) {
    let d = &p.dir;
    let conf = if p.has("Cargo.lock") { 97 } else { 92 };
    out.push(cand(
        ProbeKind::BuildCheck,
        &["cargo", "check"],
        d,
        conf,
        82,
        ProbeCost::Cheap,
        "Cargo.toml found; cargo check is fast and read-only",
    ));
    out.push(cand(
        ProbeKind::Lint,
        &["cargo", "clippy", "--all-targets", "--all-features"],
        d,
        conf,
        85,
        ProbeCost::Moderate,
        "clippy gives file/line lints beyond compile errors",
    ));
    out.push(cand(
        ProbeKind::Test,
        &["cargo", "test", "--no-fail-fast"],
        d,
        conf,
        90,
        ProbeCost::Moderate,
        "cargo test; --no-fail-fast surfaces all failures",
    ));
    out.push(cand(
        ProbeKind::FormatCheck,
        &["cargo", "fmt", "--check"],
        d,
        conf,
        40,
        ProbeCost::Cheap,
        "cargo fmt --check is read-only",
    ));
}

// ---- Go ----
fn build_go(p: &ProjectDir, out: &mut Vec<ProbeCandidate>) {
    let d = &p.dir;
    let conf = if p.has("go.sum") { 95 } else { 90 };
    out.push(cand(
        ProbeKind::BuildCheck,
        &["go", "build", "./..."],
        d,
        conf,
        78,
        ProbeCost::Cheap,
        "go.mod found; build checks compilation",
    ));
    out.push(cand(
        ProbeKind::Lint,
        &["go", "vet", "./..."],
        d,
        conf,
        80,
        ProbeCost::Cheap,
        "go vet is a fast built-in static check",
    ));
    out.push(cand(
        ProbeKind::Test,
        &["go", "test", "./..."],
        d,
        conf,
        90,
        ProbeCost::Moderate,
        "go test across all packages",
    ));
    // external tools: lower confidence (may not be installed / configured)
    out.push(cand(
        ProbeKind::StaticAnalysis,
        &["golangci-lint", "run"],
        d,
        60,
        85,
        ProbeCost::Moderate,
        "golangci-lint if available",
    ));
    out.push(cand(
        ProbeKind::StaticAnalysis,
        &["staticcheck", "./..."],
        d,
        55,
        82,
        ProbeCost::Moderate,
        "staticcheck if available",
    ));
}

// ---- Python ----
fn build_python(p: &ProjectDir, out: &mut Vec<ProbeCandidate>) {
    let d = &p.dir;
    // package-manager prefix selection
    let (prefix, pm_conf): (Vec<&str>, u8) = if p.has("uv.lock") {
        (vec!["uv", "run"], 95)
    } else if p.has("poetry.lock") || pyproject_has(p, "[tool.poetry]") {
        (vec!["poetry", "run"], 92)
    } else {
        (vec![], 80)
    };
    let with = |extra: &[&str]| -> Vec<String> {
        prefix
            .iter()
            .chain(extra.iter())
            .map(|s| s.to_string())
            .collect()
    };

    // typecheck
    // config-gated tools all share the same (strong) confidence, so the run-first
    // TIER (typecheck < lint < test) decides order, not arbitrary confidence gaps.
    let tool_conf = pm_conf.min(90);
    if pyproject_has(p, "mypy") || p.has("mypy.ini") {
        push_owned(
            out,
            ProbeKind::Typecheck,
            with(&["mypy", "."]),
            d,
            tool_conf,
            88,
            ProbeCost::Cheap,
            "mypy configured",
        );
    }
    if p.has("pyrightconfig.json") || pyproject_has(p, "pyright") {
        push_owned(
            out,
            ProbeKind::Typecheck,
            with(&["pyright"]),
            d,
            tool_conf,
            88,
            ProbeCost::Cheap,
            "pyright configured",
        );
    }
    // lint
    if p.has("ruff.toml") || pyproject_has(p, "ruff") {
        push_owned(
            out,
            ProbeKind::Lint,
            with(&["ruff", "check", "."]),
            d,
            pm_conf.min(90),
            85,
            ProbeCost::Cheap,
            "ruff configured; fast file/line lints",
        );
    }
    if pyproject_has(p, "flake8") {
        push_owned(
            out,
            ProbeKind::Lint,
            with(&["flake8", "."]),
            d,
            70,
            80,
            ProbeCost::Cheap,
            "flake8 configured",
        );
    }
    // tests
    if p.has("pytest.ini") || pyproject_has(p, "pytest") || has_tests_dir(d) {
        push_owned(
            out,
            ProbeKind::Test,
            with(&["python", "-m", "pytest", "-q"]),
            d,
            pm_conf.min(90),
            90,
            ProbeCost::Moderate,
            "pytest configured / tests dir present",
        );
    }
    if p.has("tox.ini") {
        push_owned(
            out,
            ProbeKind::Test,
            vec!["tox".into()],
            d,
            75,
            82,
            ProbeCost::Expensive,
            "tox.ini present",
        );
    }
    if p.has("noxfile.py") {
        push_owned(
            out,
            ProbeKind::Test,
            vec!["nox".into(), "-s".into(), "tests".into()],
            d,
            70,
            82,
            ProbeCost::Expensive,
            "noxfile.py present",
        );
    }
    // if nothing config-gated matched but it's clearly python, offer a low-conf pytest
    if !out.iter().any(|c| c.working_dir == *d) {
        push_owned(
            out,
            ProbeKind::Test,
            with(&["python", "-m", "pytest", "-q"]),
            d,
            55,
            90,
            ProbeCost::Moderate,
            "python project; pytest is the common test runner",
        );
    }
}

fn has_tests_dir(dir: &Path) -> bool {
    dir.join("tests").is_dir() || dir.join("test").is_dir()
}

fn pyproject_has(p: &ProjectDir, needle: &str) -> bool {
    if !p.has("pyproject.toml") {
        return false;
    }
    std::fs::read_to_string(p.dir.join("pyproject.toml"))
        .map(|s| s.contains(needle))
        .unwrap_or(false)
}

// ---- JS / TS (with package.json script resolution + vetting) ----
fn build_js(_root: &Path, p: &ProjectDir, out: &mut Vec<ProbeCandidate>) {
    let d = &p.dir;
    let (pm, pm_conf) = js_package_manager(p);

    // 1. Resolve declared scripts (highest confidence — they're defined) and VET bodies.
    let scripts = read_scripts(&d.join("package.json"));
    const GOOD: &[&str] = &[
        "typecheck",
        "type-check",
        "tsc",
        "lint",
        "check",
        "test",
        "test:unit",
        "unit",
        "verify",
        "ci",
        "format:check",
        "fmt:check",
    ];
    for name in GOOD {
        if let Some(body) = scripts.get(*name) {
            let vet = probe_classifier::classify_command(body);
            // Only recognized probe scripts become candidates.
            if matches!(vet.kind, CmdKind::Unknown) {
                continue;
            }
            let kind = map_kind(vet.kind, name);
            let mut c = cand(
                kind,
                &[pm, "run", name],
                d,
                pm_conf,
                value_for(kind),
                cost_for(kind),
                format!("package.json script `{name}` → `{}`", short(body)),
            );
            c.mutates_code = vet.mutates_code;
            c.may_hang = vet.may_hang;
            c.may_need_services = vet.may_need_services;
            // Vet the WHOLE body: a segment that installs deps / mutates / brings up
            // services taints the script even if another segment is a valid probe
            // (e.g. `npm install && jest`). Mark Risky so it's filtered from the safe
            // set but still visible (flagged) in `discover_all`.
            if probe_classifier::has_unsafe_segment(body) {
                c.cost = ProbeCost::Risky;
                c.reason = format!("{} — UNSAFE body (install/mutate/service)", c.reason);
            }
            out.push(c);
        }
    }

    // 2. Direct tool candidates gated on config presence (lower conf than declared scripts).
    if p.has("tsconfig.json") {
        out.push(cand(
            ProbeKind::Typecheck,
            &[pm, "exec", "tsc", "--noEmit"],
            d,
            pm_conf.min(85),
            90,
            ProbeCost::Cheap,
            "tsconfig.json present; tsc --noEmit is the canonical typecheck",
        ));
    }
    if p.files
        .iter()
        .any(|f| f.starts_with(".eslintrc") || f.starts_with("eslint.config."))
    {
        out.push(cand(
            ProbeKind::Lint,
            &[pm, "exec", "eslint", "."],
            d,
            pm_conf.min(82),
            82,
            ProbeCost::Cheap,
            "eslint config present",
        ));
    }
    if p.has("biome.json") || p.has("biome.jsonc") {
        out.push(cand(
            ProbeKind::Lint,
            &[pm, "exec", "biome", "check", "."],
            d,
            pm_conf.min(82),
            82,
            ProbeCost::Cheap,
            "biome config present",
        ));
    }
    if p.files.iter().any(|f| f.starts_with("vitest.config.")) {
        out.push(cand(
            ProbeKind::Test,
            &[pm, "exec", "vitest", "run"],
            d,
            pm_conf.min(80),
            88,
            ProbeCost::Moderate,
            "vitest config present; `run` avoids watch mode",
        ));
    } else if p.files.iter().any(|f| f.starts_with("jest.config.")) {
        out.push(cand(
            ProbeKind::Test,
            &[pm, "exec", "jest", "--runInBand"],
            d,
            pm_conf.min(80),
            88,
            ProbeCost::Moderate,
            "jest config present",
        ));
    }
    if p.files.iter().any(|f| f.starts_with("playwright.config.")) {
        let mut c = cand(
            ProbeKind::Test,
            &[pm, "exec", "playwright", "test"],
            d,
            pm_conf.min(75),
            70,
            ProbeCost::Risky,
            "playwright e2e — needs browsers/services",
        );
        c.may_need_services = true;
        out.push(c);
    }
}

/// Choose the package manager from lockfiles / the `packageManager` field.
fn js_package_manager(p: &ProjectDir) -> (&'static str, u8) {
    // packageManager field wins when explicit
    if let Ok(raw) = std::fs::read_to_string(p.dir.join("package.json")) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(pmf) = json.get("packageManager").and_then(|v| v.as_str()) {
                if pmf.starts_with("pnpm") {
                    return ("pnpm", 96);
                } else if pmf.starts_with("yarn") {
                    return ("yarn", 96);
                } else if pmf.starts_with("bun") {
                    return ("bun", 96);
                } else if pmf.starts_with("npm") {
                    return ("npm", 96);
                }
            }
        }
    }
    if p.has("pnpm-lock.yaml") {
        ("pnpm", 95)
    } else if p.has("yarn.lock") {
        ("yarn", 95)
    } else if p.has("bun.lock") || p.has("bun.lockb") {
        ("bun", 95)
    } else if p.has("package-lock.json") {
        ("npm", 95)
    } else {
        ("npm", 78) // default, lower confidence
    }
}

fn read_scripts(pkg: &Path) -> std::collections::BTreeMap<String, String> {
    let mut m = std::collections::BTreeMap::new();
    if let Ok(raw) = std::fs::read_to_string(pkg) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(obj) = json.get("scripts").and_then(|s| s.as_object()) {
                for (k, v) in obj {
                    if let Some(s) = v.as_str() {
                        m.insert(k.clone(), s.to_string());
                    }
                }
            }
        }
    }
    m
}

/// Map the classifier's command-kind + the script name to a discovery kind.
fn map_kind(k: CmdKind, name: &str) -> ProbeKind {
    match k {
        CmdKind::Test => ProbeKind::Test,
        CmdKind::Lint => ProbeKind::Lint,
        CmdKind::Typecheck => ProbeKind::Typecheck,
        CmdKind::FormatCheck => ProbeKind::FormatCheck,
        CmdKind::BuildCheck => ProbeKind::BuildCheck,
        CmdKind::Unknown => name_kind(name),
    }
}

fn name_kind(name: &str) -> ProbeKind {
    let n = name.to_ascii_lowercase();
    if n.contains("typecheck") || n.contains("type-check") || n == "tsc" {
        ProbeKind::Typecheck
    } else if n.contains("lint") {
        ProbeKind::Lint
    } else if n.contains("format") || n.contains("fmt") {
        ProbeKind::FormatCheck
    } else if n.contains("test") || n == "unit" {
        ProbeKind::Test
    } else {
        ProbeKind::BuildCheck
    }
}

fn value_for(k: ProbeKind) -> u8 {
    match k {
        ProbeKind::Typecheck => 90,
        ProbeKind::BuildCheck => 82,
        ProbeKind::Lint | ProbeKind::StaticAnalysis => 82,
        ProbeKind::Test => 88,
        ProbeKind::FormatCheck => 40,
        ProbeKind::Unknown => 0,
    }
}

fn cost_for(k: ProbeKind) -> ProbeCost {
    match k {
        ProbeKind::Typecheck | ProbeKind::Lint | ProbeKind::FormatCheck | ProbeKind::BuildCheck => {
            ProbeCost::Cheap
        }
        ProbeKind::StaticAnalysis => ProbeCost::Moderate,
        ProbeKind::Test => ProbeCost::Moderate,
        ProbeKind::Unknown => ProbeCost::Moderate,
    }
}

// ---- JVM / .NET / PHP / Ruby / Elixir ----
fn build_jvm(p: &ProjectDir, out: &mut Vec<ProbeCandidate>) {
    let d = &p.dir;
    if p.has("build.gradle") || p.has("build.gradle.kts") {
        let g = if p.has("gradlew") {
            "./gradlew"
        } else {
            "gradle"
        };
        out.push(cand(
            ProbeKind::BuildCheck,
            &[g, "check"],
            d,
            88,
            82,
            ProbeCost::Expensive,
            "gradle check (compile+verify)",
        ));
        out.push(cand(
            ProbeKind::Test,
            &[g, "test"],
            d,
            88,
            88,
            ProbeCost::Expensive,
            "gradle test",
        ));
    }
    if p.has("pom.xml") {
        let m = if p.has("mvnw") { "./mvnw" } else { "mvn" };
        out.push(cand(
            ProbeKind::Test,
            &[m, "test"],
            d,
            85,
            88,
            ProbeCost::Expensive,
            "maven test",
        ));
        out.push(cand(
            ProbeKind::StaticAnalysis,
            &[m, "checkstyle:check"],
            d,
            60,
            80,
            ProbeCost::Moderate,
            "checkstyle if configured",
        ));
    }
}

fn build_dotnet(p: &ProjectDir, out: &mut Vec<ProbeCandidate>) {
    let d = &p.dir;
    out.push(cand(
        ProbeKind::BuildCheck,
        &["dotnet", "build"],
        d,
        88,
        82,
        ProbeCost::Moderate,
        "dotnet build checks compilation",
    ));
    out.push(cand(
        ProbeKind::Test,
        &["dotnet", "test"],
        d,
        85,
        88,
        ProbeCost::Expensive,
        "dotnet test",
    ));
    out.push(cand(
        ProbeKind::FormatCheck,
        &["dotnet", "format", "--verify-no-changes"],
        d,
        70,
        45,
        ProbeCost::Cheap,
        "read-only format check",
    ));
}

fn build_php(p: &ProjectDir, out: &mut Vec<ProbeCandidate>) {
    let d = &p.dir;
    if p.has("phpstan.neon") {
        out.push(cand(
            ProbeKind::StaticAnalysis,
            &["vendor/bin/phpstan", "analyse"],
            d,
            85,
            85,
            ProbeCost::Moderate,
            "phpstan configured",
        ));
    }
    if p.has("psalm.xml") {
        out.push(cand(
            ProbeKind::StaticAnalysis,
            &["vendor/bin/psalm"],
            d,
            82,
            85,
            ProbeCost::Moderate,
            "psalm configured",
        ));
    }
    if p.has("phpunit.xml") || p.has("phpunit.xml.dist") {
        out.push(cand(
            ProbeKind::Test,
            &["vendor/bin/phpunit"],
            d,
            85,
            88,
            ProbeCost::Moderate,
            "phpunit configured",
        ));
    }
}

fn build_ruby(p: &ProjectDir, out: &mut Vec<ProbeCandidate>) {
    let d = &p.dir;
    out.push(cand(
        ProbeKind::Lint,
        &["bundle", "exec", "rubocop"],
        d,
        70,
        82,
        ProbeCost::Cheap,
        "rubocop if in Gemfile",
    ));
    out.push(cand(
        ProbeKind::Test,
        &["bundle", "exec", "rspec"],
        d,
        70,
        88,
        ProbeCost::Moderate,
        "rspec if in Gemfile",
    ));
}

fn build_elixir(p: &ProjectDir, out: &mut Vec<ProbeCandidate>) {
    let d = &p.dir;
    out.push(cand(
        ProbeKind::Test,
        &["mix", "test"],
        d,
        85,
        88,
        ProbeCost::Moderate,
        "mix.exs present",
    ));
}

// ---- glue (config/CI), repo-wide ----
fn build_glue(p: &ProjectDir, out: &mut Vec<ProbeCandidate>) {
    let d = &p.dir;
    if p.has_glob(".sh") {
        out.push(cand(
            ProbeKind::Lint,
            &["shellcheck"],
            d,
            55,
            70,
            ProbeCost::Cheap,
            "shell scripts present",
        ));
    }
    if p.has(".github-workflows") {
        out.push(cand(
            ProbeKind::Lint,
            &["actionlint"],
            d,
            55,
            65,
            ProbeCost::Cheap,
            "GitHub workflows present",
        ));
    }
    if p.has_glob(".tf") {
        out.push(cand(
            ProbeKind::BuildCheck,
            &["terraform", "validate"],
            d,
            60,
            70,
            ProbeCost::Cheap,
            "terraform files present",
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn push_owned(
    out: &mut Vec<ProbeCandidate>,
    kind: ProbeKind,
    command: Vec<String>,
    dir: &Path,
    confidence: u8,
    expected_value: u8,
    cost: ProbeCost,
    reason: impl Into<String>,
) {
    out.push(ProbeCandidate {
        kind,
        command,
        working_dir: dir.to_path_buf(),
        confidence,
        expected_value,
        cost,
        mutates_code: false,
        may_hang: false,
        may_need_services: false,
        reason: reason.into(),
    });
}

fn short(s: &str) -> String {
    let s = s.trim();
    if s.chars().count() <= 60 {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(60).collect::<String>())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn scratch() -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "probe_disc_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
    fn write(dir: &Path, name: &str, body: &str) {
        let full = dir.join(name);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(full, body).unwrap();
    }
    fn cmds(cs: &[ProbeCandidate]) -> Vec<String> {
        cs.iter().map(|c| c.command.join(" ")).collect()
    }

    #[test]
    fn rust_probe_discovery() {
        let d = scratch();
        write(&d, "Cargo.toml", "[package]\nname='x'\n");
        write(&d, "Cargo.lock", "");
        let got = discover(&d);
        let c = cmds(&got);
        assert!(c.contains(&"cargo check".to_string()), "{c:?}");
        assert!(c.contains(&"cargo clippy --all-targets --all-features".to_string()));
        // fast build check ranks before the test suite
        let ci = c.iter().position(|x| x == "cargo check").unwrap();
        let ti = c
            .iter()
            .position(|x| x == "cargo test --no-fail-fast")
            .unwrap();
        assert!(ci < ti, "build check should rank before tests: {c:?}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn go_probe_discovery() {
        let d = scratch();
        write(&d, "go.mod", "module x\n");
        let c = cmds(&discover(&d));
        assert!(c.contains(&"go vet ./...".to_string()), "{c:?}");
        assert!(c.contains(&"go test ./...".to_string()));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn python_probe_discovery_prefers_uv_and_config_gated() {
        let d = scratch();
        write(
            &d,
            "pyproject.toml",
            "[tool.mypy]\n[tool.ruff]\n[tool.pytest.ini_options]\n",
        );
        write(&d, "uv.lock", "");
        let got = discover(&d);
        let c = cmds(&got);
        assert!(
            c.iter().any(|x| x == "uv run mypy ."),
            "uv-prefixed mypy: {c:?}"
        );
        assert!(c.iter().any(|x| x == "uv run ruff check ."), "{c:?}");
        assert!(c.iter().any(|x| x == "uv run python -m pytest -q"), "{c:?}");
        // typecheck/lint rank before the test
        let mi = c.iter().position(|x| x.contains("mypy")).unwrap();
        let pi = c.iter().position(|x| x.contains("pytest")).unwrap();
        assert!(mi < pi, "typecheck before test: {c:?}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn js_package_manager_detection() {
        let d = scratch();
        write(&d, "package.json", r#"{"scripts":{"test":"jest"}}"#);
        write(&d, "pnpm-lock.yaml", "");
        let (pm, conf) = js_package_manager(&ProjectDir {
            dir: d.clone(),
            files: ["package.json".into(), "pnpm-lock.yaml".into()]
                .into_iter()
                .collect(),
        });
        assert_eq!(pm, "pnpm");
        assert!(conf >= 95);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn package_json_script_resolution_and_unsafe_rejection() {
        let d = scratch();
        write(
            &d,
            "package.json",
            r#"{"packageManager":"yarn@4","scripts":{
                "typecheck":"tsc --noEmit",
                "lint":"eslint .",
                "test":"vitest --watch",
                "fmtfix":"prettier --write ."
            }}"#,
        );
        write(&d, "yarn.lock", "");
        let all = discover_all(&d);
        let safe = discover(&d);
        let sc = cmds(&safe);
        // good scripts resolved with the right PM
        assert!(sc.iter().any(|x| x == "yarn run typecheck"), "{sc:?}");
        assert!(sc.iter().any(|x| x == "yarn run lint"), "{sc:?}");
        // vitest --watch script must NOT appear in the safe set (may hang)
        assert!(
            !sc.iter().any(|x| x == "yarn run test"),
            "watch test should be filtered: {sc:?}"
        );
        // but it's present (flagged) in the unfiltered set
        let watch = all
            .iter()
            .find(|c| c.command == vec!["yarn", "run", "test"]);
        assert!(
            watch.map(|c| c.may_hang).unwrap_or(false),
            "watch test flagged may_hang"
        );
        // fmtfix is not in the GOOD names list → not discovered
        assert!(!sc.iter().any(|x| x.contains("fmtfix")));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn monorepo_subdirectory_discovery() {
        let d = scratch();
        write(&d, "package.json", r#"{"private":true}"#); // root, no scripts
        write(&d, "packages/api/Cargo.toml", "[package]\nname='api'\n");
        write(
            &d,
            "packages/web/package.json",
            r#"{"scripts":{"typecheck":"tsc --noEmit"}}"#,
        );
        write(&d, "packages/web/pnpm-lock.yaml", "");
        let got = discover(&d);
        // rust candidate anchored in packages/api
        let rust = got
            .iter()
            .find(|c| c.command == vec!["cargo", "check"])
            .expect("rust");
        assert!(
            rust.working_dir.ends_with("packages/api"),
            "{:?}",
            rust.working_dir
        );
        // web typecheck anchored in packages/web, using pnpm
        let web = got
            .iter()
            .find(|c| c.command == vec!["pnpm", "run", "typecheck"])
            .expect("web typecheck");
        assert!(web.working_dir.ends_with("packages/web"));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn ranking_prefers_confident_cheap_localized() {
        let d = scratch();
        write(&d, "Cargo.toml", "[package]\nname='x'\n");
        write(&d, "Cargo.lock", "");
        let got = discover(&d);
        // first candidate should be a cheap high-confidence check, not the test suite
        assert!(
            matches!(got[0].kind, ProbeKind::BuildCheck | ProbeKind::Lint),
            "{:?}",
            got[0]
        );
        assert!(got[0].confidence >= 90);
        // sorted by the documented key
        for w in got.windows(2) {
            assert!(
                w[0].sort_key() <= w[1].sort_key(),
                "not sorted: {:?} then {:?}",
                w[0].command,
                w[1].command
            );
        }
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn install_and_mutation_never_selected() {
        // A script that installs deps must never be offered as a probe.
        let d = scratch();
        write(
            &d,
            "package.json",
            r#"{"scripts":{"test":"npm install && jest"}}"#,
        );
        write(&d, "package-lock.json", "");
        let sc = cmds(&discover(&d));
        assert!(
            !sc.iter().any(|x| x == "npm run test"),
            "install-bearing script rejected: {sc:?}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn project_types_summary() {
        let d = scratch();
        write(&d, "Cargo.toml", "[package]\nname='x'\n");
        write(&d, "package.json", r#"{"scripts":{"test":"vitest run"}}"#);
        let mut t = project_types(&d);
        t.sort();
        assert_eq!(t, vec!["javascript", "rust"]);
        std::fs::remove_dir_all(&d).ok();
    }
}
