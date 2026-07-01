//! Translate hallucinated tool names into proper `shell` invocations.
//!
//! Small local models (notably qwen3.5:9b) habitually emit shell command
//! names — `ls`, `rg`, `cat`, `git`, `pytest`, etc. — as tool names. They
//! pattern-match on their training data, where shell access typically appears
//! under those names directly. The translation below catches that and rewrites
//! the call into a real `shell` invocation so Codex's tool registry can run
//! it. Same set runs in regular and local-only modes.
//!
//! Why this isn't a "band-aid" by AGENTS.md's definition: the upstream fix
//! is the model's training, which we can't reach. This translation sits at
//! the boundary between the model's dialect and Codex's tool registry —
//! exactly the layer where translation belongs.

use base64::Engine;
use serde_json::Value as JsonValue;

/// base64 engine for the write_file→shell massage (standard alphabet, no wrap).
const B64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::STANDARD;

/// Marker that tags a `shell` command we synthesized from a `write_file` call, so
/// the inbound pass can recognize and re-present it. Followed by the base64 path.
const SHEPHARD_WRITE_MARKER: &str = "# shephard-write:";

/// Names the local model may emit as tool names that should actually be
/// executed via the `shell` tool. Comprehensive Linux developer environment
/// coverage; not all entries need to be in the LightCoder whitelist for the
/// alias to fire — these aliases run AFTER the model has emitted a call.
///
/// Excluded by design: interactive editors (`vim`, `nano`, `emacs`, `less`,
/// `more`, `top`, `htop`) — they hang on a non-interactive shell and silently
/// time out. We let those fall through unaliased so the failure is visible.
pub const SHELL_COMMAND_ALIASES: &[&str] = &[
    // --- File system navigation / inspection ---
    "ls",
    "dir",
    "tree",
    "stat",
    "file",
    "find",
    "locate",
    "which",
    "whereis",
    "type",
    "command",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "pwd",
    "cd",
    // --- File reading ---
    "cat",
    "head",
    "tail",
    "tac",
    "nl",
    "wc",
    "hexdump",
    "xxd",
    "od",
    "strings",
    // --- File / dir manipulation ---
    "touch",
    "mkdir",
    "rmdir",
    "rm",
    "cp",
    "mv",
    "ln",
    "chmod",
    "chown",
    "chgrp",
    "install",
    "truncate",
    // --- Text processing ---
    "echo",
    "printf",
    "sed",
    "awk",
    "tr",
    "cut",
    "paste",
    "sort",
    "uniq",
    "comm",
    "diff",
    "patch",
    "tee",
    "xargs",
    "fmt",
    "fold",
    "expand",
    "unexpand",
    "column",
    "rev",
    "split",
    "csplit",
    // --- Search ---
    "grep",
    "rg",
    "ag",
    "ack",
    "fgrep",
    "egrep",
    // --- Compression / archives ---
    "tar",
    "gzip",
    "gunzip",
    "zcat",
    "gzcat",
    "zip",
    "unzip",
    "bzip2",
    "bunzip2",
    "bzcat",
    "xz",
    "unxz",
    "xzcat",
    "zstd",
    "unzstd",
    // --- Version control ---
    "git",
    "hg",
    "svn",
    // --- Network (read-only / safe) ---
    "curl",
    "wget",
    "ping",
    "host",
    "dig",
    "nslookup",
    "traceroute",
    "mtr",
    // --- Network (potentially destructive — still safe to alias; sandbox enforces) ---
    "ssh",
    "scp",
    "sftp",
    "rsync",
    "nc",
    "netcat",
    "telnet",
    // --- Process / system info ---
    "ps",
    "kill",
    "killall",
    "pgrep",
    "pkill",
    "jobs",
    "free",
    "df",
    "du",
    "mount",
    "umount",
    "lsblk",
    "lscpu",
    "lsof",
    "uptime",
    "who",
    "whoami",
    "id",
    "hostname",
    "uname",
    "date",
    "cal",
    // --- Shell / env ---
    "env",
    "export",
    "set",
    "unset",
    "alias",
    "source",
    "eval",
    "exec",
    "bash",
    "sh",
    "zsh",
    "fish",
    "dash",
    // --- Package managers (system) ---
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "pacman",
    "brew",
    "snap",
    "flatpak",
    "rpm",
    "dpkg",
    // --- Package managers (language) ---
    "pip",
    "pip3",
    "pipx",
    "poetry",
    "conda",
    "mamba",
    "uv",
    "npm",
    "yarn",
    "pnpm",
    "bunx",
    "npx",
    "bundle",
    "bundler",
    "gem",
    "composer",
    "nuget",
    // --- Languages / interpreters ---
    "python",
    "python3",
    "ruby",
    "perl",
    "node",
    "deno",
    "bun",
    "java",
    "javac",
    "kotlin",
    "kotlinc",
    "scala",
    "scalac",
    "sbt",
    "go",
    "gofmt",
    "rustc",
    "gcc",
    "g++",
    "clang",
    "clang++",
    "cc",
    "ld",
    "as",
    "ghc",
    "stack",
    "cabal",
    "ocaml",
    "ocamlfind",
    "dune",
    "php",
    "lua",
    "luac",
    "dart",
    "swift",
    "swiftc",
    "julia",
    "Rscript",
    "R",
    "nim",
    "zig",
    "v",
    "crystal",
    // --- Build systems ---
    "make",
    "cmake",
    "ninja",
    "meson",
    "mvn",
    "gradle",
    "ant",
    "cargo",
    "nix",
    "nix-build",
    "nix-shell",
    "bazel",
    "buck",
    "buck2",
    "pants",
    "just",
    // --- Test runners ---
    "pytest",
    "unittest",
    "tox",
    "jest",
    "vitest",
    "mocha",
    "tap",
    "ava",
    "rspec",
    "minitest",
    "phpunit",
    "pest",
    // --- Cloud / infra / containers ---
    "docker",
    "podman",
    "buildah",
    "skopeo",
    "kubectl",
    "helm",
    "kustomize",
    "k9s",
    "aws",
    "gcloud",
    "az",
    "doctl",
    "linode",
    "terraform",
    "pulumi",
    "ansible",
    "salt",
    "puppet",
    "sam",
    "serverless",
    "vercel",
    "netlify",
    "fly",
    "railway",
    // --- Data / formats ---
    "jq",
    "yq",
    "tomlq",
    "fx",
    "base64",
    "uuencode",
    "uudecode",
    // --- Crypto / hashing ---
    "md5sum",
    "sha1sum",
    "sha256sum",
    "sha512sum",
    "cksum",
    "openssl",
    "gpg",
    "ssh-keygen",
    // --- Misc / coordination ---
    "sleep",
    "watch",
    "time",
    "timeout",
    "yes",
    "true",
    "false",
    "sudo",
    "su",
    "doas",
    "screen",
    "tmux",
    "nohup",
    "disown",
    "history",
    "tput",
    "clear",
    "reset",
    // --- Accessibility / introspection of *this* sandbox ---
    "ulimit",
    "umask",
    "trap",
];

/// Returns true if `name` is a recognized shell-command alias.
pub fn is_shell_command_alias(name: &str) -> bool {
    SHELL_COMMAND_ALIASES.contains(&name)
}

/// Result of translating a tool call.
pub struct TranslatedCall {
    /// The new tool name — always `shell`.
    pub name: &'static str,
    /// New JSON arguments for the `shell` tool: `{ "command": ["bash", "-lc", ...] }`.
    pub args: JsonValue,
    /// The reconstructed shell command line, for logging.
    pub command_line: String,
}

/// Render a block of text as `apply_patch` hunk lines, each prefixed with
/// `prefix` (`-` for the lines to remove, `+` for the lines to add). A single
/// trailing newline is stripped so we don't emit a spurious bare-prefix line.
/// Empty input yields no lines (a pure insertion or deletion).
fn prefix_hunk_lines(text: &str, prefix: char) -> String {
    if text.is_empty() {
        return String::new();
    }
    text.trim_end_matches('\n')
        .split('\n')
        .map(|line| format!("{prefix}{line}\n"))
        .collect()
}

/// Translate a content-based `edit_file` call into an `apply_patch` Update
/// hunk. Local models handle "find this exact snippet, replace it with that"
/// far more reliably than the unified-diff/hunk format — there are no `@@`
/// headers, no per-line prefixes, and no surrounding context for them to
/// reproduce from memory (the #1 source of "Failed to find context" loops).
/// The `old_string` lines become the `-` lines, which `seek_sequence`
/// fuzzy-locates in the file; `new_string` lines become the `+` lines.
///
/// Accepts `{path|file, old_string|old, new_string|new}`. `new_string` may be
/// empty (a deletion). Returns `None` if `path` or `old_string` is missing /
/// empty (an edit with no anchor can't be located).
pub fn normalize_edit_file_call(args: &JsonValue) -> Option<TranslatedCall> {
    let obj = args.as_object()?;
    let get = |keys: &[&str]| -> Option<String> {
        keys.iter()
            .find_map(|k| obj.get(*k).and_then(|v| v.as_str()))
            .map(str::to_string)
    };
    let path = get(&["path", "file", "file_path", "filename"])?;
    let old = get(&["old_string", "old", "old_str", "search"])?;
    let new = get(&["new_string", "new", "new_str", "replace"]).unwrap_or_default();
    if path.is_empty() || old.is_empty() {
        return None;
    }
    let minus = prefix_hunk_lines(&old, '-');
    let plus = prefix_hunk_lines(&new, '+');
    let body = format!("*** Begin Patch\n*** Update File: {path}\n{minus}{plus}*** End Patch");
    Some(TranslatedCall {
        name: "apply_patch",
        args: serde_json::json!({ "input": body }),
        command_line: format!("edit_file -> apply_patch (Update {path})"),
    })
}

/// Translate a `write_file` call into a `shell` invocation that writes the
/// content to disk, **creating or overwriting** the file. We route through
/// `shell` (not `apply_patch`) for two reasons: `apply_patch`'s `Add File` now
/// refuses to overwrite an existing file (the intended guard), and small models
/// reliably botch the JSON/heredoc escaping of large file content. The content
/// is single-quote-escaped (`'` → `'\''`) so ANY bytes — newlines, quotes, `$`,
/// backticks — are written verbatim, and the command runs under the same
/// sandbox as every other `shell` call. Parent directories are created.
///
/// Accepts `{path|file, content|contents|text}`. Returns `None` if `path` is
/// missing/empty or contains a single quote (which we can't safely quote).
pub fn normalize_write_file_call(args: &JsonValue) -> Option<TranslatedCall> {
    let obj = args.as_object()?;
    let get = |keys: &[&str]| -> Option<String> {
        keys.iter()
            .find_map(|k| obj.get(*k).and_then(|v| v.as_str()))
            .map(str::to_string)
    };
    let path = get(&["path", "file", "file_path", "filename"])?;
    if path.is_empty() || path.contains('\'') {
        return None;
    }
    let content = get(&["content", "contents", "text", "body"]).unwrap_or_default();
    // Wrap content in single quotes; the only sequence that needs escaping
    // inside single quotes is the single quote itself.
    let escaped = content.replace('\'', "'\\''");
    // Echo a confirmation on success: `printf … > file` is silent, and a small
    // model with no positive feedback assumes the write failed and retries (seen
    // live: "let me use write_file again" loops). The byte count + path tells it
    // the write landed.
    let cmd = format!(
        "mkdir -p \"$(dirname '{path}')\" 2>/dev/null; printf '%s' '{escaped}' > '{path}' \
         && printf 'write_file: wrote %s bytes to %s\\n' \"$(wc -c < '{path}')\" '{path}'"
    );
    Some(TranslatedCall {
        name: "shell",
        args: serde_json::json!({ "command": ["bash", "-lc", cmd] }),
        command_line: format!("write_file -> shell (overwrite {path})"),
    })
}

/// Wrap a string for safe use as a single shell argument: surround with single
/// quotes, escaping any interior single quote as `'\''`.
fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Parent directory of a path (everything up to the last `/`), or `""` if none.
fn parent_dir(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[..i],
        None => "",
    }
}

/// Lower a `write_file` call to the **agent-agnostic** substrate: a `shell`
/// invocation that writes the content via base64. The model emits the high-level
/// `write_file`; we translate it to `shell` — the one primitive every coding
/// harness exposes — and the inbound pass ([`parse_shephard_write`]) re-presents
/// the recorded shell call AS `write_file`, so the model only ever sees its own
/// tool. base64 makes the write byte-exact and immune to the ENTIRE shell
/// escaping / quoting / heredoc-marker / trailing-newline bug-class (the payload
/// is pure `[A-Za-z0-9+/=]`), so any file content round-trips. A
/// `# shephard-write:<path_b64>` sentinel makes the call recognizable statelessly
/// (survives process restarts). Parent dirs are created; a byte-count line is
/// echoed so a small model gets positive confirmation the write landed.
///
/// Accepts `{path|file|file_path|filename, content|contents|text|body}`. Returns
/// `None` only if `path` is missing/empty (caller then keeps the real handler).
pub fn write_file_to_base64_shell(args: &JsonValue) -> Option<TranslatedCall> {
    let obj = args.as_object()?;
    let get = |keys: &[&str]| -> Option<String> {
        keys.iter()
            .find_map(|k| obj.get(*k).and_then(|v| v.as_str()))
            .map(str::to_string)
    };
    let path = get(&["path", "file", "file_path", "filename"])?;
    if path.is_empty() {
        return None;
    }
    let content = get(&["content", "contents", "text", "body"]).unwrap_or_default();
    let content_b64 = B64.encode(content.as_bytes());
    let path_b64 = B64.encode(path.as_bytes());
    let q = shell_single_quote(&path);
    let dir = parent_dir(&path);
    let mkdir = if dir.is_empty() {
        String::new()
    } else {
        format!("mkdir -p {} && ", shell_single_quote(dir))
    };
    let cmd = format!(
        "{mkdir}printf %s '{content_b64}' | base64 -d > {q} && \
         printf 'write_file: wrote %s bytes to %s\\n' \"$(wc -c < {q})\" {q}  {SHEPHARD_WRITE_MARKER}{path_b64}"
    );
    Some(TranslatedCall {
        name: "shell",
        args: serde_json::json!({ "command": ["bash", "-lc", cmd] }),
        command_line: format!("write_file -> shell base64 (overwrite {path})"),
    })
}

/// Inbound half of the write_file massage: recognize a `shell` command string
/// produced by [`write_file_to_base64_shell`] and recover the original
/// `(path, content)`. Detection + extraction are purely from the command text
/// (stateless), so re-presentation survives restarts. `None` for any other
/// command.
pub fn parse_shephard_write(command: &str) -> Option<(String, String)> {
    // Path: base64 after the sentinel marker (no shell-special chars, unambiguous).
    let path_b64 = command.rsplit_once(SHEPHARD_WRITE_MARKER)?.1.trim();
    let path = String::from_utf8(B64.decode(path_b64).ok()?).ok()?;
    // Content: base64 between `printf %s '` and the closing quote.
    let after = command.split_once("printf %s '")?.1;
    let content_b64 = after.split_once('\'')?.0;
    let content = String::from_utf8(B64.decode(content_b64).ok()?).ok()?;
    Some((path, content))
}

/// Best-effort recovery of a `write_file`/`create_file` call whose JSON arguments
/// the model botched. A small model cannot reliably JSON-escape an entire file as
/// a string argument: it emits raw (unescaped) newlines and bare double-quotes in
/// the `content` value, so `serde_json` rejects the whole object and the write is
/// lost. Rather than re-prompt it to "escape better" (which it can't), we parse the
/// two fields we need out of the raw text and rebuild a clean `{path, content}`
/// object the real handler can ingest.
///
/// Heuristics (robust to the common failure, not a general JSON repair):
/// - `path` is short and single-line — read to the next unescaped quote.
/// - `content` is assumed to be the LAST field — take everything from its opening
///   quote to the last quote before the closing brace, treating the interior as
///   raw text. Recognized escapes the model *did* emit (`\n`, `\t`, `\"`, `\\`) are
///   still decoded, so fully-escaped, fully-raw, and mixed content all work.
///
/// Returns `None` if the fields can't be located (caller then leaves the malformed
/// call alone, so the handler surfaces a normal error).
pub fn recover_write_file_args(raw: &str) -> Option<JsonValue> {
    let path_at = value_open_quote(raw, &["path", "file", "file_path", "filename"])?;
    let path_rest = &raw[path_at..];
    let path_end = next_unescaped_quote(path_rest)?;
    let path = decode_lenient(&path_rest[..path_end]);
    if path.trim().is_empty() {
        return None;
    }

    let content_at = value_open_quote(
        raw,
        &["content", "contents", "text", "body", "data", "file_text"],
    )?;
    let tail = &raw[content_at..];
    // content is the last field → its closing quote is the last quote in the tail.
    let close = tail.rfind('"')?;
    let content = decode_lenient(&tail[..close]);

    Some(serde_json::json!({ "path": path, "content": content }))
}

/// Byte index just AFTER the opening quote of the first matching `"key": "…"`
/// pair in `raw`, or `None` if no key matches / the value isn't a quoted string.
fn value_open_quote(raw: &str, keys: &[&str]) -> Option<usize> {
    for key in keys {
        let pat = format!("\"{key}\"");
        if let Some(k) = raw.find(&pat) {
            let after_key = k + pat.len();
            let rest = &raw[after_key..];
            if let Some(colon) = rest.find(':') {
                let after_colon = &rest[colon + 1..];
                if let Some(q) = after_colon.find('"') {
                    return Some(after_key + colon + 1 + q + 1);
                }
            }
        }
    }
    None
}

/// Byte index of the first `"` not preceded by an odd number of backslashes.
fn next_unescaped_quote(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let mut bs = 0;
            let mut j = i;
            while j > 0 && bytes[j - 1] == b'\\' {
                bs += 1;
                j -= 1;
            }
            if bs % 2 == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// Decode the standard JSON string escapes a model may have emitted, while
/// passing raw (already-literal) characters through untouched. Unknown escapes
/// keep their backslash so nothing is silently dropped.
fn decode_lenient(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('/') => out.push('/'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// A local model sometimes emits `exec_command` with `cmd` set to `shell`'s ARRAY
/// form — `["bash","-lc","..."]`, as a real array OR a stringified one — instead
/// of the plain command STRING `exec_command` expects. The runner then tries to
/// execute a program literally named `[` and fails with "No such file or
/// directory". Detect that shape and route it to `shell`, which takes the array
/// natively. Returns `None` for a normal string `cmd` (passes through unchanged).
pub fn normalize_exec_command_array(args: &JsonValue) -> Option<TranslatedCall> {
    let cmd = args.get("cmd").or_else(|| args.get("command"))?;
    let arr: Vec<String> = match cmd {
        JsonValue::Array(_) => serde_json::from_value(cmd.clone()).ok()?,
        JsonValue::String(s) => {
            let t = s.trim_start();
            if !t.starts_with('[') {
                return None; // a normal command string — leave exec_command alone
            }
            serde_json::from_str(t).ok()?
        }
        _ => return None,
    };
    if arr.is_empty() {
        return None;
    }
    Some(TranslatedCall {
        name: "shell",
        args: serde_json::json!({ "command": arr }),
        command_line: "exec_command (array cmd) -> shell".to_string(),
    })
}

/// Normalize a read-only `read_file` call into a `shell` invocation that prints
/// the file (optionally a line range). The model supplies only a path and
/// optional bounds; the command itself is built by the harness, so a read-only
/// role cannot smuggle arbitrary shell through it. Paths containing a single
/// quote are refused rather than risk breaking the quoting.
pub fn normalize_read_file_call(args: &JsonValue) -> Option<TranslatedCall> {
    let obj = args.as_object()?;
    let get_str = |keys: &[&str]| -> Option<String> {
        keys.iter()
            .find_map(|k| obj.get(*k).and_then(|v| v.as_str()))
            .map(str::to_string)
    };
    let get_u = |keys: &[&str]| -> Option<u64> {
        keys.iter()
            .find_map(|k| obj.get(*k).and_then(|v| v.as_u64()))
    };
    let path = get_str(&["path", "file", "file_path", "filename"])?;
    if path.is_empty() || path.contains('\'') {
        return None;
    }
    let cmd = match (
        get_u(&["start_line", "start", "from"]),
        get_u(&["end_line", "end", "to"]),
    ) {
        (Some(s), Some(e)) => format!("sed -n '{s},{e}p' '{path}'"),
        (Some(s), None) => format!("sed -n '{s},$p' '{path}'"),
        _ => format!("cat '{path}'"),
    };
    Some(TranslatedCall {
        name: "shell",
        args: serde_json::json!({ "command": ["bash", "-lc", cmd.clone()] }),
        command_line: format!("read_file -> shell ({cmd})"),
    })
}

/// Normalize an `apply_patch` invocation. Two normalizations apply, in order:
///
/// 1. **Unified-diff translation** — when the model emits a standard unified
///    diff (the format `git diff` produces, with `--- a/path` / `+++ b/path`
///    headers and `@@ -L,N +L,N @@` hunks), translate it to Codex's native
///    patch format. Models reach for unified diff because that's what their
///    training corpus is full of; rather than fight that prior, accept it.
/// 2. **Prefix repair** — local models often emit hunk bodies WITHOUT the
///    required `+`/`-`/space prefix on each line (they think they're
///    pasting a file body, not a diff). Detect that and add the missing
///    prefix. Also auto-appends `*** End Patch` when missing.
///
/// Convert a pure single-file `*** Add File:` patch into a `write_file` call.
/// `apply_patch` is being retired for local models (the 9B can't reliably produce
/// matching context); a file-creating patch is just a whole-file write, so routing
/// it to the robust `write_file` handler is both more reliable and avoids the
/// "Cannot add: already exists" failure (write_file overwrites). Returns `None` for
/// anything that isn't a clean single Add — Update/Delete/multi-file/unified-diff
/// patches fall through to normal `apply_patch` normalization.
pub fn apply_patch_add_to_write_file(args: &JsonValue) -> Option<TranslatedCall> {
    let input = args
        .get("input")
        .or_else(|| args.get("patch"))
        .and_then(|v| v.as_str())?;
    if input.contains("*** Update File:") || input.contains("*** Delete File:") {
        return None;
    }
    let adds: Vec<&str> = input
        .lines()
        .filter_map(|l| l.strip_prefix("*** Add File: "))
        .collect();
    if adds.len() != 1 {
        return None; // zero, or multi-file — let the normal path handle it
    }
    let path = adds[0].trim().to_string();
    if path.is_empty() || path.contains('\n') {
        return None;
    }
    // Body = the '+'-prefixed lines after the Add File header, '+' stripped.
    let mut content = String::new();
    let mut in_body = false;
    for line in input.lines() {
        if line.starts_with("*** Add File:") {
            in_body = true;
            continue;
        }
        if line.starts_with("*** ") {
            in_body = false; // *** End Patch or another header
            continue;
        }
        if in_body && let Some(rest) = line.strip_prefix('+') {
            content.push_str(rest);
            content.push('\n');
        }
    }
    Some(TranslatedCall {
        name: "write_file",
        args: serde_json::json!({ "path": path, "content": content }),
        command_line: format!("apply_patch Add -> write_file ({path})"),
    })
}

/// Returns `Some(translated)` only when at least one normalization fired.
pub fn normalize_apply_patch_call(args: &JsonValue) -> Option<TranslatedCall> {
    let obj = args.as_object()?;
    let input = obj
        .get("input")
        .or_else(|| obj.get("patch"))
        .and_then(|v| v.as_str())?;

    let mut working = input.to_string();
    let mut applied: Vec<&str> = Vec::new();

    if let Some(translated) = translate_unified_diff_to_codex(&working) {
        working = translated;
        applied.push("unified-diff translation");
    }
    if let Some(collapsed) = collapse_repeated_patch_wrappers(&working) {
        working = collapsed;
        applied.push("collapsed repeated wrappers");
    }
    if let Some(fixed_add) = fix_add_file_blocks(&working) {
        working = fixed_add;
        applied.push("stripped @@/- from Add File");
    }
    if let Some(fixed) = fix_apply_patch_body(&working) {
        working = fixed;
        applied.push("fixed prefixes");
    }

    if applied.is_empty() {
        return None;
    }

    let mut new_args = obj.clone();
    new_args.insert(
        "input".to_string(),
        serde_json::Value::String(working.clone()),
    );
    new_args.remove("patch");

    Some(TranslatedCall {
        name: "apply_patch",
        args: serde_json::Value::Object(new_args),
        command_line: format!(
            "apply_patch ({}, {} bytes)",
            applied.join(" + "),
            working.len()
        ),
    })
}

/// If `input` looks like a standard unified diff (`--- a/path` / `+++ b/path`
/// with `@@ -L,N +L,N @@` hunk headers), translate it into Codex's native
/// patch format. Returns `None` for inputs that aren't unified diffs (which
/// includes inputs that are already in Codex format).
///
/// Translations applied:
/// - File pairs `--- a/<path>` + `+++ b/<path>` → `*** Update File: <path>`
/// - File pair `--- /dev/null` + `+++ b/<path>` → `*** Add File: <path>`
/// - File pair `--- a/<path>` + `+++ /dev/null` → `*** Delete File: <path>`
/// - Hunk header `@@ -L,N +L,N @@ <ctx>` → `@@ <ctx>` (Codex matches by
///   context, not line numbers; the optional anchor text is preserved when
///   the model included one)
/// - `@@ -L,N +L,N @@` (no anchor) → `@@`
/// - Body lines (`+`, `-`, ` `) pass through unchanged
/// - Wrapped with `*** Begin Patch` / `*** End Patch`
///
/// The path-prefix conventions `a/` and `b/` come from `git diff`; they're
/// stripped since the working directory is implicit. Bare `<path>` (no `a/`
/// or `b/` prefix, as `diff -u` produces) is also accepted.
pub fn translate_unified_diff_to_codex(input: &str) -> Option<String> {
    let lines: Vec<&str> = input.lines().collect();
    if !looks_like_unified_diff(&lines) {
        return None;
    }

    let mut out = String::with_capacity(input.len() + 64);
    out.push_str("*** Begin Patch\n");

    let mut i = 0;
    let mut produced_any_file = false;
    while i < lines.len() {
        let line = lines[i];

        // Skip git's noise headers ("diff --git ...", "index abc..def", etc.)
        if line.starts_with("diff --git ")
            || line.starts_with("index ")
            || line.starts_with("similarity index")
            || line.starts_with("rename from ")
            || line.starts_with("rename to ")
            || line.starts_with("new file mode")
            || line.starts_with("deleted file mode")
            || line.starts_with("old mode")
            || line.starts_with("new mode")
        {
            i += 1;
            continue;
        }

        // File header pair: --- followed by +++.
        if let Some(old_path_raw) = line.strip_prefix("--- ") {
            let next = lines.get(i + 1)?;
            let new_path_raw = next.strip_prefix("+++ ")?;
            let header = file_header(old_path_raw, new_path_raw)?;
            out.push_str(&header);
            out.push('\n');
            produced_any_file = true;
            i += 2;
            continue;
        }

        // Hunk header: @@ -L,N +L,N @@ [optional anchor]
        if let Some(rest) = line.strip_prefix("@@") {
            let translated = translate_hunk_header(rest);
            out.push_str(&translated);
            out.push('\n');
            i += 1;
            continue;
        }

        // Anything else in a unified-diff context is body content (+, -, or
        // space-prefixed) or a "\ No newline at end of file" marker. Drop
        // the no-newline marker; pass the rest through verbatim.
        if line.starts_with("\\ No newline") {
            i += 1;
            continue;
        }
        out.push_str(line);
        out.push('\n');
        i += 1;
    }

    if !produced_any_file {
        // Looked unified-diff-ish but no file headers — bail out and let the
        // input pass through unchanged.
        return None;
    }

    out.push_str("*** End Patch\n");
    Some(out)
}

/// Returns true iff the input has the structural markers of a unified diff:
/// at least one `--- ` / `+++ ` file-header pair followed by a `@@` hunk
/// header. Inputs that already start with `*** Begin Patch` are explicitly
/// rejected so we don't double-process Codex-format input.
fn looks_like_unified_diff(lines: &[&str]) -> bool {
    if lines.iter().any(|l| l.starts_with("*** Begin Patch")) {
        return false;
    }
    let mut saw_minus_header = false;
    let mut saw_plus_header_after = false;
    let mut saw_hunk_header_after = false;
    for line in lines {
        if !saw_minus_header && line.starts_with("--- ") {
            saw_minus_header = true;
            continue;
        }
        if saw_minus_header && !saw_plus_header_after && line.starts_with("+++ ") {
            saw_plus_header_after = true;
            continue;
        }
        if saw_plus_header_after && line.starts_with("@@") {
            saw_hunk_header_after = true;
            break;
        }
    }
    saw_minus_header && saw_plus_header_after && saw_hunk_header_after
}

/// Build the Codex file header (`*** Add File:`, `*** Update File:`, or
/// `*** Delete File:`) from a unified-diff `--- ` / `+++ ` pair. Returns
/// `None` when the pair is unparseable.
fn file_header(old_path_raw: &str, new_path_raw: &str) -> Option<String> {
    let old_path = strip_diff_path_decoration(old_path_raw);
    let new_path = strip_diff_path_decoration(new_path_raw);
    let old_is_null = old_path == "/dev/null";
    let new_is_null = new_path == "/dev/null";
    match (old_is_null, new_is_null) {
        (true, true) => None,
        (true, false) => Some(format!("*** Add File: {new_path}")),
        (false, true) => Some(format!("*** Delete File: {old_path}")),
        (false, false) => Some(format!("*** Update File: {new_path}")),
    }
}

/// Remove the `a/`/`b/` git prefix and any trailing tab-delimited timestamp
/// metadata that `diff -u` appends.
fn strip_diff_path_decoration(raw: &str) -> String {
    let trimmed = raw.trim();
    // `diff -u` emits "path\tYYYY-MM-DD HH:MM:SS.NNN +TZ"; cut at the tab.
    let no_tab = trimmed.split('\t').next().unwrap_or(trimmed).trim();
    let stripped = no_tab
        .strip_prefix("a/")
        .or_else(|| no_tab.strip_prefix("b/"))
        .unwrap_or(no_tab);
    stripped.to_string()
}

/// Translate the portion of a hunk header that follows the leading `@@`.
/// Examples:
///   ` -17,7 +17,7 @@`                     → `@@`
///   ` -17,7 +17,7 @@ def my_function():`  → `@@ def my_function():`
///   ``                                     → `@@` (already empty)
fn translate_hunk_header(rest: &str) -> String {
    // Strip a leading ` -L[,N] +L[,N] @@` segment if present, then preserve
    // any anchor text the model put after the second `@@`.
    let trimmed = rest.trim_start();
    if let Some(after_minus) = trimmed.strip_prefix('-') {
        // Skip "L[,N] +L[,N] @@" and read the optional trailing anchor.
        if let Some(after_at_at) = find_segment_after_at_at(after_minus) {
            let anchor = after_at_at.trim();
            if anchor.is_empty() {
                return "@@".to_string();
            } else {
                return format!("@@ {anchor}");
            }
        }
    }
    // Couldn't recognize the line-number form; pass through with the leading
    // `@@` re-attached so the existing anchor-line semantics still work.
    if rest.is_empty() || rest == " " {
        "@@".to_string()
    } else {
        format!("@@{rest}")
    }
}

/// Helper for `translate_hunk_header`: given the text after the leading `-`
/// (so it starts with `L,N +L,N @@ ...`), return the substring after the
/// closing `@@`. Returns `None` if no closing `@@` is found.
fn find_segment_after_at_at(s: &str) -> Option<&str> {
    s.find("@@").map(|idx| &s[idx + 2..])
}

/// Normalize the text after `@@` in a Codex-envelope hunk header. Strips a
/// leading ` -L[,N] +L[,N] @@` unified-diff segment if present, preserving
/// any anchor text the model put after the second `@@`. The return value
/// does NOT include the leading `@@` — the caller concatenates it back.
///
/// Examples:
///   ``                                     → ``                (no change)
///   ` def my_function():`                  → ` def my_function():` (no change)
///   ` -1,6 +1,6 @@`                        → ``                (stripped)
///   ` -17,7 +17,7 @@ def my_function():`   → ` def my_function():`
fn normalize_codex_hunk_header(rest: &str) -> String {
    let trimmed = rest.trim_start();
    let Some(after_minus) = trimmed.strip_prefix('-') else {
        return rest.to_string();
    };
    // Expect digits immediately after the `-`; otherwise it's a real
    // anchor line that happens to start with `-` (unusual but possible).
    if !after_minus
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit())
    {
        return rest.to_string();
    }
    // Require a closing `@@` somewhere after, otherwise this isn't a
    // line-number header — could be raw content starting with `-`.
    let Some(at_at_idx) = after_minus.find("@@") else {
        return rest.to_string();
    };
    let after_at_at = &after_minus[at_at_idx + 2..];
    let anchor = after_at_at.trim();
    if anchor.is_empty() {
        String::new()
    } else {
        format!(" {anchor}")
    }
}

/// Pre-pass: collapse multiple `*** Begin Patch ... *** End Patch` wrappers
/// into a single one. Some local models emit one wrapper per file when
/// patching multiple files; the apply_patch parser only accepts a single
/// wrapper containing multiple Add/Update/Delete operations. Returns
/// `Some(rewritten)` if a collapse was needed, otherwise `None`.
fn collapse_repeated_patch_wrappers(input: &str) -> Option<String> {
    let begin_count = input.matches("*** Begin Patch").count();
    if begin_count <= 1 {
        return None;
    }
    // Walk lines: keep the first `*** Begin Patch`, drop every subsequent
    // `*** Begin Patch` and every non-final `*** End Patch`, keep the last
    // `*** End Patch` (or none if missing — the prefix-fixer will add one).
    let end_count = input.matches("*** End Patch").count();
    let mut seen_begin = 0usize;
    let mut seen_end = 0usize;
    let mut out = String::with_capacity(input.len());
    for raw_line in input.split_inclusive('\n') {
        let trimmed = raw_line.trim_end_matches(['\n', '\r']);
        if trimmed == "*** Begin Patch" {
            seen_begin += 1;
            if seen_begin > 1 {
                continue; // drop duplicate wrapper opener
            }
        } else if trimmed == "*** End Patch" {
            seen_end += 1;
            if seen_end < end_count {
                continue; // drop intermediate wrapper closer
            }
        }
        out.push_str(raw_line);
    }
    Some(out)
}

/// Pre-pass: when an `*** Add File: <path>` block is followed by `@@` hunk
/// headers and `-` lines (Update File–style), strip them and keep only
/// the `+` lines as the new file's content. Add File creates a new file
/// and only accepts `+` lines — `@@` and `-` are rejected by the parser.
/// Returns `Some(rewritten)` if a fix was applied, else `None`.
fn fix_add_file_blocks(input: &str) -> Option<String> {
    let mut changed = false;
    let mut out = String::with_capacity(input.len());
    let mut in_add_file = false;
    for raw_line in input.split_inclusive('\n') {
        let trimmed = raw_line.trim_end_matches(['\n', '\r']);
        if trimmed.starts_with("*** Add File:") {
            in_add_file = true;
            out.push_str(raw_line);
            continue;
        }
        if trimmed.starts_with("*** End Patch")
            || trimmed.starts_with("*** Update File:")
            || trimmed.starts_with("*** Delete File:")
            || trimmed.starts_with("*** End of File")
        {
            in_add_file = false;
            out.push_str(raw_line);
            continue;
        }
        if in_add_file {
            // In Add File: drop `@@` headers and `-` lines outright.
            // Keep `+` lines (real content), and keep blank lines.
            if trimmed.starts_with("@@") {
                changed = true;
                continue;
            }
            if trimmed.starts_with('-') {
                changed = true;
                continue;
            }
        }
        out.push_str(raw_line);
    }
    if changed { Some(out) } else { None }
}

/// Walk an apply_patch body and prefix any bare content lines with `+`,
/// and auto-append the `*** End Patch` terminator when missing. Returns
/// `None` if no fix was needed.
fn fix_apply_patch_body(input: &str) -> Option<String> {
    let mut output = String::with_capacity(input.len());
    let mut in_hunk = false;
    let mut changed = false;

    for raw_line in input.split_inclusive('\n') {
        // Strip the trailing newline if present so we can match cleanly,
        // remembering whether to re-add it.
        let (line, newline) = match raw_line.strip_suffix('\n') {
            Some(stripped) => (stripped, "\n"),
            None => (raw_line, ""),
        };

        // Patch envelope markers — never modify, but they reset hunk state.
        if line.starts_with("*** Begin Patch")
            || line.starts_with("*** End Patch")
            || line.starts_with("*** Add File:")
            || line.starts_with("*** Update File:")
            || line.starts_with("*** Delete File:")
            || line.starts_with("*** End of File")
        {
            in_hunk = line.starts_with("*** Add File:") || line.starts_with("*** Update File:");
            output.push_str(line);
            output.push_str(newline);
            continue;
        }

        // Hunk context markers (`@@ ... @@`) start a hunk window but are
        // themselves headers, not content. If the model emitted a
        // unified-diff-style header like `@@ -1,6 +1,6 @@` (or
        // `@@ -17,7 +17,7 @@ def foo():`) inside an otherwise-Codex patch,
        // strip the line-number segment — Codex apply_patch treats whatever
        // follows `@@ ` as a literal anchor line, and the line-number form
        // will always fail to match. This is the hybrid case the full
        // unified-diff translator skips because the envelope itself is
        // already Codex format.
        if line.starts_with("@@") {
            in_hunk = true;
            let rest = &line[2..];
            let normalized_header = normalize_codex_hunk_header(rest);
            let new_header_line = if normalized_header == rest {
                line.to_string()
            } else {
                changed = true;
                format!("@@{normalized_header}")
            };
            output.push_str(&new_header_line);
            output.push_str(newline);
            continue;
        }

        if !in_hunk {
            output.push_str(line);
            output.push_str(newline);
            continue;
        }

        // Inside a hunk: lines starting with +, -, or a leading space are
        // already correctly prefixed. Empty lines are also fine (they
        // represent context blank lines and Codex's parser accepts them).
        // Any other line is bare content that the model forgot to prefix.
        let already_prefixed = line.starts_with('+')
            || line.starts_with('-')
            || line.starts_with(' ')
            || line.is_empty();
        if already_prefixed {
            output.push_str(line);
            output.push_str(newline);
            continue;
        }

        output.push('+');
        output.push_str(line);
        output.push_str(newline);
        changed = true;
    }

    // Auto-append `*** End Patch` if the body has at least one `*** Begin Patch`
    // but no closing terminator. Models commonly forget the closing marker.
    let trimmed_end = output.trim_end_matches(['\n', '\r']);
    let has_begin = output.contains("*** Begin Patch");
    let has_end = trimmed_end.ends_with("*** End Patch")
        || output.contains("\n*** End Patch\n")
        || output.contains("\n*** End Patch");
    if has_begin && !has_end {
        if !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str("*** End Patch\n");
        changed = true;
    }

    if changed { Some(output) } else { None }
}

/// Translate a model-emitted tool call to a properly-shaped `shell` call.
///
/// Returns `Some(translated)` in two cases:
///
/// 1. The model called a recognized shell-command alias (e.g. `ls`, `git`):
///    builds a full command line from the alias name + heuristically-extracted
///    args, wrapped as `shell({"command": ["bash", "-lc", "<cmd>"]})`.
///
/// 2. The model called `shell` itself but passed `command` as a string instead
///    of the required array (e.g. `shell({"command": "ls -la"})`): wraps it as
///    `shell({"command": ["bash", "-lc", "ls -la"]})` so Codex's strict tool
///    schema accepts it.
///
/// Returns `None` for anything else — the call passes through unchanged.
///
/// Argument extraction is heuristic — different models pack the args
/// differently. We try the common conventions in order:
///   - `command`/`cmd`/`args`/`argv`/`input` → string or array of args
///   - `path`/`file`/`filename`/`target`/`dir`/`url`/`query`/`pattern` → single positional
///   - everything else → flag-style `--key=value`
pub fn translate_to_shell_call(name: &str, args: &JsonValue) -> Option<TranslatedCall> {
    if name == "shell" {
        return normalize_shell_args(args);
    }
    if !is_shell_command_alias(name) {
        return None;
    }
    let arg_str = extract_args_string(args);
    let command_line = if arg_str.is_empty() {
        name.to_string()
    } else {
        format!("{name} {arg_str}")
    };
    Some(TranslatedCall {
        name: "shell",
        args: serde_json::json!({
            "command": ["bash", "-lc", command_line.clone()],
        }),
        command_line,
    })
}

/// Normalize a `shell` call's `command` field. The schema expects an array of
/// strings (typically `["bash", "-lc", "<command>"]`); local models commonly
/// produce two malformed shapes:
///
/// - String instead of array: `{"command": "ls -la"}` — wrap as
///   `["bash", "-lc", "ls -la"]`.
/// - Double-wrapped array: `["bash", "-lc", "[\"bash\",\"-lc\",\"<cmd>\"]"]`
///   where the third element is the literal JSON of an inner bash invocation
///   — unwrap the inner command line.
///
/// Returns `None` when the call already conforms to the schema.
fn normalize_shell_args(args: &JsonValue) -> Option<TranslatedCall> {
    let obj = args.as_object()?;
    let command = obj.get("command")?;

    // Array case: check for both already-correct and double-wrapped shapes.
    if let Some(arr) = command.as_array() {
        if !arr.iter().all(|v| v.is_string()) {
            return None;
        }
        // Detect double-wrap: ["bash", "-lc", "[\"bash\",\"-lc\",\"<cmd>\"]"]
        if let Some(inner_cmd) = detect_double_wrap(arr) {
            let mut new_args = obj.clone();
            new_args.insert(
                "command".to_string(),
                serde_json::json!(["bash", "-lc", inner_cmd.clone()]),
            );
            return Some(TranslatedCall {
                name: "shell",
                args: serde_json::Value::Object(new_args),
                command_line: inner_cmd,
            });
        }
        // Otherwise the call already conforms.
        return None;
    }

    // String form: wrap with bash -lc — but first unwrap if the string is
    // itself a JSON-encoded shell array, the most common malformed shape we
    // see from local models (`command: "[\"bash\", \"-lc\", \"ls\"]"`).
    let command_str = command
        .as_str()
        .or_else(|| obj.get("cmd").and_then(|v| v.as_str()))?;
    let command_line =
        unwrap_json_shell_string(command_str).unwrap_or_else(|| command_str.trim().to_string());
    if command_line.is_empty() {
        return None;
    }

    // Preserve any other fields the caller passed (e.g. `workdir`, `timeout_ms`).
    let mut new_args = obj.clone();
    new_args.insert(
        "command".to_string(),
        serde_json::json!(["bash", "-lc", command_line.clone()]),
    );
    new_args.remove("cmd");

    Some(TranslatedCall {
        name: "shell",
        args: serde_json::Value::Object(new_args),
        command_line,
    })
}

/// If `s` is a JSON array of strings whose first two elements look like a
/// shell + flag (e.g. `["bash", "-lc", "<cmd>"]`), return the joined inner
/// command line. Otherwise return `None` and let the caller treat `s` as a
/// raw shell line.
fn unwrap_json_shell_string(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return None;
    }
    // Same two-pass approach as detect_double_wrap: try strict parse first,
    // then re-escape control chars inside strings before retrying. Models
    // commonly include literal newlines (heredoc bodies) which break the
    // strict JSON parse.
    if let Some(parsed) = parse_shell_array(trimmed) {
        return Some(parsed);
    }
    let escaped = escape_control_chars_in_strings(trimmed);
    parse_shell_array(&escaped)
}

/// Detect the model's "double-wrapped bash" mistake. Returns the inner command
/// line if the array looks like `["bash", "-lc", "[\"bash\",\"-lc\",\"<cmd>\"]"]`
/// or a close variant.
fn detect_double_wrap(arr: &[JsonValue]) -> Option<String> {
    // Need at least bash + -lc + payload.
    let last = arr.last()?.as_str()?.trim();
    if !last.starts_with('[') || !last.ends_with(']') {
        return None;
    }

    // Strict JSON parse first — covers the clean case.
    if let Some(parsed) = parse_shell_array(last) {
        return Some(parsed);
    }

    // Strict JSON failed. The most common reason is unescaped control
    // characters inside the inner heredoc body (literal `\n`, `\t`, `\r`).
    // Re-escape control chars inside string literals before re-parsing —
    // serde_json will unescape them back to literal characters in the
    // resulting `String` values.
    let escaped = escape_control_chars_in_strings(last);
    parse_shell_array(&escaped)
}

/// Parse a string of the form `["shell", "flag", "cmd"...]` and return the
/// joined `cmd...` if it matches the shell-prefix shape. Returns `None` on any
/// parse failure or if the array doesn't look like a shell invocation.
fn parse_shell_array(s: &str) -> Option<String> {
    let inner: Vec<JsonValue> = serde_json::from_str(s).ok()?;
    if inner.len() < 3 {
        return None;
    }
    let inner_strs: Vec<&str> = inner.iter().filter_map(|v| v.as_str()).collect();
    if inner_strs.len() != inner.len() {
        return None;
    }
    let shell_like = matches!(
        inner_strs[0],
        "bash" | "sh" | "zsh" | "/bin/bash" | "/bin/sh"
    );
    let flag_like = matches!(inner_strs[1], "-c" | "-lc" | "-l");
    if !shell_like || !flag_like {
        return None;
    }
    Some(inner_strs[2..].join(" "))
}

/// Walk the input and escape `\n`, `\r`, `\t` inside JSON string literals.
/// Tracks whether we're currently inside a `"..."` to avoid touching
/// structural whitespace. Backslash-escaped quotes are honored.
fn escape_control_chars_in_strings(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut in_string = false;
    let mut prev = '\0';
    for c in s.chars() {
        if c == '"' && prev != '\\' {
            in_string = !in_string;
            out.push(c);
        } else if in_string {
            match c {
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                other => out.push(other),
            }
        } else {
            out.push(c);
        }
        prev = c;
    }
    out
}

fn extract_args_string(args: &JsonValue) -> String {
    if args.is_null() {
        return String::new();
    }

    // Some models emit arguments as a raw string rather than an object.
    if let Some(s) = args.as_str() {
        return s.trim().to_string();
    }

    let Some(obj) = args.as_object() else {
        return String::new();
    };
    if obj.is_empty() {
        return String::new();
    }

    // 1. Common "rest of command" fields.
    for key in &[
        "command",
        "cmd",
        "args",
        "argv",
        "input",
        "expression",
        "code",
    ] {
        if let Some(v) = obj.get(*key) {
            if let Some(s) = v.as_str() {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    return trimmed.to_string();
                }
            }
            if let Some(arr) = v.as_array() {
                let joined = arr
                    .iter()
                    .filter_map(|a| a.as_str().map(shell_quote_if_needed))
                    .collect::<Vec<_>>()
                    .join(" ");
                if !joined.is_empty() {
                    return joined;
                }
            }
        }
    }

    // 2. Single-positional path-shaped fields.
    for key in &[
        "path",
        "file",
        "filename",
        "filepath",
        "target",
        "dir",
        "directory",
        "folder",
        "url",
        "uri",
        "query",
        "pattern",
        "search",
    ] {
        if let Some(v) = obj.get(*key).and_then(|v| v.as_str()) {
            let trimmed = v.trim();
            if !trimmed.is_empty() {
                return shell_quote_if_needed(trimmed);
            }
        }
    }

    // 3. Last resort — flag-style serialization. This catches things like
    //    `chmod({"mode": "755", "file": "x"})` → `--mode=755 --file=x`. Not
    //    perfect but better than dropping args entirely.
    obj.iter()
        .filter_map(|(k, v)| {
            v.as_str()
                .or_else(|| v.as_bool().map(|b| if b { "true" } else { "false" }))
                .map(|s| format!("--{k}={}", shell_quote_if_needed(s)))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Quote a single argument if it contains shell metacharacters.
fn shell_quote_if_needed(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    let needs_quote = s.chars().any(|c| {
        matches!(
            c,
            ' ' | '\t'
                | '\n'
                | '"'
                | '\''
                | '$'
                | '`'
                | '\\'
                | '|'
                | '&'
                | ';'
                | '<'
                | '>'
                | '('
                | ')'
                | '#'
                | '*'
                | '?'
                | '['
                | ']'
                | '!'
                | '{'
                | '}'
        )
    });
    if !needs_quote {
        return s.to_string();
    }
    // Single-quote and escape any embedded single quotes.
    let escaped = s.replace('\'', r"'\''");
    format!("'{escaped}'")
}

// --- Leaked tool-call recovery (Hermes / ChatML `<tool_call>` format) --------
//
// Qwen-family models (e.g. Qwopus) emit tool calls as:
//   <tool_call>
//   {"name": "exec_command", "arguments": {"cmd": "..."}}
//   </tool_call>
// When the server's chat template doesn't parse this into structured
// `tool_calls` (it should with `--jinja`, but doesn't reliably for every turn),
// the call leaks into the assistant's *text* content. The harness then sees
// zero tool calls and the model's action silently vanishes — and worse, the
// turn can be accepted as a completion. Recovering the call here turns it back
// into a real tool call. The proper fix is server-side parsing; this is the
// safety net.

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Cheap pre-check: does the content carry a leaked `<tool_call>` block?
pub fn has_leaked_tool_call(content: &str) -> bool {
    content.contains(TOOL_CALL_OPEN)
}

/// Parse every `<tool_call>{json}</tool_call>` block in `content` into the
/// Ollama wire shape (`{"function": {"name", "arguments": <string>}}`) so they
/// can go through the same `translate_native_tool_calls` path as real calls.
pub fn parse_leaked_tool_calls(content: &str) -> Vec<JsonValue> {
    let mut out = Vec::new();
    let mut rest = content;
    while let Some(open) = rest.find(TOOL_CALL_OPEN) {
        let after = &rest[open + TOOL_CALL_OPEN.len()..];
        let Some(close) = after.find(TOOL_CALL_CLOSE) else {
            break;
        };
        if let Some(call) = parse_one_leaked_call(after[..close].trim()) {
            out.push(call);
        }
        rest = &after[close + TOOL_CALL_CLOSE.len()..];
    }
    out
}

fn parse_one_leaked_call(inner: &str) -> Option<JsonValue> {
    // Preferred shape: Hermes JSON — `{"name":..,"arguments":{..}}`.
    if let Ok(v) = serde_json::from_str::<JsonValue>(inner)
        && let Some(name) = v.get("name").and_then(|n| n.as_str())
    {
        // `arguments` (Hermes) or `parameters` (some variants); object or string.
        let args_string = match v.get("arguments").or_else(|| v.get("parameters")) {
            Some(JsonValue::String(s)) => s.clone(),
            Some(other) => serde_json::to_string(other).ok()?,
            None => "{}".to_string(),
        };
        return Some(serde_json::json!({
            "function": { "name": name, "arguments": args_string }
        }));
    }
    // Fallback: XML-function shape some finetunes (Ornith, Hermes-2-Pro,
    // Qwen-Agent) emit instead — `<function=NAME><parameter=KEY>VALUE</parameter>…</function>`.
    parse_xml_function_call(inner)
}

/// Parse the XML-style tool call `<function=NAME><parameter=KEY>VALUE</parameter>…`
/// into the Ollama wire shape. Tolerates both `<function=NAME>` and
/// `<function name="NAME">` (likewise for parameters). Numeric/bool values become
/// JSON scalars; everything else stays a string so multi-line shell commands
/// survive intact.
fn parse_xml_function_call(inner: &str) -> Option<JsonValue> {
    const FN_TAG: &str = "<function";
    const P_TAG: &str = "<parameter";
    const P_CLOSE: &str = "</parameter>";

    let fstart = inner.find(FN_TAG)?;
    let after_fn = &inner[fstart + FN_TAG.len()..];
    let head_end = after_fn.find('>')?;
    let name = tag_key(&after_fn[..head_end])?;
    if name.is_empty() {
        return None;
    }

    let mut map = serde_json::Map::new();
    let mut rest = &after_fn[head_end + 1..];
    while let Some(p) = rest.find(P_TAG) {
        let after_tag = &rest[p + P_TAG.len()..];
        let Some(tag_end) = after_tag.find('>') else {
            break;
        };
        let key = tag_key(&after_tag[..tag_end]);
        let body = &after_tag[tag_end + 1..];
        let Some(vclose) = body.find(P_CLOSE) else {
            break;
        };
        if let Some(key) = key.filter(|k| !k.is_empty()) {
            map.insert(key, parse_param_value(body[..vclose].trim()));
        }
        rest = &body[vclose + P_CLOSE.len()..];
    }

    let args_string = serde_json::to_string(&JsonValue::Object(map)).ok()?;
    Some(serde_json::json!({
        "function": { "name": name, "arguments": args_string }
    }))
}

/// Pull the identifier out of an opening-tag remainder, handling both
/// `=NAME` (from `<function=NAME>`) and ` name="NAME"` attribute styles.
fn tag_key(tag_attrs: &str) -> Option<String> {
    let key = tag_attrs
        .split('=')
        .nth(1)?
        .trim()
        .trim_matches('"')
        .trim()
        .to_string();
    Some(key)
}

/// A parameter value is a JSON scalar when it cleanly parses as one (so
/// `max_output_tokens` stays an int), otherwise a string (so a multi-line
/// shell command is preserved verbatim).
fn parse_param_value(raw: &str) -> JsonValue {
    match raw {
        "true" => return JsonValue::Bool(true),
        "false" => return JsonValue::Bool(false),
        _ => {}
    }
    if let Ok(n) = raw.parse::<i64>() {
        return JsonValue::from(n);
    }
    JsonValue::String(raw.to_string())
}

/// Remove the `<tool_call>...</tool_call>` blocks from `content` so the leaked
/// JSON doesn't also show up as prose once it's been promoted to a real call.
pub fn strip_leaked_tool_calls(content: &str) -> String {
    let mut out = String::new();
    let mut rest = content;
    while let Some(open) = rest.find(TOOL_CALL_OPEN) {
        out.push_str(&rest[..open]);
        let after = &rest[open + TOOL_CALL_OPEN.len()..];
        match after.find(TOOL_CALL_CLOSE) {
            Some(close) => rest = &after[close + TOOL_CALL_CLOSE.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell_cmd_of(t: &TranslatedCall) -> String {
        // The translated shell call carries {command: ["bash","-lc", CMD]}.
        t.args
            .get("command")
            .and_then(|c| c.as_array())
            .and_then(|a| a.last())
            .and_then(|s| s.as_str())
            .unwrap()
            .to_string()
    }

    #[test]
    fn write_file_base64_round_trips_hostile_content() {
        // Content packed with everything that breaks naive shell escaping:
        // single + double quotes, raw newlines, $, backticks, and the literal
        // sentinel/heredoc-marker text itself.
        let nasty = "line1 'sq' \"dq\" $VAR `cmd`\nEOF\n# shephard-write:fake\nprintf %s 'x'\n";
        let path = "src/a b/wei'rd.py"; // space + single quote in the path
        let t = write_file_to_base64_shell(&serde_json::json!({ "path": path, "content": nasty }))
            .expect("a normal write_file must translate");
        assert_eq!(t.name, "shell");
        let cmd = shell_cmd_of(&t);
        // It must actually be a base64 write to the right (quoted) path.
        assert!(cmd.contains("| base64 -d > "), "uses base64 decode: {cmd}");
        assert!(cmd.contains("mkdir -p "), "creates the parent dir: {cmd}");
        // The inbound parse recovers the ORIGINAL path + content byte-exact.
        let (got_path, got_content) =
            parse_shephard_write(&cmd).expect("the sentinel'd command must parse back");
        assert_eq!(got_path, path);
        assert_eq!(got_content, nasty);
    }

    #[test]
    fn write_file_base64_needs_a_path() {
        assert!(write_file_to_base64_shell(&serde_json::json!({ "content": "x" })).is_none());
        assert!(
            write_file_to_base64_shell(&serde_json::json!({ "path": "", "content": "x" }))
                .is_none()
        );
    }

    #[test]
    fn parse_shephard_write_ignores_other_shell_commands() {
        assert!(parse_shephard_write("pytest -q && ls -la").is_none());
        assert!(parse_shephard_write("printf %s 'aGk=' | base64 -d > x").is_none()); // no sentinel
    }

    #[test]
    fn exec_command_array_cmd_routes_to_shell() {
        // Stringified array (the observed failure: cmd is a JSON array string).
        let t = normalize_exec_command_array(
            &serde_json::json!({"cmd": "[\"bash\", \"-lc\", \"pytest -q\"]"}),
        )
        .expect("array-shaped cmd should route to shell");
        assert_eq!(t.name, "shell");
        assert_eq!(
            t.args["command"],
            serde_json::json!(["bash", "-lc", "pytest -q"])
        );
        // A real array value works too.
        let t2 = normalize_exec_command_array(&serde_json::json!({"cmd": ["ls", "-la"]}))
            .expect("real array should route to shell");
        assert_eq!(t2.args["command"], serde_json::json!(["ls", "-la"]));
        // A normal string command is left alone (passes through as exec_command).
        assert!(normalize_exec_command_array(&serde_json::json!({"cmd": "ls -la"})).is_none());
    }

    #[test]
    fn apply_patch_add_becomes_write_file() {
        let args = serde_json::json!({"input": "*** Begin Patch\n*** Add File: src/new.py\n+import os\n+\n+def main():\n+    pass\n*** End Patch"});
        let t = apply_patch_add_to_write_file(&args).expect("a pure Add should convert");
        assert_eq!(t.name, "write_file");
        assert_eq!(t.args["path"], "src/new.py");
        assert_eq!(t.args["content"], "import os\n\ndef main():\n    pass\n");
    }

    #[test]
    fn apply_patch_update_is_not_converted() {
        // Updates need context matching — they must NOT be turned into a blind
        // overwrite (that would destroy the rest of the file).
        let args = serde_json::json!({"input": "*** Begin Patch\n*** Update File: a.py\n-old\n+new\n*** End Patch"});
        assert!(apply_patch_add_to_write_file(&args).is_none());
        // Multi-file patch also falls through.
        let multi =
            serde_json::json!({"input": "*** Add File: a.py\n+x\n*** Add File: b.py\n+y\n"});
        assert!(apply_patch_add_to_write_file(&multi).is_none());
    }

    #[test]
    fn recover_write_file_handles_raw_newlines_and_bare_quotes() {
        // The dominant 9B failure: file content dumped with LITERAL newlines and
        // unescaped inner double-quotes → invalid JSON that serde rejects.
        let raw = "{\"path\": \"app.py\", \"content\": \"def greet():\n    print(\"hi\")\n    return 0\"}";
        assert!(
            serde_json::from_str::<JsonValue>(raw).is_err(),
            "precondition: invalid JSON"
        );
        let recovered = recover_write_file_args(raw).expect("should recover");
        assert_eq!(recovered["path"], "app.py");
        let content = recovered["content"].as_str().unwrap();
        assert!(
            content.contains("print(\"hi\")"),
            "inner quotes preserved: {content:?}"
        );
        assert!(
            content.contains("def greet():\n"),
            "raw newline preserved: {content:?}"
        );
    }

    #[test]
    fn recover_write_file_decodes_escaped_content() {
        // Fully-escaped (valid-ish) content also decodes to the same result.
        let raw = r#"{"path":"a.txt","content":"line1\nline2\ttabbed"}"#;
        let recovered = recover_write_file_args(raw).expect("should recover");
        assert_eq!(recovered["content"], "line1\nline2\ttabbed");
    }

    #[test]
    fn recover_write_file_none_without_fields() {
        assert!(recover_write_file_args(r#"{"foo": 1}"#).is_none());
        assert!(recover_write_file_args("not json at all").is_none());
    }

    #[test]
    fn recovers_leaked_hermes_tool_call() {
        // The exact shape that leaked in a real session: the model wrote its
        // pytest call as text instead of a structured call.
        let content = "I'll verify now.\n<tool_call>\n{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"pytest -q\"}}\n</tool_call>\nDone.";
        assert!(has_leaked_tool_call(content));
        let calls = parse_leaked_tool_calls(content);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "exec_command");
        // arguments must be a STRING (Ollama wire shape), re-parseable to the object.
        let args: JsonValue =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["cmd"], "pytest -q");
        // Stripping removes the block but keeps the prose.
        let stripped = strip_leaked_tool_calls(content);
        assert!(!stripped.contains("<tool_call>"));
        assert!(stripped.contains("I'll verify now."));
        assert!(stripped.contains("Done."));
    }

    #[test]
    fn recovers_leaked_xml_function_tool_call() {
        // The exact Ornith leak: <tool_call><function=NAME><parameter=KEY>…
        let leaked = "<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncd /tmp && python3 -c \"print('hi')\"\n</parameter>\n<parameter=max_output_tokens>\n5000\n</parameter>\n</function>\n</tool_call>";
        assert!(has_leaked_tool_call(leaked));
        let calls = parse_leaked_tool_calls(leaked);
        assert_eq!(calls.len(), 1, "should recover one call");
        let f = &calls[0]["function"];
        assert_eq!(f["name"], "exec_command");
        // arguments is a JSON string; parse it back and check the fields.
        let args: serde_json::Value =
            serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["cmd"], "cd /tmp && python3 -c \"print('hi')\"");
        assert_eq!(args["max_output_tokens"], 5000); // numeric, not "5000"
        // The block is stripped from the visible content.
        assert!(strip_leaked_tool_calls(leaked).is_empty());
    }

    #[test]
    fn recovers_real_world_xml_leak_with_pipes_and_redirects() {
        // The EXACT shape Ornith leaked in session 019f05ae: a multi-line shell
        // command with a pipe and a `2>&1` redirect inside the parameter body.
        // The `>` in `2>&1` must NOT be mistaken for a tag close.
        let leaked = "<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncd /home/jesse/src/codex.test.site && python3 -m pytest test_lambda_handler.py -v 2>&1 | tail -40\n</parameter>\n</function>\n</tool_call>";
        assert!(has_leaked_tool_call(leaked));
        let calls = parse_leaked_tool_calls(leaked);
        assert_eq!(calls.len(), 1, "must recover the real leak");
        assert_eq!(calls[0]["function"]["name"], "exec_command");
        let args: serde_json::Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(
            args["cmd"],
            "cd /home/jesse/src/codex.test.site && python3 -m pytest test_lambda_handler.py -v 2>&1 | tail -40"
        );
    }

    #[test]
    fn malformed_leaked_tool_call_is_detected_but_not_parsed() {
        // The real failure: a heredoc whose quotes/newlines break the JSON. The
        // detector must still flag it (so the harness nudges instead of treating
        // the turn as complete), but the parser must gracefully yield nothing
        // rather than panic or recover garbage.
        let bad = "<tool_call>\n{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"python3 << 'EOF'\nprint('x')\"\nEOF\n\"}}\n</tool_call>";
        assert!(has_leaked_tool_call(bad));
        assert!(parse_leaked_tool_calls(bad).is_empty());
    }

    #[test]
    fn leaked_parser_handles_multiple_and_ignores_plain_text() {
        assert!(!has_leaked_tool_call("just prose, no tools"));
        assert!(parse_leaked_tool_calls("just prose").is_empty());
        let two = "<tool_call>{\"name\":\"a\",\"arguments\":{\"x\":1}}</tool_call> mid <tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
        let calls = parse_leaked_tool_calls(two);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["function"]["name"], "a");
        assert_eq!(calls[1]["function"]["name"], "b");
    }

    #[test]
    fn recognizes_common_unix_commands() {
        for name in ["ls", "cat", "grep", "rg", "find", "git", "make", "cargo"] {
            assert!(is_shell_command_alias(name), "missing alias for {name}");
        }
    }

    #[test]
    fn ignores_known_codex_tools() {
        // None of these should be in the shell-command alias set.
        // (`shell` itself has a separate normalization path.)
        for name in [
            "shell",
            "apply_patch",
            "list_dir",
            "view_image",
            "local_web_search",
        ] {
            assert!(
                !is_shell_command_alias(name),
                "should not alias {name} (it's a real Codex tool)"
            );
        }
    }

    #[test]
    fn shell_with_string_command_gets_normalized_to_array() {
        let result =
            translate_to_shell_call("shell", &serde_json::json!({"command": "ls -la"})).unwrap();
        assert_eq!(result.name, "shell");
        assert_eq!(result.command_line, "ls -la");
        assert_eq!(
            result.args,
            serde_json::json!({"command": ["bash", "-lc", "ls -la"]})
        );
    }

    #[test]
    fn shell_with_correct_array_command_passes_through() {
        let result = translate_to_shell_call(
            "shell",
            &serde_json::json!({"command": ["bash", "-lc", "ls"]}),
        );
        assert!(
            result.is_none(),
            "correct shell shape should not be re-translated"
        );
    }

    #[test]
    fn shell_double_wrap_is_unwrapped() {
        let result = translate_to_shell_call(
            "shell",
            &serde_json::json!({
                "command": ["bash", "-lc", "[\"bash\",\"-lc\",\"cat foo.txt\"]"]
            }),
        )
        .unwrap();
        assert_eq!(result.command_line, "cat foo.txt");
        assert_eq!(
            result.args,
            serde_json::json!({"command": ["bash", "-lc", "cat foo.txt"]})
        );
    }

    #[test]
    fn shell_double_wrap_with_multiline_heredoc_is_unwrapped() {
        // Real failure pattern: model double-wraps a heredoc that contains
        // embedded newlines. The naive `serde_json::from_str` fails because
        // JSON requires control characters in strings to be escaped — the
        // raw `\n` in the inner string is invalid JSON.
        let inner = "cat > handle-resolver.ts << 'EOF'\nimport { fetch } from 'undici'\n\ninterface X {}\nEOF";
        let wrapped_third = format!("[\"bash\",\"-lc\",\"{inner}\"]");
        let result = translate_to_shell_call(
            "shell",
            &serde_json::json!({"command": ["bash", "-lc", wrapped_third]}),
        );
        assert!(
            result.is_some(),
            "double-wrap with heredoc/newlines should still be detected"
        );
        let result = result.unwrap();
        assert_eq!(result.command_line, inner);
    }

    /// Mirrors the exact byte shape we observed in the wild from qwen3.5:9b —
    /// a path with multiple slashes, no extra whitespace, the kind of payload
    /// that revealed the original detection bug.
    #[test]
    fn shell_double_wrap_observed_in_wild_is_unwrapped() {
        let inner = "wc -l /home/jesse/src/codex.test.site/tests/test_lambda.py";
        let wrapped_third = format!("[\"bash\",\"-lc\",\"{inner}\"]");
        let result = translate_to_shell_call(
            "shell",
            &serde_json::json!({"command": ["bash", "-lc", wrapped_third]}),
        )
        .expect("double-wrap should be detected");
        assert_eq!(result.command_line, inner);
    }

    #[test]
    fn shell_legitimate_array_brackets_in_command_pass_through() {
        // A command line that legitimately contains array-like syntax (e.g.
        // `python -c '[1,2,3]'`) should NOT be unwrapped — the inner is not a
        // shell-prefixed JSON array.
        let result = translate_to_shell_call(
            "shell",
            &serde_json::json!({
                "command": ["bash", "-lc", "python -c 'print([1,2,3])'"]
            }),
        );
        assert!(result.is_none(), "real shell command should pass through");
    }

    #[test]
    fn shell_with_string_holding_json_shell_array_is_unwrapped() {
        // The model passes `command` as a string, but the string is itself a
        // JSON-encoded shell array — the actual failure mode observed with
        // qwen3.5:9b in local-only mode.
        let result = translate_to_shell_call(
            "shell",
            &serde_json::json!({
                "command": "[\"bash\",\"-lc\",\"wc -l tests/test_lambda.py\"]"
            }),
        )
        .unwrap();
        assert_eq!(result.command_line, "wc -l tests/test_lambda.py");
        assert_eq!(
            result.args,
            serde_json::json!({"command": ["bash", "-lc", "wc -l tests/test_lambda.py"]})
        );
    }

    #[test]
    fn shell_with_string_that_looks_arrayish_but_isnt_passes_through() {
        // A string command that LOOKS like it has brackets but isn't a JSON
        // shell array (e.g. `python -c '[1,2,3]'`) should be wrapped normally.
        let result = translate_to_shell_call(
            "shell",
            &serde_json::json!({"command": "python -c 'print([1,2,3])'"}),
        )
        .unwrap();
        assert_eq!(result.command_line, "python -c 'print([1,2,3])'");
    }

    #[test]
    fn apply_patch_with_missing_plus_prefix_gets_fixed() {
        // The model commonly drops the `+` prefix on add-file content lines.
        let input =
            "*** Begin Patch\n*** Add File: hello.py\nimport sys\nprint('hi')\n*** End Patch\n";
        let result = normalize_apply_patch_call(&serde_json::json!({"input": input})).unwrap();
        let fixed = result.args.get("input").unwrap().as_str().unwrap();
        assert!(
            fixed.contains("+import sys\n"),
            "missing + on import line: {fixed}"
        );
        assert!(
            fixed.contains("+print('hi')\n"),
            "missing + on print line: {fixed}"
        );
        assert!(fixed.contains("*** Begin Patch"));
        assert!(fixed.contains("*** Add File: hello.py"));
    }

    #[test]
    fn apply_patch_missing_end_marker_gets_appended() {
        // Real failure observed: model emits a complete patch body but forgets
        // the trailing `*** End Patch`. Auto-append.
        let input = "*** Begin Patch\n*** Add File: test.ts\n+const a = 1\n+export {}";
        let result = normalize_apply_patch_call(&serde_json::json!({"input": input})).unwrap();
        let fixed = result.args.get("input").unwrap().as_str().unwrap();
        assert!(
            fixed.trim_end().ends_with("*** End Patch"),
            "should auto-append *** End Patch:\n{fixed}"
        );
    }

    #[test]
    fn apply_patch_already_correct_passes_through() {
        let input =
            "*** Begin Patch\n*** Add File: hello.py\n+import sys\n+print('hi')\n*** End Patch\n";
        let result = normalize_apply_patch_call(&serde_json::json!({"input": input}));
        assert!(
            result.is_none(),
            "well-formed patch should not be rewritten"
        );
    }

    #[test]
    fn apply_patch_update_preserves_context_and_minus_lines() {
        let input = "*** Begin Patch\n*** Update File: foo.py\n@@\n-old\n new content line\n+new\n*** End Patch\n";
        // Only `new content line` (without the leading space) would need fixing,
        // but it already has a leading space → context line. Nothing should change.
        let result = normalize_apply_patch_call(&serde_json::json!({"input": input}));
        assert!(result.is_none());
    }

    #[test]
    fn edit_file_translates_to_update_hunk() {
        let call = serde_json::json!({
            "path": "src/a.rs",
            "old_string": "let x = 1;\nlet y = 2;",
            "new_string": "let x = 10;\nlet y = 20;"
        });
        let t = normalize_edit_file_call(&call).unwrap();
        assert_eq!(t.name, "apply_patch");
        let body = t.args.get("input").unwrap().as_str().unwrap();
        assert_eq!(
            body,
            "*** Begin Patch\n*** Update File: src/a.rs\n-let x = 1;\n-let y = 2;\n+let x = 10;\n+let y = 20;\n*** End Patch"
        );
    }

    #[test]
    fn edit_file_empty_new_string_is_a_deletion() {
        let call =
            serde_json::json!({"path": "a.rs", "old_string": "dead_line();", "new_string": ""});
        let body = normalize_edit_file_call(&call)
            .unwrap()
            .args
            .get("input")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(
            body,
            "*** Begin Patch\n*** Update File: a.rs\n-dead_line();\n*** End Patch"
        );
    }

    #[test]
    fn edit_file_requires_path_and_nonempty_old_string() {
        assert!(normalize_edit_file_call(&serde_json::json!({"path": "a.rs"})).is_none());
        assert!(
            normalize_edit_file_call(&serde_json::json!({"old_string": "x", "new_string": "y"}))
                .is_none()
        );
        assert!(
            normalize_edit_file_call(
                &serde_json::json!({"path": "a.rs", "old_string": "", "new_string": "y"})
            )
            .is_none()
        );
    }

    #[test]
    fn edit_file_accepts_alias_keys() {
        let call = serde_json::json!({"file": "a.rs", "old": "alpha", "new": "beta"});
        let body = normalize_edit_file_call(&call)
            .unwrap()
            .args
            .get("input")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(
            body,
            "*** Begin Patch\n*** Update File: a.rs\n-alpha\n+beta\n*** End Patch"
        );
    }

    #[test]
    fn write_file_translates_to_shell_overwrite() {
        // Content with a single quote must be escaped so it writes verbatim and
        // can't break out of the shell quoting.
        let call =
            serde_json::json!({"path": "src/new.rs", "content": "fn main() { let s = 'x'; }\n"});
        let t = normalize_write_file_call(&call).unwrap();
        assert_eq!(t.name, "shell");
        let cmd = t.args["command"][2].as_str().unwrap();
        assert!(cmd.contains("> 'src/new.rs'"), "writes to the path: {cmd}");
        assert!(cmd.contains("mkdir -p"), "creates parent dirs: {cmd}");
        // The embedded single quote is escaped as '\'' (no raw break-out).
        assert!(cmd.contains("'\\''x'\\''"), "single quotes escaped: {cmd}");
        // Confirms success so the model knows the write landed.
        assert!(
            cmd.contains("write_file: wrote"),
            "echoes a confirmation: {cmd}"
        );
    }

    #[test]
    fn write_file_refuses_path_with_single_quote() {
        let call = serde_json::json!({"path": "weird'name.rs", "content": "x"});
        assert!(normalize_write_file_call(&call).is_none());
    }

    #[test]
    fn unified_diff_translation_basic_update() {
        let input = "\
--- a/handler.py
+++ b/handler.py
@@ -17,7 +17,7 @@
             \"body\": json.dumps({\"error\": \"Missing 'handle' in event\"})
         }

-    url = f\"https://api.handle.me/resolve/{handle}\"
+    url = f\"https://api.handle.me/handles/{handle}\"

     try:
         response = requests.get(url)
";
        let translated = translate_unified_diff_to_codex(input).unwrap();
        assert!(translated.starts_with("*** Begin Patch\n"));
        assert!(translated.contains("*** Update File: handler.py\n"));
        assert!(translated.contains("@@\n")); // hunk header collapsed (no anchor)
        assert!(translated.contains("-    url = f\"https://api.handle.me/resolve/{handle}\""));
        assert!(translated.contains("+    url = f\"https://api.handle.me/handles/{handle}\""));
        assert!(translated.trim_end().ends_with("*** End Patch"));
    }

    #[test]
    fn unified_diff_translation_preserves_anchor_after_hunk_header() {
        let input = "\
--- a/lib.py
+++ b/lib.py
@@ -1,3 +1,3 @@ def my_function():
-    foo()
+    bar()
     return None
";
        let translated = translate_unified_diff_to_codex(input).unwrap();
        assert!(translated.contains("@@ def my_function():\n"));
    }

    #[test]
    fn unified_diff_translation_dev_null_means_add_file() {
        let input = "\
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,2 @@
+import sys
+print('hi')
";
        let translated = translate_unified_diff_to_codex(input).unwrap();
        assert!(translated.contains("*** Add File: new_file.py\n"));
        assert!(!translated.contains("*** Update File:"));
    }

    #[test]
    fn unified_diff_translation_dev_null_means_delete_file() {
        let input = "\
--- a/old_file.py
+++ /dev/null
@@ -1,2 +0,0 @@
-import sys
-print('hi')
";
        let translated = translate_unified_diff_to_codex(input).unwrap();
        assert!(translated.contains("*** Delete File: old_file.py\n"));
    }

    #[test]
    fn unified_diff_translation_skips_git_noise_headers() {
        let input = "\
diff --git a/foo.py b/foo.py
index abc1234..def5678 100644
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-old
+new
";
        let translated = translate_unified_diff_to_codex(input).unwrap();
        assert!(!translated.contains("diff --git"));
        assert!(!translated.contains("index abc"));
        assert!(translated.contains("*** Update File: foo.py\n"));
    }

    #[test]
    fn unified_diff_translation_returns_none_for_codex_format() {
        let input = "*** Begin Patch\n*** Add File: hello.py\n+import sys\n*** End Patch\n";
        assert!(translate_unified_diff_to_codex(input).is_none());
    }

    #[test]
    fn unified_diff_translation_returns_none_for_unrelated_text() {
        let input = "Hello world, this is not a diff at all.";
        assert!(translate_unified_diff_to_codex(input).is_none());
    }

    #[test]
    fn unified_diff_strips_no_newline_marker() {
        let input = "\
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-old
+new
\\ No newline at end of file
";
        let translated = translate_unified_diff_to_codex(input).unwrap();
        assert!(!translated.contains("\\ No newline"));
    }

    #[test]
    fn normalize_pipeline_handles_unified_diff_end_to_end() {
        let input = "\
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-old
+new
";
        let result = normalize_apply_patch_call(&serde_json::json!({"input": input})).unwrap();
        let body = result.args.get("input").unwrap().as_str().unwrap();
        assert!(body.starts_with("*** Begin Patch\n"));
        assert!(body.contains("*** Update File: foo.py\n"));
        assert!(body.trim_end().ends_with("*** End Patch"));
        assert!(result.command_line.contains("unified-diff translation"));
    }

    #[test]
    fn codex_hunk_header_with_line_numbers_gets_stripped() {
        let input = "\
*** Begin Patch
*** Update File: handler.py
@@ -1,6 +1,6 @@
 import requests
 import os

-API_BASE_URL = \"https://api.handle.me/resolve/\"
+API_BASE_URL = \"https://api.handle.me/handles/\"

*** End Patch
";
        let result = normalize_apply_patch_call(&serde_json::json!({"input": input})).unwrap();
        let body = result.args.get("input").unwrap().as_str().unwrap();
        // The hybrid hunk header should have been collapsed to bare `@@`.
        assert!(body.contains("\n@@\n"), "expected bare `@@`, got:\n{body}");
        assert!(!body.contains("@@ -1,6 +1,6 @@"));
        // Content around the change is preserved verbatim.
        assert!(body.contains("-API_BASE_URL = \"https://api.handle.me/resolve/\""));
        assert!(body.contains("+API_BASE_URL = \"https://api.handle.me/handles/\""));
    }

    #[test]
    fn codex_hunk_header_with_line_numbers_and_anchor_preserves_anchor() {
        let input = "\
*** Begin Patch
*** Update File: lib.py
@@ -17,7 +17,7 @@ def my_function():
-    foo()
+    bar()
*** End Patch
";
        let result = normalize_apply_patch_call(&serde_json::json!({"input": input})).unwrap();
        let body = result.args.get("input").unwrap().as_str().unwrap();
        // Anchor text is preserved; line numbers are gone.
        assert!(
            body.contains("@@ def my_function():"),
            "expected `@@ def my_function():`, got:\n{body}"
        );
        assert!(!body.contains("-17,7"));
    }

    #[test]
    fn codex_hunk_header_with_real_anchor_is_untouched() {
        // A legitimate `@@ <anchor>` form (no line numbers) must pass through
        // unchanged.
        let input = "\
*** Begin Patch
*** Update File: foo.py
@@ def bar():
-    old
+    new
*** End Patch
";
        // Since this patch is already well-formed, normalize should return None.
        let result = normalize_apply_patch_call(&serde_json::json!({"input": input}));
        assert!(
            result.is_none(),
            "well-formed @@ anchor should not be rewritten"
        );
    }

    #[test]
    fn codex_hunk_header_bare_at_at_is_untouched() {
        let input = "\
*** Begin Patch
*** Update File: foo.py
@@
-    old
+    new
*** End Patch
";
        let result = normalize_apply_patch_call(&serde_json::json!({"input": input}));
        assert!(result.is_none(), "bare `@@` should not be rewritten");
    }

    #[test]
    fn unified_diff_with_bare_path_no_a_b_prefix() {
        // diff -u (without git) emits paths without the a/ b/ prefix.
        let input = "\
--- foo.py\t2026-04-22 02:00:00.000 +0000
+++ foo.py\t2026-04-22 02:01:00.000 +0000
@@ -1 +1 @@
-old
+new
";
        let translated = translate_unified_diff_to_codex(input).unwrap();
        assert!(translated.contains("*** Update File: foo.py\n"));
    }

    #[test]
    fn shell_normalization_preserves_extra_fields() {
        let result = translate_to_shell_call(
            "shell",
            &serde_json::json!({"command": "cargo test", "workdir": "/tmp/foo", "timeout_ms": 60000}),
        )
        .unwrap();
        assert_eq!(result.args.get("workdir").unwrap(), "/tmp/foo");
        assert_eq!(result.args.get("timeout_ms").unwrap(), 60000);
    }

    #[test]
    fn empty_args_runs_bare_command() {
        let t = translate_to_shell_call("ls", &serde_json::json!({})).unwrap();
        assert_eq!(t.command_line, "ls");
        assert_eq!(
            t.args,
            serde_json::json!({"command": ["bash", "-lc", "ls"]})
        );
    }

    #[test]
    fn command_field_string_is_used_as_args() {
        let t = translate_to_shell_call("ls", &serde_json::json!({"command": "-la"})).unwrap();
        assert_eq!(t.command_line, "ls -la");
    }

    #[test]
    fn args_array_is_joined() {
        let t = translate_to_shell_call(
            "git",
            &serde_json::json!({"argv": ["status", "--porcelain"]}),
        )
        .unwrap();
        assert_eq!(t.command_line, "git status --porcelain");
    }

    #[test]
    fn path_field_used_for_cat() {
        let t = translate_to_shell_call("cat", &serde_json::json!({"path": "src/foo.py"})).unwrap();
        assert_eq!(t.command_line, "cat src/foo.py");
    }

    #[test]
    fn pattern_field_used_for_grep() {
        let t = translate_to_shell_call("grep", &serde_json::json!({"pattern": "TODO"})).unwrap();
        assert_eq!(t.command_line, "grep TODO");
    }

    #[test]
    fn paths_with_spaces_are_quoted() {
        let t =
            translate_to_shell_call("cat", &serde_json::json!({"path": "my file.txt"})).unwrap();
        assert_eq!(t.command_line, "cat 'my file.txt'");
    }

    #[test]
    fn flag_style_fallback() {
        let t = translate_to_shell_call(
            "chmod",
            &serde_json::json!({"mode": "755", "file": "run.sh"}),
        )
        .unwrap();
        // file= field is a path, takes priority over mode= flag — exact result
        // depends on iteration order, but the path-shaped field wins.
        assert_eq!(t.command_line, "chmod run.sh");
    }

    #[test]
    fn unknown_tool_returns_none() {
        let result = translate_to_shell_call("apply_patch", &serde_json::json!({}));
        assert!(result.is_none());
    }

    #[test]
    fn args_string_form_is_accepted() {
        let t = translate_to_shell_call("ls", &serde_json::json!("-la /tmp")).unwrap();
        assert_eq!(t.command_line, "ls -la /tmp");
    }
}
