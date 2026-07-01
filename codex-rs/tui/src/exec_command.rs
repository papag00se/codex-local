use std::path::Path;
use std::path::PathBuf;

use codex_shell_command::parse_command::extract_shell_command;
use dirs::home_dir;
use shlex::try_join;

pub(crate) fn escape_command(command: &[String]) -> String {
    try_join(command.iter().map(String::as_str)).unwrap_or_else(|_| command.join(" "))
}

pub(crate) fn strip_bash_lc_and_escape(command: &[String]) -> String {
    let display = if let Some((_, script)) = extract_shell_command(command) {
        shephard_write_display(script).unwrap_or_else(|| script.to_string())
    } else {
        escape_command(command)
    };
    cap_command_display(display)
}

/// Hard cap on the command string handed to the renderer. The exec cell
/// syntax-highlights AND re-wraps this on EVERY frame (`transcript_lines` in
/// `exec_cell/render.rs`). A single tool call can carry tens of KB on one line — a
/// base64 file write, an inlined payload — and highlighting + wrapping a 30 KB line
/// across many cells every frame pegs the render thread and freezes the UI. The
/// full command is never shown in the cell anyway (it snippets), so bounding it
/// here makes the render cost O(1) for ANY command, whether or not we recognize it.
/// This is the durable fix; `shephard_write_display` above is just a nicer label
/// for the common case.
fn cap_command_display(s: String) -> String {
    const MAX: usize = 2000;
    if s.len() <= MAX {
        return s;
    }
    let mut end = MAX;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… [truncated for display — {} bytes]", &s[..end], s.len())
}

/// The local `write_file` massage lowers a file write to a
/// `printf '<base64>' | base64 -d > path` shell command (byte-exact, escaping-proof).
/// Those are tens of KB of base64 on a single line; the TUI stores every command it
/// runs and re-renders the whole history each frame, so a handful of these pegs the
/// render thread and freezes the UI. Collapse them to a one-line `write_file <path>`
/// for display. Detected by the `# shephard-write:` sentinel the massage appends;
/// the path is read from the `> '<path>'` redirect (no base64 decode needed).
fn shephard_write_display(script: &str) -> Option<String> {
    if !script.contains("# shephard-write:") {
        return None;
    }
    let after = script.split("base64 -d >").nth(1)?.trim_start();
    let path = if let Some(rest) = after.strip_prefix('\'') {
        rest.split('\'').next().unwrap_or(rest)
    } else {
        after.split_whitespace().next().unwrap_or(after)
    };
    Some(format!("write_file {}", path.trim()))
}

pub(crate) fn split_command_string(command: &str) -> Vec<String> {
    let Some(parts) = shlex::split(command) else {
        return vec![command.to_string()];
    };
    match shlex::try_join(parts.iter().map(String::as_str)) {
        Ok(round_trip)
            if round_trip == command
                || (!command.contains(":\\")
                    && shlex::split(&round_trip).as_ref() == Some(&parts)) =>
        {
            parts
        }
        _ => vec![command.to_string()],
    }
}

/// If `path` is absolute and inside $HOME, return the part *after* the home
/// directory; otherwise, return the path as-is. Note if `path` is the homedir,
/// this will return and empty path.
pub(crate) fn relativize_to_home<P>(path: P) -> Option<PathBuf>
where
    P: AsRef<Path>,
{
    let path = path.as_ref();
    if !path.is_absolute() {
        // If the path is not absolute, we can’t do anything with it.
        return None;
    }

    let home_dir = home_dir()?;
    let rel = path.strip_prefix(&home_dir).ok()?;
    Some(rel.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_command() {
        let args = vec!["foo".into(), "bar baz".into(), "weird&stuff".into()];
        let cmdline = escape_command(&args);
        assert_eq!(cmdline, "foo 'bar baz' 'weird&stuff'");
    }

    #[test]
    fn base64_write_collapses_to_write_file() {
        // A shephard-write (write_file lowered to base64 shell) must display as a
        // one-liner, not tens of KB of base64 that freezes the render thread.
        let script = "mkdir -p '/home/x/proj' && printf %s 'SGVsbG8gd29ybGQ=' | base64 -d > '/home/x/proj/foo.py' && printf 'write_file: wrote %s bytes to %s\\n' \"$(wc -c < '/home/x/proj/foo.py')\" '/home/x/proj/foo.py'  # shephard-write:L2hvbWUveC9wcm9qL2Zvby5weQ==";
        let args = vec!["bash".into(), "-lc".into(), script.into()];
        assert_eq!(
            strip_bash_lc_and_escape(&args),
            "write_file /home/x/proj/foo.py"
        );
    }

    #[test]
    fn huge_command_is_capped_so_render_stays_cheap() {
        // A 30 KB single-line command with NO shephard sentinel must still be
        // bounded — highlighting + wrapping it every frame is what pegs the UI.
        // The cap is what makes the fix durable for any command, not just writes.
        let script = format!("echo '{}'", "A".repeat(30_000));
        let args = vec!["bash".into(), "-lc".into(), script.into()];
        let out = strip_bash_lc_and_escape(&args);
        assert!(out.len() < 2_200, "must be capped, got {} bytes", out.len());
        assert!(out.contains("truncated for display"));
    }

    #[test]
    fn test_strip_bash_lc_and_escape() {
        // Test bash
        let args = vec!["bash".into(), "-lc".into(), "echo hello".into()];
        let cmdline = strip_bash_lc_and_escape(&args);
        assert_eq!(cmdline, "echo hello");

        // Test zsh
        let args = vec!["zsh".into(), "-lc".into(), "echo hello".into()];
        let cmdline = strip_bash_lc_and_escape(&args);
        assert_eq!(cmdline, "echo hello");

        // Test absolute path to zsh
        let args = vec!["/usr/bin/zsh".into(), "-lc".into(), "echo hello".into()];
        let cmdline = strip_bash_lc_and_escape(&args);
        assert_eq!(cmdline, "echo hello");

        // Test absolute path to bash
        let args = vec!["/bin/bash".into(), "-lc".into(), "echo hello".into()];
        let cmdline = strip_bash_lc_and_escape(&args);
        assert_eq!(cmdline, "echo hello");
    }

    #[test]
    fn split_command_string_round_trips_shell_wrappers() {
        let command =
            shlex::try_join(["/bin/zsh", "-lc", r#"python3 -c 'print("Hello, world!")'"#])
                .expect("round-trippable command");
        assert_eq!(
            split_command_string(&command),
            vec![
                "/bin/zsh".to_string(),
                "-lc".to_string(),
                r#"python3 -c 'print("Hello, world!")'"#.to_string(),
            ]
        );
    }

    #[test]
    fn split_command_string_preserves_non_roundtrippable_windows_commands() {
        let command = r#"C:\Program Files\Git\bin\bash.exe -lc "echo hi""#;
        assert_eq!(split_command_string(command), vec![command.to_string()]);
    }
}
