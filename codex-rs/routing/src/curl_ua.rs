//! Best-effort User-Agent injection for `curl` invocations.
//!
//! Two entry points:
//!   * [`inject_curl_user_agent_argv`] — mutates an argv `Vec<String>` when
//!     `argv[0]` is `curl` and no UA flag is present.
//!   * [`inject_curl_user_agent_str`] — returns a new command string with
//!     `--user-agent '<default>'` injected after each unguarded `curl`
//!     token. Conservative: if any UA flag already appears anywhere in the
//!     string we leave it alone rather than risk a double-injection.
//!
//! The shell tool dispatches either argv (`ShellHandler`) or a free-form
//! shell string (`ShellCommandHandler`), so both shapes need coverage. We
//! don't try to parse nested shells, command substitution, or eval-style
//! constructs; those keep their original command verbatim.

use crate::local_web_search::DEFAULT_USER_AGENT;
use regex::Regex;
use std::sync::OnceLock;

/// Inject `-A <DEFAULT_USER_AGENT>` after `curl` in `argv` when `argv[0]` is
/// a bare `curl` invocation with no explicit User-Agent. Returns `true` if
/// the argv was mutated.
pub fn inject_curl_user_agent_argv(argv: &mut Vec<String>) -> bool {
    if !is_curl_basename(argv.first().map(String::as_str)) {
        return false;
    }
    if argv_has_user_agent(&argv[1..]) {
        return false;
    }
    argv.insert(1, "-A".to_string());
    argv.insert(2, DEFAULT_USER_AGENT.to_string());
    true
}

/// Return a copy of `command` with `--user-agent '<DEFAULT_USER_AGENT>'`
/// inserted after each bare `curl` token, unless the string already
/// contains a UA flag — in which case the original string is returned
/// unchanged.
pub fn inject_curl_user_agent_str(command: &str) -> String {
    if ua_flag_regex().is_match(command) {
        return command.to_string();
    }
    if !curl_token_regex().is_match(command) {
        return command.to_string();
    }
    curl_token_regex()
        .replace_all(command, |caps: &regex::Captures<'_>| {
            let prefix = &caps[1];
            format!("{prefix}curl --user-agent '{DEFAULT_USER_AGENT}' ",)
        })
        .into_owned()
}

fn is_curl_basename(arg0: Option<&str>) -> bool {
    let Some(arg0) = arg0 else {
        return false;
    };
    let basename = arg0.rsplit(['/', '\\']).next().unwrap_or(arg0);
    basename == "curl" || basename.eq_ignore_ascii_case("curl.exe")
}

fn argv_has_user_agent(args: &[String]) -> bool {
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "-A" || a == "--user-agent" {
            return true;
        }
        if a.starts_with("--user-agent=") {
            return true;
        }
        if a == "-H" || a == "--header" {
            if let Some(next) = args.get(i + 1)
                && header_name_is_user_agent(next)
            {
                return true;
            }
        } else if let Some(rest) = a.strip_prefix("-H") {
            if !rest.is_empty() && header_name_is_user_agent(rest) {
                return true;
            }
        } else if let Some(rest) = a.strip_prefix("--header=") {
            if header_name_is_user_agent(rest) {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn header_name_is_user_agent(value: &str) -> bool {
    let trimmed = value.trim().trim_start_matches(['\'', '"']);
    let Some(colon) = trimmed.find(':') else {
        return false;
    };
    trimmed[..colon].trim().eq_ignore_ascii_case("user-agent")
}

fn ua_flag_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Detect any of:
        //   -A <value>                     (followed by whitespace)
        //   --user-agent <value>           (with or without =)
        //   -H/--header "User-Agent: ..."  (case-insensitive header name)
        Regex::new(
            r#"(?x)
            (?:^|[\s;|&`(])                   # boundary before the flag
            (?:
                -A\s+\S                       # -A <value>
              | --user-agent(?:\s+|=)\S       # --user-agent <value>
              | (?:-H|--header)(?:\s+|=)
                ['"]?\s*[Uu][Ss][Ee][Rr]-[Aa][Gg][Ee][Nn][Tt]\s*:
            )
            "#,
        )
        .expect("static regex must compile")
    })
}

fn curl_token_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        // Bare `curl` token: at string start, or after one of the shell
        // pipeline/separator characters we recognize. The preceding
        // boundary *and* any trailing whitespace up to `curl` are folded
        // into the capture so the replacement can emit them verbatim — we
        // don't want to collapse `; curl` into `;curl`.
        Regex::new(r"(^\s*|[;|&`(\n]\s*)curl\s+").expect("static regex must compile")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn argv_injects_when_curl_has_no_ua() {
        let mut argv = s(&["curl", "https://example.com"]);
        assert!(inject_curl_user_agent_argv(&mut argv));
        assert_eq!(argv[0], "curl");
        assert_eq!(argv[1], "-A");
        assert_eq!(argv[2], DEFAULT_USER_AGENT);
        assert_eq!(argv[3], "https://example.com");
    }

    #[test]
    fn argv_respects_explicit_dash_a() {
        let mut argv = s(&["curl", "-A", "mybot/1.0", "https://example.com"]);
        assert!(!inject_curl_user_agent_argv(&mut argv));
        assert_eq!(argv, s(&["curl", "-A", "mybot/1.0", "https://example.com"]));
    }

    #[test]
    fn argv_respects_long_flag() {
        let mut argv = s(&["curl", "--user-agent", "mybot", "https://e.x"]);
        assert!(!inject_curl_user_agent_argv(&mut argv));
    }

    #[test]
    fn argv_respects_long_flag_with_equals() {
        let mut argv = s(&["curl", "--user-agent=mybot", "https://e.x"]);
        assert!(!inject_curl_user_agent_argv(&mut argv));
    }

    #[test]
    fn argv_respects_header_form_separate_args() {
        let mut argv = s(&["curl", "-H", "User-Agent: foo", "https://e.x"]);
        assert!(!inject_curl_user_agent_argv(&mut argv));
    }

    #[test]
    fn argv_respects_header_form_joined() {
        let mut argv = s(&["curl", "-HUser-Agent: foo", "https://e.x"]);
        assert!(!inject_curl_user_agent_argv(&mut argv));
    }

    #[test]
    fn argv_ignores_non_curl() {
        let mut argv = s(&["wget", "https://example.com"]);
        assert!(!inject_curl_user_agent_argv(&mut argv));
    }

    #[test]
    fn argv_accepts_absolute_path_to_curl() {
        let mut argv = s(&["/usr/bin/curl", "https://example.com"]);
        assert!(inject_curl_user_agent_argv(&mut argv));
        assert_eq!(argv[1], "-A");
    }

    #[test]
    fn str_injects_into_simple_curl() {
        let out = inject_curl_user_agent_str("curl https://example.com");
        assert!(out.starts_with("curl --user-agent '"));
        assert!(out.contains("https://example.com"));
    }

    #[test]
    fn str_injects_after_pipeline() {
        let out = inject_curl_user_agent_str("curl https://a | jq .");
        assert!(out.starts_with("curl --user-agent '"));
        assert!(out.contains("| jq ."));
    }

    #[test]
    fn str_injects_after_semicolon() {
        let out = inject_curl_user_agent_str("echo hi; curl https://a");
        assert!(out.contains("; curl --user-agent '"));
    }

    #[test]
    fn str_leaves_curl_with_existing_ua_alone() {
        let input = "curl -A 'custom/1.0' https://example.com";
        assert_eq!(inject_curl_user_agent_str(input), input);
    }

    #[test]
    fn str_leaves_curl_with_long_ua_flag_alone() {
        let input = "curl --user-agent 'custom' https://example.com";
        assert_eq!(inject_curl_user_agent_str(input), input);
    }

    #[test]
    fn str_leaves_curl_with_header_ua_alone() {
        let input = "curl -H 'User-Agent: x' https://example.com";
        assert_eq!(inject_curl_user_agent_str(input), input);
    }

    #[test]
    fn str_leaves_commands_without_curl_alone() {
        let input = "ls -la && echo done";
        assert_eq!(inject_curl_user_agent_str(input), input);
    }

    #[test]
    fn str_does_not_match_curly_braces_or_partial_words() {
        let input = "curly_function && curlylookup";
        assert_eq!(inject_curl_user_agent_str(input), input);
    }
}
