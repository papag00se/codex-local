//! Table-driven tests for the trim module.
//!
//! Each test builds a small synthetic transcript via the helper constructors
//! at the bottom of this file, runs `trim_for_local`, and asserts on the
//! resulting messages, prelude, and summary counters.

use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;

use super::TrimInput;
use super::trim_for_local;

#[test]
fn empty_transcript_produces_only_system_prompt() {
    let result = trim_for_local(
        &TrimInput {
            items: &[],
            system_prompt: "You are Codex.",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert_eq!(result.system, "You are Codex.");
    assert!(result.messages.is_empty());
    assert_eq!(result.summary.original_items, 0);
}

#[test]
fn user_instructions_appear_in_prelude() {
    let result = trim_for_local(
        &TrimInput {
            items: &[user_msg("hello")],
            system_prompt: "SYS",
            user_instructions: Some("Don't use mocks."),
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.system.contains("[Persistent project context]"),
        "system: {}",
        result.system
    );
    assert!(result.system.contains("Don't use mocks."));
    assert!(result.system.contains("SYS"));
}

#[test]
fn system_prompt_is_never_stubbed_or_truncated() {
    let long = "A".repeat(20_000);
    let result = trim_for_local(
        &TrimInput {
            items: &[user_msg("hi")],
            system_prompt: &long,
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(result.system.starts_with(&long));
}

#[test]
fn active_turn_user_message_kept_verbatim() {
    let prompt = "I would like to build a hello world Lambda.";
    let result = trim_for_local(
        &TrimInput {
            items: &[user_msg(prompt)],
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert_eq!(result.messages.len(), 1);
    let content = result.messages[0]
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap();
    assert_eq!(content, prompt);
}

#[test]
fn tool_calls_and_outputs_in_active_turn_are_preserved() {
    let items = vec![
        user_msg("read auth.py"),
        function_call(
            "call_1",
            "text_editor",
            r#"{"command":"view","path":"src/auth.py"}"#,
        ),
        function_output("call_1", "<file contents>", true),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    // user message + assistant tool_call + tool output (rendered as user
    // message wrapping <tool_result>) = 3 messages.
    assert_eq!(result.messages.len(), 3, "messages: {:?}", result.messages);
    assert_eq!(result.messages[1].get("role").unwrap(), "assistant");
    assert!(result.messages[1].get("tool_calls").is_some());
    assert_eq!(result.messages[2].get("role").unwrap(), "user");
    let last = result.messages[2]
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap();
    assert!(
        last.contains("<tool_result"),
        "tool output should be wrapped in <tool_result>: {last}"
    );
    assert!(last.contains("<file contents>"));
}

#[test]
fn old_read_then_patch_drops_old_read_output() {
    let items = vec![
        user_msg("turn 1: read"),
        function_call("c1", "text_editor", r#"{"command":"view","path":"foo.py"}"#),
        function_output("c1", "OLD CONTENT", true),
        user_msg("turn 2: patch"),
        function_call(
            "c2",
            "apply_patch",
            r#"{"input":"*** Update File: foo.py\n@@\n-old\n+new\n"}"#,
        ),
        function_output("c2", "patched", true),
        user_msg("turn 3: do something else"),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.summary.stale_reads_dropped >= 1,
        "expected stale read drop, got summary {:?}",
        result.summary
    );
    // The OLD CONTENT string must not appear anywhere in the rendered output.
    let combined: String = result
        .messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !combined.contains("OLD CONTENT"),
        "stale read content leaked into messages:\n{combined}"
    );
}

#[test]
fn duplicate_grep_supersedes_older_output() {
    let args = r#"{"query":"foo","path":"."}"#;
    let items = vec![
        user_msg("turn 1: grep"),
        function_call("g1", "grep_files", args),
        function_output("g1", "OLD MATCHES", true),
        user_msg("turn 2: grep again"),
        function_call("g2", "grep_files", args),
        function_output("g2", "NEW MATCHES", true),
        user_msg("turn 3"),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.summary.superseded_outputs_dropped >= 1,
        "expected supersession, got summary {:?}",
        result.summary
    );
    let combined: String = result
        .messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!combined.contains("OLD MATCHES"));
}

#[test]
fn shell_output_with_nonzero_exit_recognized_as_failure() {
    // Codex's shell handler hardcodes `success: Some(true)`, putting the
    // actual exit code in `metadata.exit_code` inside the content. Trim
    // should detect this and treat the output as a failure, surfacing it
    // in [UNRESOLVED ERRORS] AND tagging it as <tool_error>.
    let items = vec![
        user_msg("run the broken command"),
        function_call(
            "call1",
            "shell",
            r#"{"command":["bash","-lc","rg '*.ts'"]}"#,
        ),
        function_output(
            "call1",
            r#"{"output":"rg: regex parse error","metadata":{"exit_code":2,"duration_seconds":0.1}}"#,
            true,
        ),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.system.contains("[UNRESOLVED ERRORS]"),
        "exit_code 2 should surface as unresolved error:\n{}",
        result.system
    );
    let combined: String = result
        .messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        combined.contains("<tool_error"),
        "failed output should be tagged <tool_error>:\n{combined}"
    );
}

#[test]
fn failed_shell_output_kept_even_in_old_turn() {
    let items = vec![
        user_msg("turn 1: install"),
        function_call("s1", "shell", r#"{"command":"pip install boto3"}"#),
        function_output("s1", "ERROR: pip not found", false),
        user_msg("turn 2"),
        user_msg("turn 3"),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.system.contains("[UNRESOLVED ERRORS]"),
        "system block missing unresolved errors header:\n{}",
        result.system
    );
    assert!(result.system.contains("pip not found"));
}

#[test]
fn repetition_detected_after_three_identical_calls() {
    // Same shell command called three times with identical args — typical
    // "stuck loop" pattern from local models when an API returns the same
    // error repeatedly.
    let curl_args = r#"{"command":["bash","-lc","curl -s https://api.example.com/foo"]}"#;
    let bad_output = r#"{"output":"<!DOCTYPE html><html>404</html>","metadata":{"exit_code":0,"duration_seconds":0.1}}"#;
    let items = vec![
        user_msg("check the api"),
        function_call("c1", "shell", curl_args),
        function_output("c1", bad_output, true),
        function_call("c2", "shell", curl_args),
        function_output("c2", bad_output, true),
        function_call("c3", "shell", curl_args),
        function_output("c3", bad_output, true),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.system.contains("[STOP — REPETITION DETECTED]"),
        "should surface repetition alert:\n{}",
        result.system
    );
    assert!(
        result.system.contains("3 times"),
        "should mention the count:\n{}",
        result.system
    );
    // Alert should be at the top of the system prompt, before other blocks.
    let stop_idx = result.system.find("[STOP").unwrap();
    if let Some(world_idx) = result.system.find("[World state]") {
        assert!(
            stop_idx < world_idx,
            "[STOP] alert should come before [World state]"
        );
    }
}

#[test]
fn repetition_detected_with_short_assistant_text_interleaved() {
    // Regression: local models often emit 1-2 chars of assistant content
    // alongside each tool_call (a single space, ".", or similar filler).
    // Earlier the detector broke the streak on any AssistantText, so it
    // never crossed threshold and the model looped indefinitely on the
    // same `cat <file>` / same `curl` call. The detector must now treat
    // AssistantText as transparent.
    let curl_args = r#"{"command":["bash","-lc","cat /tmp/foo.py"]}"#;
    let items = vec![
        user_msg("look at foo.py"),
        assistant_msg(" "),
        function_call("c1", "shell", curl_args),
        function_output("c1", "content", true),
        assistant_msg("."),
        function_call("c2", "shell", curl_args),
        function_output("c2", "content", true),
        assistant_msg(" "),
        function_call("c3", "shell", curl_args),
        function_output("c3", "content", true),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.system.contains("[STOP — REPETITION DETECTED]"),
        "should fire despite short assistant narration between calls:\n{}",
        result.system
    );
}

#[test]
fn no_repetition_alert_after_only_two_identical_calls() {
    // Two calls is fine — could be a legitimate retry. Don't false-positive.
    let curl_args = r#"{"command":["bash","-lc","curl -s https://api.example.com/foo"]}"#;
    let items = vec![
        user_msg("check the api"),
        function_call("c1", "shell", curl_args),
        function_output("c1", "ok", true),
        function_call("c2", "shell", curl_args),
        function_output("c2", "ok", true),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        !result.system.contains("[STOP — REPETITION DETECTED]"),
        "two calls shouldn't trigger the alert:\n{}",
        result.system
    );
}

#[test]
fn no_repetition_alert_when_calls_have_different_args() {
    let items = vec![
        user_msg("check things"),
        function_call(
            "c1",
            "shell",
            r#"{"command":["bash","-lc","curl https://api.example.com/foo"]}"#,
        ),
        function_output("c1", "x", true),
        function_call(
            "c2",
            "shell",
            r#"{"command":["bash","-lc","curl https://api.example.com/bar"]}"#,
        ),
        function_output("c2", "y", true),
        function_call(
            "c3",
            "shell",
            r#"{"command":["bash","-lc","curl https://api.example.com/baz"]}"#,
        ),
        function_output("c3", "z", true),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        !result.system.contains("[STOP — REPETITION DETECTED]"),
        "different args shouldn't trigger the alert"
    );
}

#[test]
fn repetition_injects_synthetic_tool_result_on_most_recent_call() {
    // Three identical apply_patch calls all returning the same failure: the
    // hard guard must replace the MOST RECENT call's tool-output with the
    // synthesized "stop repeating" result (in addition to the prelude alert),
    // and must do so exactly once (only the last call, not c1/c2).
    let patch = r#"{"input":"*** Begin Patch\n*** Update File: a.rs\n@@\n-x\n+y\n*** End Patch"}"#;
    let items = vec![
        user_msg("fix it"),
        function_call("c1", "apply_patch", patch),
        function_output("c1", "Failed to find context", false),
        function_call("c2", "apply_patch", patch),
        function_output("c2", "Failed to find context", false),
        function_call("c3", "apply_patch", patch),
        function_output("c3", "Failed to find context", false),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    // Reinforcing prelude alert still present.
    assert!(result.system.contains("[STOP — REPETITION DETECTED]"));
    let joined = result
        .messages
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    let hits = joined.matches("[REPEATED CALL BLOCKED BY HARNESS]").count();
    assert_eq!(
        hits, 1,
        "synthetic stop result should appear exactly once (on the last call):\n{joined}"
    );
    // And it must be the most recent call (c3), not an earlier one.
    let c3_output = result
        .messages
        .iter()
        .find(|m| m.to_string().contains("call_id=\\\"c3\\\""))
        .expect("c3 tool output message present");
    assert!(
        c3_output
            .to_string()
            .contains("[REPEATED CALL BLOCKED BY HARNESS]"),
        "c3's output should be the one overridden:\n{c3_output}"
    );
}

#[test]
fn enforces_token_budget_by_truncating_bulky_tool_output() {
    // A single active-turn tool output far larger than the budget MUST be
    // truncated so the rendered prompt fits target_ctx — overflow has to be
    // structurally impossible, not merely unlikely.
    let huge = "lorem ipsum dolor sit amet ".repeat(8000); // ~50k+ tokens
    let items = vec![
        user_msg("do the thing"),
        function_call("c1", "shell", r#"{"command":["bash","-lc","cat big.txt"]}"#),
        function_output("c1", &huge, true),
    ];
    let target = 4096usize;
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        target,
    );
    // The enforced budget is the safety-adjusted ceiling (target / SAFETY_FACTOR),
    // not the raw target — that margin is what keeps the *real* tokenized prompt
    // (which runs larger than the chars/4 estimate) under the model's window.
    let effective = (target as f64 / super::ESTIMATE_SAFETY_FACTOR) as usize;
    assert!(
        result.summary.estimated_input_tokens <= effective,
        "prompt must fit the safety-adjusted budget: {} > {effective} (raw target {target})",
        result.summary.estimated_input_tokens
    );
    // The user's actual request must survive truncation.
    assert!(
        result.messages.iter().any(|m| m
            .get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| s.contains("do the thing"))),
        "user request must be preserved"
    );
    // The bulky tool output must show the truncation marker.
    assert!(
        result.messages.iter().any(|m| m
            .get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| s.contains("truncated to fit"))),
        "bulky tool output should be truncated with a marker"
    );
}

#[test]
fn world_state_includes_stale_warning_for_old_modifications() {
    // Active turn is 5; file modified at turn 1 (4 turns ago) — should warn.
    let items = vec![
        user_msg("turn 0"),
        user_msg("turn 1: edit"),
        function_call(
            "p1",
            "apply_patch",
            r#"{"input":"*** Add File: foo.py\n+x\n"}"#,
        ),
        function_output("p1", "ok", true),
        user_msg("turn 2"),
        user_msg("turn 3"),
        user_msg("turn 4"),
        user_msg("turn 5: active"),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.system.contains("turns ago"),
        "should annotate turns_since on stale modifications:\n{}",
        result.system
    );
    assert!(
        result.system.contains("re-read with `cat <path>`")
            || result.system.contains("re-read with `cat <path>`")
            || result.system.to_lowercase().contains("re-read"),
        "should suggest re-reading stale files:\n{}",
        result.system
    );
}

#[test]
fn apply_patch_failure_gets_recovery_hint() {
    let items = vec![
        user_msg("edit foo.ts"),
        function_call(
            "p1",
            "apply_patch",
            r#"{"input":"*** Begin Patch\n*** Update File: foo.ts\n@@\n-old\n+new\n*** End Patch"}"#,
        ),
        function_output(
            "p1",
            "apply_patch verification failed: Failed to find context '@@' in /tmp/foo.ts",
            false,
        ),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    let combined: String = result
        .messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        combined.contains("→ Hint:"),
        "failed apply_patch should get a recovery hint:\n{combined}"
    );
    assert!(
        combined.contains("write_file"),
        "hint should steer to a write_file rewrite, not a re-read:\n{combined}"
    );
}

#[test]
fn failed_patch_steers_to_write_file_rewrite() {
    // Most-recent apply_patch on foo.ts failed → prelude directive + the
    // structured patch_rewrite_path signal the caller uses to force write_file.
    let base = || {
        vec![
            user_msg("add a function to foo.ts"),
            function_call(
                "p1",
                "apply_patch",
                r#"{"input":"*** Begin Patch\n*** Update File: /work/foo.ts\n-def gone():\n-    pass\n+def gone():\n+    return 1\n*** End Patch"}"#,
            ),
            function_output(
                "p1",
                "apply_patch verification failed: Failed to find expected lines in /work/foo.ts",
                false,
            ),
        ]
    };
    let input_for = |items: &[ResponseItem]| -> super::TrimResult {
        trim_for_local(
            &TrimInput {
                items,
                system_prompt: "SYS",
                user_instructions: None,
                current_files: None,
                flavor: super::super::config::ClientFlavor::Ollama,
                system_budget_pct: 0,
            },
            16384,
        )
    };

    let result = input_for(&base());
    assert_eq!(
        result.patch_rewrite_path.as_deref(),
        Some("/work/foo.ts"),
        "the failed-patch target should be surfaced for the caller to force write_file"
    );
    assert!(
        result
            .system
            .contains("[PATCH DID NOT APPLY — REWRITE THE FILE]"),
        "prelude should carry the rewrite directive:\n{}",
        result.system
    );

    // A later SUCCESSFUL write_file to the same file clears the directive.
    let mut items2 = base();
    items2.push(function_call(
        "w1",
        "write_file",
        r#"{"path":"/work/foo.ts","content":"def gone():\n    return 1\n"}"#,
    ));
    items2.push(function_output("w1", "write_file: wrote 24 bytes", true));
    assert_eq!(
        input_for(&items2).patch_rewrite_path,
        None,
        "a successful rewrite should clear the failed-patch state (no infinite directive)"
    );
}

#[test]
fn shell_regex_failure_gets_glob_hint() {
    let items = vec![
        user_msg("find ts files"),
        function_call("s1", "shell", r#"{"command":["bash","-lc","rg '*.ts'"]}"#),
        function_output(
            "s1",
            r#"{"output":"rg: regex parse error:\n    repetition operator missing expression","metadata":{"exit_code":2,"duration_seconds":0.0}}"#,
            true,
        ),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    let combined: String = result
        .messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        combined.contains("rg --files -g") || combined.contains("→ Hint:"),
        "rg regex failure should suggest --files -g:\n{combined}"
    );
}

#[test]
fn world_state_lists_modified_files() {
    let items = vec![
        user_msg("turn 1: create"),
        function_call(
            "c1",
            "apply_patch",
            r#"{"input":"*** Add File: src/lambda.py\n+import json\n"}"#,
        ),
        function_output("c1", "ok", true),
        user_msg("turn 2"),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.system.contains("Created: src/lambda.py"),
        "expected created file in world state:\n{}",
        result.system
    );
}

#[test]
fn old_assistant_text_dropped_from_messages_but_actions_recorded() {
    let items = vec![
        user_msg("turn 1: do it"),
        assistant_msg("I'll do it."),
        function_call(
            "p1",
            "apply_patch",
            r#"{"input":"*** Add File: foo.py\n+x\n"}"#,
        ),
        function_output("p1", "ok", true),
        user_msg("turn 2: now this"),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    let combined: String = result
        .messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !combined.contains("I'll do it."),
        "old assistant narration leaked:\n{combined}"
    );
    assert!(
        result.system.contains("[Actions taken]"),
        "actions block missing:\n{}",
        result.system
    );
    assert!(result.system.contains("Created foo.py"));
}

#[test]
fn old_successful_shell_output_dropped_action_receipt_kept() {
    // Shell, apply_patch, etc. are "action-only" tools: once we have an
    // [Actions taken] entry in the prelude, the raw output bytes are dead
    // weight. Drop them entirely from older turns.
    let mut long = String::new();
    for i in 0..500 {
        long.push_str(&format!("line {i}\n"));
    }
    let items = vec![
        user_msg("turn 1"),
        function_call("s1", "shell", r#"{"command":"ls -R /"}"#),
        function_output("s1", &long, true),
        user_msg("turn 2 — keep going"),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    let combined: String = result
        .messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !combined.contains("line 0") && !combined.contains("line 499"),
        "old shell output bytes leaked into messages:\n{}",
        &combined.chars().take(2000).collect::<String>()
    );
    assert!(
        result.system.contains("[Actions taken]"),
        "actions block missing:\n{}",
        result.system
    );
    assert!(
        result.system.contains("Ran `ls -R /`"),
        "shell action receipt missing:\n{}",
        result.system
    );
}

#[test]
fn old_grep_output_kept_with_match_cap() {
    // Read-shaped tools (grep, list_dir, text_editor view) keep their data
    // because the model may still reference it. Long match lists get capped.
    let mut matches = String::new();
    for i in 0..50 {
        matches.push_str(&format!("src/file_{i}.rs:1: foo\n"));
    }
    let items = vec![
        user_msg("turn 1"),
        function_call("g1", "grep_files", r#"{"query":"foo","path":"."}"#),
        function_output("g1", &matches, true),
        user_msg("turn 2 — keep going"),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    let combined: String = result
        .messages
        .iter()
        .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        combined.contains("file_0.rs"),
        "head matches not preserved:\n{}",
        &combined.chars().take(2000).collect::<String>()
    );
    assert!(
        combined.contains("more matches elided"),
        "missing grep cap marker"
    );
}

#[test]
fn turn_id_increments_only_on_user_messages() {
    use super::items::parse;
    let items = vec![
        user_msg("first"),
        assistant_msg("response"),
        user_msg("second"),
        assistant_msg("response 2"),
        function_call("c1", "shell", r#"{"command":"ls"}"#),
        function_output("c1", "ok", true),
        user_msg("third"),
    ];
    let parsed = parse(&items);
    assert_eq!(parsed.max_turn_id, 2);
    assert_eq!(parsed.items[0].turn_id(), 0); // first user
    assert_eq!(parsed.items[1].turn_id(), 0); // assistant in turn 0
    assert_eq!(parsed.items[2].turn_id(), 1); // second user
}

#[test]
fn current_file_state_block_injects_modified_file_contents() {
    use std::collections::HashMap;
    // Active turn contains an apply_patch that edits handler.py.
    let patch = "*** Begin Patch\n*** Update File: handler.py\n@@\n-old\n+new\n*** End Patch\n";
    let items = vec![
        user_msg("please edit"),
        function_call(
            "c1",
            "apply_patch",
            &format!(
                "{{\"input\":{}}}",
                serde_json::Value::String(patch.to_string())
            ),
        ),
        function_output("c1", "Success", true),
    ];
    // Caller has re-read the file and passed the current disk content.
    let mut current_files = HashMap::new();
    current_files.insert(
        "handler.py".to_string(),
        "line1\nline2\nline3\n".to_string(),
    );
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: Some(&current_files),
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.system.contains("[Current file state"),
        "missing current-file block:\n{}",
        result.system
    );
    assert!(
        result.system.contains("--- Current content of handler.py"),
        "missing file header:\n{}",
        result.system
    );
    assert!(
        result.system.contains("line1\nline2\nline3"),
        "missing file content:\n{}",
        result.system
    );
    assert!(result.system.contains("--- End of handler.py"));
}

#[test]
fn current_file_state_block_omitted_when_no_files_modified() {
    use std::collections::HashMap;
    let current_files: HashMap<String, String> = HashMap::new();
    let result = trim_for_local(
        &TrimInput {
            items: &[user_msg("hi")],
            system_prompt: "SYS",
            user_instructions: None,
            current_files: Some(&current_files),
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(!result.system.contains("[Current file state"));
}

#[test]
fn same_target_failure_repetition_detected_despite_different_args() {
    // 3 consecutive apply_patch FAILURES on handler.py with different args
    // (different `-` lines each time) should still fire the repetition
    // alert. The exact-signature detector misses this because the
    // signature hashes differ; the same-target-failure detector catches it.
    let mk_patch = |marker: &str| -> String {
        format!(
            "*** Begin Patch\n*** Update File: handler.py\n@@\n-{marker}\n+replacement\n*** End Patch\n"
        )
    };
    let items = vec![
        user_msg("fix it"),
        function_call(
            "c1",
            "apply_patch",
            &format!(
                "{{\"input\":{}}}",
                serde_json::Value::String(mk_patch("foo"))
            ),
        ),
        function_output(
            "c1",
            "apply_patch verification failed: Failed to find expected lines",
            false,
        ),
        function_call(
            "c2",
            "apply_patch",
            &format!(
                "{{\"input\":{}}}",
                serde_json::Value::String(mk_patch("bar"))
            ),
        ),
        function_output(
            "c2",
            "apply_patch verification failed: Failed to find expected lines",
            false,
        ),
        function_call(
            "c3",
            "apply_patch",
            &format!(
                "{{\"input\":{}}}",
                serde_json::Value::String(mk_patch("baz"))
            ),
        ),
        function_output(
            "c3",
            "apply_patch verification failed: Failed to find expected lines",
            false,
        ),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.system.contains("[NO PROGRESS — DIAGNOSE"),
        "expected forced-diagnosis alert (same-target thrash):\n{}",
        result.system
    );
    assert!(
        result.system.contains("consecutive failures on handler.py"),
        "expected failure-streak summary:\n{}",
        result.system
    );
}

#[test]
fn unproductive_recurrence_forces_diagnosis_on_interleaved_thrash() {
    // The model re-runs the SAME failing test 3x with DIFFERENT edits between
    // each run. The runs are never consecutive (an edit sits between them), so
    // the exact-repeat detector misses it — the windowed recurrence detector
    // should catch the repeated test and force a diagnosis instead.
    let test_cmd = r#"{"cmd":"pytest test_x.py"}"#;
    let edit = |s: &str| format!(r#"{{"cmd":"sed -i {s} test_x.py"}}"#);
    let fail = "FAILED test_x.py::test_foo - AssertionError: 1 != 2";
    let items = vec![
        user_msg("make the test pass"),
        function_call("e1", "exec_command", &edit("'1s/a/b/'")),
        function_output("e1", "ok", true),
        function_call("t1", "exec_command", test_cmd),
        function_output("t1", fail, false),
        function_call("e2", "exec_command", &edit("'2s/c/d/'")),
        function_output("e2", "ok", true),
        function_call("t2", "exec_command", test_cmd),
        function_output("t2", fail, false),
        function_call("e3", "exec_command", &edit("'3s/e/f/'")),
        function_output("e3", "ok", true),
        function_call("t3", "exec_command", test_cmd),
        function_output("t3", fail, false),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        result.system.contains("[NO PROGRESS — DIAGNOSE"),
        "expected forced-diagnosis on interleaved thrash:\n{}",
        result.system
    );
    // Not a byte-identical consecutive loop, so the plain STOP banner must NOT fire.
    assert!(!result.system.contains("[STOP — REPETITION DETECTED]"));
}

#[test]
fn productive_recurrence_does_not_force_diagnosis() {
    // Same command runs 3x interleaved with edits, but SUCCEEDS every time
    // (e.g. keeping a passing test green while refactoring, or routine re-runs of
    // ls/grep/git status). That is healthy, not thrash — the forced-diagnosis
    // nudge must NOT fire just because a signature recurs.
    let test_cmd = r#"{"cmd":"pytest test_x.py"}"#;
    let edit = |s: &str| format!(r#"{{"cmd":"sed -i {s} test_x.py"}}"#);
    let pass = "1 passed in 0.01s";
    let items = vec![
        user_msg("keep the test green while refactoring"),
        function_call("e1", "exec_command", &edit("'1s/a/b/'")),
        function_output("e1", "ok", true),
        function_call("t1", "exec_command", test_cmd),
        function_output("t1", pass, true),
        function_call("e2", "exec_command", &edit("'2s/c/d/'")),
        function_output("e2", "ok", true),
        function_call("t2", "exec_command", test_cmd),
        function_output("t2", pass, true),
        function_call("e3", "exec_command", &edit("'3s/e/f/'")),
        function_output("e3", "ok", true),
        function_call("t3", "exec_command", test_cmd),
        function_output("t3", pass, true),
    ];
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    assert!(
        !result.system.contains("[NO PROGRESS — DIAGNOSE"),
        "successful repeats must not force diagnosis:\n{}",
        result.system
    );
    assert!(!result.system.contains("[STOP — REPETITION DETECTED]"));
}

#[test]
fn escalation_excises_the_loop_and_reframes() {
    // A byte-identical call repeated past the escalation threshold should be
    // PRUNED from the rendered messages (so the model can't copy it out of its
    // own context) and replaced by a context-reset reframe in the prelude.
    let probe = r#"{"cmd":"python3 -c \"print(1)\""}"#;
    let mut items = vec![user_msg("make it pass")];
    for i in 0..7 {
        items.push(function_call(&format!("c{i}"), "exec_command", probe));
        items.push(function_output(&format!("c{i}"), "1", true));
    }
    let result = trim_for_local(
        &TrimInput {
            items: &items,
            system_prompt: "SYS",
            user_instructions: None,
            current_files: None,
            flavor: super::super::config::ClientFlavor::Ollama,
            system_budget_pct: 0,
        },
        16384,
    );
    // The reframe fired.
    assert!(
        result.system.contains("[HARNESS — STUCK; LOOP REMOVED"),
        "expected context-reset reframe:\n{}",
        result.system
    );
    // Every loop call+output was excised from the messages.
    let leaks = result
        .messages
        .iter()
        .filter(|m| m.to_string().contains("print(1)"))
        .count();
    assert_eq!(
        leaks, 0,
        "loop calls must be pruned from messages:\n{result:#?}"
    );
    let tool_call_msgs = result
        .messages
        .iter()
        .filter(|m| m.get("tool_calls").is_some())
        .count();
    assert_eq!(tool_call_msgs, 0, "no loop tool-calls should remain");
}

// --- helpers ------------------------------------------------------------

fn user_msg(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        end_turn: None,
        phase: None,
    }
}

fn assistant_msg(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: text.to_string(),
        }],
        end_turn: None,
        phase: None,
    }
}

fn function_call(call_id: &str, name: &str, arguments: &str) -> ResponseItem {
    ResponseItem::FunctionCall {
        id: None,
        name: name.to_string(),
        namespace: None,
        arguments: arguments.to_string(),
        call_id: call_id.to_string(),
    }
}

fn function_output(call_id: &str, content: &str, success: bool) -> ResponseItem {
    ResponseItem::FunctionCallOutput {
        call_id: call_id.to_string(),
        output: FunctionCallOutputPayload {
            body: codex_protocol::models::FunctionCallOutputBody::Text(content.to_string()),
            success: Some(success),
        },
    }
}

#[test]
fn drop_oldest_until_fit_drops_front_and_keeps_recent() {
    use serde_json::json;
    // Five assistant messages, each ~400 chars of protected content (tool-data
    // truncation can't touch assistant messages). With a tight budget the
    // last-resort pass must drop the OLDEST until it fits, keeping the newest.
    let big = "x".repeat(400);
    let mut messages = vec![
        json!({"role": "assistant", "content": format!("OLDEST {big}")}),
        json!({"role": "assistant", "content": format!("second {big}")}),
        json!({"role": "assistant", "content": format!("third {big}")}),
        json!({"role": "assistant", "content": format!("fourth {big}")}),
        json!({"role": "user", "content": "NEWEST request"}),
    ];
    let dropped = super::drop_oldest_until_fit("sys", &mut messages, 60);
    assert!(dropped > 0, "should have dropped oldest content");
    // The newest message is always retained.
    assert_eq!(messages.last().unwrap()["content"], "NEWEST request");
    // The oldest was dropped first.
    assert_ne!(
        messages.first().unwrap()["content"],
        json!(format!("OLDEST {big}"))
    );
    // It actually fits now (or is down to the single final message).
    let fit = super::estimate_messages_tokens(&messages) + crate::metrics::estimate_tokens("sys")
        <= 60
        || messages.len() == 1;
    assert!(fit, "must converge to fit or a single message");
}

#[test]
fn drop_oldest_strips_leading_orphan_tool_message() {
    use serde_json::json;
    // After dropping an assistant turn, a leading `tool` result would be orphaned;
    // it must be stripped so the wire payload stays well-formed.
    let big = "y".repeat(800);
    let mut messages = vec![
        json!({"role": "assistant", "content": format!("a {big}")}),
        json!({"role": "tool", "content": format!("orphan {big}")}),
        json!({"role": "user", "content": "final"}),
    ];
    super::drop_oldest_until_fit("", &mut messages, 20);
    assert_ne!(
        messages.first().unwrap()["role"],
        json!("tool"),
        "no leading orphan tool message"
    );
}

#[test]
fn compress_system_prompt_keeps_head_and_tail_when_over_budget() {
    // Build a big system prompt with a distinctive head and tail.
    let body = "MIDDLE ".repeat(4000); // ~28k chars
    let text = format!("HEAD-ROLE-FRAMING\n{body}\nTAIL-OUTPUT-RULES");
    // Budget it tiny (100 est tokens ≈ 400 chars) so it must compress.
    let out = super::compress_system_prompt(&text, 100);
    assert!(out.len() < text.len(), "should shrink");
    assert!(out.contains("HEAD-ROLE-FRAMING"), "keeps the head");
    assert!(out.contains("TAIL-OUTPUT-RULES"), "keeps the tail");
    assert!(out.contains("middle elided"), "marks the elision");
}

#[test]
fn compress_system_prompt_untouched_when_within_budget_or_disabled() {
    let text = "You are a careful coding agent. Always use tools.";
    // Generous budget → unchanged.
    assert_eq!(super::compress_system_prompt(text, 10_000), text);
    // usize::MAX (disabled) → unchanged.
    assert_eq!(super::compress_system_prompt(text, usize::MAX), text);
}

#[test]
fn drop_oldest_always_keeps_a_user_message() {
    use serde_json::json;
    // A huge turn: user request (oldest) + many big assistant/tool messages.
    let big = "z".repeat(2000);
    let mut messages = vec![
        json!({"role": "user", "content": "Build the Ada handle resolver"}),
        json!({"role": "assistant", "content": format!("a {big}")}),
        json!({"role": "tool", "content": format!("t {big}")}),
        json!({"role": "assistant", "content": format!("b {big}")}),
        json!({"role": "tool", "content": format!("u {big}")}),
    ];
    super::drop_oldest_until_fit("sys", &mut messages, 30); // tiny budget → drop almost all
    // Even though the user message was the OLDEST (dropped first), one survives —
    // otherwise Ornith's template raises "No user query found" (a 500).
    assert!(
        messages.iter().any(|m| m["role"] == "user"),
        "a user message must always survive: {messages:?}"
    );
}
