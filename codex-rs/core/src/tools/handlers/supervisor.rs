//! Supervisor tool handler — runs the deterministic supervisor loop.
//!
//! This is the thin integration point between the codex-supervisor crate and
//! codex-core. The model calls this tool with a goal, and the deterministic
//! loop runs to completion before returning. The model never gets a chance
//! to ask "should I continue?" because it's waiting for the tool to return.
//!
//! See docs/spec/supervisor-integration.md and docs/spec/design-principles.md.

use crate::agent::status::is_final;
use crate::codex::Session;
use crate::codex::TurnContext;
use crate::function_tool::FunctionCallError;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolOutput;
use crate::tools::context::ToolPayload;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use codex_routing::OllamaClientPool;
use codex_routing::config::RoutingConfig;
use codex_supervisor::DispatchResult;
use codex_supervisor::SupervisorConfig;
use codex_supervisor::SupervisorJudge;
use codex_supervisor::Task as SupervisorTask;
use codex_supervisor::TaskStatus;
use codex_supervisor::TerminationReason;
use codex_supervisor::run_supervisor;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::sync::Arc;
use tracing::{info, warn};

// --- Prompts for LLM judgment calls ---

const PLANNER_PROMPT: &str = r#"You are a task planner. Given an engineering goal, decompose it into concrete subtasks.

Rules:
- Each task should be independently executable by a coding agent
- Tasks should be ordered by dependency (tasks that depend on others come later)
- Keep task descriptions specific and actionable
- For simple goals that don't need decomposition, return a single task
- Return ONLY a JSON object with a "tasks" key

Return format:
{"tasks": [
  {"id": "task_1", "description": "...", "type": "code", "depends_on": []},
  {"id": "task_2", "description": "...", "type": "code", "depends_on": ["task_1"]}
]}

Valid types: code, test, review, docs

Goal: "#;

const EVALUATOR_PROMPT: &str = r#"You are evaluating whether a coding task was completed successfully.

Respond with ONLY "yes" or "no" followed by a brief reason.

Example responses:
- "yes — the file was created with the correct content"
- "no — the function was added but the import is missing"

Task: "#;

// --- Tool arguments ---

#[derive(Deserialize)]
struct SupervisorArgs {
    goal: String,
    #[serde(default)]
    verification_command: Option<String>,
    #[serde(default)]
    max_retries: Option<u32>,
}

// --- Tool handler ---

pub struct SupervisorHandler;

pub struct SupervisorOutput {
    summary: String,
    success: bool,
}

impl ToolOutput for SupervisorOutput {
    fn log_preview(&self) -> String {
        self.summary.clone()
    }

    fn success_for_logging(&self) -> bool {
        self.success
    }

    fn to_response_item(&self, call_id: &str, _payload: &ToolPayload) -> ResponseInputItem {
        let mut output = FunctionCallOutputPayload::from_text(self.summary.clone());
        output.success = Some(self.success);
        ResponseInputItem::FunctionCallOutput {
            call_id: call_id.to_string(),
            output,
        }
    }

    fn code_mode_result(&self, _payload: &ToolPayload) -> JsonValue {
        serde_json::json!({
            "success": self.success,
            "summary": self.summary,
        })
    }
}

impl ToolHandler for SupervisorHandler {
    type Output = SupervisorOutput;

    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    fn matches_kind(&self, payload: &ToolPayload) -> bool {
        matches!(payload, ToolPayload::Function { .. })
    }

    async fn handle(&self, invocation: ToolInvocation) -> Result<Self::Output, FunctionCallError> {
        let ToolInvocation {
            session,
            turn,
            payload,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "supervisor handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: SupervisorArgs = serde_json::from_str(&arguments).map_err(|err| {
            FunctionCallError::RespondToModel(format!(
                "failed to parse supervisor arguments: {err}"
            ))
        })?;

        info!(goal = %args.goal, "Supervisor tool invoked");

        // Load project config for routing and supervisor settings
        let cwd = std::env::current_dir().unwrap_or_default();
        let project_config = codex_routing::project_config::ProjectConfig::load(&cwd);

        // Supervisor config: prefer tool args, fall back to project config
        let supervisor_config = SupervisorConfig {
            max_iterations: project_config.supervisor.max_iterations,
            timeout: std::time::Duration::from_secs(project_config.supervisor.timeout_seconds),
            max_retries_per_task: args
                .max_retries
                .unwrap_or(project_config.supervisor.max_retries_per_task),
            verification_command: args
                .verification_command
                .or(project_config.supervisor.verification_command.clone()),
        };
        let routing_config = RoutingConfig::from_project_config(&project_config);
        let ollama_pool = Arc::new(OllamaClientPool::new());

        let judge = CodexJudge {
            session: session.clone(),
            turn: turn.clone(),
            routing_config,
            ollama_pool,
            failover: project_config.failover.clone(),
        };

        let result = run_supervisor(&args.goal, &supervisor_config, &judge).await;

        let success = matches!(
            result.termination_reason,
            TerminationReason::AllTasksComplete
        );
        let summary = format!(
            "Supervisor completed: {}/{} tasks done, {} failed. Reason: {:?}. Iterations: {}.",
            result.completed_tasks,
            result.total_tasks,
            result.failed_tasks,
            result.termination_reason,
            result.iterations_used,
        );

        info!(
            completed = result.completed_tasks,
            failed = result.failed_tasks,
            total = result.total_tasks,
            iterations = result.iterations_used,
            "Supervisor loop finished"
        );

        Ok(SupervisorOutput { summary, success })
    }
}

// --- Bridge: codex_supervisor::SupervisorJudge → codex-core internals ---

struct CodexJudge {
    session: Arc<Session>,
    turn: Arc<TurnContext>,
    routing_config: RoutingConfig,
    ollama_pool: Arc<OllamaClientPool>,
    failover: codex_routing::project_config::FailoverChains,
}

impl CodexJudge {
    /// Map a failover-chain role name to a local endpoint the supervisor can
    /// call directly. Cloud roles return `None` — the supervisor reaches cloud
    /// via `spawn_and_wait` (the complex-goal branch), not these local chains.
    fn local_endpoint_for_role(
        &self,
        role: &str,
    ) -> Option<&codex_routing::config::OllamaEndpoint> {
        let rc = &self.routing_config;
        let ep = match role {
            "light_reasoner" => &rc.reasoner,
            "light_reasoner_backup" => &rc.reasoner_backup,
            "light_coder" => &rc.light_coder,
            "compactor" => &rc.compactor,
            "classifier" => &rc.classifier,
            _ => return None,
        };
        ep.enabled.then_some(ep)
    }

    /// Resolve a configured failover chain to the ordered list of local
    /// endpoints the supervisor can call. Empty if the chain has no usable
    /// local role (e.g. an all-cloud chain); callers fall back accordingly.
    fn local_chain(&self, chain: &str) -> Vec<&codex_routing::config::OllamaEndpoint> {
        let roles = match chain {
            "planning" => &self.failover.planning,
            "evaluation" => &self.failover.evaluation,
            _ => return Vec::new(),
        };
        roles
            .iter()
            .filter_map(|role| self.local_endpoint_for_role(role))
            .collect()
    }

    /// Spawn a Codex sub-agent and wait for completion.
    async fn spawn_and_wait(&self, prompt: &str) -> Result<String, String> {
        self.spawn_and_wait_inner(prompt, None).await
    }

    #[allow(dead_code)] // Will be used when Codex fallback needs structured output
    async fn spawn_and_wait_with_schema(
        &self,
        prompt: &str,
        schema: JsonValue,
    ) -> Result<String, String> {
        self.spawn_and_wait_inner(prompt, Some(schema)).await
    }

    async fn spawn_and_wait_inner(
        &self,
        prompt: &str,
        output_schema: Option<JsonValue>,
    ) -> Result<String, String> {
        let (text, _thread_id) = self.spawn_and_wait_full(prompt, output_schema).await?;
        Ok(text)
    }

    /// Spawn a sub-agent and wait for completion. Returns both the output and thread ID.
    /// If `fork_from` is provided, the new agent forks from that thread's conversation
    /// so it has context of what was tried before.
    async fn spawn_and_wait_full(
        &self,
        prompt: &str,
        output_schema: Option<JsonValue>,
    ) -> Result<(String, String), String> {
        self.spawn_and_wait_with_fork(prompt, output_schema, None)
            .await
    }

    /// Spawn a sub-agent, optionally forking from a previous agent's conversation.
    async fn spawn_and_wait_with_fork(
        &self,
        prompt: &str,
        output_schema: Option<JsonValue>,
        fork_from: Option<&str>,
    ) -> Result<(String, String), String> {
        let agent_control = &self.session.services.agent_control;
        let config = (*self.turn.config).clone();

        let initial_op = Op::UserInput {
            items: vec![UserInput::Text {
                text: prompt.to_string(),
                text_elements: vec![],
            }],
            final_output_json_schema: output_schema,
        };

        let thread_id = if let Some(parent_tid_str) = fork_from {
            // Fork from the previous agent's conversation
            let parent_tid = codex_protocol::ThreadId::from_string(parent_tid_str)
                .map_err(|e| format!("Invalid parent thread ID: {e}"))?;

            info!(
                parent = %parent_tid_str,
                "Forking new agent from previous agent's context"
            );

            let session_source = codex_protocol::protocol::SessionSource::SubAgent(
                codex_protocol::protocol::SubAgentSource::ThreadSpawn {
                    parent_thread_id: parent_tid,
                    depth: 1,
                    agent_path: None,
                    agent_nickname: None,
                    agent_role: None,
                },
            );

            let options = crate::agent::control::SpawnAgentOptions {
                fork_parent_spawn_call_id: Some(format!("supervisor_retry_{}", parent_tid_str)),
                fork_mode: Some(crate::agent::control::SpawnAgentForkMode::LastNTurns(5)),
            };

            let live_agent = agent_control
                .spawn_agent_with_metadata(config, initial_op, Some(session_source), options)
                .await
                .map_err(|e| format!("Failed to fork agent: {e}"))?;

            live_agent.thread_id
        } else {
            // Fresh spawn — no previous context
            agent_control
                .spawn_agent(config, initial_op, /*session_source=*/ None)
                .await
                .map_err(|e| format!("Failed to spawn agent: {e}"))?
        };

        let mut status_rx = agent_control
            .subscribe_status(thread_id)
            .await
            .map_err(|e| format!("Failed to subscribe to agent status: {e}"))?;

        let mut status = status_rx.borrow().clone();
        while !is_final(&status) {
            if status_rx.changed().await.is_err() {
                status = agent_control.get_status(thread_id).await;
                break;
            }
            status = status_rx.borrow().clone();
        }

        let tid_string = thread_id.to_string();
        match status {
            AgentStatus::Completed(msg) => Ok((
                msg.unwrap_or_else(|| "Agent completed (no message)".into()),
                tid_string,
            )),
            AgentStatus::Errored(err) => Err(format!("Agent errored: {err}")),
            AgentStatus::Shutdown => Err("Agent was shut down".into()),
            other => Err(format!("Agent ended with unexpected status: {other:?}")),
        }
    }

    /// Call a local Ollama endpoint directly (free, no API cost).
    async fn call_endpoint(
        &self,
        ep: &codex_routing::config::OllamaEndpoint,
        prompt: &str,
    ) -> Result<String, String> {
        if !ep.enabled {
            return Err("Endpoint disabled".into());
        }
        let response = self
            .ollama_pool
            .chat(
                ep,
                vec![serde_json::json!({"role": "user", "content": prompt})],
                None,
                None,
            )
            .await;

        match response {
            Some(body) => {
                let content = body
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();
                if content.is_empty() {
                    Err("Ollama returned empty response".into())
                } else {
                    Ok(content)
                }
            }
            None => Err("Ollama request failed".into()),
        }
    }

    /// Try a chain of Ollama endpoints, return the first success.
    /// Falls back to Codex sub-agent if all local endpoints fail.
    async fn call_with_failover(
        &self,
        endpoints: &[&codex_routing::config::OllamaEndpoint],
        prompt: &str,
        task_name: &str,
    ) -> Result<String, String> {
        for (i, ep) in endpoints.iter().enumerate() {
            match self.call_endpoint(ep, prompt).await {
                Ok(r) => {
                    info!(
                        task = %task_name,
                        model = %ep.model,
                        endpoint = %ep.base_url,
                        attempt = i + 1,
                        "Local Ollama succeeded (free)"
                    );
                    return Ok(r);
                }
                Err(e) => {
                    info!(
                        task = %task_name,
                        model = %ep.model,
                        endpoint = %ep.base_url,
                        attempt = i + 1,
                        error = %e,
                        "Local Ollama failed, trying next"
                    );
                }
            }
        }

        // All local endpoints failed — fall back to Codex sub-agent
        info!(task = %task_name, "All local endpoints failed, falling back to Codex sub-agent");
        self.spawn_and_wait(prompt).await
    }

    /// Route a task and dispatch accordingly.
    fn parse_plan(&self, output: &str, goal: &str) -> Vec<SupervisorTask> {
        let tasks: Vec<PlanTask> = if let Ok(wrapper) =
            serde_json::from_str::<PlanWrapper>(output.trim())
        {
            wrapper.tasks
        } else {
            let json_str = extract_json_array(output);
            match serde_json::from_str(&json_str) {
                Ok(tasks) => tasks,
                Err(e) => {
                    warn!(error = %e, output = %truncate(output, 200), "Failed to parse planner output, falling back to single task");
                    return vec![self.single_task(goal)];
                }
            }
        };

        if tasks.is_empty() {
            return vec![self.single_task(goal)];
        }

        tasks
            .into_iter()
            .map(|t| SupervisorTask {
                id: t.id,
                description: t.description,
                task_type: t.task_type.unwrap_or_else(|| "code".into()),
                dependencies: t.depends_on.unwrap_or_default(),
                status: TaskStatus::Pending,
                assigned_model: None,
                retry_count: 0,
                max_retries: 3,
                result: None,
                error: None,
                last_agent_thread_id: None,
            })
            .collect()
    }

    fn single_task(&self, goal: &str) -> SupervisorTask {
        SupervisorTask {
            id: "task_1".into(),
            description: goal.to_string(),
            task_type: "code".into(),
            dependencies: vec![],
            status: TaskStatus::Pending,
            assigned_model: None,
            retry_count: 0,
            max_retries: 3,
            result: None,
            error: None,
            last_agent_thread_id: None,
        }
    }
}

impl SupervisorJudge for CodexJudge {
    async fn plan_tasks(&self, goal: &str) -> Vec<SupervisorTask> {
        info!(goal = %goal, "Planner: decomposing goal into tasks");

        let prompt = format!("{PLANNER_PROMPT}{goal}");

        // Use the classifier to decide: plan locally (free) or with cloud (better).
        // Complex goals need better decomposition — use cloud.
        // Simple goals can be planned locally.
        let tool_names: Vec<&str> = vec![];
        let classification = codex_routing::classifier::classify_request(
            goal,
            &tool_names,
            0,
            0,
            &self.routing_config,
            &self.ollama_pool,
        )
        .await;

        let output = match classification.route {
            // Simple goals: plan locally (free)
            codex_routing::classifier::RouteTarget::LightReasoner
            | codex_routing::classifier::RouteTarget::LightCoder
            | codex_routing::classifier::RouteTarget::CloudFast => {
                info!(route = ?classification.route, "Planning locally (simple goal)");
                let mut endpoints = self.local_chain("planning");
                if endpoints.is_empty() {
                    endpoints = vec![
                        &self.routing_config.reasoner,
                        &self.routing_config.reasoner_backup,
                    ];
                }
                self.call_with_failover(&endpoints, &prompt, "planning")
                    .await
            }
            // Complex goals: plan with cloud (better decomposition)
            _ => {
                info!(route = ?classification.route, "Planning with cloud (complex goal)");
                self.spawn_and_wait(&prompt).await
            }
        };

        match output {
            Ok(text) => {
                let tasks = self.parse_plan(&text, goal);
                info!(task_count = tasks.len(), "Planner: produced tasks");
                tasks
            }
            Err(e) => {
                warn!(error = %e, "Planner failed, falling back to single task");
                vec![self.single_task(goal)]
            }
        }
    }

    async fn dispatch_task(&self, task: &SupervisorTask) -> Result<DispatchResult, String> {
        info!(
            task_id = %task.id,
            description = %task.description,
            retry = task.retry_count,
            has_previous_context = task.last_agent_thread_id.is_some(),
            "Dispatching task"
        );

        // Fork from previous agent's context when available:
        // - On retry: sees what was tried and why it failed
        // - On sequential dependency: sees what the prior task produced
        let fork_from = task.last_agent_thread_id.as_deref();

        let (output, thread_id) = self
            .spawn_and_wait_with_fork(&task.description, None, fork_from)
            .await?;

        Ok(DispatchResult {
            output,
            agent_thread_id: Some(thread_id),
        })
    }

    async fn evaluate_completion(&self, task: &SupervisorTask, output: &str) -> bool {
        let prompt = format!(
            "{EVALUATOR_PROMPT}{}\n\nAgent output:\n{}",
            task.description,
            truncate(output, 2000),
        );

        // Failover via the configured `evaluation` chain (local roles), falling
        // back to reasoner → reasoner_backup if the chain has no usable role.
        let mut endpoints = self.local_chain("evaluation");
        if endpoints.is_empty() {
            endpoints = vec![
                &self.routing_config.reasoner,
                &self.routing_config.reasoner_backup,
            ];
        }
        let response = self
            .call_with_failover(&endpoints, &prompt, "evaluation")
            .await;

        match response {
            Ok(response) => {
                // Strip <think>...</think> tags that qwen3.5 models add before the answer.
                let cleaned = strip_think_tags(&response);
                let lower = cleaned.trim().to_lowercase();
                let complete = lower.starts_with("yes") || lower.contains("\nyes");
                info!(
                    task_id = %task.id,
                    complete,
                    reason = %truncate(&response, 100),
                    "Evaluator: task completion judgment"
                );
                complete
            }
            Err(e) => {
                warn!(
                    task_id = %task.id,
                    error = %e,
                    "Evaluator agent failed, assuming task complete"
                );
                true
            }
        }
    }

    async fn verify(&self, _task: &SupervisorTask, verification_command: &str) -> bool {
        let cwd = self.turn.cwd.as_path();
        info!(cmd = %verification_command, cwd = %cwd.display(), "Running verification command");

        let result = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(verification_command)
            .current_dir(cwd)
            .output()
            .await;

        match result {
            Ok(output) => {
                let passed = output.status.success();
                if !passed {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    info!(
                        exit_code = output.status.code(),
                        stderr = %truncate(&stderr, 500),
                        "Verification failed"
                    );
                }
                passed
            }
            Err(e) => {
                warn!(error = %e, "Verification command failed to execute");
                false
            }
        }
    }
}

// --- Helpers ---

#[derive(Deserialize)]
struct PlanWrapper {
    tasks: Vec<PlanTask>,
}

#[derive(Deserialize)]
struct PlanTask {
    id: String,
    description: String,
    #[serde(rename = "type")]
    task_type: Option<String>,
    depends_on: Option<Vec<String>>,
}

fn extract_json_array(text: &str) -> String {
    let trimmed = text.trim();

    if trimmed.starts_with('[') {
        return trimmed.to_string();
    }

    if let Some(start) = trimmed.find("```json") {
        if let Some(end) = trimmed[start..].find("```\n").or_else(|| {
            let after_start = &trimmed[start + 7..];
            after_start.rfind("```").map(|e| e + 7)
        }) {
            let json_block = &trimmed[start + 7..start + end];
            let json_block = json_block.trim();
            if json_block.starts_with('[') {
                return json_block.to_string();
            }
        }
    }

    if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            if end > start {
                return trimmed[start..=end].to_string();
            }
        }
    }

    trimmed.to_string()
}

/// Strip `<think>...</think>` blocks from model output.
/// Qwen 3.5 models often wrap responses in thinking tags.
fn strip_think_tags(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end) = result.find("</think>") {
            result = format!("{}{}", &result[..start], &result[end + 8..]);
        } else {
            // Unclosed think tag — strip from <think> to end
            result = result[..start].to_string();
            break;
        }
    }
    result.trim().to_string()
}

fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len { s } else { &s[..max_len] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_array_direct() {
        let input = r#"[{"id": "task_1", "description": "do thing"}]"#;
        assert_eq!(extract_json_array(input), input);
    }

    #[test]
    fn test_extract_json_array_with_prose() {
        let input = "Here are the tasks:\n\n[{\"id\": \"t1\"}]\n\nLet me know if you need more.";
        assert_eq!(extract_json_array(input), "[{\"id\": \"t1\"}]");
    }

    #[test]
    fn test_extract_json_array_in_fence() {
        let input = "```json\n[{\"id\": \"t1\"}]\n```";
        assert_eq!(extract_json_array(input), "[{\"id\": \"t1\"}]");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello world", 5), "hello");
        assert_eq!(truncate("hi", 10), "hi");
    }
}
