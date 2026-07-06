//! Project-level configuration for multi-agent routing.
//!
//! Loaded from `.codex-multi/config.toml` in the working directory.
//! Separate from `~/.codex/config.toml` — does not affect the base Codex config.
//!
//! See docs/spec/model-routing-strategy.md.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// A single model endpoint entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub provider: String,
    #[serde(default)]
    pub endpoint: Option<String>,
    pub model: String,
    #[serde(default = "default_weight")]
    pub weight: u32,
    #[serde(default = "default_reasoning")]
    pub reasoning: String,
    #[serde(default, alias = "num_ctx")]
    pub trim_budget: Option<usize>,
}

fn default_weight() -> u32 {
    100
}
fn default_reasoning() -> String {
    "off".into()
}

/// A model role — may have a single entry or weighted distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelRole {
    Single {
        provider: String,
        #[serde(default)]
        endpoint: Option<String>,
        model: String,
        #[serde(default = "default_reasoning")]
        reasoning: String,
        #[serde(default)]
        trim_budget: Option<usize>,
        /// `"focused"` (default, ~6 tools) or `"full"` (entire catalog).
        /// Override per-model when the local model can handle a larger
        /// catalog without losing focus.
        #[serde(default)]
        tool_subset: Option<String>,
        /// Hard ceiling on output tokens per response — the CONVENTIONAL
        /// `max_tokens` meaning (OpenAI `max_tokens` / Ollama `num_predict`).
        /// Unset (the default) — or explicitly `0` — means NO cap: the model
        /// generates into whatever room the window has. Setting it opts into a
        /// hard cap, which risks truncating a large `write_file` mid-content, so
        /// leave it unset unless you specifically want a ceiling. Runaway is
        /// bounded by the live rumination detector + `timeout_seconds`, not this.
        #[serde(default)]
        max_tokens: Option<usize>,
        /// Tokens of the context window to RESERVE for the model's output when
        /// sizing the input budget (see `effective_window`): input is trimmed to
        /// `n_ctx − output_reserve − margin`, so the model always has ≥ this much
        /// room to generate. This is NOT a hard cap (that's `max_tokens`) — it's
        /// the input/output split of a fixed window. Unset → a built-in default.
        /// Give file-writing roles (coder) a generous value so big writes fit.
        #[serde(default)]
        output_reserve: Option<usize>,
        /// Per-request HTTP timeout in seconds. Defaults to 300. Raise
        /// for slow reasoning models, lower for fast-fail behavior.
        #[serde(default)]
        timeout_seconds: Option<u64>,
        /// Sampler overrides. When unset, `temperature` falls back to a
        /// reasoning-based default (0.0 off / 0.1 on) and the rest are omitted
        /// from the request (server default applies). Some local models (e.g.
        /// Gemma 4) degenerate / repeat at the stock sampler and need explicit
        /// values — Gemma 4's author recommends `temperature = 1.0`,
        /// `top_p = 0.95`, `top_k = 64`, `repeat_penalty = 1.1`.
        #[serde(default)]
        temperature: Option<f64>,
        #[serde(default)]
        top_p: Option<f64>,
        #[serde(default)]
        top_k: Option<u64>,
        #[serde(default)]
        repeat_penalty: Option<f64>,
        /// Tool-call constraint for OpenAI-compatible servers (e.g. llama.cpp).
        /// Unset (the default) sends no `tool_choice` — i.e. `"auto"`: the model
        /// is free to call a tool or answer in text, and the FORMAT is
        /// unconstrained (the source of leaked/malformed tool calls). Set to
        /// `"required"` to force + grammar-constrain a valid tool call (note:
        /// this forces a call EVERY turn — use only for roles that should always
        /// act, as it removes text-only completion). Set to a specific function
        /// name to force that tool. See docs/spec/local-coder-massaging.md §25.
        #[serde(default)]
        tool_choice: Option<String>,
    },
    Weighted {
        entries: Vec<ModelEntry>,
    },
}

/// Routing behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingBehavior {
    #[serde(default = "default_strategy")]
    pub strategy: String,
    #[serde(default)]
    pub compaction_model: Option<String>,
    /// When true, never dispatch to a cloud provider. Cloud classifier routes
    /// are remapped to local equivalents and cloud roles are stripped from
    /// failover chains. If no local model can serve the request, an error is
    /// surfaced to the user instead of silently falling back to cloud.
    /// Can also be enabled via the `CODEX_LOCAL_ONLY` env var or the
    /// `--local-only` CLI flag.
    #[serde(default)]
    pub local_only: bool,
    /// Max share of the local model's context budget the **system prompt**
    /// (base instructions + state prelude) may occupy, as a percentage. When the
    /// system exceeds it, the trimmer compresses it instead of letting it crowd
    /// out the conversation. `0` disables system compression. Default 20.
    /// Generic by design — bounds *any* incoming system prompt, not just Codex's,
    /// so the same logic works when this fork becomes a harness-agnostic service.
    #[serde(default = "default_system_budget_pct")]
    pub system_budget_pct: u8,
}

impl Default for RoutingBehavior {
    fn default() -> Self {
        Self {
            strategy: default_strategy(),
            compaction_model: None,
            local_only: false,
            system_budget_pct: default_system_budget_pct(),
        }
    }
}

fn default_strategy() -> String {
    "cost_first".into()
}

fn default_system_budget_pct() -> u8 {
    20
}

/// Supervisor behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisorBehavior {
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries_per_task: u32,
    #[serde(default)]
    pub verification_command: Option<String>,
}

fn default_max_iterations() -> u32 {
    50
}
fn default_timeout() -> u64 {
    7200
}
fn default_max_retries() -> u32 {
    3
}

impl Default for SupervisorBehavior {
    fn default() -> Self {
        Self {
            max_iterations: default_max_iterations(),
            timeout_seconds: default_timeout(),
            max_retries_per_task: default_max_retries(),
            verification_command: None,
        }
    }
}

/// Failover chain configuration.
/// Failover chain configuration — defines escalation paths per task type.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FailoverChains {
    #[serde(default)]
    pub reasoning: Vec<String>,
    #[serde(default)]
    pub coding: Vec<String>,
    #[serde(default)]
    pub classification: Vec<String>,
    #[serde(default)]
    pub compaction: Vec<String>,
    #[serde(default)]
    pub planning: Vec<String>,
    #[serde(default)]
    pub evaluation: Vec<String>,

    /// Controls how failures are handled before walking the chains.
    #[serde(default)]
    pub behavior: FailoverBehavior,
}

/// Failover behavior — controls retry and escalation parameters.
///
/// Failure types:
/// F1 (rate limit): retry same model with backoff, then walk chain
/// F2 (quota exhausted): walk chain immediately
/// F3 (model unavailable): walk chain immediately
/// F4 (model not found): walk chain immediately, log config error
/// F5 (auth failure): hard-fail, don't retry
/// F6 (timeout): retry same model once, then walk chain
/// F7 (quality failure): walk chain immediately (same model = same garbage)
/// F8 (context overflow): walk chain (need larger context model)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverBehavior {
    /// F1 + F6: how many times to retry the same model before walking chain
    #[serde(default = "default_fo_retry_attempts")]
    pub retry_same_attempts: u32,

    /// Backoff between retries of the same model (ms)
    #[serde(default = "default_fo_backoff")]
    pub retry_same_backoff_ms: u64,

    /// F1: if no retry-after header, wait this long (ms)
    #[serde(default = "default_fo_rate_limit_wait")]
    pub rate_limit_default_wait_ms: u64,

    /// F1: maximum wait for a rate limit, even if retry-after says longer
    #[serde(default = "default_fo_rate_limit_max")]
    pub rate_limit_max_wait_ms: u64,

    /// F6: request timeout (ms)
    #[serde(default = "default_fo_timeout")]
    pub timeout_ms: u64,
}

fn default_fo_retry_attempts() -> u32 {
    2
}
fn default_fo_backoff() -> u64 {
    1000
}
fn default_fo_rate_limit_wait() -> u64 {
    5000
}
fn default_fo_rate_limit_max() -> u64 {
    30000
}
fn default_fo_timeout() -> u64 {
    30000
}

impl Default for FailoverBehavior {
    fn default() -> Self {
        Self {
            retry_same_attempts: default_fo_retry_attempts(),
            retry_same_backoff_ms: default_fo_backoff(),
            rate_limit_default_wait_ms: default_fo_rate_limit_wait(),
            rate_limit_max_wait_ms: default_fo_rate_limit_max(),
            timeout_ms: default_fo_timeout(),
        }
    }
}

/// Usage preservation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageConfig {
    #[serde(default = "default_warn_threshold")]
    pub primary_warn_threshold: f64,
    #[serde(default = "default_true")]
    pub prefer_secondary: bool,
}

fn default_warn_threshold() -> f64 {
    0.7
}
fn default_true() -> bool {
    true
}

impl Default for UsageConfig {
    fn default() -> Self {
        Self {
            primary_warn_threshold: default_warn_threshold(),
            prefer_secondary: true,
        }
    }
}

/// Agent role configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRole {
    pub nickname: String,
    pub instructions: String,
}

/// Configuration for the `local_web_search` tool (Brave Search backend).
///
/// When `brave_api_key` is empty (or the section is missing), the tool is
/// considered disabled — calls return an error message instead of hitting
/// the network.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchConfig {
    /// Brave Search API subscription token. Empty string means the tool is
    /// disabled. Drop the value into `.codex-multi/config.toml` directly:
    ///
    /// ```toml
    /// [search]
    /// brave_api_key = "BSAxxxxxxxxxxx"
    /// ```
    #[serde(default)]
    pub brave_api_key: String,

    /// Number of results per query, clamped to the Brave API limit of 1-20.
    #[serde(default = "default_results_per_query")]
    pub results_per_query: usize,
}

fn default_results_per_query() -> usize {
    10
}

/// CLI binary paths for external provider dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliBinaries {
    /// Path to `claude` CLI binary. Default: "claude" (assumes on PATH).
    #[serde(default = "default_claude_binary")]
    pub claude: String,
}

fn default_claude_binary() -> String {
    "claude".into()
}

impl Default for CliBinaries {
    fn default() -> Self {
        Self {
            claude: default_claude_binary(),
        }
    }
}

/// The full project-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectConfig {
    #[serde(default)]
    pub models: std::collections::HashMap<String, ModelRole>,
    #[serde(default)]
    pub roles: std::collections::HashMap<String, AgentRole>,
    #[serde(default)]
    pub routing: RoutingBehavior,
    #[serde(default)]
    pub supervisor: SupervisorBehavior,
    #[serde(default)]
    pub failover: FailoverChains,
    #[serde(default)]
    pub usage: UsageConfig,
    #[serde(default)]
    pub cli: CliBinaries,
    #[serde(default)]
    pub search: SearchConfig,
}

impl ProjectConfig {
    /// Load from `.codex-multi/config.toml` in the given directory.
    /// Returns default config if the file doesn't exist.
    pub fn load(dir: &Path) -> Self {
        let config_path = dir.join(".codex-multi").join("config.toml");
        if !config_path.exists() {
            return Self::default();
        }

        match std::fs::read_to_string(&config_path) {
            Ok(content) => match toml::from_str::<ProjectConfig>(&content) {
                Ok(config) => {
                    info!(path = %config_path.display(), "Loaded project config");
                    config
                }
                Err(e) => {
                    warn!(
                        path = %config_path.display(),
                        error = %e,
                        "Failed to parse project config, using defaults"
                    );
                    Self::default()
                }
            },
            Err(e) => {
                warn!(
                    path = %config_path.display(),
                    error = %e,
                    "Failed to read project config, using defaults"
                );
                Self::default()
            }
        }
    }

    /// Get a model role by name.
    pub fn get_model(&self, name: &str) -> Option<&ModelRole> {
        self.models.get(name)
    }

    /// Get the failover chain for a task type.
    pub fn failover_chain(&self, task_type: &str) -> &[String] {
        match task_type {
            "reasoning" => &self.failover.reasoning,
            "coding" => &self.failover.coding,
            "classification" => &self.failover.classification,
            "compaction" => &self.failover.compaction,
            "planning" => &self.failover.planning,
            "evaluation" => &self.failover.evaluation,
            _ => &[],
        }
    }
}

/// Returns true if a role name refers to a cloud-dispatched role.
/// Used by local_only mode to strip cloud entries from failover chains.
pub fn is_cloud_role(role: &str) -> bool {
    matches!(
        role,
        "cloud_fast" | "cloud_mini" | "cloud_reasoner" | "cloud_coder"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProjectConfig::default();
        assert_eq!(config.supervisor.max_iterations, 50);
        assert_eq!(config.supervisor.timeout_seconds, 7200);
        assert_eq!(config.routing.strategy, "cost_first");
        assert!(config.usage.prefer_secondary);
    }

    #[test]
    fn test_parse_single_model() {
        let toml = r#"
[models.classifier]
provider = "ollama"
endpoint = "http://localhost:11434"
model = "qwen3.5-9b:iq4_xs"
reasoning = "off"
"#;
        let config: ProjectConfig = toml::from_str(toml).unwrap();
        assert!(config.models.contains_key("classifier"));
    }

    #[test]
    fn test_parse_weighted_model() {
        let toml = r#"
[models.cloud_coder]
entries = [
    { provider = "openai", model = "gpt-5.3-codex-spark", weight = 40, reasoning = "low" },
    { provider = "openai", model = "gpt-5.4", weight = 30, reasoning = "medium" },
    { provider = "anthropic", model = "opus-4.6", weight = 30, reasoning = "medium" },
]
"#;
        let config: ProjectConfig = toml::from_str(toml).unwrap();
        match config.models.get("cloud_coder") {
            Some(ModelRole::Weighted { entries }) => {
                assert_eq!(entries.len(), 3);
                assert_eq!(entries[0].weight, 40);
                assert_eq!(entries[1].model, "gpt-5.4");
            }
            _ => panic!("Expected weighted model"),
        }
    }

    #[test]
    fn test_parse_full_config() {
        let toml = r#"
[models.classifier]
provider = "ollama"
endpoint = "http://localhost:11434"
model = "qwen3.5-9b:iq4_xs"

[models.light_reasoner]
provider = "ollama"
endpoint = "http://localhost:11435"
model = "qwen3.5:9b"
reasoning = "on"

[models.cloud_coder]
entries = [
    { provider = "openai", model = "gpt-5.3-codex-spark", weight = 40 },
    { provider = "openai", model = "gpt-5.4", weight = 30 },
]

[routing]
strategy = "cost_first"
compaction_model = "compactor"

[supervisor]
max_iterations = 30
verification_command = "pytest tests/"

[failover]
reasoning = ["light_reasoner", "light_reasoner_backup", "cloud_reasoner", "cloud_coder"]
coding = ["light_coder", "cloud_fast", "cloud_mini", "cloud_coder"]

[usage]
primary_warn_threshold = 0.8
prefer_secondary = true
"#;
        let config: ProjectConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.supervisor.max_iterations, 30);
        assert_eq!(
            config.supervisor.verification_command,
            Some("pytest tests/".into())
        );
        assert_eq!(config.failover.reasoning.len(), 4);
        assert_eq!(config.failover.coding.len(), 4);
        assert_eq!(config.usage.primary_warn_threshold, 0.8);
    }

    #[test]
    fn test_load_nonexistent() {
        let config = ProjectConfig::load(Path::new("/nonexistent/path"));
        assert_eq!(config.supervisor.max_iterations, 50); // defaults
    }

    #[test]
    fn test_local_only_default_off() {
        let config = ProjectConfig::default();
        assert!(!config.routing.local_only);
    }

    #[test]
    fn test_local_only_parsed_from_toml() {
        let toml = r#"
[routing]
local_only = true
"#;
        let config: ProjectConfig = toml::from_str(toml).unwrap();
        assert!(config.routing.local_only);
    }

    #[test]
    fn test_is_cloud_role_classification() {
        assert!(is_cloud_role("cloud_fast"));
        assert!(is_cloud_role("cloud_mini"));
        assert!(is_cloud_role("cloud_reasoner"));
        assert!(is_cloud_role("cloud_coder"));
        assert!(!is_cloud_role("light_coder"));
        assert!(!is_cloud_role("light_reasoner"));
        assert!(!is_cloud_role("light_reasoner_backup"));
        assert!(!is_cloud_role("compactor"));
        assert!(!is_cloud_role("classifier"));
    }
}
