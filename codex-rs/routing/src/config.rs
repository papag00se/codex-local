//! Routing configuration — multi-tier model routing.
//!
//! Supports local Ollama (free), cloud secondary buckets (cheap), and
//! cloud primary buckets (conserve). See docs/spec/model-routing-strategy.md.

use serde::{Deserialize, Serialize};
use std::env;

/// How many tools to expose to a local model on the LightCoder route.
///
/// `Focused` — a small curated set (~6) for models that lose attention on
/// big tool catalogs. This is the default and what 9b-class models need.
///
/// `Full` — the entire Codex tool catalog (~120 schemas including MCP
/// connectors and multi-agent orchestration). For larger local models
/// (e.g. 30B+) that can navigate the full set without thrashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolSubset {
    Focused,
    Full,
}

impl Default for ToolSubset {
    fn default() -> Self {
        Self::Focused
    }
}

impl ToolSubset {
    /// Parse from the `tool_subset` field in `.codex-multi/config.toml`.
    /// Falls back to `Focused` for unknown values.
    pub fn from_config_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "full" => Self::Full,
            _ => Self::Focused,
        }
    }
}

/// Wire-format flavor for a local model endpoint. Both the dispatcher and
/// the response parser branch on this; callers always see a uniform
/// Ollama-shaped response regardless of which flavor was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClientFlavor {
    /// Ollama's native `/api/chat` endpoint with `options: { num_ctx, ... }`,
    /// `think: bool`, `format: "json"`, NDJSON streaming. Default for
    /// backwards compatibility with existing configs.
    #[default]
    Ollama,
    /// OpenAI-compatible `/v1/chat/completions` endpoint. Works with
    /// LM Studio, llama.cpp's `--api-server`, vLLM, and Ollama itself
    /// (which also exposes a `/v1` shim). Translates payloads to the
    /// OpenAI shape (top-level `temperature`/`max_tokens`,
    /// `response_format: {type: "json_object"}`, no `think` field) and
    /// translates responses back to the Ollama shape so callers don't
    /// need to know which flavor was used.
    OpenAICompat,
}

/// A single local-model endpoint + model. Despite the name, this is used
/// for any local-model wire flavor — see [`ClientFlavor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEndpoint {
    pub base_url: String,
    pub model: String,
    pub trim_budget: usize,
    pub temperature: f64,
    /// Per-request wall-clock timeout. `0` disables the timeout entirely
    /// (reasoning runs are legitimately unbounded). Normalized from config:
    /// a missing key defaults to 300s; an explicit `0` means "wait forever".
    pub timeout_seconds: u64,
    pub enabled: bool,
    /// Whether to enable Ollama's reasoning/`think` mode. Derived from the
    /// model role's `reasoning` config (`"on"`/`"off"`); defaults to `false`.
    /// Models that support thinking (qwen3.5, deepseek-r1, …) produce better
    /// multi-step plans when this is on, at the cost of extra latency from
    /// the thinking tokens. Ignored for [`ClientFlavor::OpenAICompat`].
    #[serde(default)]
    pub think: bool,
    /// How many tools to expose. See [`ToolSubset`]. Derived from the role's
    /// `tool_subset` config field; defaults to `Focused`.
    #[serde(default)]
    pub tool_subset: ToolSubset,
    /// Wire-format flavor — Ollama's `/api/chat` (default) or OpenAI-style
    /// `/v1/chat/completions`. Set via the project config's `provider`
    /// field: `"ollama"` → `Ollama`, `"openai-compat"` / `"openai_compat"`
    /// / `"lmstudio"` → `OpenAICompat`.
    #[serde(default)]
    pub flavor: ClientFlavor,
    /// Hard ceiling on output tokens per response. `None` = no cap, let
    /// the server decide. Maps to `max_tokens` for OpenAI-compat and
    /// `options.num_predict` for Ollama. Normalized from config: a
    /// `max_tokens = 0` in the TOML file is treated the same as omitting
    /// the key (unlimited), which is the ergonomic convention.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Optional sampler overrides. `None` = omit from the request so the
    /// server default applies. Set per-role for models that need non-stock
    /// sampling (e.g. Gemma 4: `top_p=0.95`, `top_k=64`, `repeat_penalty=1.1`).
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u64>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    /// Tool-call constraint sent to OpenAI-compatible servers. `None` omits the
    /// field (server default = `"auto"`, unconstrained format). `Some("required")`
    /// forces and grammar-constrains a valid tool call; `Some("<fn name>")`
    /// forces a specific tool. The lever for fixing local-model tool-call format
    /// at the source instead of recovering after the fact — see
    /// docs/spec/local-coder-massaging.md §25.
    #[serde(default)]
    pub tool_choice: Option<String>,
}

impl OllamaEndpoint {
    fn from_env(url_var: &str, model_var: &str, defaults: (&str, &str)) -> Self {
        Self {
            base_url: env_or(url_var, defaults.0),
            model: env_or(model_var, defaults.1),
            trim_budget: 8192,
            temperature: 0.1,
            timeout_seconds: 300,
            enabled: true,
            think: false,
            tool_subset: ToolSubset::Focused,
            flavor: ClientFlavor::Ollama,
            max_tokens: None,
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            tool_choice: None,
        }
    }
}

/// Full routing configuration across all tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    // --- Local Ollama (free) ---
    pub classifier: OllamaEndpoint,
    pub reasoner: OllamaEndpoint,
    pub reasoner_backup: OllamaEndpoint,
    pub light_coder: OllamaEndpoint,
    pub compactor: OllamaEndpoint,

    // --- Cloud secondary buckets (prefer over primary) ---
    pub codex_spark_enabled: bool,
    pub mini_enabled: bool,
    pub sonnet_enabled: bool,

    // --- Legacy compat with the route selection engine ---
    pub router: RouterModelConfig,
    pub coder: OllamaRouteConfig,

    pub codex_cli_enabled: bool,

    /// When true, never dispatch to cloud providers. See
    /// `RoutingBehavior::local_only` in `project_config.rs`.
    pub local_only: bool,
}

/// Configuration for the router model (the LLM that picks routes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterModelConfig {
    pub base_url: String,
    pub model: String,
    pub trim_budget: usize,
    pub temperature: f64,
    pub timeout_seconds: u64,
}

/// Legacy config for the route selection engine's coder/reasoner paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaRouteConfig {
    pub base_url: String,
    pub model: String,
    pub trim_budget: usize,
    pub temperature: f64,
    pub timeout_seconds: u64,
    pub enabled: bool,
}

impl RoutingConfig {
    /// Load routing config from environment variables.
    pub fn from_env() -> Self {
        let classifier = OllamaEndpoint::from_env(
            "OLLAMA_CLASSIFIER_URL",
            "OLLAMA_CLASSIFIER_MODEL",
            ("http://localhost:11434", "qwen3.5-9b:iq4_xs"),
        );

        // Reasoner and coder default to thinking ON. The classifier and
        // compactor stay off since they need fast deterministic output.
        // Users can still override per-role via `.codex-multi/config.toml`'s
        // `reasoning = "off"` field.
        let mut reasoner = OllamaEndpoint::from_env(
            "OLLAMA_REASONER_URL",
            "OLLAMA_REASONER_MODEL",
            ("http://localhost:11435", "qwen3.5:9b"),
        );
        reasoner.think = true;

        let mut reasoner_backup = OllamaEndpoint::from_env(
            "OLLAMA_REASONER_BACKUP_URL",
            "OLLAMA_REASONER_BACKUP_MODEL",
            ("http://localhost:11434", "qwen3.5:9b"),
        );
        reasoner_backup.think = true;

        let mut light_coder = OllamaEndpoint::from_env(
            "OLLAMA_CODER_URL",
            "OLLAMA_CODER_MODEL",
            (
                "http://localhost:11435",
                "qwen3.5-9b-opus-openclaw-distilled:tools",
            ),
        );
        light_coder.think = true;

        let compactor = OllamaEndpoint::from_env(
            "OLLAMA_COMPACTOR_URL",
            "OLLAMA_COMPACTOR_MODEL",
            ("http://localhost:11435", "qwen3.5-9b:iq4_xs"),
        );

        Self {
            classifier,
            reasoner: reasoner.clone(),
            reasoner_backup,
            light_coder,
            compactor,
            codex_spark_enabled: env_bool("ENABLE_CODEX_SPARK", true),
            mini_enabled: env_bool("ENABLE_GPT_MINI", true),
            sonnet_enabled: env_bool("ENABLE_SONNET", true),
            // Legacy compat for route selection engine
            router: RouterModelConfig {
                base_url: reasoner.base_url.clone(),
                model: reasoner.model.clone(),
                trim_budget: reasoner.trim_budget,
                temperature: 0.0,
                timeout_seconds: reasoner.timeout_seconds,
            },
            coder: OllamaRouteConfig {
                base_url: env_or("CODER_OLLAMA_BASE_URL", &reasoner.base_url),
                model: env_or("CODER_MODEL", &reasoner.model),
                trim_budget: env_usize("CODER_TRIM_BUDGET", 16384),
                temperature: env_f64("CODER_TEMPERATURE", 0.1),
                timeout_seconds: env_u64("CODER_TIMEOUT_SECONDS", 300),
                enabled: env_bool("ENABLE_LOCAL_CODER", true),
            },
            codex_cli_enabled: false,
            local_only: env_bool("CODEX_LOCAL_ONLY", false),
        }
    }

    /// Load routing config from a ProjectConfig (`.codex-multi/config.toml`).
    /// Falls back to from_env() for any missing fields.
    pub fn from_project_config(pc: &crate::project_config::ProjectConfig) -> Self {
        let mut config = Self::from_env();

        // Project config sets local_only; env var (already loaded into
        // config.local_only by from_env) wins if true.
        config.local_only = config.local_only || pc.routing.local_only;

        // Override from project config model roles
        if let Some(role) = pc.get_model("classifier") {
            if let Some(ep) = endpoint_from_role(role) {
                config.classifier = ep;
            }
        }
        if let Some(role) = pc.get_model("light_reasoner") {
            if let Some(ep) = endpoint_from_role(role) {
                config.reasoner = ep;
            }
        }
        if let Some(role) = pc.get_model("light_reasoner_backup") {
            if let Some(ep) = endpoint_from_role(role) {
                config.reasoner_backup = ep;
            }
        }
        if let Some(role) = pc.get_model("light_coder") {
            if let Some(ep) = endpoint_from_role(role) {
                config.light_coder = ep;
            }
        }
        if let Some(role) = pc.get_model("compactor") {
            if let Some(ep) = endpoint_from_role(role) {
                config.compactor = ep;
            }
        }

        // Update the legacy router config to match the reasoner
        config.router = RouterModelConfig {
            base_url: config.reasoner.base_url.clone(),
            model: config.reasoner.model.clone(),
            trim_budget: config.reasoner.trim_budget,
            temperature: 0.0,
            timeout_seconds: config.reasoner.timeout_seconds,
        };

        config
    }
}

/// Extract an OllamaEndpoint from a model role (single entry only).
fn endpoint_from_role(role: &crate::project_config::ModelRole) -> Option<OllamaEndpoint> {
    match role {
        crate::project_config::ModelRole::Single {
            provider,
            endpoint,
            model,
            reasoning,
            trim_budget,
            tool_subset,
            max_tokens,
            timeout_seconds,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            tool_choice,
        } => {
            let flavor = match provider.as_str() {
                "ollama" => ClientFlavor::Ollama,
                "openai-compat" | "openai_compat" | "lmstudio" | "lm-studio" | "lm_studio"
                | "openai" => ClientFlavor::OpenAICompat,
                _ => return None,
            };
            // Normalize 0 → None so `max_tokens = 0` reads as "unlimited"
            // (the convention for "disabled") rather than actually clamping
            // output to zero tokens.
            let max_tokens = max_tokens.filter(|&n| n > 0);
            Some(OllamaEndpoint {
                base_url: endpoint
                    .clone()
                    .unwrap_or_else(|| "http://127.0.0.1:11434".into()),
                model: model.clone(),
                // Unset `trim_budget` now means AUTO: 0 tells the budget logic to
                // use the server's full detected window (from /props) minus output
                // + tool reserves. A non-zero value is an explicit cap. (Was 8192,
                // which silently capped the budget far below the real window.)
                trim_budget: trim_budget.unwrap_or(0),
                temperature: (*temperature).unwrap_or(if reasoning == "off" { 0.0 } else { 0.1 }),
                timeout_seconds: timeout_seconds.unwrap_or(300),
                enabled: true,
                think: reasoning != "off",
                tool_subset: tool_subset
                    .as_deref()
                    .map(ToolSubset::from_config_str)
                    .unwrap_or_default(),
                flavor,
                max_tokens,
                top_p: *top_p,
                top_k: *top_k,
                repeat_penalty: *repeat_penalty,
                tool_choice: tool_choice.clone(),
            })
        }
        crate::project_config::ModelRole::Weighted { .. } => {
            // Weighted roles are for cloud models, not local endpoints
            None
        }
    }
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .map(|v| {
            matches!(
                v.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}
