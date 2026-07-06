//! Handler for the `local_web_search` tool — Brave Search backend.
//!
//! Loads the Brave API key from `.codex-multi/config.toml` (via the routing
//! crate's `ProjectConfig`) on each invocation. The lookup is cheap; we
//! re-read so config edits take effect without restarting Codex.

use codex_protocol::models::WebSearchAction;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::WebSearchBeginEvent;
use codex_protocol::protocol::WebSearchEndEvent;
use codex_routing::local_web_search;
use codex_routing::project_config::ProjectConfig;
use serde::Deserialize;

use crate::function_tool::FunctionCallError;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

pub struct LocalWebSearchHandler;

#[derive(Deserialize)]
struct LocalWebSearchArgs {
    query: String,
    #[serde(default)]
    count: Option<usize>,
    #[serde(default)]
    user_agent: Option<String>,
}

impl ToolHandler for LocalWebSearchHandler {
    type Output = FunctionToolOutput;

    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    async fn handle(&self, invocation: ToolInvocation) -> Result<Self::Output, FunctionCallError> {
        let ToolInvocation {
            session,
            turn,
            call_id,
            payload,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "local_web_search handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: LocalWebSearchArgs = parse_arguments(&arguments)?;
        let query = args.query.trim();
        if query.is_empty() {
            return Err(FunctionCallError::RespondToModel(
                "local_web_search: query must not be empty".to_string(),
            ));
        }

        // Rumination gate BEFORE touching the network: if this search is
        // essentially one already made THIS turn (similar stripped word-set),
        // refuse it as a 400 — return the refusal as the result and never hit the
        // API (no phantom search cell either). Keyed by session so sessions don't
        // cross-contaminate; reset each user turn.
        let session_id = session.conversation_id.to_string();
        if let local_web_search::SearchGate::Block { message } =
            local_web_search::gate_search(&session_id, &turn.sub_id, query)
        {
            return Err(FunctionCallError::RespondToModel(message));
        }

        let cwd = std::env::current_dir().unwrap_or_default();
        let project_config = ProjectConfig::load(&cwd);
        let api_key = project_config.search.brave_api_key.clone();
        let count = args
            .count
            .unwrap_or(project_config.search.results_per_query);

        // Surface the search in the UI (reusing the web-search cell) so the user
        // can see when the model reaches the network.
        session
            .send_event(
                turn.as_ref(),
                EventMsg::WebSearchBegin(WebSearchBeginEvent {
                    call_id: call_id.clone(),
                }),
            )
            .await;
        let result =
            local_web_search::search(&api_key, query, count, args.user_agent.as_deref()).await;
        session
            .send_event(
                turn.as_ref(),
                EventMsg::WebSearchEnd(WebSearchEndEvent {
                    call_id: call_id.clone(),
                    query: query.to_string(),
                    action: WebSearchAction::Search {
                        query: Some(query.to_string()),
                        queries: None,
                    },
                }),
            )
            .await;

        match result {
            Ok(results) => {
                let mut text = local_web_search::format_results(query, &results);
                // If the query names a domain, nudge toward fetching it directly.
                // This fires on the FIRST (executed) search — repeats are refused
                // upstream by the rumination gate, which carries the same steer.
                if let Some(domain) = local_web_search::first_domain_in(query) {
                    text.push_str(&format!(
                        "\n\n[hint] Your query names a domain. If you need site-specific data, \
                         fetching it directly is usually faster and more reliable than searching \
                         about it: web_fetch https://{domain} , then parse the response for what \
                         you need."
                    ));
                }
                Ok(FunctionToolOutput::from_text(text, Some(true)))
            }
            Err(e) => Err(FunctionCallError::RespondToModel(e.to_string())),
        }
    }
}
