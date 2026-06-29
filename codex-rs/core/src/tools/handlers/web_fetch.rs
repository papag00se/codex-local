//! Handler for the `web_fetch` tool — simple HTTP GET with a browser-like
//! User-Agent. Backend lives in the routing crate.

use codex_protocol::models::WebSearchAction;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::WebSearchBeginEvent;
use codex_protocol::protocol::WebSearchEndEvent;
use codex_routing::web_fetch;
use serde::Deserialize;

use crate::function_tool::FunctionCallError;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

pub struct WebFetchHandler;

#[derive(Deserialize)]
struct WebFetchArgs {
    url: String,
    #[serde(default)]
    user_agent: Option<String>,
    /// Jump to the part of the page matching this keyword/path (returns a small
    /// in-context slice instead of the whole page).
    #[serde(default)]
    find: Option<String>,
    /// Continuation token from a previous truncated result — fetches the next page.
    #[serde(default)]
    cursor: Option<String>,
}

impl ToolHandler for WebFetchHandler {
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
                    "web_fetch handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: WebFetchArgs = parse_arguments(&arguments)?;
        let url = args.url.trim();
        if url.is_empty() {
            return Err(FunctionCallError::RespondToModel(
                "web_fetch: url must not be empty".to_string(),
            ));
        }

        // Surface the fetch in the UI as an "open page" cell (reusing the
        // web-search cell), so the user can see when the model reaches the
        // network. A live cell shows during the await; it completes on return.
        session
            .send_event(
                turn.as_ref(),
                EventMsg::WebSearchBegin(WebSearchBeginEvent {
                    call_id: call_id.clone(),
                }),
            )
            .await;
        let result = web_fetch::fetch_nav(
            url,
            args.user_agent.as_deref(),
            args.find.as_deref(),
            args.cursor.as_deref(),
            web_fetch::WEB_FETCH_CONTENT_CAP_TOKENS,
        )
        .await;
        session
            .send_event(
                turn.as_ref(),
                EventMsg::WebSearchEnd(WebSearchEndEvent {
                    call_id: call_id.clone(),
                    query: url.to_string(),
                    action: WebSearchAction::OpenPage {
                        url: Some(url.to_string()),
                    },
                }),
            )
            .await;

        match result {
            Ok(text) => Ok(FunctionToolOutput::from_text(text, Some(true))),
            Err(e) => Err(FunctionCallError::RespondToModel(e.to_string())),
        }
    }
}
