//! Brave Search backend for the `local_web_search` tool.
//!
//! Single HTTP GET to the Brave Search API. No retry loop, no caching, no
//! pagination — keep it small. The handler in `codex-core` calls this and
//! formats the result for the model.

use serde::Deserialize;

const BRAVE_ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";

/// Default User-Agent sent on every Brave Search request when the caller
/// doesn't supply one. Matches what current Brave Browser sends on Linux
/// desktop — Brave intentionally identifies as Chrome to avoid being
/// fingerprinted as a separate browser.
pub const DEFAULT_USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36";

/// Result of a single Brave search call.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub description: String,
}

#[derive(Debug)]
pub enum SearchError {
    /// API key is empty / unconfigured.
    Unconfigured,
    /// HTTP error (network, non-2xx status).
    Http(String),
    /// API responded with an error body.
    Api(String),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unconfigured => {
                write!(
                    f,
                    "local_web_search: brave_api_key is not configured in .codex-multi/config.toml"
                )
            }
            Self::Http(msg) => write!(f, "local_web_search HTTP error: {msg}"),
            Self::Api(msg) => write!(f, "local_web_search API error: {msg}"),
        }
    }
}

#[derive(Deserialize)]
struct BraveResponse {
    web: Option<BraveWeb>,
}

#[derive(Deserialize)]
struct BraveWeb {
    results: Vec<BraveResult>,
}

#[derive(Deserialize)]
struct BraveResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    description: String,
}

/// Execute a Brave web search. Caller is responsible for clamping `count` to
/// the API's allowed range (1-20). `user_agent` is sent verbatim if `Some`;
/// otherwise [`DEFAULT_USER_AGENT`] is used.
pub async fn search(
    api_key: &str,
    query: &str,
    count: usize,
    user_agent: Option<&str>,
) -> Result<Vec<SearchResult>, SearchError> {
    if api_key.trim().is_empty() {
        return Err(SearchError::Unconfigured);
    }

    let count_clamped = count.clamp(1, 20);
    let client = reqwest::Client::new();
    let response = client
        .get(BRAVE_ENDPOINT)
        .header("X-Subscription-Token", api_key)
        .header("Accept", "application/json")
        .header("User-Agent", user_agent.unwrap_or(DEFAULT_USER_AGENT))
        .query(&[("q", query), ("count", &count_clamped.to_string())])
        .send()
        .await
        .map_err(|e| SearchError::Http(e.to_string()))?;

    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<unreadable body>".to_string());
        return Err(SearchError::Api(format!("HTTP {status}: {body}")));
    }

    let parsed: BraveResponse = response
        .json()
        .await
        .map_err(|e| SearchError::Http(format!("response JSON parse failed: {e}")))?;

    let results = parsed
        .web
        .map(|w| {
            w.results
                .into_iter()
                .map(|r| SearchResult {
                    title: r.title,
                    url: r.url,
                    description: r.description,
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(results)
}

/// Render search results as a compact text block suitable for a tool output.
pub fn format_results(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No results for query: {query}");
    }
    let mut out = format!("Search results for: {query}\n");
    for (i, r) in results.iter().enumerate() {
        out.push_str(&format!(
            "\n{}. {}\n   {}\n   {}\n",
            i + 1,
            r.title,
            r.url,
            r.description.trim()
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unconfigured_returns_error() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(search("", "test", 5, None));
        assert!(matches!(result, Err(SearchError::Unconfigured)));

        let result = rt.block_on(search("   ", "test", 5, None));
        assert!(matches!(result, Err(SearchError::Unconfigured)));
    }

    #[test]
    fn format_results_handles_empty() {
        let formatted = format_results("rust async", &[]);
        assert!(formatted.contains("No results"));
        assert!(formatted.contains("rust async"));
    }

    #[test]
    fn format_results_includes_each_entry() {
        let results = vec![
            SearchResult {
                title: "Title 1".to_string(),
                url: "https://example.com/1".to_string(),
                description: "Description one.".to_string(),
            },
            SearchResult {
                title: "Title 2".to_string(),
                url: "https://example.com/2".to_string(),
                description: "Description two.".to_string(),
            },
        ];
        let formatted = format_results("q", &results);
        assert!(formatted.contains("Title 1"));
        assert!(formatted.contains("https://example.com/2"));
        assert!(formatted.contains("Description one."));
    }
}
