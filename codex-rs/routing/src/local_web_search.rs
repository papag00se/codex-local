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

/// Decided BEFORE a search executes, by [`gate_search`].
pub enum SearchGate {
    /// Run the search.
    Proceed,
    /// This search is essentially one already made THIS turn — refuse it. Returned
    /// as an HTTP 400 (the native "bad request, don't repeat" idiom an HTTP tool
    /// speaks), so the model treats it as a real failure and stops re-issuing it.
    /// Search results don't change turn to turn, so a repeat is pure wasted
    /// inference; the previous result was already the right one.
    Block { message: String },
}

/// Most-recent stripped searches to keep per turn — bounds memory on a
/// pathologically search-heavy turn. Since a repeat is BLOCKED (never stored
/// twice), the list only grows with genuinely distinct searches.
const MAX_RECENT_SEARCHES: usize = 64;

/// Recent searches (normalized word-sets) per session, reset each user turn.
/// Session-scoped so sessions/sub-agents don't cross-contaminate; see
/// [`crate::guard_state`].
static SEARCH_MEM: std::sync::LazyLock<crate::guard_state::SessionTurnStore<Vec<Vec<String>>>> =
    std::sync::LazyLock::new(crate::guard_state::SessionTurnStore::new);

/// Decide whether to run this search or refuse it as a repeat of one already made
/// this turn. `session` = harness `conversation_id`, `turn` = `sub_id` (so the
/// memory resets on a new user turn). A search "matches" a prior one when their
/// normalized word-sets overlap enough — see [`searches_match`].
pub fn gate_search(session: &str, turn: &str, query: &str) -> SearchGate {
    let words = normalize_search(query);
    if words.is_empty() {
        return SearchGate::Proceed;
    }
    SEARCH_MEM.with(session, turn, |recent| {
        if recent.iter().any(|prev| searches_match(&words, prev)) {
            // If the repeated query names a domain, point at fetching it directly
            // instead of searching about it again — the exact steer a coder needs
            // when it circles a domain in web_search without ever curling it.
            let steer = match first_domain_in(query) {
                Some(d) => format!(
                    "\nThis query names a domain — stop searching ABOUT it and FETCH it directly: \
                     web_fetch https://{d} , then parse the response for what you need."
                ),
                None => String::new(),
            };
            SearchGate::Block {
                message: format!(
                    "HTTP 400 Bad Request \u{b7} web_search \"{query}\"\n\
                     You've searched this before. Make a major change to your search, or try something else.{steer}"
                ),
            }
        } else {
            recent.push(words);
            if recent.len() > MAX_RECENT_SEARCHES {
                let excess = recent.len() - MAX_RECENT_SEARCHES;
                recent.drain(0..excess);
            }
            SearchGate::Proceed
        }
    })
}

/// If `query` contains a bare domain name (e.g. `api.handle.me`), return it — used
/// to steer a coder that keeps SEARCHING a domain toward FETCHING it directly. A
/// domain is a dotted token whose final label is an alphabetic TLD (≥2 chars) that
/// is NOT a common source-file extension, so `handler.py` / `config.json` don't
/// masquerade as hosts.
pub fn first_domain_in(query: &str) -> Option<String> {
    query
        .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_')))
        .map(|t| t.trim_matches(|c: char| c == '.' || c == '-' || c == '_'))
        .find(|t| looks_like_domain(t))
        .map(str::to_string)
}

fn looks_like_domain(tok: &str) -> bool {
    const FILE_EXTS: &[&str] = &[
        "py", "rs", "js", "ts", "jsx", "tsx", "json", "md", "txt", "toml", "yaml", "yml", "go",
        "java", "cpp", "hpp", "sh", "rb", "php", "html", "css", "xml", "csv", "lock", "cfg", "ini",
        "env", "log", "sql",
    ];
    let labels: Vec<&str> = tok.split('.').collect();
    if labels.len() < 2 {
        return false;
    }
    let tld = labels[labels.len() - 1].to_ascii_lowercase();
    if tld.len() < 2
        || !tld.chars().all(|c| c.is_ascii_alphabetic())
        || FILE_EXTS.contains(&tld.as_str())
    {
        return false;
    }
    labels
        .iter()
        .all(|l| !l.is_empty() && l.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
}

/// Normalize a query to a deduped set of content words: lowercase, drop
/// apostrophes, split on non-word chars but KEEP `.`, `_`, `-` inside tokens so
/// `api.handle.me` and `get_handle` stay whole, drop stopwords, light-stem.
/// Reuses the loop detector's stopword list + stemmer.
fn normalize_search(query: &str) -> Vec<String> {
    let mut words: Vec<String> = query
        .to_lowercase()
        .replace(['\'', '\u{2019}'], "")
        .split(|c: char| !(c.is_alphanumeric() || c == '.' || c == '_' || c == '-'))
        .filter(|t| !t.is_empty() && !crate::loop_detector::is_stopword(t))
        .map(|t| crate::loop_detector::stem(t).to_string())
        .collect();
    words.sort();
    words.dedup();
    words
}

/// Is the NEW search `new_q` essentially a re-hunt of a PRIOR one (`prior`) —
/// rumination — rather than a genuine new direction or a refinement? Called as
/// `searches_match(new, prior)`; the order matters for the refinement rule.
///
/// Rules, in order:
///   - **Refinement** — `new` is longer and keeps all but at most one word of
///     `prior` (a dropped filler), adding new specificity: a real narrowing of
///     the same hunt → NOT a match.
///   - **Exact repeat** — identical word-sets (any length) → match.
///   - **Heavy overlap** — Jaccard (shared ÷ combined) ≥ 0.5 with ≥2 shared
///     words, or ≥5 shared words outright → match.
///
/// Jaccard (relative, not an absolute count) is what catches short queries
/// dominated by one anchor term (`api.handle.me` + a small reworded tail):
/// swapping or dropping tail terms keeps the sets mostly overlapping, while a
/// genuine pivot (a new focused term at the cost of others) drops the ratio, and
/// a refinement (keep everything, add specificity) is exempted outright. An
/// absolute `overlap ≥ 5` count let anchor-heavy rewordings slip through — this
/// took one real session from 1 caught re-hunt to 6.
fn searches_match(new_q: &[String], prior: &[String]) -> bool {
    use std::collections::HashSet;
    let sn: HashSet<&str> = new_q.iter().map(String::as_str).collect();
    let sp: HashSet<&str> = prior.iter().map(String::as_str).collect();
    // Refinement: the new query is LONGER and kept the bulk of the prior —
    // dropping at most one term (typically a filler like "using") while adding
    // new specificity. That's a narrowing of the same hunt, not a re-hunt.
    if sn.len() > sp.len() {
        let dropped = sp.iter().filter(|w| !sn.contains(*w)).count();
        if dropped <= 1 {
            return false;
        }
    }
    if sn == sp {
        return true; // exact repeat (any length), including a pure reorder
    }
    let overlap = sn.iter().filter(|w| sp.contains(*w)).count();
    if overlap >= 5 {
        return true;
    }
    let union = sn.len() + sp.len() - overlap;
    overlap >= 2 && union > 0 && (overlap as f64) / (union as f64) >= 0.5
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
    fn first_domain_in_recognizes_hosts_not_filenames() {
        // Real domains in a query → detected (for the fetch-directly steer).
        assert_eq!(
            first_domain_in("api.handle.me handles endpoint schema").as_deref(),
            Some("api.handle.me")
        );
        assert_eq!(
            first_domain_in("docs on example.com").as_deref(),
            Some("example.com")
        );
        assert_eq!(first_domain_in("handle.me get_handle").as_deref(), Some("handle.me"));
        // Source filenames and non-hosts must NOT masquerade as domains.
        assert_eq!(first_domain_in("read handler.py and config.json"), None);
        assert_eq!(first_domain_in("cardano staking rewards"), None);
        assert_eq!(first_domain_in("bump version to 1.5"), None);
        assert_eq!(first_domain_in("update main.rs"), None);
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

    #[test]
    fn search_sameness_rule_matches_the_examples() {
        let n = normalize_search;
        // Ex1: 4-word overlap and one side is 6 words → DIFFERENT.
        assert!(!searches_match(
            &n("api.handle.me resolve handle address endpoint get_handle"),
            &n("resolve a handle using the api.handle.me endpoint"),
        ));
        // Ex2: both ≤5 words but not the same set → DIFFERENT.
        assert!(!searches_match(
            &n("api.handle.me get_handle endpoint"),
            &n("api.handle.me resolve handle address endpoint"),
        ));
        // Ex3: identical set after stripping → SAME.
        assert!(searches_match(
            &n("api.handle.me get_handle endpoint"),
            &n("get_handle endpoint on api.handle.me"),
        ));
        // ≥5-word overlap → SAME even if the queries differ elsewhere.
        assert!(searches_match(
            &n("cardano handle resolution endpoint holder address"),
            &n("cardano handle resolution endpoint holder wallet"),
        ));
    }

    #[test]
    fn anchor_dominated_rewordings_are_caught() {
        let n = normalize_search;
        // Same api.handle.me hunt with the tail dropped — a re-hunt the old
        // absolute `overlap >= 5` rule let through (Jaccard 4/6 = 0.67).
        assert!(searches_match(
            &n("api.handle.me handles/goose response"),
            &n("api.handle.me handles/goose JSON response example"),
        ));
        // Same hunt, one tail term swapped (Jaccard 2/4 = 0.5) → caught.
        assert!(searches_match(
            &n("api.handle.me addresses endpoint"),
            &n("api.handle.me addresses stake"),
        ));
        // Genuine refinement — keep the whole prior query, add specificity → allowed.
        assert!(!searches_match(
            &n("rust async trait object safety dyn"),
            &n("rust async trait"),
        ));
        // Genuine new angle sharing only the ubiquitous anchor term → allowed.
        assert!(!searches_match(
            &n("api.handle.me get_holder list_holders"),
            &n("api.handle.me addresses stake"),
        ));
    }

    #[test]
    fn keeps_domain_and_snake_tokens_whole() {
        let w = normalize_search("api.handle.me get_handle endpoint");
        assert!(w.contains(&"api.handle.me".to_string()));
        assert!(w.contains(&"get_handle".to_string()));
    }

    #[test]
    fn gate_blocks_repeat_and_resets_on_new_turn() {
        let (s, t1, t2) = ("sess-gate-test", "turn-1", "turn-2");
        // First search runs.
        assert!(matches!(
            gate_search(s, t1, "api.handle.me get_handle endpoint"),
            SearchGate::Proceed
        ));
        // A re-phrasing that's "the same" is refused as a 400.
        match gate_search(s, t1, "get_handle endpoint on api.handle.me") {
            SearchGate::Block { message } => {
                assert!(message.contains("HTTP 400"));
                assert!(message.contains("searched this before"));
            }
            SearchGate::Proceed => panic!("a repeat search must be blocked"),
        }
        // A genuinely different search runs.
        assert!(matches!(
            gate_search(s, t1, "cardano staking rewards calculation"),
            SearchGate::Proceed
        ));
        // A new user turn is a clean slate — the earlier search runs again.
        assert!(matches!(
            gate_search(s, t2, "api.handle.me get_handle endpoint"),
            SearchGate::Proceed
        ));
    }
}
