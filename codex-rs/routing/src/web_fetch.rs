//! HTTP fetch backend for the `web_fetch` tool.
//!
//! Single GET request with a browser-like User-Agent, a size cap, and a
//! timeout. Returns the response body as text when the content type is
//! textual; otherwise returns a short placeholder describing what was
//! received. No retries, no redirects beyond reqwest's default, no caching —
//! keep it small.

use crate::local_web_search::DEFAULT_USER_AGENT;

/// Maximum bytes read from a response body. Anything beyond is truncated and
/// a notice is appended. Sized to fit comfortably in a local model's context
/// without letting a single fetch dominate the transcript.
const MAX_BODY_BYTES: usize = 512 * 1024;
/// Token budget a single fetched body may occupy after reduction. A web page
/// shouldn't eat the local model's whole window; ~4k leaves room for the rest of
/// the turn. (`find`/`cursor` will let the model pull more on demand.)
pub const WEB_FETCH_CONTENT_CAP_TOKENS: usize = 4000;

/// Per-request timeout. Matches the search tool's expectations — local
/// models shouldn't block for minutes on a single fetch.
const REQUEST_TIMEOUT_SECS: u64 = 30;

#[derive(Debug, Clone)]
pub struct FetchResult {
    pub status: u16,
    pub final_url: String,
    pub content_type: Option<String>,
    pub body: String,
    pub truncated: bool,
}

#[derive(Debug)]
pub enum FetchError {
    InvalidUrl(String),
    Http(String),
    DecodeError(String),
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidUrl(msg) => write!(f, "web_fetch: invalid URL: {msg}"),
            Self::Http(msg) => write!(f, "web_fetch HTTP error: {msg}"),
            Self::DecodeError(msg) => write!(f, "web_fetch decode error: {msg}"),
        }
    }
}

impl std::error::Error for FetchError {}

/// Build a human-readable description of a `reqwest::Error` that walks its
/// `source()` chain so the root cause (DNS lookup failure, TLS certificate
/// mismatch, connection refused, etc.) is visible in the error message we
/// surface to the model. Also tags the failure category (`connect`,
/// `timeout`, `redirect`, `body`) when reqwest can identify it.
fn describe_reqwest_error(err: &reqwest::Error) -> String {
    let mut kind_tags: Vec<&'static str> = Vec::new();
    if err.is_timeout() {
        kind_tags.push("timeout");
    }
    if err.is_connect() {
        kind_tags.push("connect");
    }
    if err.is_redirect() {
        kind_tags.push("redirect");
    }
    if err.is_body() {
        kind_tags.push("body");
    }
    if err.is_decode() {
        kind_tags.push("decode");
    }

    let mut parts: Vec<String> = Vec::new();
    parts.push(err.to_string());
    let mut src: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(err);
    let mut seen = 0usize;
    while let Some(cur) = src {
        let msg = cur.to_string();
        if !parts.iter().any(|p| p == &msg) {
            parts.push(msg);
        }
        seen += 1;
        if seen >= 5 {
            break;
        }
        src = std::error::Error::source(cur);
    }

    let chain = parts.join(" → ");
    if kind_tags.is_empty() {
        chain
    } else {
        format!("[{}] {chain}", kind_tags.join(","))
    }
}

/// Fetch `url` with a GET request. `user_agent` is sent verbatim if `Some`;
/// otherwise [`DEFAULT_USER_AGENT`] is used so ordinary websites see a
/// request that looks like a real browser.
pub async fn fetch(url: &str, user_agent: Option<&str>) -> Result<FetchResult, FetchError> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return Err(FetchError::InvalidUrl("url must not be empty".to_string()));
    }
    let parsed = reqwest::Url::parse(trimmed).map_err(|e| FetchError::InvalidUrl(e.to_string()))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(FetchError::InvalidUrl(format!(
            "unsupported scheme '{}': only http and https are allowed",
            parsed.scheme()
        )));
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()
        .map_err(|e| FetchError::Http(describe_reqwest_error(&e)))?;

    let response = client
        .get(parsed.clone())
        .header("User-Agent", user_agent.unwrap_or(DEFAULT_USER_AGENT))
        .header(
            "Accept",
            "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
        )
        .header("Accept-Language", "en-US,en;q=0.9")
        .send()
        .await
        .map_err(|e| FetchError::Http(describe_reqwest_error(&e)))?;

    let status = response.status().as_u16();
    let final_url = response.url().to_string();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let bytes = response
        .bytes()
        .await
        .map_err(|e| FetchError::Http(describe_reqwest_error(&e)))?;

    let truncated = bytes.len() > MAX_BODY_BYTES;
    let slice = if truncated {
        &bytes[..MAX_BODY_BYTES]
    } else {
        &bytes[..]
    };

    let body = if is_text_content_type(content_type.as_deref()) {
        String::from_utf8_lossy(slice).into_owned()
    } else {
        format!(
            "[non-text response: {} bytes, content-type={}]",
            bytes.len(),
            content_type.as_deref().unwrap_or("(none)")
        )
    };

    Ok(FetchResult {
        status,
        final_url,
        content_type,
        body,
        truncated,
    })
}

fn is_text_content_type(ct: Option<&str>) -> bool {
    let Some(ct) = ct else {
        // No Content-Type header: be optimistic and try to decode as text.
        return true;
    };
    let ct = ct
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    ct.starts_with("text/")
        || ct == "application/json"
        || ct == "application/xml"
        || ct == "application/xhtml+xml"
        || ct == "application/javascript"
        || ct.ends_with("+json")
        || ct.ends_with("+xml")
}

// ---------------------------------------------------------------------------
// Navigation: lossless reduce → cache by URL → find / paginate (cursor)
// ---------------------------------------------------------------------------

struct CachedDoc {
    status: u16,
    content_type: Option<String>,
    reduced: String,
}
static DOC_CACHE: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, CachedDoc>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
const DOC_CACHE_CAP: usize = 32;

/// The cursor is just the next CHARACTER offset, encoded so the model copies it
/// verbatim. Deliberately NOT hash-validated: if the cached page changed, an
/// offset still lands *somewhere* sensible (a small gap/overlap, snapped to a
/// line) rather than hard-failing — graceful degradation beats an error.
fn parse_cursor(c: &str) -> Option<usize> {
    c.strip_prefix('c')?.parse().ok()
}
fn make_cursor(offset: usize) -> String {
    format!("c{offset}")
}

fn cache_get(url: &str) -> Option<(u16, Option<String>, String)> {
    DOC_CACHE
        .lock()
        .ok()?
        .get(url)
        .map(|d| (d.status, d.content_type.clone(), d.reduced.clone()))
}

async fn fetch_reduce_cache(
    url: &str,
    user_agent: Option<&str>,
) -> Result<(u16, Option<String>, String), FetchError> {
    let result = fetch(url, user_agent).await?;
    let reduced =
        crate::content_reduce::reduce_lossless(&result.body, result.content_type.as_deref());
    if let Ok(mut cache) = DOC_CACHE.lock() {
        if cache.len() >= DOC_CACHE_CAP && !cache.contains_key(url) {
            cache.clear();
        }
        cache.insert(
            url.to_string(),
            CachedDoc {
                status: result.status,
                content_type: result.content_type.clone(),
                reduced: reduced.clone(),
            },
        );
    }
    Ok((result.status, result.content_type, reduced))
}

/// The single entry point for `web_fetch`: plain fetch, `find=` selection, and
/// `cursor=` (offset) pagination, backed by a URL-keyed cache. `cap_tokens`
/// bounds one page / one find response.
pub async fn fetch_nav(
    url: &str,
    user_agent: Option<&str>,
    find: Option<&str>,
    cursor: Option<&str>,
    cap_tokens: usize,
) -> Result<String, FetchError> {
    // find/cursor navigate the already-fetched doc → prefer the cache (no
    // re-fetch); a plain fetch always pulls fresh content.
    let (status, ct, reduced, fetched) = if find.is_some() || cursor.is_some() {
        match cache_get(url) {
            Some((s, c, r)) => (s, c, r, false),
            None => {
                let (s, c, r) = fetch_reduce_cache(url, user_agent).await?;
                (s, c, r, true)
            }
        }
    } else {
        let (s, c, r) = fetch_reduce_cache(url, user_agent).await?;
        (s, c, r, true)
    };

    // Track CONSECUTIVE non-2xx fetches. A single 404 is just reported (its body
    // may be real content — api.handle.me serves documentation ON its 404 page),
    // but 3+ failures in a row means the model is GUESSING URLs, and only then do
    // we add the stop-guessing nudge. Only real fetches move the streak; cache
    // re-serves (find/cursor) don't.
    let streak = if fetched {
        note_fetch_outcome(status)
    } else {
        current_streak()
    };

    // Always surface the real status AND the body — never suppress content, even
    // on a non-2xx. The status label makes a 404/500 unmistakable; the body still
    // comes through (paginated / find-able like any page).
    let out = if reduced.trim().is_empty() {
        render_empty_ok(url, status, ct.as_deref())
    } else if let Some(q) = find {
        let slice = crate::content_reduce::find_in(&reduced, ct.as_deref(), q, cap_tokens);
        format!(
            "{} \u{b7} {url}\nContent-Type: {}\nfind=\"{q}\"\n\n---\n{slice}",
            status_label(status),
            ct.as_deref().unwrap_or("(none)")
        )
    } else {
        let offset = cursor.and_then(parse_cursor).unwrap_or(0);
        render_page(url, status, ct.as_deref(), &reduced, offset, cap_tokens)
    };
    Ok(append_guess_hint(out, status, streak))
}

/// Consecutive non-2xx fetches in this process. A 2xx resets it; the count is how
/// many failures in a row — the signal that the model is guessing rather than
/// hitting a one-off bad URL.
static FETCH_FAILURE_STREAK: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
const GUESS_STREAK_THRESHOLD: usize = 3;

fn note_fetch_outcome(status: u16) -> usize {
    use std::sync::atomic::Ordering;
    if (200..300).contains(&status) {
        FETCH_FAILURE_STREAK.store(0, Ordering::Relaxed);
        0
    } else {
        FETCH_FAILURE_STREAK.fetch_add(1, Ordering::Relaxed) + 1
    }
}
fn current_streak() -> usize {
    FETCH_FAILURE_STREAK.load(std::sync::atomic::Ordering::Relaxed)
}

/// `HTTP 404 Not Found` / `HTTP 200 OK` / `HTTP 599` (unknown code → no phrase).
fn status_label(status: u16) -> String {
    let reason = status_reason(status);
    if reason.is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status} {reason}")
    }
}

/// After several non-2xx fetches in a row, the model is guessing — tell it to
/// stop and change tactic. Below the threshold, a failure stands on its own (the
/// status + body are enough; no scolding for a single bad URL).
fn append_guess_hint(mut out: String, status: u16, streak: usize) -> String {
    if !(200..300).contains(&status) && streak >= GUESS_STREAK_THRESHOLD {
        out.push_str(&format!(
            "\n\n\u{26a0} {streak} web_fetch calls in a row have failed (non-2xx). If you're guessing URLs, STOP — more variants on the same host will keep failing. Find the correct URL via local_web_search or a known source, or take a different step."
        ));
    }
    out
}

fn render_page(
    url: &str,
    status: u16,
    ct: Option<&str>,
    reduced: &str,
    offset: usize,
    cap_tokens: usize,
) -> String {
    let (body, next, total) = crate::content_reduce::page_from(reduced, offset, cap_tokens);
    let mut out = format!(
        "{} \u{b7} {url}\nContent-Type: {}\n--- (chars {offset}\u{2013}{next} of {total}) ---\n{body}\n",
        status_label(status),
        ct.unwrap_or("(none)"),
    );
    if next < total {
        out.push_str(&format!(
            "\n\u{26a0} More remains ({} of {total} chars left). Continue with the SAME url and:\n  cursor=\"{}\"\n(or call with find=\"<keyword>\" to jump straight to a section.)",
            total - next,
            make_cursor(next),
        ));
    }
    out
}

/// A 2xx with an empty body — say so explicitly; a bare empty string reads as
/// "nothing happened" and invites a pointless identical retry.
fn render_empty_ok(url: &str, status: u16, ct: Option<&str>) -> String {
    format!(
        "HTTP {status} {} \u{b7} {url}\nContent-Type: {}\nThe response body was EMPTY (no text content). \
         Retrying this exact URL will return the same empty result — try a different source or path.",
        status_reason(status),
        ct.unwrap_or("(none)"),
    )
}

/// Canonical reason phrase for the HTTP status codes a fetch realistically hits.
/// Unknown codes return `""` (the numeric code still carries the signal).
fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        303 => "See Other",
        304 => "Not Modified",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        408 => "Request Timeout",
        410 => "Gone",
        418 => "I'm a teapot",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_url() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(matches!(
            rt.block_on(fetch("", None)),
            Err(FetchError::InvalidUrl(_))
        ));
        assert!(matches!(
            rt.block_on(fetch("   ", None)),
            Err(FetchError::InvalidUrl(_))
        ));
    }

    #[test]
    fn rejects_non_http_schemes() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(matches!(
            rt.block_on(fetch("file:///etc/passwd", None)),
            Err(FetchError::InvalidUrl(_))
        ));
        assert!(matches!(
            rt.block_on(fetch("ftp://example.com/foo", None)),
            Err(FetchError::InvalidUrl(_))
        ));
    }

    #[test]
    fn non_2xx_keeps_the_body_and_shows_status() {
        // api.handle.me serves real docs on its 404 — the body must survive, with
        // the status made unmistakable.
        let page = render_page(
            "https://api.handle.me/openapi.json",
            404,
            Some("application/json"),
            "Not found, but here are the available endpoints: /handles, /holders",
            0,
            4000,
        );
        assert!(
            page.contains("HTTP 404 Not Found"),
            "status visible: {page}"
        );
        assert!(
            page.contains("available endpoints"),
            "body preserved: {page}"
        );
    }

    #[test]
    fn guess_hint_only_after_threshold_and_only_on_failure() {
        let base = "HTTP 404 Not Found \u{b7} https://x".to_string();
        // 1–2 failures: no scolding.
        assert!(!append_guess_hint(base.clone(), 404, 1).contains("guessing"));
        assert!(!append_guess_hint(base.clone(), 404, 2).contains("guessing"));
        // 3rd consecutive failure: the stop-guessing nudge appears.
        let hinted = append_guess_hint(base.clone(), 404, 3);
        assert!(hinted.contains("3 web_fetch calls in a row"));
        assert!(hinted.contains("guessing"));
        // A 2xx never gets the nudge, regardless of streak.
        assert!(
            !append_guess_hint("HTTP 200 OK \u{b7} x".to_string(), 200, 9).contains("guessing")
        );
    }

    #[test]
    fn status_label_formats() {
        assert_eq!(status_label(404), "HTTP 404 Not Found");
        assert_eq!(status_label(200), "HTTP 200 OK");
        assert_eq!(status_label(599), "HTTP 599");
    }

    #[test]
    fn empty_2xx_says_empty() {
        let out = render_empty_ok("https://e.com/x", 200, Some("text/plain"));
        assert!(out.contains("HTTP 200 OK"));
        assert!(out.contains("EMPTY"), "{out}");
    }

    #[test]
    fn status_reason_maps_common_codes() {
        assert_eq!(status_reason(404), "Not Found");
        assert_eq!(status_reason(200), "OK");
        assert_eq!(status_reason(503), "Service Unavailable");
        assert_eq!(status_reason(599), "");
    }

    #[test]
    fn content_type_text_detection() {
        assert!(is_text_content_type(Some("text/html")));
        assert!(is_text_content_type(Some("text/html; charset=utf-8")));
        assert!(is_text_content_type(Some("application/json")));
        assert!(is_text_content_type(Some("application/ld+json")));
        assert!(is_text_content_type(Some("application/atom+xml")));
        assert!(is_text_content_type(None));
        assert!(!is_text_content_type(Some("image/png")));
        assert!(!is_text_content_type(Some("application/octet-stream")));
    }
}
