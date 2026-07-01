//! Ollama HTTP client for routing decisions.
//!
//! This is a minimal client that calls `/api/chat` for the router model.
//! It serializes requests per endpoint using a tokio Semaphore (matching
//! the Python reference's fcntl file locks).
//!
//! See docs/spec/routing-logic-reference.md.

use crate::config::{ClientFlavor, OllamaEndpoint};
use reqwest::Client;
use serde_json::Value as JsonValue;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Semaphore};
use tracing::warn;

/// Why a request to the local server failed to start, distinguishing the one
/// case the caller can recover from — the prompt exceeding the server's context
/// window — from everything else (connection, timeout, other 4xx/5xx).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SendError {
    /// The server rejected the request because the tokenized prompt exceeds its
    /// context window. Carries the server's own numbers so the caller can
    /// re-trim to fit and retry instead of crashing.
    ContextOverflow { n_ctx: u64, n_prompt_tokens: u64 },
    /// Any other failure — connection refused/reset, timeout, auth, a different
    /// 4xx/5xx. Not recoverable by re-trimming.
    Other,
}

/// Classify a non-success response body. llama.cpp returns a 400 with
/// `type=exceed_context_size_error` and the real `n_ctx` / `n_prompt_tokens`
/// when the prompt is too large; surface those so the caller can self-correct.
fn classify_send_error(status: u16, body: &str) -> SendError {
    if status == 400 && body.contains("exceed_context_size_error") {
        if let Ok(v) = serde_json::from_str::<JsonValue>(body) {
            let err = v.get("error").unwrap_or(&v);
            if let (Some(n_ctx), Some(n_prompt_tokens)) = (
                err.get("n_ctx").and_then(JsonValue::as_u64),
                err.get("n_prompt_tokens").and_then(JsonValue::as_u64),
            ) {
                return SendError::ContextOverflow {
                    n_ctx,
                    n_prompt_tokens,
                };
            }
        }
    }
    SendError::Other
}

/// Probe a llama.cpp server's REAL context window via `/props` (the startup
/// `--ctx-size`, which a per-request `num_ctx` does NOT override). Lets us size
/// the budget from the actual window instead of a hand-set `trim_budget`. Returns
/// `None` for non-llama.cpp servers (no `/props`) or on any error — the caller
/// then falls back to its configured budget.
pub async fn probe_context_window(client: &Client, base_url: &str) -> Option<u64> {
    let base = base_url.trim_end_matches('/');
    let root = base.strip_suffix("/v1").unwrap_or(base);
    let url = format!("{root}/props");
    let resp = client
        .get(&url)
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let v: JsonValue = resp.json().await.ok()?;
    v.get("default_generation_settings")
        .and_then(|g| g.get("n_ctx"))
        .and_then(JsonValue::as_u64)
        .or_else(|| v.get("n_ctx").and_then(JsonValue::as_u64))
        .filter(|n| *n > 0)
}

/// Per-endpoint semaphore to serialize Ollama requests.
/// Ollama struggles with concurrent requests — this was discovered
/// through testing in the coding-agent-router project.
#[derive(Default)]
pub struct OllamaClientPool {
    semaphores: Mutex<HashMap<String, Arc<Semaphore>>>,
    client: Client,
    /// Tracks the last model used on each endpoint URL.
    /// Warm models avoid 10-20s cold-load penalty.
    warm_models: Mutex<HashMap<String, String>>,
}

impl OllamaClientPool {
    pub fn new() -> Self {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_default();
        Self {
            semaphores: Mutex::new(HashMap::new()),
            warm_models: Mutex::new(HashMap::new()),
            client,
        }
    }

    /// Get the last model used on an endpoint (the "warm" model).
    /// Returns None if no model has been used on this endpoint yet.
    pub async fn warm_model(&self, base_url: &str) -> Option<String> {
        let map = self.warm_models.lock().await;
        map.get(base_url).cloned()
    }

    /// Record which model was just used on an endpoint.
    async fn record_warm_model(&self, base_url: &str, model: &str) {
        let mut map = self.warm_models.lock().await;
        map.insert(base_url.to_string(), model.to_string());
    }

    /// Access the underlying HTTP client (for health checks, etc.).
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Get (or create) the semaphore for a given base URL.
    async fn semaphore_for(&self, base_url: &str) -> Arc<Semaphore> {
        let mut map = self.semaphores.lock().await;
        map.entry(base_url.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(1)))
            .clone()
    }

    /// Call the endpoint's chat completion API. Wrapper around
    /// [`Self::chat_with_tools`] for callers that don't need tool calls.
    pub async fn chat(
        &self,
        endpoint: &OllamaEndpoint,
        messages: Vec<JsonValue>,
        system: Option<&str>,
        response_format: Option<&str>,
    ) -> Option<JsonValue> {
        self.chat_with_tools(endpoint, messages, system, response_format, None)
            .await
    }

    /// POST a request with bounded retries on *transient* failures, returning
    /// the successful response (or `None`).
    ///
    /// Every local role points at the same server, so when that server
    /// momentarily refuses a connection (single slot busy / checkpointing) or
    /// returns a 5xx, failing over the chain can't route around it — the next
    /// role hits the same server. Only retrying the same server recovers it, so
    /// this lives at the pool layer where *every* local caller (coder, reasoner,
    /// classifier, completion verifier, compaction, supervisor) benefits.
    ///
    /// Retried: connection-level errors (`is_connect`) and 5xx. NOT retried:
    /// 4xx (e.g. context-overflow 400 — terminal, must escalate), timeouts (the
    /// model is slow; the caller's failover handles that), and parse errors.
    async fn send_with_retry(
        &self,
        url: &str,
        payload: &JsonValue,
        timeout_seconds: u64,
        label: &str,
    ) -> Result<reqwest::Response, SendError> {
        const ATTEMPTS: usize = 3;
        const BACKOFF_MS: u64 = 750;
        for attempt in 0..ATTEMPTS {
            let mut req = self.client.post(url).json(payload);
            if timeout_seconds > 0 {
                req = req.timeout(Duration::from_secs(timeout_seconds));
            }
            match req.send().await {
                Ok(resp) if resp.status().is_success() => return Ok(resp),
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_server_error() && attempt + 1 < ATTEMPTS {
                        warn!(label = %label, status = %status, attempt = attempt + 1,
                            "transient 5xx from local server; retrying same server after backoff");
                        tokio::time::sleep(Duration::from_millis(BACKOFF_MS)).await;
                        continue;
                    }
                    let body = resp.text().await.unwrap_or_default();
                    let snippet = body.chars().take(500).collect::<String>();
                    warn!(label = %label, url = %url, status = %status, body = %snippet,
                        "request returned non-success status");
                    return Err(classify_send_error(status.as_u16(), &body));
                }
                Err(e) if e.is_connect() && attempt + 1 < ATTEMPTS => {
                    warn!(label = %label, error = %e, attempt = attempt + 1,
                        "transient connect error to local server; retrying same server after backoff");
                    tokio::time::sleep(Duration::from_millis(BACKOFF_MS)).await;
                }
                Err(e) => {
                    warn!(label = %label, url = %url, error = %e, "request error");
                    return Err(SendError::Other);
                }
            }
        }
        Err(SendError::Other)
    }

    /// Call the endpoint's chat completion API with optional tools.
    ///
    /// Branches internally on [`OllamaEndpoint::flavor`] to build the right
    /// URL and payload shape (Ollama's `/api/chat` vs OpenAI's
    /// `/v1/chat/completions`). The returned `JsonValue` is always
    /// translated to the Ollama shape (`{message: {content, tool_calls?,
    /// thinking?}, prompt_eval_count, eval_count}`) so callers don't need
    /// to know which flavor was used.
    pub async fn chat_with_tools(
        &self,
        endpoint: &OllamaEndpoint,
        messages: Vec<JsonValue>,
        system: Option<&str>,
        response_format: Option<&str>,
        tools: Option<Vec<JsonValue>>,
    ) -> Option<JsonValue> {
        let sem = self.semaphore_for(&endpoint.base_url).await;
        let _permit = sem.acquire().await.ok()?;

        let mut payload_messages = messages;
        if let Some(sys) = system {
            payload_messages.insert(0, json!({"role": "system", "content": sys}));
        }

        let url = build_chat_url(&endpoint.base_url, endpoint.flavor);
        let payload =
            build_chat_payload(endpoint, payload_messages, response_format, tools.as_ref());

        let resp = self
            .send_with_retry(&url, &payload, endpoint.timeout_seconds, "chat")
            .await
            .ok()?;
        let body_text = resp.text().await.unwrap_or_default();
        match serde_json::from_str::<JsonValue>(&body_text) {
            Ok(body) => {
                // Some OpenAI-compat servers (and Ollama itself for some failure
                // modes) return HTTP 200 with an `{"error": ...}` body instead
                // of a real response. The translator would silently produce
                // empty content, hiding the actual problem — surface it as a
                // None so the caller's warn fires with the cause. (Not retried:
                // a 200 error body is usually terminal, e.g. context overflow.)
                if let Some(err) = body.get("error") {
                    let snippet = err.to_string().chars().take(500).collect::<String>();
                    warn!(
                        url = %url,
                        error = %snippet,
                        "chat response carried an error body — treating as failure"
                    );
                    return None;
                }
                self.record_warm_model(&endpoint.base_url, &endpoint.model)
                    .await;
                Some(translate_response_to_ollama_shape(body, endpoint.flavor))
            }
            Err(e) => {
                let snippet = body_text.chars().take(500).collect::<String>();
                warn!(
                    url = %url,
                    error = %e,
                    body = %snippet,
                    "chat response parse error"
                );
                None
            }
        }
    }

    /// Streaming chat — returns a receiver that yields content chunks as they arrive.
    /// Each chunk is a partial text delta. The final message includes token usage.
    ///
    /// Branches on [`OllamaEndpoint::flavor`]:
    /// - `Ollama`: NDJSON stream from `/api/chat`, one JSON object per line
    ///   with `{message: {content}, done, prompt_eval_count, eval_count}`.
    /// - `OpenAICompat`: Server-Sent Events from `/v1/chat/completions`,
    ///   each `data: <json>` line carrying `{choices: [{delta: {content}}]}`.
    ///   The terminator is `data: [DONE]`. We send `stream_options:
    ///   {include_usage: true}` so most servers emit a final usage chunk.
    pub async fn chat_stream(
        &self,
        endpoint: &OllamaEndpoint,
        messages: Vec<JsonValue>,
        system: Option<&str>,
    ) -> Option<tokio::sync::mpsc::Receiver<StreamChunk>> {
        let sem = self.semaphore_for(&endpoint.base_url).await;
        let _permit = sem.acquire().await.ok()?;

        let mut payload_messages = messages;
        if let Some(sys) = system {
            payload_messages.insert(0, json!({"role": "system", "content": sys}));
        }

        let payload = build_stream_payload(endpoint, payload_messages, None);
        let url = build_chat_url(&endpoint.base_url, endpoint.flavor);
        let response = self
            .send_with_retry(&url, &payload, endpoint.timeout_seconds, "stream")
            .await
            .ok()?;

        self.record_warm_model(&endpoint.base_url, &endpoint.model)
            .await;

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        match endpoint.flavor {
            ClientFlavor::Ollama => spawn_ollama_stream_reader(response, tx),
            ClientFlavor::OpenAICompat => spawn_openai_sse_reader(response, tx),
        }

        Some(rx)
    }

    /// Tool-aware streaming chat. Same as [`chat_stream`] but passes a
    /// `tools` array in the request, so the server can emit tool-call
    /// deltas via [`StreamChunk::ToolCallDelta`] during the stream.
    ///
    /// Dropping the returned receiver closes the HTTP connection, which
    /// signals the server to stop generating — this is how the rumination
    /// guard aborts in-flight inference.
    pub async fn chat_with_tools_stream(
        &self,
        endpoint: &OllamaEndpoint,
        messages: Vec<JsonValue>,
        system: Option<&str>,
        tools: Option<Vec<JsonValue>>,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamChunk>, SendError> {
        let sem = self.semaphore_for(&endpoint.base_url).await;
        let _permit = sem.acquire().await.map_err(|_| SendError::Other)?;

        let mut payload_messages = messages;
        if let Some(sys) = system {
            payload_messages.insert(0, json!({"role": "system", "content": sys}));
        }

        let payload = build_stream_payload(endpoint, payload_messages, tools.as_ref());
        let url = build_chat_url(&endpoint.base_url, endpoint.flavor);
        let response = self
            .send_with_retry(&url, &payload, endpoint.timeout_seconds, "tool-stream")
            .await?;

        self.record_warm_model(&endpoint.base_url, &endpoint.model)
            .await;

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        match endpoint.flavor {
            ClientFlavor::Ollama => spawn_ollama_stream_reader(response, tx),
            ClientFlavor::OpenAICompat => spawn_openai_sse_reader(response, tx),
        }
        Ok(rx)
    }
}

/// Build the streaming-mode request payload for the given flavor. Mirrors
/// [`build_chat_payload`] but sets `stream: true` and adds OpenAI's
/// `stream_options: {include_usage: true}` so usage tokens arrive in the
/// final SSE chunk. `tools` — when `Some` — is the same function-calling
/// schema we pass on non-streaming calls.
/// Apply optional sampler overrides (`top_p` / `top_k` / `repeat_penalty`) onto
/// an Ollama `options` object or an OpenAI-compat top-level payload — llama.cpp
/// accepts all three keys at the top level too. No-op for any value left unset,
/// so the server default applies. Models like Gemma 4 degenerate at the stock
/// sampler and need these set (see the per-role config fields).
fn apply_sampler_overrides(target: &mut JsonValue, endpoint: &OllamaEndpoint) {
    if let Some(v) = endpoint.top_p {
        target["top_p"] = json!(v);
    }
    if let Some(v) = endpoint.top_k {
        target["top_k"] = json!(v);
    }
    if let Some(v) = endpoint.repeat_penalty {
        target["repeat_penalty"] = json!(v);
    }
}

/// Normalize the outbound message list so the local server accepts it,
/// regardless of how transcript trimming/loop-excision shaped it. Two rules,
/// both enforcing the same invariant — *the list must not have, or end with,
/// stray assistant messages*:
///
/// 1. **Merge consecutive `assistant` messages.** Some servers reject adjacent
///    same-role messages ("Cannot have 2 or more assistant messages at the end
///    of the list"). A run of consecutive assistants can't contain tool/result
///    messages (different role), so merging is safe: concatenate text, union
///    `tool_calls`; any tool results still follow and stay paired by id. Only
///    `assistant` is touched (never `tool`/`user`, whose results carry ids).
/// 2. **Drop a trailing `assistant` message.** A request ending in an assistant
///    message is a "prefill", which servers reject when thinking is enabled
///    ("Assistant response prefill is incompatible with enable_thinking"). In
///    this harness the model always responds to a user/tool message, so a
///    trailing assistant is always an artifact (e.g. a dangling tool-call left
///    by loop excision) — drop it so the model responds fresh to the real last
///    input. Never empties the list.
fn normalize_outbound_messages(messages: Vec<JsonValue>) -> Vec<JsonValue> {
    let is_assistant = |m: &JsonValue| m.get("role").and_then(|r| r.as_str()) == Some("assistant");
    let mut out: Vec<JsonValue> = Vec::with_capacity(messages.len());
    for msg in messages {
        if is_assistant(&msg) && out.last().is_some_and(is_assistant) {
            merge_assistant_into_prev(out.last_mut().unwrap(), msg);
        } else {
            out.push(msg);
        }
    }
    while out.len() > 1 && out.last().is_some_and(is_assistant) {
        out.pop();
    }
    out
}

fn merge_assistant_into_prev(prev: &mut JsonValue, next: JsonValue) {
    let JsonValue::Object(next_obj) = next else {
        return;
    };
    let Some(prev_obj) = prev.as_object_mut() else {
        return;
    };
    if let Some(next_content) = next_obj.get("content").and_then(|c| c.as_str())
        && !next_content.is_empty()
    {
        let merged = match prev_obj.get("content").and_then(|c| c.as_str()) {
            Some(p) if !p.is_empty() => format!("{p}\n{next_content}"),
            _ => next_content.to_string(),
        };
        prev_obj.insert("content".to_string(), JsonValue::String(merged));
    }
    if let Some(JsonValue::Array(next_tc)) = next_obj.get("tool_calls")
        && !next_tc.is_empty()
    {
        match prev_obj.get_mut("tool_calls") {
            Some(JsonValue::Array(prev_tc)) => prev_tc.extend(next_tc.iter().cloned()),
            _ => {
                prev_obj.insert("tool_calls".to_string(), JsonValue::Array(next_tc.clone()));
            }
        }
    }
}

/// Render a configured `tool_choice` for the wire. The keyword forms
/// (`auto`/`none`/`required`/`any`) pass through as bare strings; anything else is
/// treated as a TOOL NAME and emitted in OpenAI's object form so the server forces
/// that specific tool (e.g. steering a stuck local model to `write_file`). Without
/// this, a function name would be sent as an invalid bare string and ignored.
fn tool_choice_payload(tc: &str) -> JsonValue {
    match tc {
        "auto" | "none" | "required" | "any" => json!(tc),
        name => json!({ "type": "function", "function": { "name": name } }),
    }
}

fn build_stream_payload(
    endpoint: &OllamaEndpoint,
    messages: Vec<JsonValue>,
    tools: Option<&Vec<JsonValue>>,
) -> JsonValue {
    let messages = normalize_outbound_messages(messages);
    match endpoint.flavor {
        ClientFlavor::Ollama => {
            let mut options = json!({
                "temperature": endpoint.temperature,
                "num_ctx": endpoint.trim_budget,
            });
            if let Some(n) = endpoint.max_tokens {
                options["num_predict"] = json!(n);
            }
            apply_sampler_overrides(&mut options, endpoint);
            let mut payload = json!({
                "model": &endpoint.model,
                "messages": messages,
                "stream": true,
                "options": options,
                "think": endpoint.think,
            });
            if let Some(t) = tools {
                payload["tools"] = json!(t);
            }
            payload
        }
        ClientFlavor::OpenAICompat => {
            let mut payload = json!({
                "model": &endpoint.model,
                "messages": messages,
                "stream": true,
                "temperature": endpoint.temperature,
                "stream_options": {"include_usage": true},
            });
            apply_sampler_overrides(&mut payload, endpoint);
            if let Some(n) = endpoint.max_tokens {
                payload["max_tokens"] = json!(n);
            }
            if let Some(t) = tools {
                payload["tools"] = json!(t);
                // Constrain tool-call FORMAT at the source when configured (e.g.
                // `"required"` makes llama.cpp grammar-enforce a valid call).
                // Only meaningful when tools are present.
                if let Some(tc) = &endpoint.tool_choice {
                    payload["tool_choice"] = tool_choice_payload(tc);
                }
            }
            payload
        }
    }
}

fn spawn_ollama_stream_reader(
    response: reqwest::Response,
    tx: tokio::sync::mpsc::Sender<StreamChunk>,
) {
    tokio::spawn(async move {
        use futures::StreamExt;
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_result) = byte_stream.next().await {
            let Ok(bytes) = chunk_result else { break };
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            // Process complete lines (Ollama sends one JSON object per line)
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                let Ok(obj) = serde_json::from_str::<JsonValue>(&line) else {
                    continue;
                };

                let done = obj.get("done").and_then(|d| d.as_bool()).unwrap_or(false);
                let msg = obj.get("message");
                let content = msg
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("");
                let thinking = msg
                    .and_then(|m| m.get("thinking"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("");

                if !thinking.is_empty() {
                    let _ = tx
                        .send(StreamChunk::ReasoningDelta(thinking.to_string()))
                        .await;
                }
                if !content.is_empty() {
                    let _ = tx.send(StreamChunk::Delta(content.to_string())).await;
                }
                // Ollama emits any tool_calls atomically in the final chunk
                // (not as per-arg-char deltas like OpenAI SSE). Forward them
                // as one ToolCallDelta per call, with the full argument JSON.
                if let Some(tool_calls) = msg
                    .and_then(|m| m.get("tool_calls"))
                    .and_then(|tc| tc.as_array())
                {
                    for (index, call) in tool_calls.iter().enumerate() {
                        let func = call.get("function").unwrap_or(call);
                        let name = func
                            .get("name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let args = func
                            .get("arguments")
                            .map(|v| {
                                if let Some(s) = v.as_str() {
                                    s.to_string()
                                } else {
                                    v.to_string()
                                }
                            })
                            .unwrap_or_default();
                        let _ = tx
                            .send(StreamChunk::ToolCallDelta {
                                index,
                                id: None,
                                name,
                                arguments_delta: args,
                            })
                            .await;
                    }
                }

                if done {
                    let input_tokens = obj
                        .get("prompt_eval_count")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let output_tokens = obj.get("eval_count").and_then(|v| v.as_u64()).unwrap_or(0);
                    // Ollama reports durations in nanoseconds; convert to ms.
                    // These are exact model-side times (exclude network).
                    let ns_to_ms =
                        |key: &str| obj.get(key).and_then(|v| v.as_u64()).unwrap_or(0) / 1_000_000;
                    let _ = tx
                        .send(StreamChunk::Done {
                            input_tokens,
                            output_tokens,
                            reasoning_tokens: 0, // Ollama doesn't break out reasoning tokens
                            prompt_ms: ns_to_ms("prompt_eval_duration"),
                            gen_ms: ns_to_ms("eval_duration"),
                        })
                        .await;
                    return;
                }
            }
        }
    });
}

fn spawn_openai_sse_reader(
    response: reqwest::Response,
    tx: tokio::sync::mpsc::Sender<StreamChunk>,
) {
    tokio::spawn(async move {
        use futures::StreamExt;
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut input_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;
        let mut reasoning_tokens: u64 = 0;
        // OpenAI-compat servers don't report eval durations, so measure
        // throughput by wall clock: `start` (reader spawn, ~response headers)
        // to first generated token approximates prompt-ingest; first token to
        // [DONE] approximates generation time.
        let start = std::time::Instant::now();
        let mut first_token: Option<std::time::Instant> = None;
        let wallclock_ms = |first: Option<std::time::Instant>| match first {
            Some(ft) => (
                ft.duration_since(start).as_millis() as u64,
                ft.elapsed().as_millis() as u64,
            ),
            None => (start.elapsed().as_millis() as u64, 0),
        };

        while let Some(chunk_result) = byte_stream.next().await {
            let Ok(bytes) = chunk_result else { break };
            buffer.push_str(&String::from_utf8_lossy(&bytes));

            // SSE separates events by blank lines (`\n\n`), but individual
            // `data:` lines arrive any time. We process line-by-line and
            // ignore comments / non-data lines (`event:`, `id:`, `retry:`,
            // and SSE comments starting with `:`).
            while let Some(newline_pos) = buffer.find('\n') {
                let raw_line = buffer[..newline_pos].to_string();
                buffer = buffer[newline_pos + 1..].to_string();
                let line = raw_line.trim_end_matches('\r');

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                let Some(payload) = line.strip_prefix("data:") else {
                    // Skip event:/id:/retry: and any other SSE meta lines.
                    continue;
                };
                let payload = payload.trim_start();

                if payload == "[DONE]" {
                    let (prompt_ms, gen_ms) = wallclock_ms(first_token);
                    let _ = tx
                        .send(StreamChunk::Done {
                            input_tokens,
                            output_tokens,
                            reasoning_tokens,
                            prompt_ms,
                            gen_ms,
                        })
                        .await;
                    return;
                }

                let Ok(obj) = serde_json::from_str::<JsonValue>(payload) else {
                    continue;
                };

                if let Some(usage) = obj.get("usage") {
                    if let Some(p) = usage.get("prompt_tokens").and_then(JsonValue::as_u64) {
                        input_tokens = p;
                    }
                    if let Some(c) = usage.get("completion_tokens").and_then(JsonValue::as_u64) {
                        output_tokens = c;
                    }
                    // OpenAI-compat servers report reasoning tokens under
                    // `usage.completion_tokens_details.reasoning_tokens` (LM
                    // Studio follows this). Critical for the rumination
                    // detector's budget gate.
                    if let Some(r) = usage
                        .get("completion_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                        .and_then(JsonValue::as_u64)
                    {
                        reasoning_tokens = r;
                    }
                }

                let delta = obj
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|first| first.get("delta"));

                let delta_content = delta
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("");
                let delta_reasoning = delta
                    .and_then(|d| d.get("reasoning_content"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("");

                if !delta_reasoning.is_empty() {
                    first_token.get_or_insert_with(std::time::Instant::now);
                    let _ = tx
                        .send(StreamChunk::ReasoningDelta(delta_reasoning.to_string()))
                        .await;
                }
                if !delta_content.is_empty() {
                    first_token.get_or_insert_with(std::time::Instant::now);
                    let _ = tx.send(StreamChunk::Delta(delta_content.to_string())).await;
                }

                // Tool-call deltas: each chunk carries zero or more entries
                // in `delta.tool_calls[]`. The FIRST chunk for a given
                // `index` carries `id` and `function.name`; subsequent
                // chunks carry incremental `function.arguments` string
                // fragments that the caller concatenates. We forward them
                // verbatim and let the caller accumulate.
                if let Some(tool_calls) = delta
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(|tc| tc.as_array())
                {
                    for tc in tool_calls {
                        let index =
                            tc.get("index").and_then(JsonValue::as_u64).unwrap_or(0) as usize;
                        let id = tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string());
                        let func = tc.get("function");
                        let name = func
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let arguments_delta = func
                            .and_then(|f| f.get("arguments"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let _ = tx
                            .send(StreamChunk::ToolCallDelta {
                                index,
                                id,
                                name,
                                arguments_delta,
                            })
                            .await;
                    }
                }
            }
        }

        // Stream ended without seeing [DONE] — still flush a Done event with
        // whatever usage we accumulated (often zero) so the consumer can
        // finalize.
        let (prompt_ms, gen_ms) = wallclock_ms(first_token);
        let _ = tx
            .send(StreamChunk::Done {
                input_tokens,
                output_tokens,
                reasoning_tokens,
                prompt_ms,
                gen_ms,
            })
            .await;
    });
}

/// A chunk from a streaming chat response. Unified across backends —
/// OpenAI-compat SSE and Ollama NDJSON both surface as this enum.
///
/// Richer than a plain "text delta" because rumination detection needs to
/// distinguish between `reasoning_content` (private thinking the user
/// never sees) and `content` (the final answer text), and because tool-aware
/// calls need to assemble `tool_calls` from multi-chunk `arguments` deltas.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Partial `message.content` / `choices[0].delta.content` — the
    /// user-visible assistant answer.
    Delta(String),
    /// Partial reasoning content — `choices[0].delta.reasoning_content`
    /// (OpenAI-compat) or `message.thinking` (Ollama). Kept separate from
    /// `Delta` so watchers can run rumination detection against it
    /// without having to distinguish channels by parsing `<think>` tags.
    ReasoningDelta(String),
    /// Partial tool-call information. OpenAI-compat streams tool_calls as
    /// incremental deltas keyed by `index` — the first chunk for a given
    /// index carries `id` and `name`, subsequent chunks extend the JSON
    /// string in `arguments_delta`. Callers accumulate per-index.
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: String,
    },
    /// Stream is complete, with final token usage. `reasoning_tokens`
    /// surfaces the server's reasoning-channel budget consumption when
    /// available (OpenAI-compat `usage.completion_tokens_details
    /// .reasoning_tokens`); 0 if the server didn't report it.
    ///
    /// `prompt_ms` / `gen_ms` carry the prompt-ingest and generation wall
    /// times in milliseconds so callers can compute tokens/sec. For Ollama
    /// these are the server's exact `prompt_eval_duration` / `eval_duration`
    /// (model-side GPU time, excludes network). For OpenAI-compat servers,
    /// which don't report durations, they are wall-clock measured from the
    /// reader: `prompt_ms` is time-to-first-token and `gen_ms` is
    /// first-token-to-done (so they include some network overhead). Either
    /// is 0 when the duration is unknown.
    Done {
        input_tokens: u64,
        output_tokens: u64,
        reasoning_tokens: u64,
        prompt_ms: u64,
        gen_ms: u64,
    },
}

// ---------------------------------------------------------------------------
// Flavor-aware URL / payload / response translation.
// ---------------------------------------------------------------------------

/// Build the chat endpoint URL for a base URL + flavor. Defensively strips
/// a trailing `/` and — for OpenAI-compat — a trailing `/v1`, so users who
/// write `http://host:1234`, `http://host:1234/`, or `http://host:1234/v1`
/// all end up at `http://host:1234/v1/chat/completions`.
pub(crate) fn build_chat_url(base_url: &str, flavor: ClientFlavor) -> String {
    let base = base_url.trim_end_matches('/');
    match flavor {
        ClientFlavor::Ollama => format!("{base}/api/chat"),
        ClientFlavor::OpenAICompat => {
            let base = base.strip_suffix("/v1").unwrap_or(base);
            format!("{base}/v1/chat/completions")
        }
    }
}

/// Build the request payload for a chat call, branching on flavor.
pub(crate) fn build_chat_payload(
    endpoint: &OllamaEndpoint,
    messages: Vec<JsonValue>,
    response_format: Option<&str>,
    tools: Option<&Vec<JsonValue>>,
) -> JsonValue {
    let messages = normalize_outbound_messages(messages);
    match endpoint.flavor {
        ClientFlavor::Ollama => {
            let mut options = json!({
                "temperature": endpoint.temperature,
                "num_ctx": endpoint.trim_budget,
            });
            if let Some(n) = endpoint.max_tokens {
                // Ollama's equivalent of `max_tokens` is `num_predict`.
                options["num_predict"] = json!(n);
            }
            apply_sampler_overrides(&mut options, endpoint);
            let mut payload = json!({
                "model": &endpoint.model,
                "messages": messages,
                "stream": false,
                "options": options,
                "think": endpoint.think,
            });
            if response_format == Some("json") {
                payload["format"] = json!("json");
            }
            if let Some(tools) = tools {
                payload["tools"] = json!(tools);
            }
            payload
        }
        ClientFlavor::OpenAICompat => {
            // OpenAI puts `temperature` at the top level. Our `trim_budget` is
            // client-side only — this API has no context-size field (the
            // window is fixed on the server), so it is intentionally not sent. `think` is Ollama-specific and is silently
            // dropped. `response_format: json` is intentionally NOT
            // forwarded — LM Studio (and some other OpenAI-compat servers)
            // reject the older `{"type": "json_object"}` shape, accepting
            // only `"text"` or `"json_schema"` (the latter requires a
            // real schema that we don't carry). Relying on the caller's
            // system prompt asking for "JSON only" instead; this is how
            // the coder's own tool-call path already works.
            let _ = response_format; // consumed intentionally; see above
            let mut payload = json!({
                "model": &endpoint.model,
                "messages": messages,
                "stream": false,
                "temperature": endpoint.temperature,
            });
            apply_sampler_overrides(&mut payload, endpoint);
            if let Some(n) = endpoint.max_tokens {
                payload["max_tokens"] = json!(n);
            }
            if let Some(tools) = tools {
                payload["tools"] = json!(tools);
                if let Some(tc) = &endpoint.tool_choice {
                    payload["tool_choice"] = tool_choice_payload(tc);
                }
            }
            payload
        }
    }
}

/// Translate a chat response into the Ollama shape so callers have a
/// uniform surface (`body.message.content`, `body.message.tool_calls`,
/// `body.message.thinking`, `body.prompt_eval_count`, `body.eval_count`)
/// regardless of flavor. Ollama responses are passed through unchanged;
/// OpenAI responses are rewritten.
pub(crate) fn translate_response_to_ollama_shape(
    body: JsonValue,
    flavor: ClientFlavor,
) -> JsonValue {
    match flavor {
        ClientFlavor::Ollama => body,
        ClientFlavor::OpenAICompat => openai_response_to_ollama(body),
    }
}

fn openai_response_to_ollama(body: JsonValue) -> JsonValue {
    // Extract the first choice's message, if any.
    let message = body
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|first| first.get("message"))
        .cloned()
        .unwrap_or_else(|| json!({"role": "assistant", "content": ""}));

    // Ensure we have at minimum a content string (null → empty) and pass
    // through any tool_calls verbatim (OpenAI's tool_calls shape matches
    // Ollama's well enough that downstream parsers accept either).
    let mut message_out = json!({});
    message_out["role"] = message
        .get("role")
        .cloned()
        .unwrap_or_else(|| json!("assistant"));
    message_out["content"] = match message.get("content") {
        Some(JsonValue::Null) | None => json!(""),
        Some(v) => v.clone(),
    };
    if let Some(tool_calls) = message.get("tool_calls") {
        message_out["tool_calls"] = tool_calls.clone();
    }
    // Some OpenAI-compat servers (including LM Studio's newer builds and
    // vLLM with reasoning models) expose the thinking trace on a separate
    // field. Preserve it under the same name Ollama uses.
    if let Some(reasoning) = message.get("reasoning") {
        message_out["thinking"] = reasoning.clone();
    } else if let Some(reasoning_content) = message.get("reasoning_content") {
        message_out["thinking"] = reasoning_content.clone();
    }

    let usage = body.get("usage").cloned().unwrap_or(JsonValue::Null);
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    let completion_tokens = usage
        .get("completion_tokens")
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);

    json!({
        "message": message_out,
        "prompt_eval_count": prompt_tokens,
        "eval_count": completion_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ClientFlavor, OllamaEndpoint, ToolSubset};

    #[test]
    fn normalize_merges_mid_list_consecutive_assistants() {
        // Consecutive assistants NOT at the end are merged (the 019f079f
        // "2 or more assistant" case) but kept, because a user/tool follows.
        let msgs = vec![
            json!({"role": "assistant", "content": "first"}),
            json!({"role": "assistant", "content": "second", "tool_calls": [{"id": "a"}]}),
            json!({"role": "user", "content": "go on"}),
        ];
        let out = normalize_outbound_messages(msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["role"], "assistant");
        assert_eq!(out[0]["content"], "first\nsecond");
        assert_eq!(out[0]["tool_calls"][0]["id"], "a");
        assert_eq!(out[1]["role"], "user");
    }

    #[test]
    fn normalize_drops_trailing_assistant_prefill() {
        // The 019f0b3f failure: a list ending in an assistant message is a
        // "prefill", rejected when thinking is on. Drop it so the model
        // responds fresh to the real last input. Merge-then-drop also collapses
        // a trailing run of assistants entirely.
        let one = normalize_outbound_messages(vec![
            json!({"role": "user", "content": "do it"}),
            json!({"role": "assistant", "content": "I'll do X:"}),
        ]);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0]["role"], "user");

        let run = normalize_outbound_messages(vec![
            json!({"role": "tool", "tool_call_id": "x", "content": "r"}),
            json!({"role": "assistant", "content": "a"}),
            json!({"role": "assistant", "content": "b"}),
        ]);
        assert_eq!(run.len(), 1);
        assert_eq!(run[0]["role"], "tool");
    }

    #[test]
    fn normalize_leaves_well_ordered_lists_untouched() {
        // Ends with a tool result; tool messages are never merged (their ids
        // must stay separate) and nothing trailing to drop.
        let msgs = vec![
            json!({"role": "assistant", "tool_calls": [{"id": "x"}]}),
            json!({"role": "tool", "tool_call_id": "x", "content": "r1"}),
            json!({"role": "tool", "tool_call_id": "y", "content": "r2"}),
            json!({"role": "user", "content": "next"}),
        ];
        let out = normalize_outbound_messages(msgs.clone());
        assert_eq!(out, msgs);
    }

    #[test]
    #[test]
    fn tool_choice_function_name_becomes_object_form() {
        // Keywords pass through as bare strings; a tool name must become the
        // OpenAI object form, else forcing write_file would be an invalid bare string.
        assert_eq!(
            tool_choice_payload("required"),
            serde_json::json!("required")
        );
        assert_eq!(
            tool_choice_payload("write_file"),
            serde_json::json!({ "type": "function", "function": { "name": "write_file" } })
        );
    }

    fn tool_choice_is_sent_only_when_configured_and_tools_present() {
        let tools = vec![json!({"type": "function", "function": {"name": "shell"}})];
        let mut ep = endpoint(ClientFlavor::OpenAICompat);

        // Unset → no tool_choice field (server default = "auto", unconstrained).
        let p = build_stream_payload(
            &ep,
            vec![json!({"role": "user", "content": "hi"})],
            Some(&tools),
        );
        assert!(p.get("tool_choice").is_none());

        // Set → forwarded to the server to constrain tool-call format.
        ep.tool_choice = Some("required".to_string());
        let p = build_stream_payload(
            &ep,
            vec![json!({"role": "user", "content": "hi"})],
            Some(&tools),
        );
        assert_eq!(p["tool_choice"], "required");

        // Set but NO tools → omitted (tool_choice is meaningless without tools).
        let p = build_stream_payload(&ep, vec![json!({"role": "user", "content": "hi"})], None);
        assert!(p.get("tool_choice").is_none());
    }

    #[test]
    fn classify_send_error_parses_context_overflow() {
        // The exact body llama.cpp returned in session 019f05ae.
        let body = r#"{"error":{"code":400,"message":"request (32946 tokens) exceeds the available context size (32768 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":32946,"n_ctx":32768}}"#;
        assert_eq!(
            classify_send_error(400, body),
            SendError::ContextOverflow {
                n_ctx: 32768,
                n_prompt_tokens: 32946,
            }
        );
        // A different 400 and a 5xx are both Other (not recoverable by re-trim).
        assert_eq!(
            classify_send_error(400, r#"{"error":{"type":"invalid_request"}}"#),
            SendError::Other
        );
        assert_eq!(classify_send_error(500, "internal error"), SendError::Other);
    }

    fn endpoint(flavor: ClientFlavor) -> OllamaEndpoint {
        OllamaEndpoint {
            base_url: "http://host:1234".to_string(),
            model: "m".to_string(),
            trim_budget: 2048,
            temperature: 0.1,
            timeout_seconds: 10,
            enabled: true,
            think: true,
            tool_subset: ToolSubset::Focused,
            flavor,
            max_tokens: None,
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            tool_choice: None,
        }
    }

    #[test]
    fn build_chat_url_ollama() {
        assert_eq!(
            build_chat_url("http://host:11434", ClientFlavor::Ollama),
            "http://host:11434/api/chat"
        );
    }

    #[test]
    fn build_chat_url_ollama_strips_trailing_slash() {
        assert_eq!(
            build_chat_url("http://host:11434/", ClientFlavor::Ollama),
            "http://host:11434/api/chat"
        );
    }

    #[test]
    fn build_chat_url_openai_compat() {
        assert_eq!(
            build_chat_url("http://host:1234", ClientFlavor::OpenAICompat),
            "http://host:1234/v1/chat/completions"
        );
    }

    #[test]
    fn build_chat_url_openai_compat_strips_trailing_v1() {
        assert_eq!(
            build_chat_url("http://host:1234/v1", ClientFlavor::OpenAICompat),
            "http://host:1234/v1/chat/completions"
        );
        assert_eq!(
            build_chat_url("http://host:1234/v1/", ClientFlavor::OpenAICompat),
            "http://host:1234/v1/chat/completions"
        );
    }

    #[test]
    fn ollama_payload_has_options_and_think() {
        let ep = endpoint(ClientFlavor::Ollama);
        let payload =
            build_chat_payload(&ep, vec![json!({"role":"user","content":"hi"})], None, None);
        assert_eq!(payload["model"], "m");
        assert_eq!(payload["options"]["num_ctx"], 2048);
        assert_eq!(payload["think"], true);
    }

    #[test]
    fn sampler_overrides_appear_in_payload_when_set() {
        let mut ep = endpoint(ClientFlavor::Ollama);
        ep.top_p = Some(0.95);
        ep.top_k = Some(64);
        ep.repeat_penalty = Some(1.1);
        let p = build_chat_payload(&ep, vec![json!({"role":"user","content":"hi"})], None, None);
        assert_eq!(p["options"]["top_p"], 0.95);
        assert_eq!(p["options"]["top_k"], 64);
        assert_eq!(p["options"]["repeat_penalty"], 1.1);

        let mut ep2 = endpoint(ClientFlavor::OpenAICompat);
        ep2.top_p = Some(0.95);
        ep2.repeat_penalty = Some(1.1);
        let p2 = build_chat_payload(
            &ep2,
            vec![json!({"role":"user","content":"hi"})],
            None,
            None,
        );
        assert_eq!(p2["top_p"], 0.95);
        assert_eq!(p2["repeat_penalty"], 1.1);
    }

    #[test]
    fn sampler_overrides_omitted_when_unset() {
        let ep = endpoint(ClientFlavor::Ollama);
        let p = build_chat_payload(&ep, vec![json!({"role":"user","content":"hi"})], None, None);
        assert!(p["options"].get("top_p").is_none());
        assert!(p["options"].get("repeat_penalty").is_none());
    }

    #[test]
    fn openai_payload_flat_temp_no_think_no_num_ctx() {
        let ep = endpoint(ClientFlavor::OpenAICompat);
        let payload =
            build_chat_payload(&ep, vec![json!({"role":"user","content":"hi"})], None, None);
        assert_eq!(payload["model"], "m");
        assert_eq!(payload["temperature"], 0.1);
        assert!(payload.get("options").is_none());
        assert!(payload.get("think").is_none());
        assert!(payload.get("num_ctx").is_none());
    }

    #[test]
    fn openai_payload_includes_max_tokens_when_set() {
        let mut ep = endpoint(ClientFlavor::OpenAICompat);
        ep.max_tokens = Some(8000);
        let payload = build_chat_payload(&ep, vec![], None, None);
        assert_eq!(payload["max_tokens"], 8000);
    }

    #[test]
    fn openai_payload_omits_max_tokens_when_unset() {
        let ep = endpoint(ClientFlavor::OpenAICompat);
        let payload = build_chat_payload(&ep, vec![], None, None);
        assert!(payload.get("max_tokens").is_none());
    }

    #[test]
    fn ollama_payload_uses_num_predict_for_max_tokens() {
        let mut ep = endpoint(ClientFlavor::Ollama);
        ep.max_tokens = Some(8000);
        let payload = build_chat_payload(&ep, vec![], None, None);
        assert_eq!(payload["options"]["num_predict"], 8000);
        assert!(payload.get("max_tokens").is_none()); // not top-level for Ollama
    }

    #[test]
    fn openai_payload_json_response_format_is_not_forwarded() {
        // LM Studio rejects `{type: "json_object"}`; we rely on the caller's
        // system prompt to enforce JSON output. Assert we don't emit either
        // the OpenAI field or Ollama's `format` field.
        let ep = endpoint(ClientFlavor::OpenAICompat);
        let payload = build_chat_payload(&ep, vec![], Some("json"), None);
        assert!(payload.get("response_format").is_none());
        assert!(payload.get("format").is_none());
    }

    #[test]
    fn ollama_payload_json_response_format_is_format_field() {
        let ep = endpoint(ClientFlavor::Ollama);
        let payload = build_chat_payload(&ep, vec![], Some("json"), None);
        assert_eq!(payload["format"], "json");
    }

    #[test]
    fn openai_response_translates_to_ollama_shape() {
        let openai_body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "tool_calls": [{"id": "1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 7,
                "total_tokens": 49
            }
        });
        let translated =
            translate_response_to_ollama_shape(openai_body, ClientFlavor::OpenAICompat);
        assert_eq!(translated["message"]["content"], "hello");
        assert_eq!(
            translated["message"]["tool_calls"][0]["function"]["name"],
            "x"
        );
        assert_eq!(translated["prompt_eval_count"], 42);
        assert_eq!(translated["eval_count"], 7);
    }

    #[test]
    fn openai_response_null_content_becomes_empty_string() {
        let openai_body = json!({
            "choices": [{"message": {"role": "assistant", "content": null}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0}
        });
        let translated =
            translate_response_to_ollama_shape(openai_body, ClientFlavor::OpenAICompat);
        assert_eq!(translated["message"]["content"], "");
    }

    #[test]
    fn openai_response_reasoning_mapped_to_thinking() {
        let openai_body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning": "let me think..."
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let translated =
            translate_response_to_ollama_shape(openai_body, ClientFlavor::OpenAICompat);
        assert_eq!(translated["message"]["thinking"], "let me think...");
    }

    #[test]
    fn openai_error_body_detection() {
        // We don't actually exercise the dispatcher here (would need a
        // mock HTTP server); instead, prove the structural check we rely
        // on inside chat_with_tools recognizes OpenAI error shapes.
        let body: JsonValue = serde_json::from_str(
            r#"{"error":{"message":"No models loaded","type":"invalid_request_error","param":"model"}}"#,
        )
        .unwrap();
        assert!(body.get("error").is_some());
        assert!(body.get("choices").is_none());
    }

    #[test]
    fn ollama_error_body_detection() {
        // Ollama's error shape is `{"error": "<string>"}`. Same top-level
        // `error` field, so the same check triggers.
        let body: JsonValue = serde_json::from_str(
            r#"{"error":"model 'qwopus-q6-think' not found, try pulling it first"}"#,
        )
        .unwrap();
        assert!(body.get("error").is_some());
        assert!(body.get("message").is_none());
    }

    #[test]
    fn ollama_response_passed_through_unchanged() {
        let body = json!({
            "message": {"role": "assistant", "content": "x"},
            "prompt_eval_count": 1,
            "eval_count": 2
        });
        let translated = translate_response_to_ollama_shape(body.clone(), ClientFlavor::Ollama);
        assert_eq!(translated, body);
    }
}
