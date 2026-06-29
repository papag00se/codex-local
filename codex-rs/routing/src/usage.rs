//! Usage tracking across model buckets.
//!
//! Tracks token usage per model/bucket so routing can prefer secondary
//! buckets and warn when primary is getting drained.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::warn;

/// Usage record for a single model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub request_count: u64,
    /// Tokens from requests that also reported timing. Tracked separately
    /// from `input_tokens`/`output_tokens` so tokens/sec divides only the
    /// tokens we actually have durations for (cloud requests report none).
    pub timed_input_tokens: u64,
    pub timed_output_tokens: u64,
    /// Accumulated prompt-ingest and generation wall time, in milliseconds.
    pub prompt_ms: u64,
    pub gen_ms: u64,
    /// Whether this model was served by a local endpoint. Set from the
    /// routing decision (not guessed from the model name) so locally-served
    /// models whose name happens to contain a cloud substring (e.g.
    /// `qwopus`) are still counted as local.
    pub is_local: bool,
}

impl ModelUsage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// Prompt-ingest throughput in tokens/sec, or None if no timed requests.
    pub fn input_tps(&self) -> Option<f64> {
        (self.prompt_ms > 0)
            .then(|| self.timed_input_tokens as f64 / (self.prompt_ms as f64 / 1000.0))
    }

    /// Generation throughput in tokens/sec, or None if no timed requests.
    pub fn output_tps(&self) -> Option<f64> {
        (self.gen_ms > 0).then(|| self.timed_output_tokens as f64 / (self.gen_ms as f64 / 1000.0))
    }

    /// Fold another record's counters into this one. Used to aggregate
    /// per-model entries into bucket totals.
    fn merge(&mut self, other: &ModelUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.request_count += other.request_count;
        self.timed_input_tokens += other.timed_input_tokens;
        self.timed_output_tokens += other.timed_output_tokens;
        self.prompt_ms += other.prompt_ms;
        self.gen_ms += other.gen_ms;
        self.is_local = self.is_local || other.is_local;
    }
}

/// Throughput snapshot of the most recent timed (local) model call. Drives
/// the live tokens/sec readout in the status line.
#[derive(Debug, Clone, Default)]
pub struct LastCall {
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub prompt_ms: u64,
    pub gen_ms: u64,
}

impl LastCall {
    pub fn input_tps(&self) -> Option<f64> {
        (self.prompt_ms > 0).then(|| self.input_tokens as f64 / (self.prompt_ms as f64 / 1000.0))
    }

    pub fn output_tps(&self) -> Option<f64> {
        (self.gen_ms > 0).then(|| self.output_tokens as f64 / (self.gen_ms as f64 / 1000.0))
    }
}

/// The route chosen for the in-flight request, set at dispatch time so the
/// live status line can show which model class is handling the current turn
/// (before any tokens/sec are known).
#[derive(Debug, Clone, Default)]
pub struct CurrentRoute {
    /// Routing role/bucket, e.g. `light_coder`, `cloud_reasoner`.
    pub role: String,
    /// Concrete model slug, e.g. `qwopus`, `opus-4.6`.
    pub model: String,
}

/// Tracks usage across all models, keyed by model slug.
#[derive(Debug, Default)]
pub struct UsageTracker {
    usage: Mutex<HashMap<String, ModelUsage>>,
    /// Estimated cloud tokens avoided by routing locally.
    /// This is the pre-strip token count — what the cloud model would have received.
    cloud_tokens_saved: Mutex<u64>,
    /// Throughput of the most recent timed (local) call, for the live readout.
    last_call: Mutex<Option<LastCall>>,
    /// Route chosen for the in-flight request, for the live status line.
    current_route: Mutex<Option<CurrentRoute>>,
    warn_threshold: f64,
}

impl UsageTracker {
    pub fn new(warn_threshold: f64) -> Self {
        Self {
            usage: Mutex::new(HashMap::new()),
            cloud_tokens_saved: Mutex::new(0),
            last_call: Mutex::new(None),
            current_route: Mutex::new(None),
            warn_threshold,
        }
    }

    /// Record the route chosen for the request now being dispatched.
    pub fn set_current_route(&self, role: &str, model: &str) {
        *self.current_route.lock().unwrap_or_else(|e| e.into_inner()) = Some(CurrentRoute {
            role: role.to_string(),
            model: model.to_string(),
        });
    }

    /// One-line readout for the live status row: the current route plus the
    /// most recent local call's tokens/sec (which lags the current call by one
    /// round-trip, since rates are only known once a generation completes).
    /// Returns None when no route has been dispatched yet.
    pub fn live_readout(&self) -> Option<String> {
        let route = self
            .current_route
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()?;
        let mut out = format!("{} ({})", route.role, route.model);
        // Only attach a rate if the last timed call was the same model, so we
        // don't show a stale local rate next to a different (e.g. cloud) route.
        if let Some(last) = self.last_call()
            && last.model == route.model
        {
            match (last.input_tps(), last.output_tps()) {
                (Some(i), Some(o)) => out.push_str(&format!(" · {i:.0}/{o:.0} tok/s")),
                (None, Some(o)) => out.push_str(&format!(" · {o:.0} tok/s out")),
                _ => {}
            }
        }
        Some(out)
    }

    /// Record a cloud request's token usage for a model (no timing available).
    pub fn record(&self, model: &str, input_tokens: u64, output_tokens: u64) {
        self.record_timed(
            model,
            input_tokens,
            output_tokens,
            /*prompt_ms*/ 0,
            /*gen_ms*/ 0,
        );
    }

    /// Record a request's token usage and (for local calls) its prompt-ingest
    /// and generation wall times in milliseconds. A request with timing is
    /// treated as locally served and contributes to tokens/sec; a request
    /// with both durations zero is treated as cloud and only bumps counts.
    pub fn record_timed(
        &self,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        prompt_ms: u64,
        gen_ms: u64,
    ) {
        let timed = prompt_ms > 0 || gen_ms > 0;
        let mut usage = self.usage.lock().unwrap_or_else(|e| e.into_inner());
        let entry = usage.entry(model.to_string()).or_default();
        entry.input_tokens += input_tokens;
        entry.output_tokens += output_tokens;
        entry.request_count += 1;
        if timed {
            entry.is_local = true;
            entry.timed_input_tokens += input_tokens;
            entry.timed_output_tokens += output_tokens;
            entry.prompt_ms += prompt_ms;
            entry.gen_ms += gen_ms;
            drop(usage);
            *self.last_call.lock().unwrap_or_else(|e| e.into_inner()) = Some(LastCall {
                model: model.to_string(),
                input_tokens,
                output_tokens,
                prompt_ms,
                gen_ms,
            });
        }
    }

    /// Throughput of the most recent timed (local) call, if any.
    pub fn last_call(&self) -> Option<LastCall> {
        self.last_call
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Record cloud tokens saved by routing a request locally.
    /// `pre_strip_tokens` is the estimated token count of the full conversation
    /// before context stripping — what the cloud model would have received.
    pub fn record_savings(&self, pre_strip_tokens: u64) {
        let mut saved = self
            .cloud_tokens_saved
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *saved += pre_strip_tokens;
    }

    /// Get total estimated cloud tokens saved.
    pub fn cloud_tokens_saved(&self) -> u64 {
        *self
            .cloud_tokens_saved
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Get usage for a specific model.
    pub fn get(&self, model: &str) -> ModelUsage {
        let usage = self.usage.lock().unwrap_or_else(|e| e.into_inner());
        usage.get(model).cloned().unwrap_or_default()
    }

    /// Get usage for all models.
    pub fn all(&self) -> HashMap<String, ModelUsage> {
        let usage = self.usage.lock().unwrap_or_else(|e| e.into_inner());
        usage.clone()
    }

    /// Get total usage across primary cloud bucket models.
    pub fn primary_usage(&self) -> ModelUsage {
        self.bucket_total(|model, u| !u.is_local && is_primary_model(model))
    }

    /// Get total usage across secondary cloud bucket models.
    pub fn secondary_usage(&self) -> ModelUsage {
        self.bucket_total(|model, u| !u.is_local && !is_primary_model(model))
    }

    /// Get total local (free) usage.
    pub fn local_usage(&self) -> ModelUsage {
        self.bucket_total(|_model, u| u.is_local)
    }

    /// Aggregate per-model entries matching `keep` into a single total.
    fn bucket_total(&self, keep: impl Fn(&str, &ModelUsage) -> bool) -> ModelUsage {
        let usage = self.usage.lock().unwrap_or_else(|e| e.into_inner());
        let mut total = ModelUsage::default();
        for (model, u) in usage.iter() {
            if keep(model, u) {
                total.merge(u);
            }
        }
        total
    }

    /// Summary string for logging and /stats display.
    pub fn summary(&self) -> String {
        let local = self.local_usage();
        let secondary = self.secondary_usage();
        let primary = self.primary_usage();
        let saved = self.cloud_tokens_saved();
        let total_req = local.request_count + secondary.request_count + primary.request_count;
        let local_pct = if total_req > 0 {
            (local.request_count as f64 / total_req as f64) * 100.0
        } else {
            0.0
        };
        // tok/s is only meaningful where we have timing (the local bucket).
        let speed = |u: &ModelUsage| match (u.input_tps(), u.output_tps()) {
            (Some(i), Some(o)) => format!("   ({i:.1} tok/s in, {o:.1} tok/s out)"),
            (Some(i), None) => format!("   ({i:.1} tok/s in)"),
            (None, Some(o)) => format!("   ({o:.1} tok/s out)"),
            (None, None) => String::new(),
        };
        let last = match self.last_call() {
            Some(c) => format!(
                "\nLast local call ({}): {} tok/s in, {} tok/s out",
                c.model,
                c.input_tps()
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "—".into()),
                c.output_tps()
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "—".into()),
            ),
            None => String::new(),
        };
        format!(
            "Routing stats this session:\n\
             \n\
             Local (free):  {} requests, {} tokens{}\n\
             Secondary:     {} requests, {} tokens\n\
             Primary:       {} requests, {} tokens\n\
             \n\
             Cloud tokens saved: ~{}\n\
             Local routing rate: {:.0}% of requests{}",
            local.request_count,
            local.total_tokens(),
            speed(&local),
            secondary.request_count,
            secondary.total_tokens(),
            primary.request_count,
            primary.total_tokens(),
            saved,
            local_pct,
            last,
        )
    }

    /// Check if primary usage exceeds the warning threshold.
    /// Returns a warning message if so.
    pub fn check_primary_threshold(&self, estimated_daily_budget: u64) -> Option<String> {
        if estimated_daily_budget == 0 {
            return None;
        }
        let primary = self.primary_usage();
        let ratio = primary.total_tokens() as f64 / estimated_daily_budget as f64;
        if ratio >= self.warn_threshold {
            let msg = format!(
                "Primary bucket usage at {:.0}% of estimated daily budget ({}/{} tokens). Consider routing more to secondary models.",
                ratio * 100.0,
                primary.total_tokens(),
                estimated_daily_budget,
            );
            warn!("{}", msg);
            Some(msg)
        } else {
            None
        }
    }
}

fn is_primary_model(model: &str) -> bool {
    model.contains("gpt-5.4") && !model.contains("mini") && !model.contains("spark")
        || model.contains("opus")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_get() {
        let tracker = UsageTracker::new(0.7);
        tracker.record("gpt-5.4", 1000, 200);
        tracker.record("gpt-5.4", 500, 100);
        let usage = tracker.get("gpt-5.4");
        assert_eq!(usage.input_tokens, 1500);
        assert_eq!(usage.output_tokens, 300);
        assert_eq!(usage.request_count, 2);
    }

    #[test]
    fn test_bucket_classification() {
        let tracker = UsageTracker::new(0.7);
        tracker.record("gpt-5.4", 1000, 200);
        tracker.record("gpt-5.3-codex-spark", 2000, 400);
        // Local is determined by route (timing present), not the model name —
        // even a local model whose name contains a cloud substring.
        tracker.record_timed("qwopus", 3000, 600, 100, 200);

        assert_eq!(tracker.primary_usage().request_count, 1);
        assert_eq!(tracker.secondary_usage().request_count, 1);
        assert_eq!(tracker.local_usage().request_count, 1);
    }

    #[test]
    fn test_tokens_per_second() {
        let tracker = UsageTracker::new(0.7);
        // 3000 input tokens in 1000 ms = 3000 tok/s; 600 output in 2000 ms = 300 tok/s.
        tracker.record_timed("qwopus", 3000, 600, 1000, 2000);
        let local = tracker.local_usage();
        assert_eq!(local.input_tps(), Some(3000.0));
        assert_eq!(local.output_tps(), Some(300.0));

        // Cloud requests carry no timing and must not contribute to tok/s.
        tracker.record("gpt-5.4", 9999, 9999);
        assert_eq!(tracker.primary_usage().input_tps(), None);

        let last = tracker.last_call().expect("a timed call was recorded");
        assert_eq!(last.model, "qwopus");
        assert_eq!(last.output_tps(), Some(300.0));
    }

    #[test]
    fn test_live_readout() {
        let tracker = UsageTracker::new(0.7);
        // Nothing dispatched yet.
        assert_eq!(tracker.live_readout(), None);

        // Route set but no completed call yet: route label only.
        tracker.set_current_route("light_coder", "qwopus");
        assert_eq!(
            tracker.live_readout(),
            Some("light_coder (qwopus)".to_string())
        );

        // After a timed call on the same model, append its rate.
        tracker.record_timed("qwopus", 3000, 600, 1000, 2000);
        assert_eq!(
            tracker.live_readout(),
            Some("light_coder (qwopus) · 3000/300 tok/s".to_string())
        );

        // Switching to a cloud route whose model differs drops the stale rate.
        tracker.set_current_route("cloud_coder", "opus-4.6");
        assert_eq!(
            tracker.live_readout(),
            Some("cloud_coder (opus-4.6)".to_string())
        );
    }

    #[test]
    fn test_threshold_warning() {
        let tracker = UsageTracker::new(0.7);
        tracker.record("gpt-5.4", 8000, 2000);
        // 10000 tokens against budget of 12000 = 83% > 70% threshold
        assert!(tracker.check_primary_threshold(12000).is_some());
        // Against larger budget: 10000/100000 = 10% < 70%
        assert!(tracker.check_primary_threshold(100000).is_none());
    }

    #[test]
    fn test_summary() {
        let tracker = UsageTracker::new(0.7);
        tracker.record_timed("qwopus", 100, 50, 100, 200);
        tracker.record("gpt-5.3-codex-spark", 200, 100);
        tracker.record("gpt-5.4", 300, 150);
        tracker.record_savings(5000);
        let s = tracker.summary();
        assert!(s.contains("1 requests, 150 tokens")); // local
        assert!(s.contains("1 requests, 300 tokens")); // secondary
        assert!(s.contains("1 requests, 450 tokens")); // primary
        assert!(s.contains("Cloud tokens saved: ~5000"));
        assert!(s.contains("33%")); // 1 local out of 3 total
    }
}
