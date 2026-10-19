//! Retries for stream uploads.
//!
//! Stream ingest is idempotent per chunk (a RHEED shard lands at a fixed key; a tool-state
//! chunk rewrites a fixed slice), so repeating a failed upload never duplicates data. Only
//! failures that can clear on their own are retried: transport errors, timeouts, 5xx, 429, 408
//! and S3's 400 `RequestTimeout`.

use std::fmt;
use std::future::Future;
use std::time::Duration;

use anyhow::{Error, Result};
use reqwest::StatusCode;
use tracing::{debug, warn};

pub const FIRST_RETRY_DELAY: Duration = Duration::from_millis(500);
/// Backoff stops doubling here, so a long retry budget polls steadily instead of idling for minutes.
pub const MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

/// A non-2xx HTTP response, kept typed so the retry loop can tell transient from final.
#[derive(Debug)]
pub struct HttpStatusError {
    pub status: StatusCode,
    pub body: String,
}

impl fmt::Display for HttpStatusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HTTP {}: {}", self.status, self.body)
    }
}

impl std::error::Error for HttpStatusError {}

impl HttpStatusError {
    /// A status the same request can clear on a later attempt: 5xx, 429, 408, and S3's 400
    /// `RequestTimeout`, which it sends when an upload's body stalls for about 20 s.
    fn is_transient(&self) -> bool {
        self.status.is_server_error()
            || self.status == StatusCode::TOO_MANY_REQUESTS
            || self.status == StatusCode::REQUEST_TIMEOUT
            || (self.status == StatusCode::BAD_REQUEST && self.body.contains("RequestTimeout"))
    }
}

/// True when the server definitively refused the request: a 4xx that is not transient.
pub fn is_rejection(error: &Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<HttpStatusError>()
            .is_some_and(|http| http.status.is_client_error() && !http.is_transient())
    })
}

fn is_transient(error: &Error) -> bool {
    for cause in error.chain() {
        if let Some(http) = cause.downcast_ref::<HttpStatusError>() {
            return http.is_transient();
        }
        if let Some(req) = cause.downcast_ref::<reqwest::Error>() {
            return req.status().map_or(true, |s| {
                s.is_server_error() || s == StatusCode::TOO_MANY_REQUESTS
            });
        }
    }
    false
}

/// Run `op` up to `attempts` times with exponential backoff from 0.5 s (capped at 30 s), retrying transient
/// failures only. A failure that is final, or the last attempt's, is logged at warn and
/// returned.
pub async fn with_retries<T, F, Fut>(what: &str, attempts: u32, mut op: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut delay = FIRST_RETRY_DELAY;
    for attempt in 1..=attempts {
        match op().await {
            Ok(value) => return Ok(value),
            Err(error) if attempt < attempts && is_transient(&error) => {
                debug!("{what} failed (attempt {attempt}), retrying in {delay:?}: {error:#}");
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(MAX_RETRY_DELAY);
            }
            Err(error) => {
                warn!("{what} failed after {attempt} attempt(s): {error:#}");
                return Err(error);
            }
        }
    }
    unreachable!("the final attempt always returns")
}
