//! The lab computer's clock measured against AtomCloud's reference clock.
//!
//! Lab PCs drift by seconds to minutes, so a stream never sends the host clock as it reads: every
//! timestamp leaves converted to reference UTC, exactly as the gateway does (its `labClock.ts`).
//! The host clock itself is never changed.
//!
//! The offset comes from NTP's four-timestamp exchange over HTTPS (`GET /gateway/time`, public and
//! free of database work so nothing skews it). That measures the round trip in both directions: the
//! offset is the mean of the two one-way differences, so its error is at most half the round trip.
//! Like NTP, the estimate keeps the lowest-delay recent sample, steps when the host clock has
//! jumped, and otherwise slews, so consecutive timestamps never run backwards.
//!
//! One clock is shared by every streamer of an endpoint in the process, so RHEED and tool-state
//! streams are corrected by the same offset. Set `ATOMSCALE_SKIP_CLOCK_SYNC` to send host time
//! unchanged.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use tracing::{debug, info, warn};

/// Environment variable that turns clock correction off.
pub const SKIP_ENV: &str = "ATOMSCALE_SKIP_CLOCK_SYNC";

const BURST_SAMPLES: usize = 4;
const BURST_SPACING: Duration = Duration::from_secs(2);
const SAMPLE_INTERVAL: Duration = Duration::from_secs(5 * 60);
const SAMPLE_WINDOW: usize = 8;
const EXCHANGE_TIMEOUT: Duration = Duration::from_secs(10);
/// ntpd's step threshold is 128 ms; HTTPS round trips are noisier than NTP's.
const STEP_THRESHOLD_MS: f64 = 1_000.0;
/// ntpd's maximum slew rate (500 ppm).
const MAX_SLEW_MS_PER_SECOND: f64 = 0.5;
/// How long `initialize` waits for the first measurement before streaming uncorrected.
const FIRST_ATTEMPT_WAIT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Sample {
    /// Reference UTC minus host clock.
    offset_ms: f64,
    delay_ms: f64,
    measured_at_ms: f64,
}

/// A measurement of the clock offset: reference UTC minus host, and the bound on its error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClockStatus {
    pub offset_ms: f64,
    pub uncertainty_ms: f64,
    pub measured_at_unix_ms: f64,
}

#[derive(Default)]
struct ClockState {
    samples: Vec<Sample>,
    offset_ms: Option<f64>,
    target_ms: f64,
    adjusted_at_ms: f64,
    best: Option<Sample>,
    attempted: bool,
}

impl ClockState {
    fn record(&mut self, sample: Sample, now_ms: f64) {
        if self.best.is_some_and(|best| {
            (sample.offset_ms - best.offset_ms).abs() > STEP_THRESHOLD_MS + sample.delay_ms
        }) {
            // The host clock jumped: earlier samples describe a clock that no longer exists.
            self.samples.clear();
        }
        self.samples.push(sample);
        if self.samples.len() > SAMPLE_WINDOW {
            self.samples.remove(0);
        }
        // Ties go to the newer sample, so a drifting host clock is followed.
        let best = self
            .samples
            .iter()
            .copied()
            .reduce(|best, next| {
                if next.delay_ms <= best.delay_ms {
                    next
                } else {
                    best
                }
            })
            .expect("a sample was just recorded");
        self.best = Some(best);
        // Like NTP's startup burst, the first few samples set the clock outright: slewing away a
        // noisy first estimate would take minutes.
        let current = self.current_offset(now_ms);
        if current.is_none_or(|current| {
            self.samples.len() <= BURST_SAMPLES
                || (best.offset_ms - current).abs() > STEP_THRESHOLD_MS
        }) {
            self.offset_ms = Some(best.offset_ms);
            self.adjusted_at_ms = now_ms;
        }
        self.target_ms = best.offset_ms;
    }

    fn current_offset(&mut self, now_ms: f64) -> Option<f64> {
        let offset = self.offset_ms?;
        let allowed = (now_ms - self.adjusted_at_ms).max(0.0) / 1_000.0 * MAX_SLEW_MS_PER_SECOND;
        let next = offset + (self.target_ms - offset).clamp(-allowed, allowed);
        self.offset_ms = Some(next);
        self.adjusted_at_ms = now_ms;
        Some(next)
    }
}

/// One NTP exchange as a sample, or `None` when the host clock was stepped during it.
fn sample_from_exchange(
    sent_ms: f64,
    received_ms: f64,
    server_received_ms: f64,
    server_transmitted_ms: f64,
) -> Option<Sample> {
    let server_held = server_transmitted_ms - server_received_ms;
    let delay_ms = received_ms - sent_ms - server_held;
    // A negative round trip means the host clock was stepped during the exchange.
    if delay_ms < 0.0 || server_held < 0.0 {
        return None;
    }
    Some(Sample {
        offset_ms: ((server_received_ms - sent_ms) + (server_transmitted_ms - received_ms)) / 2.0,
        delay_ms,
        measured_at_ms: server_transmitted_ms,
    })
}

fn host_now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64() * 1_000.0)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TimeResponse {
    received_at_unix_ms: f64,
    transmitted_at_unix_ms: f64,
}

/// The correction applied to one timestamp: reference UTC minus the host clock, and the bound on
/// its error (half the round trip of the sample it came from). Recorded beside every stamp so a
/// consumer can undo or refine it later.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Correction {
    pub offset_ms: i64,
    pub uncertainty_ms: f64,
}

pub struct LabClock {
    state: Mutex<ClockState>,
    attempted: Condvar,
}

impl LabClock {
    fn new() -> Self {
        Self {
            state: Mutex::new(ClockState::default()),
            attempted: Condvar::new(),
        }
    }

    /// Reference UTC minus host clock, in milliseconds; 0 until the first measurement.
    pub fn offset_ms(&self) -> f64 {
        let mut state = self.state.lock().expect("lab clock mutex poisoned");
        state.current_offset(host_now_ms()).unwrap_or(0.0)
    }

    /// A host-clock time as reference UTC, with the correction that was applied so the caller
    /// can record it beside the stamp. Unchanged, with no correction, until the first
    /// measurement. The offset is applied in whole milliseconds, so `host + offset == utc`.
    pub fn correct_ms(&self, host_ms: i64) -> (i64, Option<Correction>) {
        let mut state = self.state.lock().expect("lab clock mutex poisoned");
        let (Some(offset), Some(best)) = (state.current_offset(host_now_ms()), state.best) else {
            return (host_ms, None);
        };
        let offset_ms = offset.round() as i64;
        let uncertainty_ms = (best.delay_ms / 2.0 * 10.0).round() / 10.0;
        (
            host_ms + offset_ms,
            Some(Correction {
                offset_ms,
                uncertainty_ms,
            }),
        )
    }

    pub fn status(&self) -> Option<ClockStatus> {
        let mut state = self.state.lock().expect("lab clock mutex poisoned");
        let offset_ms = state.current_offset(host_now_ms())?;
        let best = state.best?;
        Some(ClockStatus {
            offset_ms,
            uncertainty_ms: best.delay_ms / 2.0,
            measured_at_unix_ms: best.measured_at_ms,
        })
    }

    /// Block until the first measurement has been attempted, so the first chunk is corrected.
    /// Returns early on failure, and after the wait, so streaming is never held up for long.
    pub fn wait_for_first_attempt(&self) {
        let state = self.state.lock().expect("lab clock mutex poisoned");
        let _ = self
            .attempted
            .wait_timeout_while(state, FIRST_ATTEMPT_WAIT, |state| !state.attempted);
    }

    async fn measure(&self, client: &reqwest::Client, url: &str) -> Result<()> {
        let sent = host_now_ms();
        let response = client
            .get(url)
            .header("Cache-Control", "no-store")
            .send()
            .await?;
        let received = host_now_ms();
        if !response.status().is_success() {
            bail!("reference time request failed ({})", response.status());
        }
        let server: TimeResponse = response
            .json()
            .await
            .context("reference time response was not the expected shape")?;
        let Some(sample) = sample_from_exchange(
            sent,
            received,
            server.received_at_unix_ms,
            server.transmitted_at_unix_ms,
        ) else {
            return Ok(());
        };
        let mut state = self.state.lock().expect("lab clock mutex poisoned");
        state.record(sample, host_now_ms());
        Ok(())
    }

    fn finish_attempt(&self) {
        self.state
            .lock()
            .expect("lab clock mutex poisoned")
            .attempted = true;
        self.attempted.notify_all();
    }

    async fn run(self: Arc<Self>, client: reqwest::Client, url: String) {
        let mut failing = false;
        let mut measured_before = false;
        let mut taken = 0_usize;
        loop {
            match self.measure(&client, &url).await {
                Ok(()) => {
                    failing = false;
                    if !measured_before {
                        if let Some(status) = self.status() {
                            measured_before = true;
                            let (amount, side) = (
                                status.offset_ms.abs(),
                                if status.offset_ms > 0.0 {
                                    "behind"
                                } else {
                                    "ahead of"
                                },
                            );
                            info!(
                                "[lab_clock] this computer's clock is {amount:.0} ms {side} AtomCloud's (±{:.0} ms); streams carry corrected times",
                                status.uncertainty_ms.ceil()
                            );
                        }
                    }
                }
                Err(error) => {
                    // Once per failure streak, and only once the clock has been measured: the
                    // offset already measured keeps correcting streams meanwhile.
                    if !failing && measured_before {
                        warn!("[lab_clock] unable to measure this computer's clock against AtomCloud: {error:#}");
                    } else {
                        debug!("[lab_clock] measurement failed: {error:#}");
                    }
                    failing = true;
                }
            }
            self.finish_attempt();
            taken += 1;
            tokio::time::sleep(if taken < BURST_SAMPLES {
                BURST_SPACING
            } else {
                SAMPLE_INTERVAL
            })
            .await;
        }
    }
}

/// `LabClock::status` as the dict the streamers return from `clock_status()`.
pub fn clock_status_dict<'py>(
    py: pyo3::Python<'py>,
    clock: Option<&LabClock>,
) -> pyo3::PyResult<Option<pyo3::Bound<'py, pyo3::types::PyDict>>> {
    let Some(status) = clock.and_then(LabClock::status) else {
        return Ok(None);
    };
    use pyo3::types::PyDictMethods;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("offset_ms", status.offset_ms)?;
    dict.set_item("uncertainty_ms", status.uncertainty_ms)?;
    dict.set_item("measured_at_unix_ms", status.measured_at_unix_ms)?;
    Ok(Some(dict))
}

type Registry = Mutex<HashMap<String, Arc<LabClock>>>;

/// The process-wide clock for `endpoint`, started on first use; `None` when correction is off.
pub fn lab_clock_for(endpoint: &str) -> Option<Arc<LabClock>> {
    if std::env::var_os(SKIP_ENV).is_some_and(|value| !value.is_empty() && value != "0") {
        return None;
    }
    static CLOCKS: OnceLock<Registry> = OnceLock::new();
    let mut clocks = CLOCKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("lab clock registry poisoned");
    if let Some(clock) = clocks.get(endpoint) {
        return Some(clock.clone());
    }
    let clock = Arc::new(LabClock::new());
    let url = format!("{}/gateway/time", endpoint.trim_end_matches('/'));
    let started = {
        let clock = clock.clone();
        // Its own thread and runtime: the clock outlives any one streamer's runtime.
        std::thread::Builder::new()
            .name("atomscale-lab-clock".into())
            .spawn(move || {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                let Ok(client) = reqwest::Client::builder().timeout(EXCHANGE_TIMEOUT).build()
                else {
                    return;
                };
                runtime.block_on(clock.run(client, url));
            })
    };
    if started.is_err() {
        return None;
    }
    clocks.insert(endpoint.to_string(), clock.clone());
    Some(clock)
}

#[cfg(test)]
impl LabClock {
    /// A clock part way through slewing from `offset_ms` toward `target_ms`, for streamer tests.
    pub(crate) fn slewing(offset_ms: f64, target_ms: f64) -> Self {
        let clock = Self::new();
        {
            let mut state = clock.state.lock().expect("lab clock mutex poisoned");
            state.offset_ms = Some(offset_ms);
            state.target_ms = target_ms;
            state.adjusted_at_ms = host_now_ms();
            state.attempted = true;
        }
        clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host 5 s behind the reference, with `up`/`down` one-way delays and 1 ms server hold.
    fn exchange(offset_ms: f64, up: f64, down: f64) -> Sample {
        let sent = 1_000_000.0;
        let server_received = sent + up + offset_ms;
        let server_transmitted = server_received + 1.0;
        let received = sent + up + 1.0 + down;
        sample_from_exchange(sent, received, server_received, server_transmitted).unwrap()
    }

    #[test]
    fn a_symmetric_round_trip_recovers_the_offset_exactly() {
        let sample = exchange(5_000.0, 20.0, 20.0);
        assert!((sample.offset_ms - 5_000.0).abs() < 1e-6, "{sample:?}");
        assert!((sample.delay_ms - 40.0).abs() < 1e-6);
    }

    #[test]
    fn an_asymmetric_path_errs_by_at_most_half_the_round_trip() {
        let sample = exchange(5_000.0, 10.0, 70.0);
        assert!((sample.offset_ms - 5_000.0).abs() <= sample.delay_ms / 2.0 + 1e-6);
    }

    #[test]
    fn a_stepped_host_clock_during_the_exchange_is_discarded() {
        assert_eq!(sample_from_exchange(100.0, 90.0, 0.0, 1.0), None);
        assert_eq!(sample_from_exchange(100.0, 200.0, 50.0, 40.0), None);
    }

    #[test]
    fn the_lowest_delay_recent_sample_wins_and_the_burst_steps_outright() {
        let mut state = ClockState::default();
        state.record(
            Sample {
                offset_ms: 4_900.0,
                delay_ms: 400.0,
                measured_at_ms: 0.0,
            },
            0.0,
        );
        assert_eq!(
            state.current_offset(0.0),
            Some(4_900.0),
            "the first sample sets the clock"
        );
        state.record(
            Sample {
                offset_ms: 5_000.0,
                delay_ms: 20.0,
                measured_at_ms: 1.0,
            },
            1.0,
        );
        state.record(
            Sample {
                offset_ms: 5_100.0,
                delay_ms: 300.0,
                measured_at_ms: 2.0,
            },
            2.0,
        );
        assert_eq!(state.best.unwrap().offset_ms, 5_000.0);
        assert_eq!(
            state.current_offset(2.0),
            Some(5_000.0),
            "burst samples step, not slew"
        );
    }

    #[test]
    fn after_the_burst_a_small_change_slews_and_never_runs_backwards() {
        let mut state = ClockState::default();
        for index in 0..=BURST_SAMPLES {
            state.record(
                Sample {
                    offset_ms: 5_000.0,
                    delay_ms: 20.0 + index as f64,
                    measured_at_ms: 0.0,
                },
                0.0,
            );
        }
        state.record(
            Sample {
                offset_ms: 5_050.0,
                delay_ms: 5.0,
                measured_at_ms: 1.0,
            },
            0.0,
        );
        // 10 s at 0.5 ms/s moves 5 ms toward the 50 ms difference, not all of it.
        let after_ten_seconds = state.current_offset(10_000.0).unwrap();
        assert!(
            (after_ten_seconds - 5_005.0).abs() < 1e-6,
            "{after_ten_seconds}"
        );
        // A reading earlier than the last adjustment cannot move the offset backwards.
        assert_eq!(state.current_offset(5_000.0), Some(after_ten_seconds));
    }

    #[test]
    fn a_host_clock_jump_steps_and_drops_stale_samples() {
        let mut state = ClockState::default();
        for _ in 0..=BURST_SAMPLES {
            state.record(
                Sample {
                    offset_ms: 5_000.0,
                    delay_ms: 20.0,
                    measured_at_ms: 0.0,
                },
                0.0,
            );
        }
        state.record(
            Sample {
                offset_ms: 65_000.0,
                delay_ms: 25.0,
                measured_at_ms: 1.0,
            },
            1.0,
        );
        assert_eq!(state.samples.len(), 1);
        assert_eq!(state.current_offset(1.0), Some(65_000.0));
    }

    #[test]
    fn a_corrected_stamp_reports_the_correction_that_produced_it() {
        let clock = LabClock::new();
        assert_eq!(
            clock.correct_ms(1_000),
            (1_000, None),
            "unmeasured passes through"
        );
        clock.state.lock().unwrap().record(
            Sample {
                offset_ms: 5_000.4,
                delay_ms: 20.0,
                measured_at_ms: 0.0,
            },
            host_now_ms(),
        );
        let (utc, correction) = clock.correct_ms(1_000);
        let correction = correction.expect("measured");
        assert_eq!(correction.offset_ms, 5_000);
        assert_eq!(correction.uncertainty_ms, 10.0);
        assert_eq!(
            utc,
            1_000 + correction.offset_ms,
            "host + offset == stamp, exactly"
        );
    }
}
