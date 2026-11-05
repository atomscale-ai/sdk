use anyhow::Result;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;
use tracing::debug;

use crate::clock::{clock_status_dict, lab_clock_for, ClockStatus, LabClock};
use crate::initialize::{link_new_stream, resolve_stream_links};
use crate::retry::{with_retries, HttpStatusError};
use crate::utils::init_tracing_once;
use crate::warn_py;

/// Request body for stream initialization.
#[derive(Serialize, Debug)]
struct InitializeRequest {
    stream_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    synth_source_id: Option<i64>,
    points_per_chunk: usize,
    project_id: Option<String>,
}

/// Response from stream initialization.
#[derive(Deserialize, Debug)]
struct InitializeResponse {
    data_id: String,
    processed_data_id: String,
}

/// Payload for a single time series chunk.
#[derive(Serialize, Debug)]
struct ChunkPayload {
    data_id: String,
    chunk_index: usize,
    channel_name: String,
    timestamps: Vec<f64>,
    values: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    units: Option<String>,
    /// The conversion applied to `timestamps`; `null` when they are the host clock as read.
    clock: Option<ClockStatus>,
}

/// One channel's data inside a multi-channel chunk.
#[derive(Serialize, Debug)]
struct ChannelPayload {
    timestamps: Vec<f64>,
    values: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    units: Option<String>,
}

/// Payload for every channel of one chunk, sent as a single request.
#[derive(Serialize, Debug)]
struct MultiChunkPayload {
    data_id: String,
    chunk_index: usize,
    channels: BTreeMap<String, ChannelPayload>,
    /// The conversion applied to every channel's timestamps; `null` when they are as read.
    clock: Option<ClockStatus>,
}

/// TimeseriesStreamer(api_key: str, endpoint: Optional[str] = None, points_per_chunk: int = 100)
///
/// A high-performance Python interface (via PyO3) for **real-time instrument time series
/// streaming** into the Atomscale platform. This class takes **chunks of scalar time series data**
/// and uploads them asynchronously for storage and analysis.
///
/// Typical usage:
///
/// 1) **Instantiate** the streamer with API key and optional endpoint
/// 2) **initialize(...)** to create the remote data item and get data_id
/// 3) **push(...)** repeatedly to send data chunks (fire-and-forget, non-blocking)
///
/// Notes
/// -----
/// - Each push() spawns an async upload task and returns immediately; flush() waits for them.
/// - Chunks are ordered server-side by chunk_index, so out-of-order arrival is handled.
/// - Multiple channels can be streamed for the same data_id.
///
/// Args:
///     api_key (str): Your Atomscale API key.
///     endpoint (Optional[str]): Base API URL. Defaults to `"https://api.atomscale.ai"`.
///     points_per_chunk (int): Expected number of points per chunk. Defaults to 100.
///
/// Raises:
///     RuntimeError: If the HTTP client or async runtime cannot be constructed.
#[pyclass]
pub struct TimeseriesStreamer {
    api_key: String,
    endpoint: String,
    client: Client,
    rt: Runtime,
    points_per_chunk: usize,
    /// This computer's clock against AtomCloud's; every timestamp leaves as reference UTC.
    /// `None` when correction is switched off.
    clock: Option<Arc<LabClock>>,
    uploads: Arc<UploadTracker>,
}

#[pymethods]
impl TimeseriesStreamer {
    #[new]
    #[pyo3(signature = (api_key, endpoint=None, points_per_chunk=100, verbosity=None))]
    #[pyo3(text_signature = "(api_key, endpoint=None, points_per_chunk=100, verbosity=None)")]
    fn new(
        api_key: String,
        endpoint: Option<String>,
        points_per_chunk: usize,
        verbosity: Option<u8>,
    ) -> PyResult<Self> {
        let verbosity = verbosity.unwrap_or(0);
        init_tracing_once(verbosity);

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let endpoint = endpoint.unwrap_or_else(|| "https://api.atomscale.ai".to_string());

        debug!("[timeseries_stream] init: base_url={}", endpoint);

        let clock = lab_clock_for(&endpoint);

        Ok(Self {
            api_key,
            endpoint,
            client,
            rt,
            points_per_chunk,
            clock,
            uploads: Arc::default(),
        })
    }

    /// initialize(self, stream_name: Optional[str] = None, synth_source_id: Optional[int] = None, ...) -> str
    ///
    /// Initialize a new time series stream on the server.
    ///
    /// Creates data_catalogue and processed_tool_state_catalogue entries.
    /// Returns data_id to use for subsequent push() calls.
    ///
    /// Args:
    ///     stream_name (Optional[str]): Human-readable name for the stream.
    ///     synth_source_id (Optional[int]): Growth instrument ID to link. Must belong to your
    ///         organization. Use list_instruments() to see available instruments.
    ///     physical_sample (Optional[str]): Name or UUID of a physical sample to associate with the
    ///         data item. If a UUID is provided, it must match an existing sample. If a name is
    ///         provided, it is matched case-insensitively against existing samples, or a new sample
    ///         is created if no match is found.
    ///     project_id (Optional[str]): UUID of a project to associate with the stream. When
    ///         provided along with `physical_sample`, the project's `tracking_physical_sample_id`
    ///         configuration is automatically updated to link the physical sample to the project
    ///         for growth monitoring.
    ///     tags (Optional[List[str]]): List of tag names or UUIDs to attach to the data item. Names
    ///         are matched case-insensitively against existing org tags; unknown names are created.
    ///         UUIDs must reference existing tags.
    ///
    /// Returns:
    ///     str: The data_id for this stream.
    ///
    /// Raises:
    ///     RuntimeError: If the stream cannot be created or linked, or the instrument is not
    ///         found. The physical sample and tags are resolved before the stream is created, so
    ///         a bad one creates nothing. If linking fails after creation, the stream is
    ///         finalized and the error names its `data_id`.
    #[pyo3(signature = (stream_name=None, synth_source_id=None, physical_sample=None, project_id=None, tags=None))]
    #[pyo3(
        text_signature = "(stream_name=None, synth_source_id=None, physical_sample=None, project_id=None, tags=None)"
    )]
    fn initialize(
        &self,
        py: Python<'_>,
        stream_name: Option<String>,
        synth_source_id: Option<i64>,
        physical_sample: Option<String>,
        project_id: Option<String>,
        tags: Option<Vec<String>>,
    ) -> PyResult<String> {
        // Measure the lab clock before the first point is stamped; a failure streams uncorrected.
        if let Some(clock) = &self.clock {
            py.detach(|| clock.wait_for_first_attempt());
        }

        let physical_sample = physical_sample
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let project_id = project_id
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        // Every round trip below runs with the GIL released: the caller's other Python threads
        // (a camera loop, a feed reader) keep running while it waits on the network.
        let links = py
            .detach(|| {
                self.rt.block_on(resolve_stream_links(
                    &self.client,
                    &self.endpoint,
                    &self.api_key,
                    physical_sample.as_deref(),
                    project_id.clone(),
                    tags.as_deref().unwrap_or_default(),
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

        let request = InitializeRequest {
            stream_name,
            synth_source_id,
            points_per_chunk: self.points_per_chunk,
            project_id,
        };

        let url = format!("{}/tool-state/stream/initialize", self.endpoint);

        let result = py.detach(|| {
            self.rt.block_on(async {
                let resp = self
                    .client
                    .post(&url)
                    .header("X-API-KEY", &self.api_key)
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .send()
                    .await?;

                let status = resp.status();
                if !status.is_success() {
                    let text = resp.text().await.unwrap_or_default();
                    return Err(anyhow::anyhow!("HTTP {}: {}", status, text));
                }

                let response: InitializeResponse = resp.json().await?;
                Ok(response)
            })
        });

        let data_id = match result {
            Ok(response) => {
                debug!(
                    "[timeseries_stream] initialized: data_id={}",
                    response.data_id
                );
                response.data_id
            }
            Err(e) => return Err(PyRuntimeError::new_err(e.to_string())),
        };

        let end_url = self.finalize_url(&data_id);
        py.detach(|| {
            self.rt.block_on(link_new_stream(
                &self.client,
                &self.endpoint,
                &self.api_key,
                &data_id,
                &links,
                &end_url,
            ))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

        Ok(data_id)
    }

    /// clock_status(self) -> Optional[dict]
    ///
    /// This computer's clock measured against AtomCloud's reference clock, or `None` when
    /// correction is off (`ATOMSCALE_SKIP_CLOCK_SYNC`) or no measurement has succeeded yet.
    /// Pushed timestamps are read as this computer's clock and sent as reference UTC:
    /// `offset_ms` (reference minus host) is the correction in effect.
    fn clock_status<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        clock_status_dict(py, self.clock.as_deref())
    }

    /// push(self, data_id: str, chunk_index: int, channel_name: str, timestamps: list[float], values: list[float], ...) -> None
    ///
    /// Push a single chunk of time series data for a channel. This method is **fire-and-forget**:
    /// it spawns an async upload task and returns immediately.
    ///
    /// Args:
    ///     data_id (str): The stream identifier returned by `initialize()`.
    ///     chunk_index (int): Zero-based index of this chunk for ordering.
    ///     channel_name (str): Name of the data channel (e.g., "temperature", "pressure").
    ///     timestamps (list[float]): Unix epoch timestamps in seconds.
    ///     values (list[float]): Measured values corresponding to timestamps.
    ///     units (Optional[str]): Units for the values.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If timestamps/values have different lengths.
    #[pyo3(signature = (data_id, chunk_index, channel_name, timestamps, values, units=None))]
    #[pyo3(text_signature = "(data_id, chunk_index, channel_name, timestamps, values, units=None)")]
    fn push(
        &self,
        data_id: String,
        chunk_index: usize,
        channel_name: String,
        timestamps: Vec<f64>,
        values: Vec<f64>,
        units: Option<String>,
    ) -> PyResult<()> {
        if timestamps.len() != values.len() {
            return Err(PyRuntimeError::new_err(
                "timestamps and values must have the same length",
            ));
        }

        let (timestamps, clock) = self.reference_utc_secs(timestamps);
        let payload = ChunkPayload {
            data_id,
            chunk_index,
            channel_name,
            timestamps,
            values,
            units,
            clock,
        };

        // Dropping the handle leaves the upload running; flush() and finalize() still wait for it.
        drop(self.spawn_chunk_upload(payload));
        Ok(())
    }

    /// push_multi(self, data_id: str, chunk_index: int, channels: dict[str, dict]) -> None
    ///
    /// Push data for multiple channels at once. All channels of the chunk are uploaded together
    /// in one async request, so the server writes the shared time axis once per chunk.
    ///
    /// Args:
    ///     data_id (str): The stream identifier returned by `initialize()`.
    ///     chunk_index (int): Zero-based index of this chunk.
    ///     channels (dict[str, dict]): Mapping of channel_name to channel data dict.
    ///         Each channel dict should have:
    ///         - "timestamps" (list[float]): Unix epoch timestamps in seconds
    ///         - "values" (list[float]): Measured values
    ///         - "units" (str, optional): Units for the values
    ///
    /// Example:
    ///     streamer.push_multi(data_id, 0, {
    ///         "temperature": {"timestamps": [0.0, 0.1], "values": [25.0, 25.1], "units": "C"},
    ///         "pressure": {"timestamps": [0.0, 0.1], "values": [1.0, 1.1]},
    ///     })
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If any channel has mismatched lengths.
    #[pyo3(signature = (data_id, chunk_index, channels))]
    #[pyo3(text_signature = "(data_id, chunk_index, channels)")]
    fn push_multi(
        &self,
        data_id: String,
        chunk_index: usize,
        channels: Bound<PyDict>,
    ) -> PyResult<()> {
        let mut host_channels = Vec::with_capacity(channels.len());

        for (key, value) in channels.iter() {
            let channel_name: String = key.extract().map_err(|e| {
                PyRuntimeError::new_err(format!("channel name must be a string: {}", e))
            })?;

            let channel_dict = value.cast::<PyDict>().map_err(|_| {
                PyRuntimeError::new_err(format!(
                    "channel '{}': expected a dict with 'timestamps', 'values', and optional 'units'",
                    channel_name
                ))
            })?;

            let timestamps: Vec<f64> = channel_dict
                .get_item("timestamps")?
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "channel '{}': missing 'timestamps' key",
                        channel_name
                    ))
                })?
                .extract()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "channel '{}': 'timestamps' must be a list of floats: {}",
                        channel_name, e
                    ))
                })?;

            let values: Vec<f64> = channel_dict
                .get_item("values")?
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "channel '{}': missing 'values' key",
                        channel_name
                    ))
                })?
                .extract()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "channel '{}': 'values' must be a list of floats: {}",
                        channel_name, e
                    ))
                })?;

            let units: Option<String> = match channel_dict.get_item("units")? {
                Some(u) => Some(u.extract().map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "channel '{}': 'units' must be a string: {}",
                        channel_name, e
                    ))
                })?),
                None => None,
            };

            if timestamps.len() != values.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "channel '{}': timestamps and values must have the same length",
                    channel_name
                )));
            }
            // The server rejects an empty chunk; sent alone it failed only its own channel,
            // so leave it out rather than fail the whole batch.
            if timestamps.is_empty() {
                continue;
            }

            host_channels.push((
                channel_name,
                ChannelPayload {
                    timestamps,
                    values,
                    units,
                },
            ));
        }

        if host_channels.is_empty() {
            return Ok(());
        }
        let (channels, clock) = channels_to_reference_utc(self.clock.as_deref(), host_channels);
        drop(self.spawn_multi_upload(MultiChunkPayload {
            data_id,
            chunk_index,
            channels,
            clock,
        }));
        Ok(())
    }

    /// run(self, data_id: str, channel_name: str, data_iter: Iterable[tuple[list[float], list[float]]], units: Optional[str] = None) -> None
    ///
    /// **Iterator mode.** Stream time series data by iterating over chunks. The chunk_index is
    /// automatically assigned based on iteration order.
    ///
    /// Each yielded item should be a tuple of `(timestamps, values)` where both are lists of floats.
    ///
    /// This method **blocks until all upload tasks complete**.
    ///
    /// Args:
    ///     data_id (str): The stream identifier returned by `initialize()`.
    ///     channel_name (str): Name of the data channel (e.g., "temperature", "pressure").
    ///     data_iter (Iterable[tuple[list[float], list[float]]]): Iterator yielding (timestamps, values) tuples.
    ///     units (Optional[str]): Units for the values.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If any chunk has mismatched lengths.
    #[pyo3(signature = (data_id, channel_name, data_iter, units=None))]
    #[pyo3(text_signature = "(data_id, channel_name, data_iter, units=None)")]
    fn run(
        &self,
        data_id: String,
        channel_name: String,
        data_iter: Bound<PyAny>,
        units: Option<String>,
    ) -> PyResult<()> {
        self.run_internal(data_id, channel_name, data_iter, units)
    }

    /// flush(self, data_id: str, timeout_s: Optional[float] = None) -> dict
    ///
    /// Wait until every chunk pushed so far for `data_id` has finished uploading: landed, or
    /// failed after its retries. push() and push_multi() return before their upload is done;
    /// call this to know the data arrived.
    ///
    /// Args:
    ///     data_id (str): The stream identifier returned by `initialize()`.
    ///     timeout_s (Optional[float]): How long to wait, in seconds. Defaults to 120, longer
    ///         than one upload's full retry budget.
    ///
    /// Returns:
    ///     dict: What became of the chunks pushed since the last flush() or finalize():
    ///     `chunks_uploaded`, `chunks_failed`, `chunks_pending` (still uploading when the wait
    ///     ended; they are reported by the next flush), `drained` (whether the wait finished in
    ///     time) and `lost`, a list of `(chunk_index, reason)` for the chunks that failed.
    ///
    /// Raises:
    ///     ValueError: If `timeout_s` is negative.
    #[pyo3(signature = (data_id, timeout_s=None))]
    #[pyo3(text_signature = "(data_id, timeout_s=None)")]
    fn flush<'py>(
        &self,
        py: Python<'py>,
        data_id: String,
        timeout_s: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let timeout = wait_limit(timeout_s, "timeout_s")?;
        let report = py.detach(|| self.uploads.drain(&data_id, timeout));
        report.to_dict(py)
    }

    /// finalize(self, data_id: str, drain_timeout_s: Optional[float] = None) -> dict
    ///
    /// Finalize the stream, marking it as complete on the server.
    ///
    /// Call this after all data has been pushed to signal that the stream is finished. It first
    /// waits for the stream's uploads still in flight (as flush() does): the server refuses a
    /// chunk that arrives after its stream is finalized. If any chunk was not uploaded, one
    /// `RuntimeWarning` says how many, once the stream is finalized.
    ///
    /// Args:
    ///     data_id (str): The stream identifier returned by `initialize()`.
    ///     drain_timeout_s (Optional[float]): How long to wait for uploads in flight, in
    ///         seconds. Defaults to 120.
    ///
    /// Returns:
    ///     dict: The same report as flush().
    ///
    /// Raises:
    ///     ValueError: If `drain_timeout_s` is negative.
    ///     RuntimeError: If the request fails. The report is kept for the next flush() or
    ///         finalize().
    #[pyo3(signature = (data_id, drain_timeout_s=None))]
    #[pyo3(text_signature = "(data_id, drain_timeout_s=None)")]
    fn finalize<'py>(
        &self,
        py: Python<'py>,
        data_id: String,
        drain_timeout_s: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let timeout = wait_limit(drain_timeout_s, "drain_timeout_s")?;
        let report = py.detach(|| self.uploads.drain(&data_id, timeout));

        let url = self.finalize_url(&data_id);
        let result = py.detach(|| {
            self.rt.block_on(async {
                let resp = self
                    .client
                    .post(&url)
                    .header("X-API-KEY", &self.api_key)
                    .send()
                    .await?;

                let status = resp.status();
                if !status.is_success() {
                    let text = resp.text().await.unwrap_or_default();
                    return Err(anyhow::anyhow!("HTTP {}: {}", status, text));
                }

                Ok(())
            })
        });

        if let Err(e) = result {
            self.uploads.restore(&data_id, report);
            return Err(PyRuntimeError::new_err(e.to_string()));
        }
        debug!("[timeseries_stream] finalized: data_id={}", data_id);

        // Warn only once the stream is finalized: with warnings turned into errors, a warning
        // raised first would leave the stream open.
        let not_uploaded = report.lost.len() + report.pending;
        if not_uploaded > 0 {
            warn_py(
                py,
                &format!(
                    "{not_uploaded} chunk(s) of stream {data_id} were not uploaded: {}",
                    report.describe_losses()
                ),
            )?;
        }
        report.to_dict(py)
    }
}

impl TimeseriesStreamer {
    fn finalize_url(&self, data_id: &str) -> String {
        format!("{}/tool-state/stream/{data_id}/finalize", self.endpoint)
    }

    /// Host-clock Unix seconds as reference UTC, with the conversion applied (unchanged, with
    /// none, when correction is off or not yet measured).
    fn reference_utc_secs(&self, timestamps: Vec<f64>) -> (Vec<f64>, Option<ClockStatus>) {
        let clock = self.clock.as_deref().and_then(LabClock::status);
        (to_reference_utc(timestamps, clock), clock)
    }

    /// Internal implementation of run() that handles iteration and task management.
    fn run_internal(
        &self,
        data_id: String,
        channel_name: String,
        data_iter: Bound<PyAny>,
        units: Option<String>,
    ) -> PyResult<()> {
        debug!(
            "[timeseries_stream] run: starting for channel '{}'",
            channel_name
        );

        let py = data_iter.py();
        let iter = PyIterator::from_object(&data_iter)?;
        let mut handles = Vec::new();

        let t_total0 = Instant::now();

        for (chunk_index, item) in iter.enumerate() {
            let t0 = Instant::now();
            let obj = item?;

            // Extract (timestamps, values) tuple from Python
            let (timestamps, values): (Vec<f64>, Vec<f64>) = obj.extract().map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "chunk {}: expected (timestamps, values) tuple of (list[float], list[float]): {}",
                    chunk_index, e
                ))
            })?;

            if timestamps.len() != values.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "chunk {}: timestamps and values must have the same length",
                    chunk_index
                )));
            }

            debug!(
                "[timeseries_stream] chunk#{}: prepared in {:.2?} → {} points; spawning task…",
                chunk_index,
                t0.elapsed(),
                timestamps.len()
            );

            let (timestamps, clock) = self.reference_utc_secs(timestamps);
            let payload = ChunkPayload {
                data_id: data_id.clone(),
                chunk_index,
                channel_name: channel_name.clone(),
                timestamps,
                values,
                units: units.clone(),
                clock,
            };

            let handle = self.spawn_chunk_upload(payload);
            handles.push(handle);
        }

        debug!(
            "[timeseries_stream] run: spawned {} upload task(s); awaiting…",
            handles.len()
        );

        // Wait for all tasks to finish, with the GIL released while the uploads run.
        let res = py.detach(|| {
            self.rt.block_on(async {
                for (i, h) in handles.into_iter().enumerate() {
                    match h.await {
                        Ok(Ok(())) => debug!("[timeseries_stream] task#{}: completed", i),
                        Ok(Err(e)) => {
                            debug!("[timeseries_stream] task#{}: ERROR: {}", i, e);
                            return Err(e);
                        }
                        Err(e) => {
                            debug!("[timeseries_stream] task#{}: JOIN ERROR: {}", i, e);
                            return Err(anyhow::anyhow!(e));
                        }
                    }
                }
                Ok::<(), anyhow::Error>(())
            })
        });

        if let Err(e) = res {
            return Err(PyRuntimeError::new_err(e.to_string()));
        }

        let total_dur = t_total0.elapsed();
        debug!(
            "[timeseries_stream] run: all tasks done in {:.2?}",
            total_dur
        );
        Ok(())
    }

    /// Spawn the upload of one channel's chunk and return its JoinHandle. flush() and finalize()
    /// wait for it whether or not the handle is kept.
    fn spawn_chunk_upload(&self, payload: ChunkPayload) -> JoinHandle<Result<()>> {
        let ticket = self.uploads.start(
            &payload.data_id,
            payload.chunk_index,
            Some(payload.channel_name.clone()),
        );
        self.spawn_upload("chunk", ticket, payload)
    }

    /// Spawn the upload of every channel of one chunk, in one request.
    fn spawn_multi_upload(&self, payload: MultiChunkPayload) -> JoinHandle<Result<()>> {
        debug!(
            "[timeseries_stream] spawn: chunk {} for {} channel(s)",
            payload.chunk_index,
            payload.channels.len()
        );
        let ticket = self
            .uploads
            .start(&payload.data_id, payload.chunk_index, None);
        self.spawn_upload("chunk/multi", ticket, payload)
    }

    fn spawn_upload<T: Serialize + Send + Sync + 'static>(
        &self,
        route: &str,
        ticket: UploadTicket,
        payload: T,
    ) -> JoinHandle<Result<()>> {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let url = format!("{}/tool-state/stream/{route}", self.endpoint);

        self.rt.spawn(async move {
            debug!("[timeseries_stream] uploading {}", ticket.label());
            let result = post_chunk(&client, &url, &payload, &api_key).await;
            match &result {
                Ok(()) => debug!(
                    "[timeseries_stream] {} uploaded successfully",
                    ticket.label()
                ),
                Err(e) => debug!(
                    "[timeseries_stream] {} upload failed: {}",
                    ticket.label(),
                    e
                ),
            }
            ticket.finish(&result);
            result
        })
    }
}

/// Host-clock Unix seconds as reference UTC; unchanged without a measurement.
fn to_reference_utc(mut timestamps: Vec<f64>, clock: Option<ClockStatus>) -> Vec<f64> {
    if let Some(clock) = clock {
        let offset_s = clock.offset_ms / 1_000.0;
        timestamps.iter_mut().for_each(|t| *t += offset_s);
    }
    timestamps
}

/// The channels of one chunk moved onto reference UTC with a single reading of the clock, which
/// is returned as the record of the conversion applied. While the offset slews it changes between
/// reads, so channels sent with the same timestamps would otherwise land a float step apart, and
/// the server files those as different rows.
fn channels_to_reference_utc(
    clock: Option<&LabClock>,
    channels: Vec<(String, ChannelPayload)>,
) -> (BTreeMap<String, ChannelPayload>, Option<ClockStatus>) {
    let reading = clock.and_then(LabClock::status);
    let channels = channels
        .into_iter()
        .map(|(name, mut channel)| {
            channel.timestamps = to_reference_utc(channel.timestamps, reading);
            (name, channel)
        })
        .collect();
    (channels, reading)
}

/// How long flush() and finalize() wait for uploads by default: longer than one upload's full
/// retry budget (3 attempts of up to 30 s each, plus backoff).
const DEFAULT_FLUSH_TIMEOUT: Duration = Duration::from_secs(120);

fn wait_limit(seconds: Option<f64>, name: &str) -> PyResult<Duration> {
    seconds
        .map(|seconds| {
            Duration::try_from_secs_f64(seconds)
                .map_err(|_| PyValueError::new_err(format!("{name} must be non-negative")))
        })
        .transpose()
        .map(|limit| limit.unwrap_or(DEFAULT_FLUSH_TIMEOUT))
}

/// What became of one stream's uploads since it was last flushed.
#[derive(Default)]
struct StreamUploads {
    in_flight: usize,
    uploaded: usize,
    lost: Vec<(usize, String)>,
}

/// The uploads push(), push_multi() and run() start, per stream. Uploads are fire-and-forget, so
/// without this a caller cannot tell a chunk that landed from one still in flight or lost.
#[derive(Default)]
struct UploadTracker {
    streams: Mutex<HashMap<String, StreamUploads>>,
    finished: Condvar,
}

impl UploadTracker {
    /// Count an upload as in flight until its ticket is finished or dropped.
    fn start(
        self: &Arc<Self>,
        data_id: &str,
        chunk_index: usize,
        channel: Option<String>,
    ) -> UploadTicket {
        self.streams
            .lock()
            .expect("upload tracker mutex poisoned")
            .entry(data_id.to_string())
            .or_default()
            .in_flight += 1;
        UploadTicket {
            tracker: Arc::clone(self),
            data_id: data_id.to_string(),
            chunk_index,
            channel,
            finished: false,
        }
    }

    fn finish(&self, data_id: &str, chunk_index: usize, outcome: Result<(), String>) {
        {
            let mut streams = self.streams.lock().expect("upload tracker mutex poisoned");
            let stream = streams.entry(data_id.to_string()).or_default();
            stream.in_flight = stream.in_flight.saturating_sub(1);
            match outcome {
                Ok(()) => stream.uploaded += 1,
                Err(reason) => stream.lost.push((chunk_index, reason)),
            }
        }
        self.finished.notify_all();
    }

    /// Wait up to `timeout` for `data_id`'s uploads in flight to finish, then take the record of
    /// what its uploads did since the last drain.
    fn drain(&self, data_id: &str, timeout: Duration) -> UploadReport {
        let streams = self.streams.lock().expect("upload tracker mutex poisoned");
        let (mut streams, _) = self
            .finished
            .wait_timeout_while(streams, timeout, |streams| {
                streams
                    .get(data_id)
                    .is_some_and(|stream| stream.in_flight > 0)
            })
            .expect("upload tracker mutex poisoned");
        let Some(stream) = streams.get_mut(data_id) else {
            return UploadReport::default();
        };
        let report = UploadReport {
            uploaded: std::mem::take(&mut stream.uploaded),
            pending: stream.in_flight,
            lost: std::mem::take(&mut stream.lost),
        };
        if stream.in_flight == 0 {
            streams.remove(data_id);
        }
        report
    }

    /// Put back a drained report that could not be handed over, ahead of anything recorded
    /// since, so the next drain reports it.
    fn restore(&self, data_id: &str, report: UploadReport) {
        let mut streams = self.streams.lock().expect("upload tracker mutex poisoned");
        let stream = streams.entry(data_id.to_string()).or_default();
        stream.uploaded += report.uploaded;
        stream.lost.splice(0..0, report.lost);
    }
}

/// One upload in flight. Finishing it records the outcome; dropping it unfinished (the task was
/// cancelled, e.g. by the streamer shutting down) records it as lost.
struct UploadTicket {
    tracker: Arc<UploadTracker>,
    data_id: String,
    chunk_index: usize,
    channel: Option<String>,
    finished: bool,
}

impl UploadTicket {
    fn label(&self) -> String {
        match &self.channel {
            Some(channel) => format!("chunk {} for '{channel}'", self.chunk_index),
            None => format!("chunk {}", self.chunk_index),
        }
    }

    fn finish(mut self, result: &Result<()>) {
        self.finished = true;
        let outcome = result.as_ref().map(|_| ()).map_err(|e| self.reason(e));
        self.tracker
            .finish(&self.data_id, self.chunk_index, outcome);
    }

    fn reason(&self, error: impl std::fmt::Display) -> String {
        match &self.channel {
            Some(channel) => format!("channel '{channel}': {error:#}"),
            None => format!("{error:#}"),
        }
    }
}

impl Drop for UploadTicket {
    fn drop(&mut self) {
        if !self.finished {
            let reason = self.reason("upload cancelled before it finished");
            self.tracker
                .finish(&self.data_id, self.chunk_index, Err(reason));
        }
    }
}

/// What flush() and finalize() report about a stream's uploads.
#[derive(Debug, Default)]
struct UploadReport {
    uploaded: usize,
    pending: usize,
    lost: Vec<(usize, String)>,
}

impl UploadReport {
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("chunks_uploaded", self.uploaded)?;
        dict.set_item("chunks_failed", self.lost.len())?;
        dict.set_item("chunks_pending", self.pending)?;
        dict.set_item("drained", self.pending == 0)?;
        dict.set_item("lost", self.lost.clone())?;
        Ok(dict)
    }

    /// The counts behind the chunks not uploaded, and the first few failures.
    fn describe_losses(&self) -> String {
        const LISTED: usize = 5;
        let mut text = format!(
            "{} failed after retries, {} still uploading",
            self.lost.len(),
            self.pending
        );
        for (chunk_index, reason) in self.lost.iter().take(LISTED) {
            text.push_str(&format!("; chunk {chunk_index}: {reason}"));
        }
        if self.lost.len() > LISTED {
            text.push_str("; …");
        }
        text
    }
}

/// Upload attempts per tool-state chunk; see [`with_retries`] for what is retried.
const CHUNK_UPLOAD_ATTEMPTS: u32 = 3;

/// POST a chunk payload (single- or multi-channel) to the API endpoint, with retries.
async fn post_chunk<T: Serialize>(
    client: &Client,
    url: &str,
    payload: &T,
    api_key: &str,
) -> Result<()> {
    with_retries(
        &format!("[timeseries_stream] upload to {url}"),
        CHUNK_UPLOAD_ATTEMPTS,
        || async {
            let resp = client
                .post(url)
                .header("X-API-KEY", api_key)
                .header("Content-Type", "application/json")
                .json(payload)
                .send()
                .await?;
            let status = resp.status();
            if status.is_success() {
                return Ok(());
            }
            let body = resp.text().await.unwrap_or_default();
            Err(HttpStatusError { status, body }.into())
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn streamer_with(clock: Option<LabClock>) -> TimeseriesStreamer {
        TimeseriesStreamer {
            api_key: "key".into(),
            endpoint: "http://127.0.0.1:1".into(),
            client: Client::new(),
            rt: tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
            points_per_chunk: 100,
            clock: clock.map(Arc::new),
            uploads: Arc::default(),
        }
    }

    fn channel(timestamps: &[f64]) -> ChannelPayload {
        ChannelPayload {
            timestamps: timestamps.to_vec(),
            values: vec![0.0; timestamps.len()],
            units: None,
        }
    }

    #[test]
    fn every_channel_of_a_chunk_shifts_by_the_same_offset_while_the_clock_slews() {
        let clock = LabClock::slewing(5_000.0, 5_300.0);
        // The hazard: a slewing clock reads differently from one moment to the next.
        let first = clock.status().unwrap().offset_ms;
        std::thread::sleep(Duration::from_millis(2));
        assert_ne!(first, clock.status().unwrap().offset_ms);

        // Small timestamps make any difference in offset visible after the addition.
        let channels = (0..2_000)
            .map(|i| (format!("channel {i}"), channel(&[1.0, 2.0])))
            .collect();
        let (corrected, record) = channels_to_reference_utc(Some(&clock), channels);

        let expected = &corrected["channel 0"].timestamps;
        assert!(expected[0] > 5.0, "{expected:?}");
        assert!(corrected.values().all(|c| &c.timestamps == expected));
        let recorded_s = record.expect("measured").offset_ms / 1_000.0;
        assert!(
            (expected[0] - (1.0 + recorded_s)).abs() < 1e-9,
            "the record must be the offset the stamps shifted by"
        );
    }

    #[test]
    fn without_correction_timestamps_are_sent_as_read() {
        let (corrected, record) =
            channels_to_reference_utc(None, vec![("t".into(), channel(&[1.5, 2.5]))]);
        assert_eq!(corrected["t"].timestamps, vec![1.5, 2.5]);
        assert_eq!(record, None);
    }

    fn measured_clock_body() -> serde_json::Value {
        json!({"offset_ms": 5_000.0, "uncertainty_ms": 20.0, "measured_at_unix_ms": 1_700_000_000_123_i64})
    }

    #[test]
    fn a_single_channel_chunk_carries_the_conversion_applied_to_its_timestamps() {
        let streamer = streamer_with(Some(LabClock::slewing(5_000.0, 5_000.0)));
        let (timestamps, clock) = streamer.reference_utc_secs(vec![1.0, 2.0]);
        let payload = ChunkPayload {
            data_id: "d".into(),
            chunk_index: 3,
            channel_name: "temp".into(),
            timestamps,
            values: vec![10.0, 11.0],
            units: None,
            clock,
        };
        assert_eq!(
            serde_json::to_value(&payload).unwrap(),
            json!({
                "data_id": "d",
                "chunk_index": 3,
                "channel_name": "temp",
                "timestamps": [6.0, 7.0],
                "values": [10.0, 11.0],
                "clock": measured_clock_body(),
            })
        );
    }

    #[test]
    fn a_multi_channel_chunk_carries_one_conversion_for_all_its_channels() {
        let (channels, clock) = channels_to_reference_utc(
            Some(&LabClock::slewing(5_000.0, 5_000.0)),
            vec![("a".into(), channel(&[1.0])), ("b".into(), channel(&[2.0]))],
        );
        let payload = MultiChunkPayload {
            data_id: "d".into(),
            chunk_index: 0,
            channels,
            clock,
        };
        assert_eq!(
            serde_json::to_value(&payload).unwrap(),
            json!({
                "data_id": "d",
                "chunk_index": 0,
                "channels": {
                    "a": {"timestamps": [6.0], "values": [0.0]},
                    "b": {"timestamps": [7.0], "values": [0.0]},
                },
                "clock": measured_clock_body(),
            })
        );
    }

    #[test]
    fn before_the_first_measurement_the_chunk_says_its_clock_is_null_and_stamps_are_raw() {
        for clock in [Some(LabClock::new()), None] {
            let streamer = streamer_with(clock);
            let (timestamps, clock) = streamer.reference_utc_secs(vec![1.5]);
            let body = serde_json::to_value(ChunkPayload {
                data_id: "d".into(),
                chunk_index: 0,
                channel_name: "temp".into(),
                timestamps,
                values: vec![1.0],
                units: None,
                clock,
            })
            .unwrap();
            assert_eq!(body["timestamps"], json!([1.5]));
            assert!(body.as_object().unwrap().contains_key("clock"));
            assert_eq!(body["clock"], json!(null));
        }
    }

    #[test]
    fn drain_waits_for_uploads_in_flight() {
        let tracker = Arc::new(UploadTracker::default());
        let ticket = tracker.start("stream", 0, None);
        let uploader = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            ticket.finish(&Ok(()));
        });

        let report = tracker.drain("stream", Duration::from_secs(10));
        uploader.join().unwrap();

        assert_eq!((report.uploaded, report.pending), (1, 0));
        assert!(report.lost.is_empty());
    }

    #[test]
    fn drain_that_times_out_reports_uploads_still_pending_and_the_next_drain_their_outcome() {
        let tracker = Arc::new(UploadTracker::default());
        let ticket = tracker.start("stream", 4, None);

        let report = tracker.drain("stream", Duration::from_millis(20));
        assert_eq!((report.uploaded, report.pending), (0, 1));

        ticket.finish(&Err(anyhow::anyhow!("HTTP 400: rejected")));
        let report = tracker.drain("stream", Duration::from_millis(20));
        assert_eq!(report.pending, 0);
        assert_eq!(report.lost, vec![(4, "HTTP 400: rejected".to_string())]);
    }

    #[test]
    fn a_dropped_upload_is_reported_lost() {
        let tracker = Arc::new(UploadTracker::default());
        drop(tracker.start("stream", 2, Some("temperature".into())));

        let report = tracker.drain("stream", Duration::ZERO);

        assert_eq!(
            report.lost,
            vec![(
                2,
                "channel 'temperature': upload cancelled before it finished".to_string()
            )]
        );
    }

    #[test]
    fn a_restored_report_comes_back_ahead_of_later_outcomes() {
        let tracker = Arc::new(UploadTracker::default());
        tracker.start("stream", 0, None).finish(&Ok(()));
        tracker
            .start("stream", 1, None)
            .finish(&Err(anyhow::anyhow!("first")));
        let report = tracker.drain("stream", Duration::ZERO);

        tracker
            .start("stream", 2, None)
            .finish(&Err(anyhow::anyhow!("second")));
        tracker.restore("stream", report);
        let report = tracker.drain("stream", Duration::ZERO);

        assert_eq!(report.uploaded, 1);
        assert_eq!(
            report.lost,
            vec![(1, "first".to_string()), (2, "second".to_string())]
        );
    }

    #[test]
    fn drain_reports_each_outcome_once_and_only_for_its_stream() {
        let tracker = Arc::new(UploadTracker::default());
        tracker.start("a", 0, None).finish(&Ok(()));
        let _other = tracker.start("b", 0, None);

        let report = tracker.drain("a", Duration::from_secs(10));
        assert_eq!((report.uploaded, report.pending), (1, 0));
        let report = tracker.drain("a", Duration::ZERO);
        assert_eq!((report.uploaded, report.pending), (0, 0));
    }
}
