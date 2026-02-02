use anyhow::Result;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;
use tracing::debug;

use crate::utils::init_tracing_once;

/// Request body for stream initialization.
#[derive(Serialize, Debug)]
struct InitializeRequest {
    stream_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    synth_source_id: Option<i64>,
    points_per_chunk: usize,
    physical_sample_id: Option<String>,
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
/// - Each push() spawns an async upload task and returns immediately.
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

        Ok(Self {
            api_key,
            endpoint,
            client,
            rt,
            points_per_chunk,
        })
    }

    /// initialize(self, stream_name: Optional[str] = None, synth_source_id: Optional[int] = None, ...) -> str
    ///
    /// Initialize a new time series stream on the server.
    ///
    /// Creates data_catalogue and processed_metrology_catalogue entries.
    /// Returns data_id to use for subsequent push() calls.
    ///
    /// Args:
    ///     stream_name (Optional[str]): Human-readable name for the stream.
    ///     synth_source_id (Optional[int]): Growth instrument ID to link. Must belong to your
    ///         organization. Use list_instruments() to see available instruments.
    ///     physical_sample_id (Optional[str]): Associated physical sample UUID.
    ///     project_id (Optional[str]): Associated project UUID.
    ///
    /// Returns:
    ///     str: The data_id for this stream.
    ///
    /// Raises:
    ///     RuntimeError: If the initialization request fails or instrument not found.
    #[pyo3(signature = (stream_name=None, synth_source_id=None, physical_sample_id=None, project_id=None))]
    #[pyo3(
        text_signature = "(stream_name=None, synth_source_id=None, physical_sample_id=None, project_id=None)"
    )]
    fn initialize(
        &self,
        stream_name: Option<String>,
        synth_source_id: Option<i64>,
        physical_sample_id: Option<String>,
        project_id: Option<String>,
    ) -> PyResult<String> {
        let request = InitializeRequest {
            stream_name,
            synth_source_id,
            points_per_chunk: self.points_per_chunk,
            physical_sample_id,
            project_id,
        };

        let url = format!("{}/instrument-timeseries/initialize", self.endpoint);

        let result = self.rt.block_on(async {
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
        });

        match result {
            Ok(response) => {
                debug!(
                    "[timeseries_stream] initialized: data_id={}",
                    response.data_id
                );
                Ok(response.data_id)
            }
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
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

        let payload = ChunkPayload {
            data_id,
            chunk_index,
            channel_name,
            timestamps,
            values,
            units,
        };

        self.spawn_upload(payload);
        Ok(())
    }

    /// push_multi(self, data_id: str, chunk_index: int, channels: dict[str, dict]) -> None
    ///
    /// Push data for multiple channels at once. Each channel's data is uploaded as a separate
    /// async task.
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
    fn push_multi(&self, data_id: String, chunk_index: usize, channels: Bound<PyDict>) -> PyResult<()> {

        for (key, value) in channels.iter() {
            let channel_name: String = key.extract().map_err(|e| {
                PyRuntimeError::new_err(format!("channel name must be a string: {}", e))
            })?;

            let channel_dict = value.downcast::<PyDict>().map_err(|_| {
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

            let payload = ChunkPayload {
                data_id: data_id.clone(),
                chunk_index,
                channel_name,
                timestamps,
                values,
                units,
            };

            self.spawn_upload(payload);
        }
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

    /// finalize(self, data_id: str) -> None
    ///
    /// Finalize the stream, marking it as complete on the server.
    ///
    /// Call this after all data has been pushed to signal that the stream is finished.
    ///
    /// Args:
    ///     data_id (str): The stream identifier returned by `initialize()`.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If the request fails.
    #[pyo3(signature = (data_id))]
    #[pyo3(text_signature = "(data_id)")]
    fn finalize(&self, data_id: String) -> PyResult<()> {
        let url = format!("{}/instrument-timeseries/{}/finalize", self.endpoint, data_id);

        let result = self.rt.block_on(async {
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
        });

        match result {
            Ok(()) => {
                debug!("[timeseries_stream] finalized: data_id={}", data_id);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

}

impl TimeseriesStreamer {
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

            let payload = ChunkPayload {
                data_id: data_id.clone(),
                chunk_index,
                channel_name: channel_name.clone(),
                timestamps,
                values,
                units: units.clone(),
            };

            let handle = self.spawn_upload_tracked(payload);
            handles.push(handle);
        }

        debug!(
            "[timeseries_stream] run: spawned {} upload task(s); awaiting…",
            handles.len()
        );

        // Wait for all tasks to finish
        let res = self.rt.block_on(async {
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

    /// Spawn an async task to upload the chunk payload.
    /// Fire-and-forget: the JoinHandle is dropped.
    fn spawn_upload(&self, payload: ChunkPayload) {
        // Fire-and-forget: drop the JoinHandle
        let _ = self.spawn_upload_tracked(payload);
    }

    /// Spawn an async task to upload the chunk payload and return the JoinHandle.
    fn spawn_upload_tracked(&self, payload: ChunkPayload) -> JoinHandle<Result<()>> {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let url = format!("{}/instrument-timeseries/chunk", self.endpoint);

        debug!(
            "[timeseries_stream] spawn: chunk {} for channel '{}'",
            payload.chunk_index, payload.channel_name
        );

        self.rt.spawn(async move {
            debug!(
                "[timeseries_stream] uploading chunk {} for '{}'",
                payload.chunk_index, payload.channel_name
            );

            let result = post_chunk(&client, &url, &payload, &api_key).await;

            match &result {
                Ok(_) => {
                    debug!(
                        "[timeseries_stream] chunk {} for '{}' uploaded successfully",
                        payload.chunk_index, payload.channel_name
                    );
                }
                Err(e) => {
                    debug!(
                        "[timeseries_stream] chunk {} for '{}' upload failed: {}",
                        payload.chunk_index, payload.channel_name, e
                    );
                }
            }

            result
        })
    }
}

/// POST the chunk payload to the API endpoint.
async fn post_chunk(
    client: &Client,
    url: &str,
    payload: &ChunkPayload,
    api_key: &str,
) -> Result<()> {
    let resp = client
        .post(url)
        .header("X-API-KEY", api_key)
        .header("Content-Type", "application/json")
        .json(payload)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("HTTP {}: {}", status, text));
    }

    Ok(())
}
