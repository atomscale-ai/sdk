use anyhow::{anyhow, Context};
use chrono::{Local, Utc};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyModule};
use reqwest::Client;
use std::time::Instant;
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;
use tracing::debug;

mod utils;
use utils::{generic_post, init_tracing_once};

mod initialize;
use initialize::{post_for_initialization, RHEEDStreamSettings};

mod upload;
use upload::{
    numpy_frames_to_flat, package_to_zarr_bytes, post_for_presigned, put_bytes_presigned,
    FrameChunkMetadata,
};

/// RHEEDStreamer(api_key: str, endpoint: Optional[str] = None)
///
/// A thin, high-performance Python interface (via PyO3) for **real-time RHEED frame
/// streaming** into the Atomscale platform. This class takes **chunks of 8-bit frames**
/// (NumPy arrays) and uploads them for analysis while they are being captured from a camera
/// programmatically.
///
/// Typical usage:
///
/// 1) **Instantiate** the streamer
/// 2) **initialize(...)** to create the remote data item and receive `data_id`
/// 3a) **run(data_id, frames_iter)** to stream by yielding frame chunks from a generator/iterator, **or**
/// 3b) **push(data_id, chunk_idx, frames)** repeatedly to send chunks from your own loop
/// 4) **finalize(data_id)** to mark the stream complete on the server
///
/// Notes
/// -----
/// - Frame dtype is coerced to `uint8`. Shapes `(H, W)` or `(N, H, W)` are accepted; `(N,H,W)` is preferred for chunks.
/// - Packaging happens concurrently for throughput; network PUTs are async.
/// - This class is safe to call from Python; heavy work is offloaded to multithreaded async workers.
/// - See also: `finalize(...)`.
///
/// Args:
///     api_key (str): Your Atomscale API key.
///     endpoint (Optional[str]): Base API URL. Defaults to `"https://api.atomicdatasciences.com"`.
///
/// Raises:
///     RuntimeError: If the HTTP client or async runtime cannot be constructed.
#[pyclass]
pub struct RHEEDStreamer {
    api_key: String,
    endpoint: String,
    client: Client,
    rt: Runtime,

    rotating: Option<bool>,
    fps: Option<f64>,
    chunk_size: Option<usize>,
}

#[pymethods]
impl RHEEDStreamer {
    /// RHEEDStreamer(api_key: str, endpoint: Optional[str] = None)
    ///
    /// Constructor for the streaming client.
    ///
    /// Args:
    ///     api_key (str): Your AtomScale API key.
    ///     endpoint (Optional[str]): Base API URL. Defaults to `"https://api.atomicdatasciences.com"`.
    ///
    /// Returns:
    ///     RHEEDStreamer: A configured streamer ready to be initialized and used.
    ///
    /// Raises:
    ///     RuntimeError: If the HTTP client or async runtime cannot be constructed.
    #[new]
    #[pyo3(signature = (api_key, endpoint=None, verbosity=None))]
    #[pyo3(text_signature = "(api_key, endpoint=None, verbosity=None)")]
    fn new(api_key: String, endpoint: Option<String>, verbosity: Option<u8>) -> PyResult<Self> {
        let verbosity = verbosity.unwrap_or(0);
        init_tracing_once(verbosity);

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let endpoint = endpoint.unwrap_or("https://api.atomicdatasciences.com".to_string());

        debug!("[rheed_stream] init: base_url={}", endpoint);

        Ok(Self {
            api_key,
            endpoint,
            client,
            rt,
            rotating: None,
            chunk_size: None,
            fps: None,
        })
    }

    ////Initialize stream
    /// initialize(self, stream_name: Optional[str] = None, fps: float, rotations_per_min: float, chunk_size: int) -> str
    ///
    /// Creates a new **remote data item** for this stream and returns its `data_id`.
    /// Also captures runtime configuration used for subsequent chunk uploads.
    ///
    /// The rotational period (frames per rotation) is computed as:
    /// `fpr = (fps * 60.0) / rotations_per_min`. If `rotations_per_min <= 0.0`, the stream is
    /// treated as **stationary** (no rotation).
    ///
    /// After streaming via `run(...)` or `push(...)`, call `finalize(data_id)` to mark the stream as complete.
    ///
    /// Args:
    ///     stream_name (Optional[str]): Human-readable name shown in the platform. If `None` or an empty string,
    ///         a default like `"RHEED Stream @ 1:23PM"` is used.
    ///     fps (float): Capture rate in frames per second.
    ///     rotations_per_min (float): Wafer/crystal rotations per minute; use `0.0` for stationary operation.
    ///     chunk_size (int): The **intended** number of frames per chunk you will send with `run(...)` or `push(...)`.
    ///
    /// Returns:
    ///     str: The created `data_id` for this stream.
    ///
    /// Raises:
    ///     RuntimeError: If the initialization POST fails.
    #[pyo3(signature = (fps, rotations_per_min, chunk_size, stream_name=None))]
    #[pyo3(text_signature = "(fps, rotations_per_min, chunk_size, stream_name=None)")]
    fn initialize(
        &mut self,
        fps: f64,
        rotations_per_min: f64,
        chunk_size: usize,
        stream_name: Option<String>,
    ) -> PyResult<String> {
        // Guard: chunk_size must be >= ceil(2 * fps)
        let min_chunk = (2.0 * fps).ceil() as usize;
        if chunk_size < min_chunk {
            return Err(PyValueError::new_err(format!(
                "chunk_size must be at least 2×fps (>= {min_chunk}); got {chunk_size}"
            )));
        }

        // Default file name: "RHEED Stream @ #:##AM/PM"
        let default_name = format!("RHEED Stream @ {}", Local::now().format("%-I:%M%p"));

        // Falsy-style fallback (treat empty string like None)
        let stream_name = stream_name
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(default_name);

        let fpr = (fps * 60.0) / rotations_per_min;

        #[allow(clippy::redundant_field_names)]
        let settings = RHEEDStreamSettings {
            data_item_name: stream_name.clone(),
            rotational_period: fpr,
            rotations_per_min,
            fps_capture_rate: fps,
        };

        let base_endpoint = self.endpoint.clone();
        let post_url = format!("{base_endpoint}/rheed/stream/");
        let init_fut = post_for_initialization(&self.client, &post_url, &settings, &self.api_key);

        let data_id = self
            .rt
            .block_on(init_fut)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        self.fps = Some(fps);
        self.rotating = Some(rotations_per_min > 0.0);
        self.chunk_size = Some(chunk_size);

        Ok(data_id)
    }

    /// run(self, data_id: str, frames_iter: Iterable[numpy.ndarray]) -> None
    ///
    /// **Generator/iterator mode.** Iterates `frames_iter`, where each yielded item is either:
    ///
    /// - `(N, H, W)` `numpy.ndarray[uint8]`: a chunk of `N` grayscale frames, or
    /// - `(H, W)` `numpy.ndarray[uint8]`: a single frame (treated as `N = 1`)
    ///
    /// For each yielded item, the method:
    /// 1) Converts to flat `uint8` bytes,
    /// 2) Packages frames on a blocking worker thread,
    /// 3) Uploads the shard,
    /// 4) Proceeds concurrently for high throughput.
    ///
    /// This method **blocks until all spawned tasks complete**. After `run(...)` returns,
    /// call `finalize(data_id)` to mark the stream complete on the server.
    ///
    /// Args:
    ///     data_id (str): The stream data ID returned by `initialize(...)`.
    ///     frames_iter (Iterable[numpy.ndarray]): Python iterable/generator of `(N,H,W)` or `(H,W)` uint8 arrays.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If any packaging/join/upload step fails.
    ///
    /// See also:
    ///     finalize(data_id)
    #[pyo3(signature = (data_id, frames_iter))]
    #[pyo3(text_signature = "(data_id, frames_iter)")]
    fn run(&self, data_id: String, frames_iter: Bound<PyAny>) -> PyResult<()> {
        let (rotating, fps, chunk_size) = self.cfg()?;

        debug!("[rheed_stream] run: starting (concurrent: prepare→spawn package tasks)");

        let iter = PyIterator::from_object(&frames_iter)?;
        let mut handles = Vec::new();

        let t_total0 = Instant::now();

        for (idx, it) in iter.enumerate() {
            let t0 = Instant::now();
            let obj = it?; // next yielded item

            // Prepare (under GIL): convert to (flat bytes, N,H,W)
            let (flat, n, h, w) = numpy_frames_to_flat(obj)?;
            let flat_len = flat.len();

            debug!(
                "[rheed_stream] item#{idx}: prepared in {:.2?} → N,H,W=({n},{h},{w}), flat_len={}; spawning task…",
                t0.elapsed(), flat_len
            );

            // Build metadata (timing handled externally overall; we keep 'now + duration' per chunk)
            let now_ms_utc = Utc::now().timestamp_millis();
            let metadata = FrameChunkMetadata {
                data_id: data_id.clone(),
                data_stream: "rheed".to_string(),
                is_stream: 1,
                is_rotating: rotating as u8,
                raw_frame_rate: fps,
                avg_frame_rate: fps,
                chunk_size,
                dims: format!("{n},{h},{w}"),
                start_unix_ms_utc: now_ms_utc,
                end_unix_ms_utc: now_ms_utc + (((n as f64 / fps) * 1000.0) as i64),
            };

            // Spawn via private helper
            let handle = self.spawn_chunk_upload(idx, flat, n, h, w, metadata);
            handles.push(handle);
        }

        debug!(
            "[rheed_stream] run: spawned {} packaging task(s); awaiting…",
            handles.len()
        );

        // Wait for all tasks to finish; print errors if any
        let res = self.rt.block_on(async {
            for (i, h) in handles.into_iter().enumerate() {
                match h.await {
                    Ok(Ok(())) => debug!("[rheed_stream] task#{i}: completed"),
                    Ok(Err(e)) => {
                        debug!("[rheed_stream] task#{i}: ERROR: {e}");
                        return Err(e);
                    }
                    Err(e) => {
                        debug!("[rheed_stream] task#{i}: JOIN ERROR: {e}");
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
        debug!("[rheed_stream] run: all tasks done in {:.2?}", total_dur);
        Ok(())
    }

    /// push(self, data_id: str, chunk_idx: int, frames: numpy.ndarray) -> None
    ///
    /// **Callback mode.** Push a single chunk of frames that you produced externally (e.g., from a
    /// camera callback). Call repeatedly for subsequent chunks.
    ///
    /// The `frames` argument may be `(N,H,W)` or `(H,W)`; dtype is coerced to `uint8`.
    ///
    /// After your last `push(...)`, call **`finalize(data_id)`** to mark the stream as complete on the server.
    ///
    /// Args:
    ///     data_id (str): The remote data identifier returned by `initialize(...)`.
    ///     chunk_idx (int): Zero-based index of this chunk (used in Zarr shard path).
    ///     frames (numpy.ndarray): `(N,H,W)` or `(H,W)` grayscale frames as `uint8`.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If packaging or upload fails internally.
    ///
    /// See also:
    ///     finalize(data_id)
    #[pyo3(signature = (data_id, chunk_idx, frames))]
    #[pyo3(text_signature = "(data_id, chunk_idx, frames)")]
    fn push(&self, data_id: String, chunk_idx: usize, frames: Bound<PyAny>) -> PyResult<()> {
        let (rotating, fps, chunk_size) = self.cfg()?;

        // Prepare (under GIL)
        let (flat, n, h, w) = numpy_frames_to_flat(frames)?;

        // Build metadata (same model as generator mode)
        let now_ms_utc = Utc::now().timestamp_millis();
        let metadata = FrameChunkMetadata {
            data_id: data_id.clone(),
            data_stream: "rheed".to_string(),
            is_stream: 1,
            is_rotating: rotating as u8,
            raw_frame_rate: fps,
            avg_frame_rate: fps,
            chunk_size,
            dims: format!("{n},{h},{w}"),
            start_unix_ms_utc: now_ms_utc,
            end_unix_ms_utc: now_ms_utc + (((n as f64 / fps) * 1000.0) as i64),
        };

        // Spawn via private helper; detach by dropping the handle
        self.spawn_chunk_upload(chunk_idx, flat, n, h, w, metadata);
        Ok(())
    }

    /// finalize(self, data_id: str) -> None
    ///
    /// Explicitly **closes** the remote stream for the given `data_id`. This signals to the
    /// server that no further chunks will be uploaded and allows any downstream jobs
    /// (e.g., indexing, aggregation, or post-processing) to begin.
    ///
    /// Typical use:
    /// - After `run(...)` returns (it waits for all chunk tasks), call `finalize(data_id)`.
    /// - In `push(...)` mode, call `finalize(data_id)` only after you have pushed your last
    ///   chunk **and** ensured any in-flight uploads have finished (since `push(...)`
    ///   detaches tasks).
    ///
    /// Notes
    /// -----
    /// - The operation performs a single HTTP POST to the `.../end` endpoint.
    /// - It is safe to call more than once; the server may treat it as idempotent,
    ///   but repeated calls are unnecessary.
    ///
    /// Args:
    ///     data_id (str): The stream identifier returned by `initialize(...)`.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If the finalization POST fails.
    ///
    /// See also:
    ///     run(data_id, frames_iter), push(data_id, chunk_idx, frames)
    #[pyo3(signature = (data_id))]
    #[pyo3(text_signature = "(data_id)")]
    fn finalize(&mut self, data_id: String) -> PyResult<()> {
        let base_endpoint = self.endpoint.clone();
        let post_url = format!("{base_endpoint}/rheed/stream/{data_id}/end");
        let final_fut = generic_post(&self.client, &post_url, &self.api_key);

        self.rt
            .block_on(final_fut)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }
}

// Private and rust only access
impl RHEEDStreamer {
    /// cfg(self) -> Tuple[bool, float, int]
    ///
    /// Internal helper: validates that `initialize(...)` populated required runtime config.
    ///
    /// Returns:
    ///     Tuple[bool, float, int]: `(rotating, fps, chunk_size)`
    ///
    /// Raises:
    ///     RuntimeError: If any required field is missing (i.e., `initialize(...)` not called).
    fn cfg(&self) -> PyResult<(bool, f64, usize)> {
        Ok((
            self.rotating.ok_or_else(|| {
                PyRuntimeError::new_err("rotating is not set; call initialize(...).")
            })?,
            self.fps
                .ok_or_else(|| PyRuntimeError::new_err("fps is not set; call initialize(...)."))?,
            self.chunk_size.ok_or_else(|| {
                PyRuntimeError::new_err("chunk size is not set; call initialize(...).")
            })?,
        ))
    }

    /// spawn_chunk_upload(self, chunk_idx: int, flat: bytes, n: int, h: int, w: int, metadata: FrameChunkMetadata) -> asyncio.Task
    ///
    /// Internal helper: packages a prepared chunk to Zarr and uploads via a presigned URL.
    /// Not exposed to Python.
    fn spawn_chunk_upload(
        &self,
        chunk_idx: usize,
        flat: Vec<u8>,
        n: usize,
        h: usize,
        w: usize,
        metadata: FrameChunkMetadata,
    ) -> JoinHandle<anyhow::Result<()>> {
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let post_url = format!(
            "{}/data_entries/raw_data/staged/upload_urls/",
            self.endpoint
        );
        let zarr_shard_key = format!("frames.zarr/frames/c/{chunk_idx}/0/0");

        debug!(
            "[rheed_stream] spawn#{chunk_idx}: queued (flat={} bytes, dims={n}x{h}x{w})",
            flat.len()
        );

        self.rt.spawn(async move {
            debug!("[rheed_stream] spawn#{chunk_idx}: requesting presign + packaging…");

            let url_fut =
                post_for_presigned(&client, &post_url, &zarr_shard_key, &metadata, &api_key);
            let shard_handle =
                tokio::task::spawn_blocking(move || package_to_zarr_bytes(&flat, n, h, w));

            // 🔎 If either future errors, print it so you see why we never reach PUT.
            let (url, shard): (String, Vec<u8>) = match tokio::try_join!(
                async {
                    let u = url_fut.await.context("presigned URL request failed")?;
                    debug!("[rheed_stream] spawn#{chunk_idx}: presign OK");
                    Ok::<_, anyhow::Error>(u)
                },
                async {
                    let bytes = shard_handle
                        .await
                        .map_err(|e| anyhow!("shard join error: {e}"))?
                        .context("shard worker failed")?;
                    debug!(
                        "[rheed_stream] spawn#{chunk_idx}: packaging OK ({} bytes)",
                        bytes.len()
                    );
                    Ok::<_, anyhow::Error>(bytes)
                }
            ) {
                Ok(v) => v,
                Err(e) => {
                    debug!("[rheed_stream] spawn#{chunk_idx}: presign/packaging ERROR:\n{e:#}");
                    return Err(e);
                }
            };

            debug!("[rheed_stream] spawn#{chunk_idx}: PUT start…");
            if let Err(e) = put_bytes_presigned(&client, &url, &shard).await {
                debug!("[rheed_stream] spawn#{chunk_idx}: PUT ERROR:\n{e:#}");
                return Err(e.context("PUT bytes failed"));
            }
            debug!("[rheed_stream] spawn#{chunk_idx}: PUT done");

            Ok(())
        })
    }
}

#[pymodule]
fn rheed_stream(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<RHEEDStreamer>()?;
    Ok(())
}
