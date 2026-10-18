use anyhow::{anyhow, Context};
use chrono::{Local, Utc};
use pyo3::exceptions::{PyRuntimeError, PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule, PyTuple};
use reqwest::Client;
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::runtime::Runtime;
use tokio::sync::{watch, Semaphore};
use tokio::task::JoinHandle;
use tracing::debug;

mod clock;
use clock::{clock_status_dict, lab_clock_for, LabClock};

mod utils;
use utils::{generic_post, init_tracing_once};

mod timeseries;
use timeseries::TimeseriesStreamer;

mod initialize;
use initialize::{
    link_new_stream, post_for_initialization, resolve_stream_links, RHEEDStreamSettings,
};

mod retry;
use retry::with_retries;

mod upload;
use upload::{
    numpy_frames_to_flat, package_to_zarr_bytes, post_for_presigned, put_bytes_presigned,
    FrameChunkMetadata,
};

/// Compute the `end_unix_ms_utc` for a chunk of `n` frames starting at `start_ms`
/// at declared fps. Lifted into a free function so it can be unit-tested
/// without spinning up an HTTP/runtime/python stack.
#[inline]
fn chunk_end(start_ms: i64, n: usize, fps: f64) -> i64 {
    start_ms + ((n as f64 / fps) * 1000.0) as i64
}

/// The capture start of a chunk the caller handed over at `completed_ms`.
///
/// A chunk is pushed once its frames exist, so the moment it arrives here is when it finished
/// being captured, not when it started: its first frame was captured one chunk span earlier.
/// Stamping the arrival as the start would put every chunk, and the stream's alignment with
/// other wall-clock data, a chunk span late.
#[inline]
fn capture_start_for_completed_chunk(completed_ms: i64, n: usize, fps: f64) -> i64 {
    completed_ms - (chunk_end(0, n, fps))
}

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
///     endpoint (Optional[str]): Base API URL. Defaults to `"https://api.atomscale.ai"`.
///
/// Raises:
///     RuntimeError: If the HTTP client or async runtime cannot be constructed.
#[pyclass]
pub struct RHEEDStreamer {
    api_key: String,
    endpoint: String,
    client: Client,
    rt: Runtime,

    /// Per-`data_id` config keyed by the `data_id` returned from
    /// `initialize(...)`. Concurrent producers for different data_ids do not
    /// share fps/rotating/chunk_size — each stream has its own entry.
    streams: Mutex<HashMap<String, PerStreamState>>,

    /// Bytes of chunk data (in MiB permits) that uploads may hold at once; see
    /// [`UPLOAD_BUDGET_MIB`].
    upload_budget: Arc<Semaphore>,

    /// This computer's clock against AtomCloud's; every stamp leaves as reference UTC.
    /// `None` when correction is switched off.
    clock: Option<Arc<LabClock>>,
}

/// Per-`data_id` runtime config. One `RHEEDStreamer` instance can service
/// many concurrent streams (one per data_id); this struct holds the config
/// that must not be shared across them.
#[derive(Clone, Debug)]
struct PerStreamState {
    rotating: bool,
    fps: f64,
    chunk_size: usize,
    /// Detached `push(...)` uploads, so `finalize(...)` can wait for them.
    uploads: Arc<UploadLedger>,
}

/// Uploads a stream's `push(...)` calls have detached and not yet finished.
///
/// `push` returns as soon as a chunk is queued, so without this `finalize` could close the stream
/// while its last shards are still on the wire, and an upload that exhausted its retries would
/// vanish without a trace.
#[derive(Debug)]
struct UploadLedger {
    pending: watch::Sender<usize>,
    failed: Mutex<Vec<(usize, String)>>,
}

impl UploadLedger {
    fn new() -> Self {
        Self {
            pending: watch::channel(0).0,
            failed: Mutex::new(Vec::new()),
        }
    }

    fn started(&self) {
        self.pending.send_modify(|pending| *pending += 1);
    }

    fn finished(&self, chunk_idx: usize, outcome: anyhow::Result<()>) {
        if let Err(error) = outcome {
            self.failed
                .lock()
                .expect("upload ledger mutex poisoned")
                .push((chunk_idx, format!("{error:#}")));
        }
        self.pending.send_modify(|pending| *pending -= 1);
    }

    async fn wait_idle(&self) {
        let mut pending = self.pending.subscribe();
        // The sender lives in `self`, so the channel cannot close while we wait.
        let _ = pending.wait_for(|pending| *pending == 0).await;
    }

    fn take_failed(&self) -> Vec<(usize, String)> {
        std::mem::take(&mut *self.failed.lock().expect("upload ledger mutex poisoned"))
    }
}

#[pymethods]
impl RHEEDStreamer {
    /// RHEEDStreamer(api_key: str, endpoint: Optional[str] = None)
    ///
    /// Constructor for the streaming client.
    ///
    /// Args:
    ///     api_key (str): Your AtomScale API key.
    ///     endpoint (Optional[str]): Base API URL. Defaults to `"https://api.atomscale.ai"`.
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

        let endpoint = endpoint.unwrap_or("https://api.atomscale.ai".to_string());

        debug!("[rheed_stream] init: base_url={}", endpoint);

        let clock = lab_clock_for(&endpoint);

        Ok(Self {
            api_key,
            endpoint,
            client,
            rt,
            streams: Mutex::new(HashMap::new()),
            upload_budget: Arc::new(Semaphore::new(UPLOAD_BUDGET_MIB as usize)),
            clock,
        })
    }

    ////Initialize stream
    /// initialize(self, fps: float, rotations_per_min: float, chunk_size: int, stream_name: Optional[str] = None, physical_sample: Optional[str] = None, project_id: Optional[str] = None) -> str
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
    ///     fps (float): Capture rate in frames per second.
    ///     rotations_per_min (float): Wafer/crystal rotations per minute; use `0.0` for stationary operation.
    ///     chunk_size (int): The **intended** number of frames per chunk you will send with `run(...)` or `push(...)`.
    ///     stream_name (Optional[str]): Human-readable name shown in the platform. If `None` or an empty string,
    ///         a default like `"RHEED Stream @ 1:23PM"` is used.
    ///     physical_sample (Optional[str]): Name or UUID of a physical sample to associate with the data item.
    ///         If a UUID is provided, it must match an existing sample. If a name is provided, it is matched
    ///         case-insensitively against existing samples, or a new sample is created if no match is found.
    ///     project_id (Optional[str]): UUID of a project to associate with the stream. When provided along with
    ///         `physical_sample`, the project's `tracking_physical_sample_id` configuration is automatically updated
    ///         to link the physical sample to the project for growth monitoring.
    ///     tags (Optional[List[str]]): List of tag names or UUIDs to attach to the data item. Names are matched
    ///         case-insensitively against existing org tags; unknown names are created. UUIDs must reference
    ///         existing tags. Tags do not set the stream's azimuth; use `azimuth`.
    ///     azimuth (Optional[str]): The azimuth a parked stream is recorded at, e.g. `"100"`. Its views are
    ///         labelled with it in place of the automatic azimuth classifier until a user edits the view.
    ///         Only valid with `rotations_per_min == 0`; a rotating stream carries several azimuths.
    ///
    /// Returns:
    ///     str: The created `data_id` for this stream.
    ///
    /// Raises:
    ///     ValueError: If `azimuth` is given for a rotating stream.
    ///     RuntimeError: If the stream cannot be created or linked. The physical sample and tags
    ///         are resolved before the stream is created, so a bad one creates nothing. If linking
    ///         fails after creation, the stream is ended and the error names its `data_id`.
    #[pyo3(signature = (fps, rotations_per_min, chunk_size, stream_name=None, physical_sample=None, project_id=None, tags=None, azimuth=None))]
    #[pyo3(
        text_signature = "(fps, rotations_per_min, chunk_size, stream_name=None, physical_sample=None, project_id=None, tags=None, azimuth=None)"
    )]
    fn initialize(
        &self,
        py: Python<'_>,
        fps: f64,
        rotations_per_min: f64,
        chunk_size: usize,
        stream_name: Option<String>,
        physical_sample: Option<String>,
        project_id: Option<String>,
        tags: Option<Vec<String>>,
        azimuth: Option<String>,
    ) -> PyResult<String> {
        // Measure the lab clock before the first chunk is stamped; a failure streams uncorrected.
        if let Some(clock) = &self.clock {
            py.detach(|| clock.wait_for_first_attempt());
        }

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

        let physical_sample = physical_sample
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let project_id = project_id
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let azimuth_label = azimuth
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        if azimuth_label.is_some() && rotations_per_min > 0.0 {
            return Err(PyValueError::new_err(
                "azimuth is for parked streams (rotations_per_min=0); a rotating stream carries several azimuths",
            ));
        }

        let fpr = (fps * 60.0) / rotations_per_min;

        #[allow(clippy::redundant_field_names)]
        let settings = RHEEDStreamSettings {
            data_item_name: stream_name.clone(),
            rotational_period: fpr,
            rotations_per_min,
            fps_capture_rate: fps,
            project_id,
            azimuth_label,
        };

        let links_fut = resolve_stream_links(
            &self.client,
            &self.endpoint,
            &self.api_key,
            physical_sample.as_deref(),
            settings.project_id.clone(),
            tags.as_deref().unwrap_or_default(),
        );
        let links = self
            .rt
            .block_on(links_fut)
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

        let post_url = format!("{}/rheed/stream/", self.endpoint);
        let init_fut = post_for_initialization(&self.client, &post_url, &settings, &self.api_key);

        let data_id = self
            .rt
            .block_on(init_fut)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let end_url = self.end_url(&data_id);
        let link_fut = link_new_stream(
            &self.client,
            &self.endpoint,
            &self.api_key,
            &data_id,
            &links,
            &end_url,
        );
        self.rt
            .block_on(link_fut)
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

        // Insert per-stream state keyed by the new data_id. State is not
        // shared with any other concurrent stream on this streamer instance.
        // Returning data_id from the platform, then inserting locally, is the
        // intended order: if the platform call fails we never create state for
        // a phantom data_id.
        {
            let mut streams = self
                .streams
                .lock()
                .expect("streams mutex poisoned");
            streams.insert(
                data_id.clone(),
                PerStreamState {
                    rotating: rotations_per_min > 0.0,
                    fps,
                    chunk_size,
                    uploads: Arc::new(UploadLedger::new()),
                },
            );
        }

        Ok(data_id)
    }

    /// run(self, data_id: str, frames_iter: Iterable) -> None
    ///
    /// **Generator/iterator mode.** Iterates `frames_iter`, where each yielded item is one of:
    ///
    /// - `(N, H, W)` `numpy.ndarray[uint8]`: a chunk of `N` grayscale frames
    /// - `(H, W)` `numpy.ndarray[uint8]`: a single frame (treated as `N = 1`)
    /// - `(frames, capture_start_ms_utc)` 2-tuple: frames as above + an explicit
    ///   capture-start timestamp (int milliseconds since UNIX epoch, UTC). Use
    ///   this when you have hardware/OS-level timestamps for each chunk
    ///   (camera trigger time, OS monotonic-to-utc conversion, etc.).
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
    /// Timestamps
    /// ----------
    /// Every chunk is stamped with when its first frame was **captured**, so the stream
    /// aligns with other lab wall-clock data. The chunk's `end_unix_ms_utc` is
    /// `start + (n / fps) * 1000`. Yield `(frames, capture_start_ms_utc)` whenever you have a
    /// real capture clock (camera trigger, OS monotonic-to-utc conversion). When you yield
    /// bare frames, each chunk is taken to have just finished capturing: `Utc::now()` is
    /// sampled the moment the iterator yields, before any GIL-held packaging work, and the
    /// start is that minus the chunk's span (`n / fps`).
    ///
    /// Args:
    ///     data_id (str): The stream data ID returned by `initialize(...)`.
    ///     frames_iter (Iterable): Python iterable/generator of `(N,H,W)` / `(H,W)`
    ///         uint8 arrays, or `(frames, capture_start_ms_utc)` tuples.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If any packaging/join/upload step fails.
    ///     ValueError: If a yielded tuple's second element is not an int.
    ///
    /// See also:
    ///     finalize(data_id)
    #[pyo3(signature = (data_id, frames_iter))]
    #[pyo3(text_signature = "(data_id, frames_iter)")]
    fn run(&self, data_id: String, frames_iter: Bound<PyAny>) -> PyResult<()> {
        let (rotating, fps, chunk_size) = self.cfg(&data_id)?;

        debug!("[rheed_stream] run: starting (concurrent: prepare→spawn package tasks)");

        let py = frames_iter.py();
        let iter = PyIterator::from_object(&frames_iter)?;
        let mut handles = Vec::new();

        let t_total0 = Instant::now();

        for (idx, it) in iter.enumerate() {
            let t0 = Instant::now();
            let obj = it?; // next yielded item

            // Sample arrival timestamp BEFORE any GIL-held packaging work.
            // Combined with `numpy_frames_to_flat` releasing the GIL during
            // its bytes copy, this keeps each chunk's stamp within
            // microseconds of when its producer yielded — even when other
            // streams are concurrently pushing on the same GIL.
            let arrival_ms = Utc::now().timestamp_millis();

            // Yielded item may be: frames | (frames, capture_start_ms_utc)
            // The latter form lets the caller stamp each chunk with its actual
            // capture time (e.g. from a hardware clock).
            let (frames_obj, caller_start_ms): (Bound<PyAny>, Option<i64>) =
                if let Ok(t) = obj.cast::<PyTuple>() {
                    if t.len() == 2 {
                        let frames = t.get_item(0)?;
                        let ts = t.get_item(1)?.extract::<i64>().map_err(|e| {
                            PyValueError::new_err(format!(
                                "second element of yielded tuple must be int (ms UTC): {e}"
                            ))
                        })?;
                        (frames, Some(ts))
                    } else {
                        (obj.clone(), None)
                    }
                } else {
                    (obj.clone(), None)
                };

            let (flat, n, h, w) = numpy_frames_to_flat(frames_obj)?;
            let flat_len = flat.len();

            debug!(
                "[rheed_stream] item#{idx}: prepared in {:.2?} → N,H,W=({n},{h},{w}), flat_len={}; spawning task…",
                t0.elapsed(), flat_len
            );

            let start_unix_ms_utc = self.reference_utc_ms(
                caller_start_ms
                    .unwrap_or_else(|| capture_start_for_completed_chunk(arrival_ms, n, fps)),
            );
            let end_unix_ms_utc = chunk_end(start_unix_ms_utc, n, fps);

            let metadata = FrameChunkMetadata {
                data_id: data_id.clone(),
                data_stream: "rheed".to_string(),
                is_stream: 1,
                is_rotating: rotating as u8,
                raw_frame_rate: fps,
                avg_frame_rate: fps,
                chunk_size,
                dims: format!("{n},{h},{w}"),
                start_unix_ms_utc,
                end_unix_ms_utc,
            };

            // Spawn via private helper
            let handle = self.spawn_chunk_upload(py, idx, flat, n, h, w, metadata);
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
            // Alternate form carries the cause chain, e.g. the HTTP status behind a failed upload.
            return Err(PyRuntimeError::new_err(format!("{e:#}")));
        }

        let total_dur = t_total0.elapsed();
        debug!("[rheed_stream] run: all tasks done in {:.2?}", total_dur);
        Ok(())
    }

    /// push(self, data_id: str, chunk_idx: int, frames: numpy.ndarray, capture_start_ms_utc: Optional[int] = None) -> None
    ///
    /// **Callback mode.** Push a single chunk of frames that you produced externally (e.g., from a
    /// camera callback). Call repeatedly for subsequent chunks.
    ///
    /// The `frames` argument may be `(N,H,W)` or `(H,W)`; dtype is coerced to `uint8`.
    ///
    /// After your last `push(...)`, call **`finalize(data_id)`** to mark the stream as complete on the server.
    ///
    /// Timestamps
    /// ----------
    /// Every chunk is stamped with when its first frame was **captured**, so the stream
    /// aligns with other lab wall-clock data. The chunk's `end_unix_ms_utc` is
    /// `start + (n / fps) * 1000`. Pass `capture_start_ms_utc` whenever you have a real
    /// capture clock (camera trigger, OS monotonic-to-utc conversion). When you do not, the
    /// chunk is taken to have just finished being captured: `push()` samples `Utc::now()` on
    /// entry, before any GIL-held packaging work, and the start is that minus the chunk's
    /// span (`n / fps`). That estimate is only as good as how promptly you push; a chunk
    /// pushed late is stamped late.
    ///
    /// Args:
    ///     data_id (str): The remote data identifier returned by `initialize(...)`.
    ///     chunk_idx (int): Zero-based index of this chunk (used in Zarr shard path).
    ///     frames (numpy.ndarray): `(N,H,W)` or `(H,W)` grayscale frames as `uint8`.
    ///     capture_start_ms_utc (Optional[int]): Capture-start timestamp in milliseconds
    ///         since UNIX epoch (UTC). Pass when you have a real capture-time clock
    ///         (camera trigger, OS monotonic-to-utc conversion). When None, the chunk is
    ///         taken to have just finished capturing: `Utc::now()` on entry minus `n / fps`.
    ///
    /// Returns:
    ///     None
    ///
    /// Raises:
    ///     RuntimeError: If packaging or upload fails internally.
    ///
    /// See also:
    ///     finalize(data_id)
    #[pyo3(signature = (data_id, chunk_idx, frames, capture_start_ms_utc=None))]
    #[pyo3(text_signature = "(data_id, chunk_idx, frames, capture_start_ms_utc=None)")]
    fn push(
        &self,
        data_id: String,
        chunk_idx: usize,
        frames: Bound<PyAny>,
        capture_start_ms_utc: Option<i64>,
    ) -> PyResult<()> {
        // Sample arrival timestamp BEFORE any GIL-held packaging work, so the
        // stamp reflects when `push()` was called (not when the GIL handed
        // back after another stream's heavy copy completed).
        let arrival_ms = Utc::now().timestamp_millis();

        let (rotating, fps, chunk_size) = self.cfg(&data_id)?;
        let py = frames.py();
        let (flat, n, h, w) = numpy_frames_to_flat(frames)?;

        let start_unix_ms_utc = self.reference_utc_ms(
            capture_start_ms_utc
                .unwrap_or_else(|| capture_start_for_completed_chunk(arrival_ms, n, fps)),
        );
        let end_unix_ms_utc = chunk_end(start_unix_ms_utc, n, fps);

        let metadata = FrameChunkMetadata {
            data_id: data_id.clone(),
            data_stream: "rheed".to_string(),
            is_stream: 1,
            is_rotating: rotating as u8,
            raw_frame_rate: fps,
            avg_frame_rate: fps,
            chunk_size,
            dims: format!("{n},{h},{w}"),
            start_unix_ms_utc,
            end_unix_ms_utc,
        };

        let ledger = self.upload_ledger(&data_id)?;
        let upload = self.spawn_chunk_upload(py, chunk_idx, flat, n, h, w, metadata);
        ledger.started();
        self.rt.spawn(async move {
            let outcome = upload
                .await
                .unwrap_or_else(|e| Err(anyhow!("upload task failed: {e}")));
            ledger.finished(chunk_idx, outcome);
        });
        Ok(())
    }

    /// clock_status(self) -> Optional[dict]
    ///
    /// This computer's clock measured against AtomCloud's reference clock, or `None` when
    /// correction is off (`ATOMSCALE_SKIP_CLOCK_SYNC`) or no measurement has succeeded yet.
    /// Every chunk is stamped in reference UTC, so `offset_ms` (reference minus host) is the
    /// correction in effect, `uncertainty_ms` bounds its error (half the best round trip) and
    /// `measured_at_unix_ms` is the reference time of the sample it came from.
    fn clock_status<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        clock_status_dict(py, self.clock.as_deref())
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
    /// - Waits for this stream's in-flight `push(...)` uploads first, so a shard still on the
    ///   wire is never cut off. A shard that failed all its retries emits a `RuntimeWarning`
    ///   naming its chunk index; the stream is still closed.
    /// - The server answers 503 while the stream's first analysis is still running; the POST to
    ///   the `.../end` endpoint is retried with backoff for about three minutes before it fails.
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
    fn finalize(&self, py: Python<'_>, data_id: String) -> PyResult<()> {
        let ledger = self.upload_ledger(&data_id)?;
        py.detach(|| self.rt.block_on(ledger.wait_idle()));
        for (chunk_idx, error) in ledger.take_failed() {
            let message = format!("shard {chunk_idx} of stream {data_id} was not uploaded: {error}");
            PyErr::warn(
                py,
                &py.get_type::<PyRuntimeWarning>(),
                &CString::new(message.replace('\0', " ")).expect("NUL bytes were replaced"),
                1,
            )?;
        }

        let post_url = self.end_url(&data_id);
        let final_fut = with_retries("[rheed_stream] finalize", FINALIZE_ATTEMPTS, || {
            generic_post(&self.client, &post_url, &self.api_key)
        });

        py.detach(|| self.rt.block_on(final_fut))
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

        // Drop the per-stream entry so the data_id is no longer addressable
        // from this streamer. Other streams' entries are untouched.
        let mut streams = self
            .streams
            .lock()
            .expect("streams mutex poisoned");
        streams.remove(&data_id);

        Ok(())
    }
}

/// Presign + PUT attempts per shard: 0.5, 1, 2 and 4 s apart, about 8 s in all.
const SHARD_UPLOAD_ATTEMPTS: u32 = 5;

/// Attempts at closing a stream. The server answers 503 until the stream's first analysis has
/// written its processed video, which can take a minute or more: 0.5, 1, 2, 4, 8, 16, then 30 s
/// apart, about three minutes in all.
const FINALIZE_ATTEMPTS: u32 = 12;

/// Chunk data uploads may hold at once, in MiB. A chunk's raw frames stay in memory until
/// its upload finishes, retries included, so an outage would otherwise buffer without
/// limit; past the budget, `push`/`run` wait for an upload to finish (backpressure). A
/// chunk larger than the whole budget still goes through, alone.
const UPLOAD_BUDGET_MIB: u32 = 1024;

// Private and rust only access
impl RHEEDStreamer {
    fn end_url(&self, data_id: &str) -> String {
        format!("{}/rheed/stream/{data_id}/end", self.endpoint)
    }

    /// A host-clock time as reference UTC (unchanged when correction is off or not yet measured).
    fn reference_utc_ms(&self, host_ms: i64) -> i64 {
        self.clock
            .as_ref()
            .map_or(host_ms, |clock| clock.to_utc_ms(host_ms))
    }

    fn upload_ledger(&self, data_id: &str) -> PyResult<Arc<UploadLedger>> {
        let streams = self.streams.lock().expect("streams mutex poisoned");
        let state = streams.get(data_id).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "no stream initialized for data_id={data_id}; call initialize(...) first."
            ))
        })?;
        Ok(state.uploads.clone())
    }

    /// cfg(self, data_id) -> Tuple[bool, float, int]
    ///
    /// Internal helper: looks up per-stream config for `data_id`. Returns the
    /// tuple `(rotating, fps, chunk_size)` from the entry inserted by
    /// `initialize(...)`. Errors if no entry exists for this `data_id`.
    fn cfg(&self, data_id: &str) -> PyResult<(bool, f64, usize)> {
        let streams = self.streams.lock().expect("streams mutex poisoned");
        let s = streams.get(data_id).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "no stream initialized for data_id={data_id}; call initialize(...) first."
            ))
        })?;
        Ok((s.rotating, s.fps, s.chunk_size))
    }

    /// spawn_chunk_upload(self, chunk_idx: int, flat: bytes, n: int, h: int, w: int, metadata: FrameChunkMetadata) -> asyncio.Task
    ///
    /// Internal helper: packages a prepared chunk to Zarr and uploads via a presigned URL.
    /// Not exposed to Python.
    fn spawn_chunk_upload(
        &self,
        py: Python<'_>,
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

        // Reserve this chunk's share of the upload budget; the GIL is released while
        // waiting so other Python threads keep running.
        let mib = (flat.len().div_ceil(1 << 20) as u32).clamp(1, UPLOAD_BUDGET_MIB);
        let budget = self.upload_budget.clone();
        let permit = py
            .detach(|| self.rt.block_on(budget.acquire_many_owned(mib)))
            .expect("upload budget semaphore is never closed");

        debug!(
            "[rheed_stream] spawn#{chunk_idx}: queued (flat={} bytes, dims={n}x{h}x{w})",
            flat.len()
        );

        self.rt.spawn(async move {
            let _permit = permit; // released when this upload finishes, however it ends
            let shard = tokio::task::spawn_blocking(move || package_to_zarr_bytes(&flat, n, h, w))
                .await
                .map_err(|e| anyhow!("shard join error: {e}"))?
                .context("shard worker failed")?;
            debug!(
                "[rheed_stream] spawn#{chunk_idx}: packaging OK ({} bytes)",
                shard.len()
            );

            // A lost shard is a hole in the recording, so a failed presign or PUT is retried
            // (fresh URL each attempt); the shard's key is fixed, so a repeat overwrites it.
            with_retries(
                &format!("[rheed_stream] shard {chunk_idx} upload"),
                SHARD_UPLOAD_ATTEMPTS,
                || async {
                    let url = post_for_presigned(
                        &client,
                        &post_url,
                        &zarr_shard_key,
                        &metadata,
                        &api_key,
                    )
                    .await
                    .context("presigned URL request failed")?;
                    put_bytes_presigned(&client, &url, &shard)
                        .await
                        .context("PUT bytes failed")
                },
            )
            .await?;
            debug!("[rheed_stream] spawn#{chunk_idx}: PUT done");
            Ok(())
        })
    }
}

#[pymodule]
fn rheed_stream(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<RHEEDStreamer>()?;
    m.add_class::<TimeseriesStreamer>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{capture_start_for_completed_chunk, chunk_end};

    /// A chunk pushed the moment it finished capturing started one span earlier.
    #[test]
    fn a_completed_chunk_started_one_span_earlier() {
        let completed = 1_700_000_010_000;
        // 240 frames at 120 fps span 2000 ms.
        let start = capture_start_for_completed_chunk(completed, 240, 120.0);
        assert_eq!(start, completed - 2_000);
        assert_eq!(chunk_end(start, 240, 120.0), completed);
    }

    /// `end_unix_ms_utc - start_unix_ms_utc` is the chunk's intra-chunk span,
    /// scaled by declared fps. This is what the streamer writes alongside each
    /// chunk's wallclock-sampled `start`.
    #[test]
    fn span_scales_with_fps() {
        let start = 1_700_000_000_000;
        // 4 frames at 2 fps → 2000 ms span.
        assert_eq!(chunk_end(start, 4, 2.0) - start, 2_000);
        // 30 frames at 30 fps → 1000 ms span.
        assert_eq!(chunk_end(start, 30, 30.0) - start, 1_000);
    }
}
