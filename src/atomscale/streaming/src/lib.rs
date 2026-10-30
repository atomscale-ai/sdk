use chrono::{Local, Utc};
use pyo3::exceptions::{PyRuntimeError, PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule, PyTuple};
use reqwest::Client;
use std::collections::HashMap;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tracing::{debug, warn};

mod clock;
use clock::{clock_status_dict, lab_clock_for, Correction, LabClock};

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
use upload::{numpy_frames_to_flat, FrameChunkMetadata, PutTimeout, ShardUploader};

mod spool;
use spool::{CloseReason, Enqueued, Spool, SpoolConfig, SpoolStatus, StreamReport};

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
/// - `push(...)` and `run(...)` write each chunk's encoded shard to a disk spool and return; upload
///   workers send the spool oldest first in the background, so a slow or failing network never
///   stalls (and so never delays the timestamps of) your acquisition loop. See `spool_status()`.
/// - This class is safe to call from Python; heavy work is offloaded to multithreaded async workers.
/// - See also: `finalize(...)`.
///
/// Args:
///     api_key (str): Your Atomscale API key.
///     endpoint (Optional[str]): Base API URL. Defaults to `"https://api.atomscale.ai"`.
///     spool_dir (Optional[str]): Where chunks wait to upload, one subfolder per stream. Defaults
///         to `$ATOMSCALE_SPOOL_DIR`, else `<user cache dir>/atomscale/rheed-spool`.
///     spool_max_bytes (Optional[int]): Cap on spooled data; past it, new chunks are dropped (and
///         counted) instead of blocking. Defaults to min(20 GB, 10% of the disk's free space).
///     spool_min_free_bytes (Optional[int]): Free disk space the spool always leaves. Default 1 GB.
///     spool_retention_s (Optional[float]): Age past which a chunk that still has not uploaded is
///         discarded. Default 24 h.
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

    /// Chunks waiting to upload, on disk; its workers upload them in the background.
    spool: Arc<Spool>,

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
    ///     verbosity (Optional[int]): Logging verbosity (0-4).
    ///     spool_dir, spool_max_bytes, spool_min_free_bytes, spool_retention_s: Disk spool
    ///         settings; see the class docstring.
    ///
    /// Chunks an earlier run of this endpoint left in the spool (after a crash, say) are
    /// uploaded in the background if their stream is still open, and discarded otherwise.
    ///
    /// Returns:
    ///     RHEEDStreamer: A configured streamer ready to be initialized and used.
    ///
    /// Raises:
    ///     RuntimeError: If the HTTP client, async runtime or spool directory cannot be set up.
    #[new]
    #[pyo3(signature = (api_key, endpoint=None, verbosity=None, spool_dir=None, spool_max_bytes=None, spool_min_free_bytes=None, spool_retention_s=None))]
    #[pyo3(
        text_signature = "(api_key, endpoint=None, verbosity=None, spool_dir=None, spool_max_bytes=None, spool_min_free_bytes=None, spool_retention_s=None)"
    )]
    fn new(
        api_key: String,
        endpoint: Option<String>,
        verbosity: Option<u8>,
        spool_dir: Option<PathBuf>,
        spool_max_bytes: Option<u64>,
        spool_min_free_bytes: Option<u64>,
        spool_retention_s: Option<f64>,
    ) -> PyResult<Self> {
        let verbosity = verbosity.unwrap_or(0);
        init_tracing_once(verbosity);

        // Shard PUTs set their own, size-scaled timeout (`PutTimeout`); the 60 s covers the
        // small API calls. Connecting is bounded on its own so an unreachable host fails fast
        // even under a long PUT timeout.
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let endpoint = endpoint.unwrap_or("https://api.atomscale.ai".to_string());

        debug!("[rheed_stream] init: base_url={}", endpoint);

        let clock = lab_clock_for(&endpoint);

        let retention = spool_retention_s
            .map(|seconds| {
                Duration::try_from_secs_f64(seconds)
                    .ok()
                    .filter(|retention| !retention.is_zero())
                    .ok_or_else(|| PyValueError::new_err("spool_retention_s must be positive"))
            })
            .transpose()?;
        let configure = |root: Option<PathBuf>| {
            let mut config = SpoolConfig::new(&endpoint, root);
            config.max_bytes = spool_max_bytes;
            if let Some(min_free_bytes) = spool_min_free_bytes {
                config.min_free_bytes = min_free_bytes;
            }
            if let Some(retention) = retention {
                config.retention = retention;
            }
            config
        };
        let spool = Spool::open(configure(spool_dir.clone()))
            .or_else(|error| {
                // Only the default location falls back; a directory the caller chose must work.
                let explicit =
                    spool_dir.is_some() || std::env::var_os(spool::SPOOL_DIR_ENV).is_some();
                if explicit {
                    return Err(error);
                }
                let fallback = std::env::temp_dir().join("atomscale-rheed-spool");
                warn!(
                    "[rheed_stream] {error:#}; spooling to {} instead",
                    fallback.display()
                );
                Spool::open(configure(Some(fallback)))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        spool.start(
            rt.handle(),
            Arc::new(ShardUploader {
                client: client.clone(),
                api_key: api_key.clone(),
                presign_url: format!("{endpoint}/data_entries/raw_data/staged/upload_urls/"),
                put_timeout: PutTimeout::default(),
            }),
        );
        debug!("[rheed_stream] spool: {}", spool.root().display());

        Ok(Self {
            api_key,
            endpoint,
            client,
            rt,
            streams: Mutex::new(HashMap::new()),
            spool,
            clock,
        })
    }

    ////Initialize stream
    /// initialize(self, fps: float, rotations_per_min: float, chunk_size: int, stream_name: Optional[str] = None, physical_sample: Optional[str] = None, project_id: Optional[str] = None, tags: Optional[List[str]] = None, azimuth: Optional[str] = None, seed_frame_count: int = 3) -> str
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
    ///     seed_frame_count (int): Number of seed views to select for a rotating stream. Defaults to 3;
    ///         must be an integer from 1 through 8.
    ///
    /// Returns:
    ///     str: The created `data_id` for this stream.
    ///
    /// Raises:
    ///     ValueError: If `azimuth` is given for a rotating stream or `seed_frame_count` is outside 1 through 8.
    ///     RuntimeError: If the stream cannot be created or linked. The physical sample and tags
    ///         are resolved before the stream is created, so a bad one creates nothing. If linking
    ///         fails after creation, the stream is ended and the error names its `data_id`.
    #[pyo3(signature = (fps, rotations_per_min, chunk_size, stream_name=None, physical_sample=None, project_id=None, tags=None, azimuth=None, seed_frame_count=3))]
    #[pyo3(
        text_signature = "(fps, rotations_per_min, chunk_size, stream_name=None, physical_sample=None, project_id=None, tags=None, azimuth=None, seed_frame_count=3)"
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
        seed_frame_count: i64,
    ) -> PyResult<String> {
        if !(1..=8).contains(&seed_frame_count) {
            return Err(PyValueError::new_err(
                "seed_frame_count must be an integer from 1 through 8",
            ));
        }

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
            seed_frame_count,
            project_id,
            azimuth_label,
        };

        // Every round trip below runs with the GIL released: the caller's other Python threads
        // (a camera loop, a feed reader) keep running while it waits on the network.
        let links = py
            .detach(|| {
                self.rt.block_on(resolve_stream_links(
                    &self.client,
                    &self.endpoint,
                    &self.api_key,
                    physical_sample.as_deref(),
                    settings.project_id.clone(),
                    tags.as_deref().unwrap_or_default(),
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

        let post_url = format!("{}/rheed/stream/", self.endpoint);
        let data_id = py
            .detach(|| {
                self.rt.block_on(post_for_initialization(
                    &self.client,
                    &post_url,
                    &settings,
                    &self.api_key,
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let end_url = self.end_url(&data_id);
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

        self.spool.open_stream(&data_id).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "stream {data_id} was created but cannot be spooled: {e:#}"
            ))
        })?;

        // Insert per-stream state keyed by the new data_id. State is not
        // shared with any other concurrent stream on this streamer instance.
        // Returning data_id from the platform, then inserting locally, is the
        // intended order: if the platform call fails we never create state for
        // a phantom data_id.
        {
            let mut streams = self.streams.lock().expect("streams mutex poisoned");
            streams.insert(
                data_id.clone(),
                PerStreamState {
                    rotating: rotations_per_min > 0.0,
                    fps,
                    chunk_size,
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
    /// For each yielded item, the method stamps it, encodes its shard and writes it to the disk
    /// spool, then asks the iterator for the next one. It never waits on uploads, which run in
    /// the background, oldest first. Once the iterator is exhausted, it waits for this stream's
    /// spool to drain (up to 10 minutes). After `run(...)` returns, call `finalize(data_id)` to
    /// mark the stream complete on the server.
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
    ///     RuntimeError: If a chunk cannot be packaged, or if any chunk of this run was lost
    ///         (dropped because the spool was full or the disk low, or refused by the backend).
    ///         Chunks still uploading when the drain wait ends only warn; `finalize` keeps
    ///         draining them.
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
        let lost_before = self.lost_chunks(&data_id);
        let mut spooled = 0_usize;

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
                "[rheed_stream] item#{idx}: prepared in {:.2?} → N,H,W=({n},{h},{w}), flat_len={}; spooling…",
                t0.elapsed(), flat_len
            );

            let (start_unix_ms_utc, correction) = self.reference_utc_ms(
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
                clock_offset_ms: correction.map(|c| c.offset_ms),
                clock_uncertainty_ms: correction.map(|c| c.uncertainty_ms),
            };

            self.spool_chunk(py, &data_id, idx, metadata, flat, n, h, w)?;
            spooled += 1;
        }

        debug!("[rheed_stream] run: spooled {spooled} chunk(s); waiting for uploads…");
        let drained = py.detach(|| {
            self.rt
                .block_on(self.spool.drain(&data_id, DEFAULT_DRAIN_TIMEOUT))
        });

        let lost = self.lost_chunks(&data_id) - lost_before;
        if lost > 0 {
            let report = self.spool.report(&data_id).unwrap_or_default();
            return Err(PyRuntimeError::new_err(format!(
                "{lost} of the {spooled} chunk(s) of this run of stream {data_id} will not be uploaded: {}",
                describe_losses(&report)
            )));
        }
        if !drained {
            let pending = self.spool.report(&data_id).map_or(0, |r| r.pending);
            warn_py(
                py,
                &format!(
                    "{pending} chunk(s) of stream {data_id} are still uploading after {} s; \
                     they stay spooled and finalize() waits for them",
                    DEFAULT_DRAIN_TIMEOUT.as_secs()
                ),
            )?;
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
    /// The chunk is stamped, encoded and written to the disk spool before this returns; it never
    /// waits on uploads. If the spool is at its byte cap or the disk at its free-space floor, the
    /// chunk is dropped instead (logged, and counted in `spool_status()` and `finalize()`), so the
    /// backend sees a missing shard rather than your loop stalling.
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
    ///     RuntimeError: If the chunk cannot be packaged.
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

        let (start_unix_ms_utc, correction) = self.reference_utc_ms(
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
            clock_offset_ms: correction.map(|c| c.offset_ms),
            clock_uncertainty_ms: correction.map(|c| c.uncertainty_ms),
        };

        self.spool_chunk(py, &data_id, chunk_idx, metadata, flat, n, h, w)
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

    /// spool_status(self) -> dict
    ///
    /// The disk spool chunks wait in before uploading. Keys: `dir`, `bytes` (spooled now),
    /// `max_bytes`, `min_free_bytes`, `free_bytes` (on its disk; `None` if unknown),
    /// `chunks_pending`, the cumulative `chunks_spooled`, `chunks_uploaded`, `chunks_dropped`
    /// (spool full, disk low or write failed), `chunks_failed` (refused by the backend) and
    /// `chunks_purged` (discarded unsent: stream finalized or abandoned, or past retention), and
    /// `streams`, the same counts per `data_id` with `closed`.
    fn spool_status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let status = py.detach(|| self.spool.status());
        spool_status_dict(py, &status)
    }

    /// finalize(self, data_id: str) -> None
    ///
    /// Explicitly **closes** the remote stream for the given `data_id`. This signals to the
    /// server that no further chunks will be uploaded and allows any downstream jobs
    /// (e.g., indexing, aggregation, or post-processing) to begin.
    ///
    /// Call it after your last `push(...)`, or after `run(...)` returns.
    ///
    /// Notes
    /// -----
    /// - First waits, up to `drain_timeout_s`, for this stream's spooled chunks to upload, so a
    ///   shard still waiting is not cut off. Then it discards whatever is left and closes the
    ///   stream. If any chunk was never uploaded (dropped, refused, or still spooled at the
    ///   timeout), it emits one `RuntimeWarning` with the count; the stream is still closed.
    /// - Also accepts a stream an earlier run left in the spool (after a crash), to close it.
    /// - The server answers 503 while the stream's first analysis is still running; the POST to
    ///   the `.../end` endpoint is retried with backoff for about three minutes before it fails.
    /// - It is safe to call more than once; the server may treat it as idempotent,
    ///   but repeated calls are unnecessary.
    ///
    /// Args:
    ///     data_id (str): The stream identifier returned by `initialize(...)`.
    ///     drain_timeout_s (Optional[float]): How long to wait for spooled chunks to upload.
    ///         Default 600 (10 minutes).
    ///
    /// Returns:
    ///     dict: What became of the stream's chunks: `chunks_spooled`, `chunks_uploaded`,
    ///     `chunks_dropped`, `chunks_failed`, `chunks_purged`, `chunks_not_uploaded` (the last
    ///     three plus dropped), `drained` (whether the wait finished in time) and `lost`, a list
    ///     of `(chunk_idx, reason)` for chunks not uploaded.
    ///
    /// Raises:
    ///     RuntimeError: If the finalization POST fails.
    ///
    /// See also:
    ///     run(data_id, frames_iter), push(data_id, chunk_idx, frames)
    #[pyo3(signature = (data_id, drain_timeout_s=None))]
    #[pyo3(text_signature = "(data_id, drain_timeout_s=None)")]
    fn finalize<'py>(
        &self,
        py: Python<'py>,
        data_id: String,
        drain_timeout_s: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if self.cfg(&data_id).is_err() && !self.spool.has_stream(&data_id) {
            return Err(PyRuntimeError::new_err(format!(
                "no stream initialized for data_id={data_id}; call initialize(...) first."
            )));
        }
        let timeout = drain_timeout_s
            .map(|seconds| {
                Duration::try_from_secs_f64(seconds)
                    .map_err(|_| PyValueError::new_err("drain_timeout_s must be non-negative"))
            })
            .transpose()?
            .unwrap_or(DEFAULT_DRAIN_TIMEOUT);
        let (drained, report) = py.detach(|| {
            let drained = self.rt.block_on(self.spool.drain(&data_id, timeout));
            let report = self
                .spool
                .close_stream(&data_id, CloseReason::Finalized)
                .unwrap_or_default();
            (drained, report)
        });
        let lost = report.counts.not_uploaded();
        if lost > 0 {
            warn_py(
                py,
                &format!(
                    "{lost} of the {} chunk(s) of stream {data_id} were not uploaded: {}",
                    report.counts.spooled + report.counts.dropped,
                    describe_losses(&report)
                ),
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
        self.streams
            .lock()
            .expect("streams mutex poisoned")
            .remove(&data_id);
        self.spool.forget_stream(&data_id);

        let dict = stream_report_dict(py, &report)?;
        dict.set_item("drained", drained)?;
        Ok(dict)
    }
}

/// Attempts at closing a stream. The server answers 503 until the stream's first analysis has
/// written its processed video, which can take a minute or more: 0.5, 1, 2, 4, 8, 16, then 30 s
/// apart, about three minutes in all.
const FINALIZE_ATTEMPTS: u32 = 12;

/// How long `finalize` (and the end of `run`) waits for a stream's spool to drain by default.
const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(600);

fn warn_py(py: Python<'_>, message: &str) -> PyResult<()> {
    warn!("[rheed_stream] {message}");
    PyErr::warn(
        py,
        &py.get_type::<PyRuntimeWarning>(),
        &CString::new(message.replace('\0', " ")).expect("NUL bytes were replaced"),
        1,
    )
}

/// The counts behind a stream's lost chunks, and the first few of them.
fn describe_losses(report: &StreamReport) -> String {
    const LISTED: usize = 5;
    let c = &report.counts;
    let mut text = format!(
        "{} dropped (spool full, disk low or write failed), {} refused by the backend, \
         {} discarded unsent",
        c.dropped, c.failed, c.purged
    );
    for loss in report.losses.iter().take(LISTED) {
        text.push_str(&format!("; chunk {}: {}", loss.chunk_idx, loss.reason));
    }
    if report.losses.len() > LISTED {
        text.push_str("; …");
    }
    text
}

fn stream_report_dict<'py>(py: Python<'py>, report: &StreamReport) -> PyResult<Bound<'py, PyDict>> {
    let c = &report.counts;
    let dict = PyDict::new(py);
    dict.set_item("chunks_spooled", c.spooled)?;
    dict.set_item("chunks_uploaded", c.uploaded)?;
    dict.set_item("chunks_dropped", c.dropped)?;
    dict.set_item("chunks_failed", c.failed)?;
    dict.set_item("chunks_purged", c.purged)?;
    dict.set_item("chunks_pending", report.pending)?;
    dict.set_item("chunks_not_uploaded", c.not_uploaded())?;
    dict.set_item(
        "lost",
        report
            .losses
            .iter()
            .map(|loss| (loss.chunk_idx, loss.reason.clone()))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("closed", report.closed.map(|reason| reason.to_string()))?;
    Ok(dict)
}

fn spool_status_dict<'py>(py: Python<'py>, status: &SpoolStatus) -> PyResult<Bound<'py, PyDict>> {
    let t = &status.totals;
    let dict = PyDict::new(py);
    dict.set_item("dir", status.root.to_string_lossy())?;
    dict.set_item("bytes", status.bytes)?;
    dict.set_item("max_bytes", status.max_bytes)?;
    dict.set_item("min_free_bytes", status.min_free_bytes)?;
    dict.set_item("free_bytes", status.free_bytes)?;
    dict.set_item("chunks_pending", status.pending)?;
    dict.set_item("chunks_spooled", t.spooled)?;
    dict.set_item("chunks_uploaded", t.uploaded)?;
    dict.set_item("chunks_dropped", t.dropped)?;
    dict.set_item("chunks_failed", t.failed)?;
    dict.set_item("chunks_purged", t.purged)?;
    let streams = PyDict::new(py);
    for (data_id, report) in &status.streams {
        streams.set_item(data_id, stream_report_dict(py, report)?)?;
    }
    dict.set_item("streams", streams)?;
    Ok(dict)
}

// Private and rust only access
impl RHEEDStreamer {
    fn end_url(&self, data_id: &str) -> String {
        format!("{}/rheed/stream/{data_id}/end", self.endpoint)
    }

    /// A host-clock time as reference UTC, with the correction applied (none when correction is off
    /// or the clock is not yet measured).
    fn reference_utc_ms(&self, host_ms: i64) -> (i64, Option<Correction>) {
        self.clock
            .as_ref()
            .map_or((host_ms, None), |clock| clock.correct_ms(host_ms))
    }

    /// Chunks of the stream that will not be uploaded so far.
    fn lost_chunks(&self, data_id: &str) -> u64 {
        self.spool
            .report(data_id)
            .map_or(0, |report| report.counts.not_uploaded())
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

    /// Encodes a chunk and writes it to the spool, with the GIL released. A chunk the spool
    /// drops is logged and counted there; only bad input is an error.
    #[allow(clippy::too_many_arguments)]
    fn spool_chunk(
        &self,
        py: Python<'_>,
        data_id: &str,
        chunk_idx: usize,
        metadata: FrameChunkMetadata,
        flat: Vec<u8>,
        n: usize,
        h: usize,
        w: usize,
    ) -> PyResult<()> {
        let t0 = Instant::now();
        let outcome = py
            .detach(|| {
                self.spool
                    .enqueue(data_id, chunk_idx, metadata, flat, n, h, w)
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;
        debug!(
            "[rheed_stream] chunk {chunk_idx}: {} in {:.2?}",
            match outcome {
                Enqueued::Spooled => "spooled",
                Enqueued::Dropped(_) => "dropped",
            },
            t0.elapsed()
        );
        Ok(())
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
