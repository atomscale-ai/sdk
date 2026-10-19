//! Disk spool between the caller and the network for RHEED shards.
//!
//! `push()` and `run()` stamp a chunk, encode its shard, write it here and return. They never
//! wait on uploads, so a slow or broken network cannot stall the caller's acquisition loop and
//! make later chunks late (their stamps are taken on entry). Upload workers send the oldest
//! entries first and delete an entry only once its upload is confirmed. A shard always lands at
//! the same key for its chunk, so uploading one again after a crash is harmless.
//!
//! Layout: `<root>/<data_id>/` holds `stream.json` (endpoint, creation and last-confirmed-upload
//! times), `.lock` (held while a streamer owns the stream) and, per chunk, `<stem>.zarr` (the
//! encoded shard) beside its `<stem>.json` sidecar. Each is written to a temp file and renamed,
//! the sidecar last, so an entry exists only once it is complete.
//!
//! Bounds: past `max_bytes` of spooled data, or when the disk would fall below `min_free_bytes`
//! free, the incoming chunk is dropped rather than blocking the caller; the backend sees it as a
//! missing shard. Entries are purged when their stream is finalized, when the stream has gone
//! [`ABANDON_AFTER`] without a confirmed upload (the backend has then closed it as
//! `stream_error`), or when they are older than `retention`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::future::Future;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::runtime::Handle;
use tokio::sync::{watch, Notify};
use tracing::{debug, info, warn};

use crate::retry::{FIRST_RETRY_DELAY, MAX_RETRY_DELAY};
use crate::upload::{package_to_zarr_bytes, FrameChunkMetadata};

/// Environment variable that overrides the spool's root directory.
pub const SPOOL_DIR_ENV: &str = "ATOMSCALE_SPOOL_DIR";
/// Default cap on spooled bytes: 20 GB, or 10% of the disk's free space if that is less.
const MAX_BYTES_CEILING: u64 = 20_000_000_000;
const MAX_BYTES_FREE_DIVISOR: u64 = 10;
/// Default free space the spool always leaves on its disk.
pub const DEFAULT_MIN_FREE_BYTES: u64 = 1_000_000_000;
/// Default age past which an entry that still has not uploaded is discarded.
pub const DEFAULT_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);
/// The backend closes a stream as `stream_error` once it has received no data for this long,
/// so entries of a stream with no confirmed upload in this time can no longer land.
pub const ABANDON_AFTER: Duration = Duration::from_secs(2 * 60 * 60);
/// Shards uploaded at once.
pub const DEFAULT_UPLOAD_WORKERS: usize = 8;

const SWEEP_INTERVAL: Duration = Duration::from_secs(60);
/// How often a stream's last-confirmed-upload time is persisted; well inside [`ABANDON_AFTER`].
const MARKER_REFRESH_MS: i64 = 60_000;
/// Lost chunks listed per stream in a report; the counts are always complete.
const MAX_LISTED_LOSSES: usize = 1_000;
/// Consecutive failed attempts before a retrying upload is logged at warn.
const WARN_AFTER_ATTEMPTS: u32 = 5;

const MARKER_FILE: &str = "stream.json";
const LOCK_FILE: &str = ".lock";
const SHARD_EXT: &str = "zarr";
const SIDECAR_EXT: &str = "json";
const TMP_EXT: &str = "tmp";
const FORMAT_VERSION: u32 = 1;

pub type FreeSpace = Arc<dyn Fn(&Path) -> io::Result<u64> + Send + Sync>;
pub type NowMs = Arc<dyn Fn() -> i64 + Send + Sync>;

/// Delay between attempts at one upload: doubles from `first` up to `max`.
#[derive(Clone, Copy, Debug)]
pub struct Backoff {
    pub first: Duration,
    pub max: Duration,
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            first: FIRST_RETRY_DELAY,
            max: MAX_RETRY_DELAY,
        }
    }
}

#[derive(Clone)]
pub struct SpoolConfig {
    pub root: PathBuf,
    /// The API endpoint uploads go to. Leftover entries are resumed only by a streamer of the
    /// same endpoint.
    pub endpoint: String,
    /// `None`: min(20 GB, 10% of the disk's free space when the spool opens).
    pub max_bytes: Option<u64>,
    pub min_free_bytes: u64,
    pub retention: Duration,
    pub abandon_after: Duration,
    pub workers: usize,
    pub backoff: Backoff,
    pub free_space: FreeSpace,
    pub now_ms: NowMs,
}

impl SpoolConfig {
    /// The defaults for `endpoint`, rooted at `root`, else `$ATOMSCALE_SPOOL_DIR`, else the
    /// per-user cache directory.
    pub fn new(endpoint: &str, root: Option<PathBuf>) -> Self {
        Self {
            root: root.unwrap_or_else(default_root),
            endpoint: endpoint.to_string(),
            max_bytes: None,
            min_free_bytes: DEFAULT_MIN_FREE_BYTES,
            retention: DEFAULT_RETENTION,
            abandon_after: ABANDON_AFTER,
            workers: DEFAULT_UPLOAD_WORKERS,
            backoff: Backoff::default(),
            free_space: Arc::new(|path| fs4::available_space(path)),
            now_ms: Arc::new(|| chrono::Utc::now().timestamp_millis()),
        }
    }
}

/// `$ATOMSCALE_SPOOL_DIR`, else `<user cache dir>/atomscale/rheed-spool`.
pub fn default_root() -> PathBuf {
    if let Some(dir) = std::env::var_os(SPOOL_DIR_ENV).filter(|dir| !dir.is_empty()) {
        return PathBuf::from(dir);
    }
    dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("atomscale")
        .join("rheed-spool")
}

/// One spooled chunk, as stored in its sidecar.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EntryMeta {
    pub version: u32,
    pub data_id: String,
    pub chunk_idx: usize,
    pub spooled_at_ms: i64,
    pub seq: u64,
    pub shard_len: u64,
    /// What the presign request carries, stamps included.
    pub metadata: FrameChunkMetadata,
}

type EntryKey = (i64, u64, String);

impl EntryMeta {
    /// Oldest first; stream identity also distinguishes entries from independent streamers.
    fn key(&self) -> EntryKey {
        (self.spooled_at_ms, self.seq, self.data_id.clone())
    }

    fn stem(&self) -> String {
        format!(
            "{:013}-{:010}-c{}",
            self.spooled_at_ms, self.seq, self.chunk_idx
        )
    }
}

/// Why an upload attempt failed, and whether another attempt could succeed.
#[derive(Debug)]
pub enum UploadError {
    Retry(anyhow::Error),
    GiveUp(anyhow::Error),
}

pub type UploadFuture<'a> = Pin<Box<dyn Future<Output = Result<(), UploadError>> + Send + 'a>>;

/// Sends one spooled shard. `Ok` means the upload is confirmed and the entry may be deleted.
pub trait Uploader: Send + Sync + 'static {
    fn upload<'a>(&'a self, entry: &'a EntryMeta, shard: Bytes) -> UploadFuture<'a>;
}

/// What `enqueue` did with a chunk.
#[derive(Debug, PartialEq)]
pub enum Enqueued {
    Spooled,
    /// Not spooled, so never uploaded; the reason is logged and counted.
    Dropped(String),
}

/// Chunks of a stream (or of the whole spool) by what became of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Written to the spool, including entries resumed from an earlier run.
    pub spooled: u64,
    pub uploaded: u64,
    /// Not spooled: the spool was full, the disk low, or the write failed.
    pub dropped: u64,
    /// Refused by the backend.
    pub failed: u64,
    /// Spooled but discarded before uploading: stream finalized or abandoned, or past retention.
    pub purged: u64,
}

impl Counts {
    pub fn not_uploaded(&self) -> u64 {
        self.dropped + self.failed + self.purged
    }

    fn add(&mut self, other: &Counts) {
        self.spooled += other.spooled;
        self.uploaded += other.uploaded;
        self.dropped += other.dropped;
        self.failed += other.failed;
        self.purged += other.purged;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fate {
    Uploaded,
    Failed,
    Purged,
}

impl Counts {
    fn record(&mut self, fate: Fate) {
        match fate {
            Fate::Uploaded => self.uploaded += 1,
            Fate::Failed => self.failed += 1,
            Fate::Purged => self.purged += 1,
        }
    }
}

/// A chunk that will never be uploaded, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Loss {
    pub chunk_idx: usize,
    pub reason: String,
}

#[derive(Clone, Debug, Default)]
pub struct StreamReport {
    pub counts: Counts,
    /// Spooled and not yet uploaded.
    pub pending: u64,
    /// The first [`MAX_LISTED_LOSSES`] chunks in `counts.not_uploaded()`.
    pub losses: Vec<Loss>,
    pub closed: Option<CloseReason>,
}

#[derive(Clone, Debug)]
pub struct SpoolStatus {
    pub root: PathBuf,
    pub bytes: u64,
    pub max_bytes: u64,
    pub min_free_bytes: u64,
    pub free_bytes: Option<u64>,
    pub pending: u64,
    pub totals: Counts,
    pub streams: Vec<(String, StreamReport)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloseReason {
    Finalized,
    Abandoned,
}

impl fmt::Display for CloseReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            CloseReason::Finalized => "finalized",
            CloseReason::Abandoned => "abandoned",
        })
    }
}

#[derive(Serialize, Deserialize)]
struct StreamMarker {
    version: u32,
    data_id: String,
    endpoint: String,
    created_at_ms: i64,
    last_upload_ms: i64,
    finalized: bool,
}

struct Entry {
    meta: Arc<EntryMeta>,
    dir: PathBuf,
    bytes: u64,
    in_flight: bool,
}

struct StreamState {
    dir: PathBuf,
    /// Held while this spool owns the stream, so no other streamer resumes its entries.
    lock: Option<File>,
    created_at_ms: i64,
    /// Creation time until the first confirmed upload.
    last_upload_ms: i64,
    marker_written_ms: i64,
    closed: Option<CloseReason>,
    counts: Counts,
    losses: Vec<Loss>,
    pending: watch::Sender<u64>,
    /// Cancels claimed uploads, including retry sleeps, when the stream closes.
    cancel: watch::Sender<bool>,
}

impl StreamState {
    fn new(dir: PathBuf, lock: File, created_at_ms: i64, last_upload_ms: i64, now: i64) -> Self {
        Self {
            dir,
            lock: Some(lock),
            created_at_ms,
            last_upload_ms,
            marker_written_ms: now,
            closed: None,
            counts: Counts::default(),
            losses: Vec::new(),
            pending: watch::channel(0).0,
            cancel: watch::channel(false).0,
        }
    }

    fn lose(&mut self, chunk_idx: usize, reason: String) {
        if self.losses.len() < MAX_LISTED_LOSSES {
            self.losses.push(Loss { chunk_idx, reason });
        }
    }

    fn report(&self) -> StreamReport {
        StreamReport {
            counts: self.counts,
            pending: *self.pending.borrow(),
            losses: self.losses.clone(),
            closed: self.closed,
        }
    }

    fn marker(&self, data_id: &str, endpoint: &str, finalized: bool) -> StreamMarker {
        StreamMarker {
            version: FORMAT_VERSION,
            data_id: data_id.to_string(),
            endpoint: endpoint.to_string(),
            created_at_ms: self.created_at_ms,
            last_upload_ms: self.last_upload_ms,
            finalized,
        }
    }
}

#[derive(Default)]
struct State {
    entries: BTreeMap<EntryKey, Entry>,
    streams: HashMap<String, StreamState>,
    /// Bytes of the entries in `entries`.
    bytes: u64,
    /// Bytes being written by `enqueue`, not yet in `entries`.
    reserved: u64,
    totals: Counts,
}

/// An entry a worker has claimed.
struct Job {
    key: EntryKey,
    meta: Arc<EntryMeta>,
    shard_path: PathBuf,
}

enum Staleness {
    /// Already removed (its stream was closed or it expired); nothing left to do.
    Gone,
    Expired,
    Abandoned,
}

pub struct Spool {
    config: SpoolConfig,
    max_bytes: u64,
    state: Mutex<State>,
    work: Notify,
    seq: AtomicU64,
}

impl Spool {
    /// Opens (creating if needed) the spool at `config.root` and takes over leftover entries:
    /// those of still-active streams of this endpoint are queued for upload; those of finalized
    /// or abandoned streams, and any past retention, are deleted. Call [`Spool::start`] to
    /// begin uploading.
    pub fn open(config: SpoolConfig) -> Result<Arc<Self>> {
        fs::create_dir_all(&config.root)
            .with_context(|| format!("cannot create spool directory {}", config.root.display()))?;
        let max_bytes = config.max_bytes.unwrap_or_else(|| {
            (config.free_space)(&config.root).map_or(MAX_BYTES_CEILING, |free| {
                (free / MAX_BYTES_FREE_DIVISOR).min(MAX_BYTES_CEILING)
            })
        });
        let spool = Arc::new(Self {
            config,
            max_bytes,
            state: Mutex::new(State::default()),
            work: Notify::new(),
            seq: AtomicU64::new(0),
        });
        spool.recover()?;
        Ok(spool)
    }

    /// Spawns the upload workers and the periodic sweep on `rt`.
    pub fn start(self: &Arc<Self>, rt: &Handle, uploader: Arc<dyn Uploader>) {
        for _ in 0..self.config.workers.max(1) {
            rt.spawn(self.clone().work(uploader.clone()));
        }
        let spool = self.clone();
        rt.spawn(async move {
            loop {
                tokio::time::sleep(SWEEP_INTERVAL).await;
                let spool = spool.clone();
                let _ = tokio::task::spawn_blocking(move || spool.sweep()).await;
            }
        });
    }

    pub fn root(&self) -> &Path {
        &self.config.root
    }

    /// Starts spooling for a new stream. Fails if another streamer already owns it.
    pub fn open_stream(&self, data_id: &str) -> Result<()> {
        if self.lock_state().streams.contains_key(data_id) {
            return Ok(());
        }
        let dir = self.stream_dir(data_id)?;
        fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create spool directory {}", dir.display()))?;
        let Some(lock) = lock_dir(&dir)? else {
            bail!("another streamer is already spooling stream {data_id}");
        };
        let now = self.now();
        let stream = StreamState::new(dir.clone(), lock, now, now, now);
        write_atomically(
            &dir.join(MARKER_FILE),
            &serde_json::to_vec(&stream.marker(data_id, &self.config.endpoint, false))?,
        )
        .with_context(|| format!("cannot write to spool directory {}", dir.display()))?;
        self.lock_state()
            .streams
            .insert(data_id.to_string(), stream);
        Ok(())
    }

    /// Whether this spool has (or had) the stream, including one resumed from an earlier run.
    pub fn has_stream(&self, data_id: &str) -> bool {
        self.lock_state().streams.contains_key(data_id)
    }

    /// Encodes a chunk's shard and writes it to the spool, or drops it when the spool is full or
    /// the disk low. Never waits on uploads. Fails only on bad input or an unknown stream.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue(
        &self,
        data_id: &str,
        chunk_idx: usize,
        metadata: FrameChunkMetadata,
        flat: Vec<u8>,
        n: usize,
        h: usize,
        w: usize,
    ) -> Result<Enqueued> {
        let shard = package_to_zarr_bytes(&flat, n, h, w)?;
        drop(flat);

        let meta = EntryMeta {
            version: FORMAT_VERSION,
            data_id: data_id.to_string(),
            chunk_idx,
            spooled_at_ms: self.now(),
            seq: self.seq.fetch_add(1, Ordering::Relaxed),
            shard_len: shard.len() as u64,
            metadata,
        };
        let sidecar = serde_json::to_vec(&meta)?;
        let need = (shard.len() + sidecar.len()) as u64;

        let dir = {
            let mut state = self.lock_state();
            let Some(stream) = state.streams.get(data_id) else {
                bail!("no spool for stream {data_id}; call initialize(...) first");
            };
            if let Some(reason) = stream.closed {
                return Ok(self.drop_chunk(
                    &mut state,
                    data_id,
                    chunk_idx,
                    format!("stream was {reason}"),
                ));
            }
            let dir = stream.dir.clone();
            let used = state.bytes + state.reserved;
            if used + need > self.max_bytes {
                return Ok(self.drop_chunk(
                    &mut state,
                    data_id,
                    chunk_idx,
                    format!(
                        "spool is full ({used} of {} bytes waiting to upload)",
                        self.max_bytes
                    ),
                ));
            }
            // An unmeasurable disk is not a reason to lose data; the byte cap still holds.
            if let Ok(free) = (self.config.free_space)(&self.config.root) {
                let free = free.saturating_sub(state.reserved);
                if free < need + self.config.min_free_bytes {
                    return Ok(self.drop_chunk(
                        &mut state,
                        data_id,
                        chunk_idx,
                        format!(
                            "disk has {free} bytes free, and the spool leaves at least {}",
                            self.config.min_free_bytes
                        ),
                    ));
                }
            }
            state.reserved += need;
            dir
        };

        let written = write_entry(&dir, &meta, &shard, &sidecar);

        let mut state = self.lock_state();
        state.reserved -= need;
        if let Err(error) = written {
            return Ok(self.drop_chunk(
                &mut state,
                data_id,
                chunk_idx,
                format!("could not write to the spool: {error:#}"),
            ));
        }
        let Some(stream) = state
            .streams
            .get_mut(data_id)
            .filter(|s| s.closed.is_none())
        else {
            // Closed while the entry was being written.
            drop(state);
            remove_entry_files(&dir, &meta);
            let mut state = self.lock_state();
            return Ok(self.drop_chunk(
                &mut state,
                data_id,
                chunk_idx,
                "stream was closed".to_string(),
            ));
        };
        stream.counts.spooled += 1;
        stream.pending.send_modify(|pending| *pending += 1);
        state.totals.spooled += 1;
        state.bytes += need;
        state.entries.insert(
            meta.key(),
            Entry {
                meta: Arc::new(meta),
                dir,
                bytes: need,
                in_flight: false,
            },
        );
        drop(state);
        self.work.notify_one();
        Ok(Enqueued::Spooled)
    }

    /// Waits until every spooled chunk of the stream has been uploaded or discarded, or until
    /// `timeout`. Returns whether it drained.
    pub async fn drain(&self, data_id: &str, timeout: Duration) -> bool {
        let pending = self
            .lock_state()
            .streams
            .get(data_id)
            .map(|stream| stream.pending.subscribe());
        let Some(mut pending) = pending else {
            return true;
        };
        // A closed channel means the stream was forgotten: nothing is left to wait for.
        let drained = tokio::time::timeout(timeout, pending.wait_for(|pending| *pending == 0))
            .await
            .is_ok();
        drained
    }

    /// Stops spooling the stream: discards its remaining entries (counted as purged) and
    /// deletes its directory. Returns the stream's report; closing twice is harmless.
    pub fn close_stream(&self, data_id: &str, reason: CloseReason) -> Option<StreamReport> {
        let (dir, lock, marker) = {
            let mut state = self.lock_state();
            let state = &mut *state;
            let stream = state.streams.get_mut(data_id)?;
            if stream.closed.is_some() {
                return Some(stream.report());
            }
            stream.closed = Some(reason);
            stream.cancel.send_replace(true);
            let keys: Vec<EntryKey> = state
                .entries
                .iter()
                .filter(|(_, entry)| entry.meta.data_id == data_id)
                .map(|(key, _)| key.clone())
                .collect();
            for key in keys {
                let entry = state.entries.remove(&key).expect("key was just listed");
                state.bytes -= entry.bytes;
                stream.counts.purged += 1;
                state.totals.purged += 1;
                stream.lose(
                    entry.meta.chunk_idx,
                    format!("not uploaded before the stream was {reason}"),
                );
            }
            stream.pending.send_replace(0);
            (
                stream.dir.clone(),
                stream.lock.take(),
                stream.marker(data_id, &self.config.endpoint, true),
            )
        };
        // Mark it finalized first, so a crash part-way through deletion still discards it.
        if let Ok(marker) = serde_json::to_vec(&marker) {
            let _ = write_atomically(&dir.join(MARKER_FILE), &marker);
        }
        drop(lock);
        remove_dir(&dir);
        self.report(data_id)
    }

    /// Drops the record of a closed stream; its counts stay in the spool totals.
    pub fn forget_stream(&self, data_id: &str) {
        let mut state = self.lock_state();
        if state
            .streams
            .get(data_id)
            .is_some_and(|stream| stream.closed.is_some())
        {
            state.streams.remove(data_id);
        }
    }

    pub fn report(&self, data_id: &str) -> Option<StreamReport> {
        self.lock_state()
            .streams
            .get(data_id)
            .map(StreamState::report)
    }

    pub fn status(&self) -> SpoolStatus {
        let free_bytes = (self.config.free_space)(&self.config.root).ok();
        let state = self.lock_state();
        let mut streams: Vec<(String, StreamReport)> = state
            .streams
            .iter()
            .map(|(data_id, stream)| (data_id.clone(), stream.report()))
            .collect();
        streams.sort_by(|a, b| a.0.cmp(&b.0));
        SpoolStatus {
            root: self.config.root.clone(),
            bytes: state.bytes,
            max_bytes: self.max_bytes,
            min_free_bytes: self.config.min_free_bytes,
            free_bytes,
            pending: state.entries.len() as u64,
            totals: state.totals,
            streams,
        }
    }

    /// Closes streams the backend has given up on and discards entries past retention. Runs
    /// every [`SWEEP_INTERVAL`] once started.
    pub fn sweep(&self) {
        let now = self.now();
        let abandon_after = self.config.abandon_after.as_millis() as i64;
        let retention = self.config.retention.as_millis() as i64;
        let (abandoned, expired): (Vec<String>, Vec<EntryKey>) = {
            let state = self.lock_state();
            (
                state
                    .streams
                    .iter()
                    .filter(|(_, s)| s.closed.is_none() && now - s.last_upload_ms > abandon_after)
                    .map(|(data_id, _)| data_id.clone())
                    .collect(),
                state
                    .entries
                    .iter()
                    .filter(|(_, e)| !e.in_flight && now - e.meta.spooled_at_ms > retention)
                    .map(|(key, _)| key.clone())
                    .collect(),
            )
        };
        for data_id in abandoned {
            self.abandon(&data_id);
        }
        for key in expired {
            self.finish(key, Fate::Purged, self.retention_reason());
        }
    }

    // ---- internals ----

    fn lock_state(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("spool mutex poisoned")
    }

    fn now(&self) -> i64 {
        (self.config.now_ms)()
    }

    fn stream_dir(&self, data_id: &str) -> Result<PathBuf> {
        let safe = !data_id.is_empty()
            && data_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if !safe {
            bail!("data_id {data_id:?} cannot name a spool directory");
        }
        Ok(self.config.root.join(data_id))
    }

    fn retention_reason(&self) -> String {
        format!(
            "not uploaded within the {} h retention limit",
            self.config.retention.as_secs_f64() / 3600.0
        )
    }

    fn drop_chunk(
        &self,
        state: &mut State,
        data_id: &str,
        chunk_idx: usize,
        reason: String,
    ) -> Enqueued {
        state.totals.dropped += 1;
        let dropped = match state.streams.get_mut(data_id) {
            Some(stream) => {
                stream.counts.dropped += 1;
                stream.lose(chunk_idx, format!("dropped: {reason}"));
                stream.counts.dropped
            }
            None => 1,
        };
        warn!(
            "[rheed_spool] dropped chunk {chunk_idx} of stream {data_id}: {reason} \
             ({dropped} dropped from this stream so far)"
        );
        Enqueued::Dropped(reason)
    }

    fn abandon(&self, data_id: &str) {
        let pending = self.report(data_id).map_or(0, |report| report.pending);
        warn!(
            "[rheed_spool] stream {data_id} has had no confirmed upload for {} h, so the backend has \
             closed it; discarding its {pending} spooled chunk(s)",
            self.config.abandon_after.as_secs_f64() / 3600.0
        );
        self.close_stream(data_id, CloseReason::Abandoned);
    }

    /// Takes the oldest entry no worker is uploading.
    fn claim(&self) -> Option<(Job, watch::Receiver<bool>)> {
        let mut state = self.lock_state();
        let (key, entry) = state
            .entries
            .iter_mut()
            .find(|(_, entry)| !entry.in_flight)?;
        entry.in_flight = true;
        let job = Job {
            key: key.clone(),
            meta: entry.meta.clone(),
            shard_path: entry_path(&entry.dir, &entry.meta, SHARD_EXT),
        };
        let cancel = state.streams.get(&job.meta.data_id)?.cancel.subscribe();
        Some((job, cancel))
    }

    /// Returns a claimed entry to the queue untouched.
    fn release(&self, key: EntryKey) {
        if let Some(entry) = self.lock_state().entries.get_mut(&key) {
            entry.in_flight = false;
        }
    }

    async fn work(self: Arc<Self>, uploader: Arc<dyn Uploader>) {
        loop {
            match self.claim() {
                Some((job, mut cancel)) => {
                    tokio::select! {
                        biased;
                        _ = cancel.wait_for(|closed| *closed) => {}
                        _ = self.upload(job, uploader.as_ref()) => {}
                    }
                }
                None => self.work.notified().await,
            }
        }
    }

    async fn upload(&self, job: Job, uploader: &dyn Uploader) {
        let path = job.shard_path.clone();
        let shard = match tokio::task::spawn_blocking(move || fs::read(path)).await {
            Ok(Ok(shard)) => Bytes::from(shard),
            Ok(Err(error)) if error.kind() == io::ErrorKind::NotFound => {
                // Purged while it waited.
                self.finish(
                    job.key,
                    Fate::Failed,
                    format!("spooled shard is missing: {error}"),
                );
                return;
            }
            Ok(Err(error)) => {
                self.finish(
                    job.key,
                    Fate::Failed,
                    format!("spooled shard unreadable: {error}"),
                );
                return;
            }
            Err(error) => {
                // The runtime is shutting down (or the read panicked): leave the entry on disk
                // for the next run rather than counting it lost.
                debug!("[rheed_spool] shard read did not complete: {error}");
                self.release(job.key);
                return;
            }
        };
        let chunk_idx = job.meta.chunk_idx;
        let data_id = &job.meta.data_id;
        let mut delay = self.config.backoff.first;
        let mut attempt = 0_u32;
        loop {
            match self.staleness(&job) {
                None => {}
                Some(Staleness::Gone) => return,
                Some(Staleness::Expired) => {
                    self.finish(job.key, Fate::Purged, self.retention_reason());
                    return;
                }
                Some(Staleness::Abandoned) => {
                    self.abandon(data_id);
                    return;
                }
            }
            attempt += 1;
            match uploader.upload(&job.meta, shard.clone()).await {
                Ok(()) => {
                    debug!("[rheed_spool] chunk {chunk_idx} of stream {data_id} uploaded (attempt {attempt})");
                    self.finish(job.key, Fate::Uploaded, String::new());
                    return;
                }
                Err(UploadError::GiveUp(error)) => {
                    warn!("[rheed_spool] chunk {chunk_idx} of stream {data_id} was refused: {error:#}");
                    self.finish(
                        job.key,
                        Fate::Failed,
                        format!("refused by the backend: {error:#}"),
                    );
                    return;
                }
                Err(UploadError::Retry(error)) => {
                    if attempt == WARN_AFTER_ATTEMPTS
                        || attempt.is_multiple_of(4 * WARN_AFTER_ATTEMPTS)
                    {
                        warn!(
                            "[rheed_spool] chunk {chunk_idx} of stream {data_id} still not uploaded after \
                             {attempt} attempts; it stays spooled and keeps retrying: {error:#}"
                        );
                    } else {
                        debug!("[rheed_spool] chunk {chunk_idx} of stream {data_id} attempt {attempt} failed, retrying in {delay:?}: {error:#}");
                    }
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(self.config.backoff.max);
                }
            }
        }
    }

    fn staleness(&self, job: &Job) -> Option<Staleness> {
        let now = self.now();
        let state = self.lock_state();
        if !state.entries.contains_key(&job.key) {
            return Some(Staleness::Gone);
        }
        let stream = state.streams.get(&job.meta.data_id)?;
        if stream.closed.is_some() {
            return Some(Staleness::Gone);
        }
        if now - stream.last_upload_ms > self.config.abandon_after.as_millis() as i64 {
            return Some(Staleness::Abandoned);
        }
        if now - job.meta.spooled_at_ms > self.config.retention.as_millis() as i64 {
            return Some(Staleness::Expired);
        }
        None
    }

    /// Removes an entry from the index and from disk, then counts what became of it, so a
    /// drain only completes once the stream's files are gone. A no-op if it was already removed.
    fn finish(&self, key: EntryKey, fate: Fate, reason: String) {
        let Some(entry) = self.lock_state().entries.remove(&key) else {
            return;
        };
        if fate == Fate::Purged {
            info!(
                "[rheed_spool] discarded chunk {} of stream {}: {reason}",
                entry.meta.chunk_idx, entry.meta.data_id
            );
        }
        remove_entry_files(&entry.dir, &entry.meta);

        let now = self.now();
        let marker = {
            let mut state = self.lock_state();
            let state = &mut *state;
            state.bytes -= entry.bytes;
            state.totals.record(fate);
            let mut marker = None;
            if let Some(stream) = state.streams.get_mut(&entry.meta.data_id) {
                stream.counts.record(fate);
                // A close in the meantime has already zeroed it.
                stream
                    .pending
                    .send_modify(|pending| *pending = pending.saturating_sub(1));
                if fate == Fate::Uploaded {
                    stream.last_upload_ms = now;
                    if stream.closed.is_none()
                        && now - stream.marker_written_ms >= MARKER_REFRESH_MS
                    {
                        stream.marker_written_ms = now;
                        marker = Some((
                            stream.dir.join(MARKER_FILE),
                            stream.marker(&entry.meta.data_id, &self.config.endpoint, false),
                        ));
                    }
                } else {
                    stream.lose(entry.meta.chunk_idx, reason);
                }
            }
            marker
        };
        if let Some((path, marker)) = marker {
            if let Ok(marker) = serde_json::to_vec(&marker) {
                // Best effort: the directory may have just been removed by a close.
                let _ = write_atomically(&path, &marker);
            }
        }
    }

    /// Takes over what earlier runs left in the spool directory.
    fn recover(&self) -> Result<()> {
        let now = self.now();
        let dirs = fs::read_dir(&self.config.root).with_context(|| {
            format!("cannot read spool directory {}", self.config.root.display())
        })?;
        for dir in dirs.flatten() {
            let path = dir.path();
            if !path.is_dir() {
                continue;
            }
            if let Err(error) = self.recover_stream(&path, now) {
                warn!(
                    "[rheed_spool] skipping spool directory {}: {error:#}",
                    path.display()
                );
            }
        }
        Ok(())
    }

    fn recover_stream(&self, dir: &Path, now: i64) -> Result<()> {
        let Some(lock) = lock_dir(dir)? else {
            debug!(
                "[rheed_spool] {} belongs to a running streamer",
                dir.display()
            );
            return Ok(());
        };
        let marker = fs::read(dir.join(MARKER_FILE))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<StreamMarker>(&bytes).ok());
        let Some(marker) = marker else {
            info!(
                "[rheed_spool] removing {}: it has no readable stream record",
                dir.display()
            );
            drop(lock);
            remove_dir(dir);
            return Ok(());
        };
        if marker.endpoint != self.config.endpoint {
            // Leave all cleanup to a streamer of this endpoint.
            return Ok(());
        }
        let data_id = marker.data_id.clone();
        if marker.finalized {
            drop(lock);
            remove_dir(dir);
            return Ok(());
        }

        let mut counts = Counts::default();
        let retention = self.config.retention.as_millis() as i64;
        let (expired, fresh): (Vec<_>, Vec<_>) = scan_entries(dir, &data_id)
            .into_iter()
            .partition(|(meta, _)| now - meta.spooled_at_ms > retention);
        let mut losses = Vec::new();
        for (meta, _) in &expired {
            info!(
                "[rheed_spool] discarded chunk {} of stream {data_id}: {}",
                meta.chunk_idx,
                self.retention_reason()
            );
            remove_entry_files(dir, meta);
            losses.push(Loss {
                chunk_idx: meta.chunk_idx,
                reason: self.retention_reason(),
            });
        }
        counts.spooled = (expired.len() + fresh.len()) as u64;
        counts.purged = expired.len() as u64;

        let last_activity = marker.created_at_ms.max(marker.last_upload_ms);
        if now - last_activity > self.config.abandon_after.as_millis() as i64 {
            if !fresh.is_empty() {
                warn!(
                    "[rheed_spool] stream {data_id} has had no confirmed upload for over {} h, so the \
                     backend has closed it; discarding its {} spooled chunk(s)",
                    self.config.abandon_after.as_secs_f64() / 3600.0,
                    fresh.len()
                );
            }
            counts.purged = counts.spooled;
            drop(lock);
            remove_dir(dir);
            self.lock_state().totals.add(&counts);
            return Ok(());
        }

        if !fresh.is_empty() {
            info!(
                "[rheed_spool] resuming upload of {} chunk(s) of stream {data_id} left by an earlier run",
                fresh.len()
            );
        }
        let mut stream = StreamState::new(
            dir.to_path_buf(),
            lock,
            marker.created_at_ms,
            last_activity,
            now,
        );
        let mut state = self.lock_state();
        let state = &mut *state;
        let pending = fresh.len() as u64;
        for (meta, bytes) in fresh {
            self.seq.fetch_max(meta.seq + 1, Ordering::Relaxed);
            state.bytes += bytes;
            state.entries.insert(
                meta.key(),
                Entry {
                    meta: Arc::new(meta),
                    dir: dir.to_path_buf(),
                    bytes,
                    in_flight: false,
                },
            );
        }
        stream.counts = counts;
        losses.truncate(MAX_LISTED_LOSSES);
        stream.losses = losses;
        stream.pending.send_replace(pending);
        state.totals.add(&counts);
        state.streams.insert(data_id, stream);
        Ok(())
    }
}

fn entry_path(dir: &Path, meta: &EntryMeta, ext: &str) -> PathBuf {
    dir.join(format!("{}.{ext}", meta.stem()))
}

/// Locks a stream directory for this process; `None` if another live streamer holds it. The
/// lock is released when the file is closed, including when the process dies.
fn lock_dir(dir: &Path) -> io::Result<Option<File>> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir.join(LOCK_FILE))?;
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(error)) => Err(error),
    }
}

fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension(format!(
        "{}.{TMP_EXT}",
        path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
    ));
    let written = File::create(&tmp)
        .and_then(|mut file| file.write_all(bytes))
        .and_then(|()| fs::rename(&tmp, path));
    if written.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    written
}

/// The shard, then its sidecar: an entry is complete once its sidecar exists.
fn write_entry(dir: &Path, meta: &EntryMeta, shard: &[u8], sidecar: &[u8]) -> io::Result<()> {
    let shard_path = entry_path(dir, meta, SHARD_EXT);
    write_atomically(&shard_path, shard)?;
    write_atomically(&entry_path(dir, meta, SIDECAR_EXT), sidecar).inspect_err(|_| {
        let _ = fs::remove_file(&shard_path);
    })
}

/// The sidecar first, so a crash part-way leaves a bare shard, which recovery deletes.
fn remove_entry_files(dir: &Path, meta: &EntryMeta) {
    for ext in [SIDECAR_EXT, SHARD_EXT] {
        if let Err(error) = fs::remove_file(entry_path(dir, meta, ext)) {
            if error.kind() != io::ErrorKind::NotFound {
                warn!("[rheed_spool] could not delete a spooled file: {error}");
            }
        }
    }
}

fn remove_dir(dir: &Path) {
    if let Err(error) = fs::remove_dir_all(dir) {
        if error.kind() != io::ErrorKind::NotFound {
            warn!("[rheed_spool] could not remove {}: {error}", dir.display());
        }
    }
}

/// The complete entries in a stream directory, with their size on disk. Deletes temp files from
/// interrupted writes and any shard without a valid sidecar (or sidecar without its shard).
fn scan_entries(dir: &Path, data_id: &str) -> Vec<(EntryMeta, u64)> {
    let Ok(files) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut entries = Vec::new();
    let mut shards = Vec::new();
    let mut kept = HashSet::new();
    for file in files.flatten() {
        let path = file.path();
        let name = file.file_name().to_string_lossy().into_owned();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default();
        if ext == TMP_EXT {
            let _ = fs::remove_file(&path);
        } else if ext == SHARD_EXT {
            shards.push(path);
        } else if ext == SIDECAR_EXT && name != MARKER_FILE {
            let meta = fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<EntryMeta>(&bytes).ok())
                .filter(|meta| meta.version == FORMAT_VERSION && meta.data_id == data_id);
            let shard_len = meta
                .as_ref()
                .and_then(|meta| fs::metadata(entry_path(dir, meta, SHARD_EXT)).ok())
                .map(|m| m.len());
            match (meta, shard_len) {
                (Some(meta), Some(len)) if len == meta.shard_len => {
                    let sidecar_len = file.metadata().map_or(0, |m| m.len());
                    kept.insert(entry_path(dir, &meta, SHARD_EXT));
                    entries.push((meta, len + sidecar_len));
                }
                _ => {
                    debug!("[rheed_spool] removing incomplete entry {}", path.display());
                    let _ = fs::remove_file(&path);
                }
            }
        }
    }
    for shard in shards {
        if !kept.contains(&shard) {
            let _ = fs::remove_file(&shard);
        }
    }
    entries.sort_by_key(|(meta, _)| meta.key());
    entries
}

#[cfg(test)]
mod tests;
