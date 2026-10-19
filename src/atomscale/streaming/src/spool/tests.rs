use super::*;

use std::sync::atomic::{AtomicI64, AtomicU32, AtomicUsize};
use std::time::Instant;

use anyhow::anyhow;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;

use crate::retry::HttpStatusError;
use crate::upload::{package_to_zarr_bytes, PutTimeout, ShardUploader};

const ENDPOINT: &str = "https://api.test";
const T0: i64 = 1_792_000_000_000;
const HOUR_MS: i64 = 60 * 60 * 1000;
const N: usize = 16;
const H: usize = 32;
const W: usize = 32;

// ---- fixtures ----

struct Fixture {
    dir: TempDir,
    clock: Arc<AtomicI64>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            dir: TempDir::new().unwrap(),
            clock: Arc::new(AtomicI64::new(T0)),
        }
    }

    fn config(&self) -> SpoolConfig {
        let mut config = SpoolConfig::new(ENDPOINT, Some(self.dir.path().to_path_buf()));
        config.max_bytes = Some(1 << 40);
        config.free_space = Arc::new(|_| Ok(1 << 50));
        let clock = self.clock.clone();
        config.now_ms = Arc::new(move || clock.load(Ordering::SeqCst));
        config.backoff = Backoff {
            first: Duration::from_millis(1),
            max: Duration::from_millis(5),
        };
        config
    }

    fn open(&self) -> Arc<Spool> {
        Spool::open(self.config()).unwrap()
    }

    fn advance(&self, ms: i64) {
        self.clock.fetch_add(ms, Ordering::SeqCst);
    }

    fn stream_dir(&self, data_id: &str) -> PathBuf {
        self.dir.path().join(data_id)
    }

    /// Complete entries (sidecars) on disk for the stream.
    fn entries_on_disk(&self, data_id: &str) -> usize {
        fs::read_dir(self.stream_dir(data_id))
            .map(|files| {
                files
                    .flatten()
                    .filter(|f| {
                        let name = f.file_name().to_string_lossy().into_owned();
                        name.ends_with(".json") && name != MARKER_FILE
                    })
                    .count()
            })
            .unwrap_or(0)
    }
}

fn runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

fn metadata(data_id: &str, chunk_idx: usize) -> FrameChunkMetadata {
    let start = T0 + chunk_idx as i64 * 2_000;
    FrameChunkMetadata {
        data_id: data_id.to_string(),
        data_stream: "rheed".to_string(),
        is_stream: 1,
        is_rotating: 0,
        raw_frame_rate: 8.0,
        avg_frame_rate: 8.0,
        chunk_size: N,
        dims: format!("{N},{H},{W}"),
        start_unix_ms_utc: start,
        end_unix_ms_utc: start + 2_000,
        clock_offset_ms: Some(12),
        clock_uncertainty_ms: Some(3.5),
    }
}

fn frames(chunk_idx: usize) -> Vec<u8> {
    (0..N * H * W)
        .map(|i| ((i / W) as u8).wrapping_add(chunk_idx as u8))
        .collect()
}

fn push(spool: &Spool, data_id: &str, chunk_idx: usize) -> Enqueued {
    spool
        .enqueue(
            data_id,
            chunk_idx,
            metadata(data_id, chunk_idx),
            frames(chunk_idx),
            N,
            H,
            W,
        )
        .unwrap()
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn s3_request_timeout() -> UploadError {
    UploadError::Retry(
        anyhow::Error::new(HttpStatusError {
            status: reqwest::StatusCode::BAD_REQUEST,
            body: "<Error><Code>RequestTimeout</Code><Message>Your socket connection to the \
                   server was not read from or written to within the timeout period.</Message></Error>"
                .to_string(),
        })
        .context("PUT bytes failed"),
    )
}

// ---- uploaders ----

/// Never finishes an upload, like a network that froze.
#[derive(Default)]
struct Stalled {
    calls: AtomicUsize,
}

impl Uploader for Stalled {
    fn upload<'a>(&'a self, _: &'a EntryMeta, _: Bytes) -> UploadFuture<'a> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(std::future::pending())
    }
}

/// Confirms every upload, after `delay`, and records it.
#[derive(Default)]
struct Recorder {
    delay: Duration,
    uploaded: Mutex<Vec<(String, usize, FrameChunkMetadata, usize)>>,
}

impl Recorder {
    fn chunks(&self, data_id: &str) -> Vec<usize> {
        self.uploaded
            .lock()
            .unwrap()
            .iter()
            .filter(|(id, ..)| id == data_id)
            .map(|(_, chunk_idx, ..)| *chunk_idx)
            .collect()
    }
}

impl Uploader for Recorder {
    fn upload<'a>(&'a self, entry: &'a EntryMeta, shard: Bytes) -> UploadFuture<'a> {
        Box::pin(async move {
            tokio::time::sleep(self.delay).await;
            self.uploaded.lock().unwrap().push((
                entry.data_id.clone(),
                entry.chunk_idx,
                entry.metadata.clone(),
                shard.len(),
            ));
            Ok(())
        })
    }
}

/// Holds each upload until the test lets one through.
struct Gate {
    open: Semaphore,
}

impl Uploader for Gate {
    fn upload<'a>(&'a self, _: &'a EntryMeta, _: Bytes) -> UploadFuture<'a> {
        Box::pin(async move {
            self.open.acquire().await.unwrap().forget();
            Ok(())
        })
    }
}

/// Refuses chunk 0 outright; stalls the rest.
struct RefuseFirst;

impl Uploader for RefuseFirst {
    fn upload<'a>(&'a self, entry: &'a EntryMeta, _: Bytes) -> UploadFuture<'a> {
        let chunk_idx = entry.chunk_idx;
        Box::pin(async move {
            if chunk_idx == 0 {
                return Err(UploadError::GiveUp(anyhow!("HTTP 404: stream not found")));
            }
            std::future::pending().await
        })
    }
}

// ---- push never blocks ----

#[test]
fn push_never_waits_for_stalled_uploads() {
    let fx = Fixture::new();
    let rt = runtime();
    let spool = fx.open();
    let stalled = Arc::new(Stalled::default());
    spool.start(rt.handle(), stalled.clone());
    spool.open_stream("s1").unwrap();

    let mut slowest = Duration::ZERO;
    for chunk_idx in 0..40 {
        let t0 = Instant::now();
        assert_eq!(push(&spool, "s1", chunk_idx), Enqueued::Spooled);
        slowest = slowest.max(t0.elapsed());
    }

    // Every worker is stuck mid-upload, and every chunk was still accepted promptly.
    wait_until("every worker to be stuck", || {
        stalled.calls.load(Ordering::SeqCst) == DEFAULT_UPLOAD_WORKERS
    });
    assert!(
        slowest < Duration::from_millis(500),
        "a push took {slowest:?}"
    );
    let status = spool.status();
    assert_eq!(status.pending, 40);
    assert_eq!(status.totals.spooled, 40);
    assert_eq!(status.totals.dropped, 0);
    assert_eq!(fx.entries_on_disk("s1"), 40);
}

// ---- bounds ----

/// Bytes one test chunk takes in the spool.
fn entry_bytes() -> u64 {
    let fx = Fixture::new();
    let spool = fx.open();
    spool.open_stream("s1").unwrap();
    assert_eq!(push(&spool, "s1", 0), Enqueued::Spooled);
    spool.status().bytes
}

#[test]
fn a_full_spool_drops_the_new_chunk_instead_of_blocking() {
    let size = entry_bytes();
    let fx = Fixture::new();
    let rt = runtime();
    let mut config = fx.config();
    config.max_bytes = Some(2 * size + size / 2);
    let spool = Spool::open(config).unwrap();
    let gate = Arc::new(Gate {
        open: Semaphore::new(0),
    });
    spool.start(rt.handle(), gate.clone());
    spool.open_stream("s1").unwrap();

    assert_eq!(push(&spool, "s1", 0), Enqueued::Spooled);
    assert_eq!(push(&spool, "s1", 1), Enqueued::Spooled);
    for chunk_idx in 2..5 {
        let t0 = Instant::now();
        match push(&spool, "s1", chunk_idx) {
            Enqueued::Dropped(reason) => assert!(reason.contains("spool is full"), "{reason}"),
            Enqueued::Spooled => panic!("chunk {chunk_idx} should not fit"),
        }
        assert!(t0.elapsed() < Duration::from_millis(500));
    }
    let report = spool.report("s1").unwrap();
    assert_eq!(report.counts.dropped, 3);
    assert_eq!(report.pending, 2);
    assert_eq!(
        report
            .losses
            .iter()
            .map(|l| l.chunk_idx)
            .collect::<Vec<_>>(),
        vec![2, 3, 4]
    );
    assert_eq!(spool.status().totals.dropped, 3);
    assert_eq!(fx.entries_on_disk("s1"), 2);

    // Once uploads catch up there is room again.
    gate.open.add_permits(2);
    assert!(rt.block_on(spool.drain("s1", Duration::from_secs(10))));
    assert_eq!(spool.status().bytes, 0);
    assert_eq!(push(&spool, "s1", 5), Enqueued::Spooled);
}

#[test]
fn a_low_disk_drops_the_new_chunk() {
    let fx = Fixture::new();
    let free = Arc::new(AtomicU64::new(10_000_000_000));
    let mut config = fx.config();
    config.min_free_bytes = 1_000_000_000;
    let reading = free.clone();
    config.free_space = Arc::new(move |_| Ok(reading.load(Ordering::SeqCst)));
    let spool = Spool::open(config).unwrap();
    spool.open_stream("s1").unwrap();

    assert_eq!(push(&spool, "s1", 0), Enqueued::Spooled);
    // Writing the next chunk would take the disk below the 1 GB floor.
    free.store(1_000_000_010, Ordering::SeqCst);
    match push(&spool, "s1", 1) {
        Enqueued::Dropped(reason) => assert!(reason.contains("disk has"), "{reason}"),
        Enqueued::Spooled => panic!("chunk 1 should have been dropped"),
    }
    assert_eq!(fx.entries_on_disk("s1"), 1);
    let status = spool.status();
    assert_eq!(status.totals.dropped, 1);
    assert_eq!(status.free_bytes, Some(1_000_000_010));
    assert_eq!(spool.report("s1").unwrap().losses[0].chunk_idx, 1);
}

#[test]
fn the_default_cap_is_a_tenth_of_free_space_up_to_20_gb() {
    let fx = Fixture::new();
    let mut config = fx.config();
    config.max_bytes = None;
    config.free_space = Arc::new(|_| Ok(50_000_000_000));
    assert_eq!(
        Spool::open(config.clone()).unwrap().status().max_bytes,
        5_000_000_000
    );
    config.free_space = Arc::new(|_| Ok(1_000_000_000_000));
    assert_eq!(
        Spool::open(config).unwrap().status().max_bytes,
        20_000_000_000
    );
}

// ---- ordering ----

#[test]
fn uploads_go_oldest_first_across_streams() {
    let fx = Fixture::new();
    let rt = runtime();
    let mut config = fx.config();
    config.workers = 1;
    let spool = Spool::open(config).unwrap();
    spool.open_stream("a").unwrap();
    spool.open_stream("b").unwrap();
    for chunk_idx in 0..4 {
        push(&spool, "a", chunk_idx);
        fx.advance(1);
        push(&spool, "b", chunk_idx);
        fx.advance(1);
    }
    let recorder = Arc::new(Recorder::default());
    spool.start(rt.handle(), recorder.clone());
    assert!(rt.block_on(spool.drain("a", Duration::from_secs(10))));
    assert!(rt.block_on(spool.drain("b", Duration::from_secs(10))));

    let order: Vec<(String, usize)> = recorder
        .uploaded
        .lock()
        .unwrap()
        .iter()
        .map(|(id, chunk_idx, ..)| (id.clone(), *chunk_idx))
        .collect();
    let expected: Vec<(String, usize)> = (0..4)
        .flat_map(|i| [("a".to_string(), i), ("b".to_string(), i)])
        .collect();
    assert_eq!(order, expected);
}

// ---- crash and restart ----

#[test]
fn entries_left_by_a_crash_are_uploaded_on_restart() {
    let fx = Fixture::new();
    {
        let rt = runtime();
        let spool = fx.open();
        spool.start(rt.handle(), Arc::new(Stalled::default()));
        spool.open_stream("s1").unwrap();
        for chunk_idx in 0..3 {
            push(&spool, "s1", chunk_idx);
        }
        // A write the crash interrupted, and a shard whose sidecar never landed.
        let dir = fx.stream_dir("s1");
        fs::write(
            dir.join("0001792000000000-0000000099-c99.zarr.tmp"),
            b"partial",
        )
        .unwrap();
        fs::write(dir.join("0001792000000000-0000000098-c98.zarr"), b"orphan").unwrap();
        // The crash: the runtime and its upload workers vanish with nothing cleaned up.
        drop(rt);
        drop(spool);
    }
    fx.advance(10 * 60 * 1000);

    let rt = runtime();
    let spool = fx.open();
    assert!(spool.has_stream("s1"));
    assert_eq!(spool.status().pending, 3);
    let recorder = Arc::new(Recorder::default());
    spool.start(rt.handle(), recorder.clone());
    assert!(rt.block_on(spool.drain("s1", Duration::from_secs(10))));

    let mut uploaded = recorder.uploaded.lock().unwrap().clone();
    uploaded.sort_by_key(|(_, chunk_idx, ..)| *chunk_idx);
    assert_eq!(uploaded.len(), 3);
    for (chunk_idx, (data_id, uploaded_idx, meta, _)) in uploaded.iter().enumerate() {
        assert_eq!((data_id.as_str(), *uploaded_idx), ("s1", chunk_idx));
        // The stamps taken at push time survive the restart unchanged.
        assert_eq!(*meta, metadata("s1", chunk_idx));
    }
    let report = spool.report("s1").unwrap();
    assert_eq!(report.counts.uploaded, 3);
    assert_eq!(report.counts.not_uploaded(), 0);

    let mut left: Vec<String> = fs::read_dir(fx.stream_dir("s1"))
        .unwrap()
        .flatten()
        .map(|f| f.file_name().to_string_lossy().into_owned())
        .collect();
    left.sort();
    assert_eq!(left, vec![LOCK_FILE, MARKER_FILE]);

    // The resumed stream can be closed like one this run started.
    let report = spool.close_stream("s1", CloseReason::Finalized).unwrap();
    assert_eq!(report.counts.uploaded, 3);
    assert!(!fx.stream_dir("s1").exists());
}

#[test]
fn a_running_streamer_keeps_its_streams() {
    let fx = Fixture::new();
    let first = fx.open();
    first.open_stream("s1").unwrap();
    push(&first, "s1", 0);

    let second = fx.open();
    assert!(!second.has_stream("s1"));
    assert_eq!(second.status().pending, 0);
    assert!(second.open_stream("s1").is_err());
    assert_eq!(fx.entries_on_disk("s1"), 1);
}

// ---- staleness ----

#[test]
fn restart_discards_finalized_and_abandoned_streams() {
    let fx = Fixture::new();
    {
        let spool = fx.open();
        // Closed by finalize, but the crash came before its directory was deleted.
        spool.open_stream("finalized").unwrap();
        push(&spool, "finalized", 0);
        let marker = StreamMarker {
            version: FORMAT_VERSION,
            data_id: "finalized".to_string(),
            endpoint: ENDPOINT.to_string(),
            created_at_ms: T0,
            last_upload_ms: T0,
            finalized: true,
        };
        fs::write(
            fx.stream_dir("finalized").join(MARKER_FILE),
            serde_json::to_vec(&marker).unwrap(),
        )
        .unwrap();
        // Nothing confirmed since T0.
        spool.open_stream("quiet").unwrap();
        push(&spool, "quiet", 0);
        push(&spool, "quiet", 1);
        // Started 2 h 50 min in, so active 10 minutes before the restart.
        fx.advance(2 * HOUR_MS + 50 * 60 * 1000);
        spool.open_stream("busy").unwrap();
        push(&spool, "busy", 0);
    }
    fx.advance(10 * 60 * 1000);

    let spool = fx.open();
    assert!(!fx.stream_dir("finalized").exists());
    assert!(!fx.stream_dir("quiet").exists());
    assert!(!spool.has_stream("quiet"));
    assert!(spool.has_stream("busy"));
    let status = spool.status();
    assert_eq!(status.pending, 1);
    assert_eq!(status.totals.purged, 2);
}

#[test]
fn restart_discards_entries_past_retention() {
    let fx = Fixture::new();
    let mut config = fx.config();
    config.retention = Duration::from_secs(60 * 60);
    config.abandon_after = Duration::from_secs(48 * 60 * 60);
    {
        let spool = Spool::open(config.clone()).unwrap();
        spool.open_stream("s1").unwrap();
        push(&spool, "s1", 0);
        fx.advance(90 * 60 * 1000);
        push(&spool, "s1", 1);
    }
    fx.advance(30 * 60 * 1000);

    let spool = Spool::open(config).unwrap();
    let report = spool.report("s1").unwrap();
    assert_eq!(report.pending, 1);
    assert_eq!(report.counts.purged, 1);
    assert_eq!(report.losses[0].chunk_idx, 0);
    assert!(report.losses[0].reason.contains("retention"));
    assert_eq!(fx.entries_on_disk("s1"), 1);
}

#[test]
fn a_stream_with_no_confirmed_upload_for_two_hours_is_abandoned() {
    let fx = Fixture::new();
    let rt = runtime();
    let spool = fx.open();
    spool.start(rt.handle(), Arc::new(Stalled::default()));
    spool.open_stream("s1").unwrap();
    for chunk_idx in 0..3 {
        push(&spool, "s1", chunk_idx);
    }

    fx.advance(HOUR_MS);
    spool.sweep();
    assert_eq!(spool.report("s1").unwrap().pending, 3);

    fx.advance(HOUR_MS + 1);
    spool.sweep();
    let report = spool.report("s1").unwrap();
    assert_eq!(report.closed, Some(CloseReason::Abandoned));
    assert_eq!(report.counts.purged, 3);
    assert_eq!(report.pending, 0);
    assert!(!fx.stream_dir("s1").exists());
    assert_eq!(spool.status().bytes, 0);
    match push(&spool, "s1", 3) {
        Enqueued::Dropped(reason) => assert!(reason.contains("abandoned"), "{reason}"),
        Enqueued::Spooled => panic!("an abandoned stream should not spool"),
    }
}

#[test]
fn sweep_discards_entries_past_retention() {
    let fx = Fixture::new();
    let mut config = fx.config();
    config.retention = Duration::from_secs(60);
    let spool = Spool::open(config).unwrap();
    spool.open_stream("s1").unwrap();
    push(&spool, "s1", 0);
    fx.advance(30_000);
    push(&spool, "s1", 1);
    fx.advance(31_000);
    spool.sweep();

    let report = spool.report("s1").unwrap();
    assert_eq!(report.counts.purged, 1);
    assert_eq!(report.pending, 1);
    assert_eq!(fx.entries_on_disk("s1"), 1);
}

// ---- finalize ----

#[test]
fn finalize_drains_the_spool_then_reports() {
    let fx = Fixture::new();
    let rt = runtime();
    let mut config = fx.config();
    config.workers = 2;
    let spool = Spool::open(config).unwrap();
    let recorder = Arc::new(Recorder {
        delay: Duration::from_millis(20),
        ..Default::default()
    });
    spool.start(rt.handle(), recorder.clone());
    spool.open_stream("s1").unwrap();
    for chunk_idx in 0..6 {
        push(&spool, "s1", chunk_idx);
    }

    assert!(rt.block_on(spool.drain("s1", Duration::from_secs(10))));
    let report = spool.close_stream("s1", CloseReason::Finalized).unwrap();
    assert_eq!(report.counts.spooled, 6);
    assert_eq!(report.counts.uploaded, 6);
    assert_eq!(report.counts.not_uploaded(), 0);
    assert!(report.losses.is_empty());
    let mut chunks = recorder.chunks("s1");
    chunks.sort();
    assert_eq!(chunks, (0..6).collect::<Vec<_>>());
    assert!(!fx.stream_dir("s1").exists());
}

#[test]
fn finalize_reports_every_chunk_it_could_not_upload() {
    let size = entry_bytes();
    let fx = Fixture::new();
    let rt = runtime();
    let mut config = fx.config();
    config.max_bytes = Some(2 * size + size / 2);
    let spool = Spool::open(config).unwrap();
    spool.start(rt.handle(), Arc::new(RefuseFirst));
    spool.open_stream("s1").unwrap();
    push(&spool, "s1", 0);
    push(&spool, "s1", 1);
    wait_until("chunk 0 to be refused", || {
        spool.report("s1").unwrap().counts.failed == 1
    });
    // Room for one more: chunk 3 finds the spool full.
    push(&spool, "s1", 2);
    assert!(matches!(push(&spool, "s1", 3), Enqueued::Dropped(_)));

    // Chunks 1 and 2 never get through: the drain times out.
    assert!(!rt.block_on(spool.drain("s1", Duration::from_millis(100))));
    let report = spool.close_stream("s1", CloseReason::Finalized).unwrap();
    let c = report.counts;
    assert_eq!(
        (c.spooled, c.uploaded, c.dropped, c.failed, c.purged),
        (3, 0, 1, 1, 2)
    );
    assert_eq!(c.not_uploaded(), 4);
    assert_eq!(report.pending, 0);
    let mut lost: Vec<usize> = report.losses.iter().map(|l| l.chunk_idx).collect();
    lost.sort();
    assert_eq!(lost, vec![0, 1, 2, 3]);
    assert!(!fx.stream_dir("s1").exists());
    assert_eq!(spool.status().bytes, 0);

    // Closing again changes nothing.
    assert_eq!(
        spool
            .close_stream("s1", CloseReason::Finalized)
            .unwrap()
            .counts,
        c
    );
}

// ---- retries ----

/// Fails every attempt with S3's 400 RequestTimeout until `until` has passed.
struct FailsFor {
    until: tokio::time::Instant,
    attempts: AtomicU32,
}

impl Uploader for FailsFor {
    fn upload<'a>(&'a self, _: &'a EntryMeta, _: Bytes) -> UploadFuture<'a> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        let failing = tokio::time::Instant::now() < self.until;
        Box::pin(async move {
            if failing {
                Err(s3_request_timeout())
            } else {
                Ok(())
            }
        })
    }
}

#[test]
fn an_upload_keeps_retrying_well_past_the_old_eight_seconds() {
    let fx = Fixture::new();
    let mut config = fx.config();
    config.backoff = Backoff::default(); // the real 0.5 s doubling to 30 s
    let spool = Spool::open(config).unwrap();
    spool.open_stream("s1").unwrap();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .unwrap();
    rt.block_on(async {
        let start = tokio::time::Instant::now();
        let uploader = Arc::new(FailsFor {
            until: start + Duration::from_secs(120),
            attempts: AtomicU32::new(0),
        });
        spool.start(&Handle::current(), uploader.clone());
        push(&spool, "s1", 0);

        assert!(spool.drain("s1", Duration::from_secs(3600)).await);
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_secs(120), "{elapsed:?}");
        // 0.5 + 1 + 2 + 4 + 8 + 16 + 30 + 30 + 30 + 30 s: no attempt waits more than 30 s.
        assert!(elapsed < Duration::from_secs(152), "{elapsed:?}");
        assert!(uploader.attempts.load(Ordering::SeqCst) >= 10);
    });
    let report = spool.report("s1").unwrap();
    assert_eq!(report.counts.uploaded, 1);
    assert_eq!(report.counts.failed, 0);
}

// ---- the real uploader against a fake backend ----

#[derive(Default)]
struct FakeBackend {
    presigns: AtomicUsize,
    puts: AtomicUsize,
    /// PUTs answered with S3's 400 RequestTimeout before one succeeds.
    put_timeouts: usize,
    presign_status: Option<u16>,
    /// How long the nth PUT waits before answering.
    put_delays: Vec<Duration>,
}

impl FakeBackend {
    fn respond(&self, method: &str, path: &str, port: u16) -> (u16, String, Duration) {
        let (status, body) = self.answer(method, path, port);
        let delay = match method {
            "PUT" => self.put_delays.get(self.puts.load(Ordering::SeqCst) - 1),
            _ => None,
        };
        (status, body, delay.copied().unwrap_or_default())
    }

    fn answer(&self, method: &str, path: &str, port: u16) -> (u16, String) {
        match method {
            "POST" if path.starts_with("/data_entries/raw_data/staged/upload_urls/") => {
                let n = self.presigns.fetch_add(1, Ordering::SeqCst);
                match self.presign_status {
                    Some(status) => (status, r#"{"detail":"Not Found"}"#.to_string()),
                    None => (
                        200,
                        format!(
                            r#"[{{"url":"http://127.0.0.1:{port}/bucket/shard?sig={n}","part":0}}]"#
                        ),
                    ),
                }
            }
            "PUT" if path.starts_with("/bucket/shard") => {
                if self.puts.fetch_add(1, Ordering::SeqCst) < self.put_timeouts {
                    (
                        400,
                        "<Error><Code>RequestTimeout</Code></Error>".to_string(),
                    )
                } else {
                    (200, String::new())
                }
            }
            _ => (500, "unexpected request".to_string()),
        }
    }
}

/// A minimal HTTP/1.1 server: one request per connection.
async fn serve(backend: Arc<FakeBackend>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let backend = backend.clone();
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut buf = [0_u8; 8192];
                let header_end = loop {
                    let read = socket.read(&mut buf).await.unwrap();
                    if read == 0 {
                        return;
                    }
                    request.extend_from_slice(&buf[..read]);
                    if let Some(at) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        break at + 4;
                    }
                };
                let head = String::from_utf8_lossy(&request[..header_end]).into_owned();
                let content_length = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                while request.len() < header_end + content_length {
                    let read = socket.read(&mut buf).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buf[..read]);
                }
                let mut first = head.lines().next().unwrap_or_default().split(' ');
                let (method, path) = (
                    first.next().unwrap_or_default(),
                    first.next().unwrap_or_default(),
                );
                let (status, body, delay) = backend.respond(method, path, port);
                tokio::time::sleep(delay).await;
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    port
}

/// The client's own timeout, standing in for the real 60 s.
const CLIENT_TIMEOUT: Duration = Duration::from_millis(300);

fn spool_against(
    fx: &Fixture,
    rt: &Runtime,
    backend: Arc<FakeBackend>,
    put_timeout: PutTimeout,
) -> Arc<Spool> {
    let port = rt.block_on(serve(backend));
    let spool = fx.open();
    spool.start(
        rt.handle(),
        Arc::new(ShardUploader {
            client: reqwest::Client::builder()
                .timeout(CLIENT_TIMEOUT)
                .build()
                .unwrap(),
            api_key: "test-key".to_string(),
            presign_url: format!(
                "http://127.0.0.1:{port}/data_entries/raw_data/staged/upload_urls/"
            ),
            put_timeout,
        }),
    );
    spool.open_stream("s1").unwrap();
    spool
}

#[test]
fn an_s3_request_timeout_is_retried_with_a_fresh_url() {
    let fx = Fixture::new();
    let rt = runtime();
    let backend = Arc::new(FakeBackend {
        put_timeouts: 6,
        ..Default::default()
    });
    let spool = spool_against(&fx, &rt, backend.clone(), PutTimeout::default());
    push(&spool, "s1", 0);

    assert!(rt.block_on(spool.drain("s1", Duration::from_secs(10))));
    assert_eq!(spool.report("s1").unwrap().counts.uploaded, 1);
    // Seven attempts (more than the old five), each with its own presigned URL.
    assert_eq!(backend.puts.load(Ordering::SeqCst), 7);
    assert_eq!(backend.presigns.load(Ordering::SeqCst), 7);
    assert_eq!(fx.entries_on_disk("s1"), 0);
}

#[test]
fn a_chunk_the_backend_refuses_is_not_retried() {
    let fx = Fixture::new();
    let rt = runtime();
    let backend = Arc::new(FakeBackend {
        presign_status: Some(404),
        ..Default::default()
    });
    let spool = spool_against(&fx, &rt, backend.clone(), PutTimeout::default());
    push(&spool, "s1", 0);

    assert!(rt.block_on(spool.drain("s1", Duration::from_secs(10))));
    let report = spool.report("s1").unwrap();
    assert_eq!(report.counts.failed, 1);
    assert!(
        report.losses[0].reason.contains("404"),
        "{:?}",
        report.losses
    );
    assert_eq!(backend.presigns.load(Ordering::SeqCst), 1);
    assert_eq!(backend.puts.load(Ordering::SeqCst), 0);
    assert_eq!(fx.entries_on_disk("s1"), 0);
}

#[test]
fn the_put_timeout_grows_with_the_shard() {
    let timeout = PutTimeout::default();
    // A 240-frame 640×480 shard compresses to about 38 MB.
    assert_eq!(timeout.for_len(38_000_000), Duration::from_secs(440));
    assert_eq!(timeout.for_len(0), Duration::from_secs(60));
}

#[test]
fn a_slow_put_gets_time_in_proportion_to_its_size() {
    let fx = Fixture::new();
    let rt = runtime();
    let slow = Duration::from_millis(1_500);
    let put_timeout = PutTimeout {
        base: Duration::from_millis(100),
        min_bytes_per_sec: 1_000,
    };
    let shard_len = package_to_zarr_bytes(&frames(0), N, H, W).unwrap().len();
    // Longer than the client's own timeout, shorter than the one sized to this shard.
    assert!(CLIENT_TIMEOUT < slow && slow < put_timeout.for_len(shard_len));
    let backend = Arc::new(FakeBackend {
        put_delays: vec![slow],
        ..Default::default()
    });
    let spool = spool_against(&fx, &rt, backend.clone(), put_timeout);
    push(&spool, "s1", 0);

    assert!(rt.block_on(spool.drain("s1", Duration::from_secs(10))));
    assert_eq!(spool.report("s1").unwrap().counts.uploaded, 1);
    assert_eq!(backend.puts.load(Ordering::SeqCst), 1);
}

#[test]
fn a_frozen_put_times_out_and_is_retried() {
    let fx = Fixture::new();
    let rt = runtime();
    let put_timeout = PutTimeout {
        base: Duration::from_millis(200),
        min_bytes_per_sec: u64::MAX,
    };
    let backend = Arc::new(FakeBackend {
        put_delays: vec![Duration::from_secs(60)],
        ..Default::default()
    });
    let spool = spool_against(&fx, &rt, backend.clone(), put_timeout);
    push(&spool, "s1", 0);

    let t0 = Instant::now();
    assert!(rt.block_on(spool.drain("s1", Duration::from_secs(10))));
    assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
    assert_eq!(spool.report("s1").unwrap().counts.uploaded, 1);
    assert_eq!(backend.puts.load(Ordering::SeqCst), 2);
    assert_eq!(backend.presigns.load(Ordering::SeqCst), 2);
}

// Independent producers can generate identical (time, sequence) keys.
#[test]
fn recovery_keeps_entries_from_independent_streamers() {
    let fx = Fixture::new();
    let a = fx.open();
    a.open_stream("a").unwrap();
    push(&a, "a", 0);
    let b = fx.open();
    b.open_stream("b").unwrap();
    push(&b, "b", 0);
    drop(a);
    drop(b);
    let recovered = fx.open();
    assert_eq!(recovered.report("a").unwrap().pending, 1);
    assert_eq!(recovered.report("b").unwrap().pending, 1);
    assert_eq!(
        recovered.status().pending,
        2,
        "both stream entries must be in the upload queue"
    );
    let rt = runtime();
    let recorder = Arc::new(Recorder::default());
    recovered.start(rt.handle(), recorder.clone());
    for id in ["a", "b"] {
        assert!(rt.block_on(recovered.drain(id, Duration::from_secs(10))));
        assert_eq!(recorder.chunks(id), vec![0]);
        assert_eq!(fx.entries_on_disk(id), 0);
    }
    assert_eq!(recovered.status().bytes, 0);
}

#[test]
fn another_endpoint_leaves_spooled_entries_untouched() {
    let fx = Fixture::new();
    {
        let spool = fx.open();
        spool.open_stream("production").unwrap();
        push(&spool, "production", 0);
    }
    fx.advance(2_000);
    let mut other = fx.config();
    other.endpoint = "https://dev.test".into();
    other.retention = Duration::from_secs(1);
    let _dev = Spool::open(other).unwrap();
    assert_eq!(
        fx.entries_on_disk("production"),
        1,
        "dev startup must not delete prod shards"
    );
}

struct OldStreamStalls {
    calls: AtomicUsize,
}

impl Uploader for OldStreamStalls {
    fn upload<'a>(&'a self, entry: &'a EntryMeta, _: Bytes) -> UploadFuture<'a> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            if entry.data_id == "old" {
                std::future::pending().await
            } else {
                Ok(())
            }
        })
    }
}

#[test]
fn closing_stream_frees_its_workers() {
    for reason in [CloseReason::Finalized, CloseReason::Abandoned] {
        let fx = Fixture::new();
        let rt = runtime();
        let mut config = fx.config();
        config.workers = 2;
        let spool = Spool::open(config).unwrap();
        let uploader = Arc::new(OldStreamStalls {
            calls: AtomicUsize::new(0),
        });
        spool.start(rt.handle(), uploader.clone());
        spool.open_stream("old").unwrap();
        for idx in 0..2 {
            push(&spool, "old", idx);
        }
        wait_until("old uploads to start", || {
            uploader.calls.load(Ordering::SeqCst) == 2
        });
        let report = spool.close_stream("old", reason).unwrap();
        assert_eq!(report.counts.purged, 2);
        spool.forget_stream("old");
        spool.open_stream("new").unwrap();
        push(&spool, "new", 0);
        assert!(
            rt.block_on(spool.drain("new", Duration::from_secs(1))),
            "closed stream still consumes the workers"
        );
        assert_eq!(spool.report("new").unwrap().counts.uploaded, 1);
    }
}
