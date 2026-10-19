use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::Duration;
use zarrs::{
    array::{
        codec::{array_to_bytes::sharding::ShardingCodecBuilder, bytes_to_bytes::zstd::ZstdCodec},
        data_type, ArrayBuilder,
    },
    storage::store::MemoryStore,
};

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use tracing::debug;

use crate::retry::{is_rejection, HttpStatusError};
use crate::spool::{EntryMeta, UploadError, UploadFuture, Uploader};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "snake_case")] // Ensures JSON fields are snake_case (e.g., data_id)
pub struct FrameChunkMetadata {
    pub data_id: String,
    pub data_stream: String,
    pub is_stream: u8,
    pub is_rotating: u8,
    pub raw_frame_rate: f64,
    pub avg_frame_rate: f64,
    pub chunk_size: usize,
    pub dims: String,
    pub start_unix_ms_utc: i64,
    pub end_unix_ms_utc: i64,
    /// The clock correction already applied to the two stamps above: reference UTC minus the
    /// host clock, in ms, so `host time = start_unix_ms_utc - clock_offset_ms`. Absent when the
    /// stamps are the host clock as read (correction off, or the clock never measured).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_offset_ms: Option<i64>,
    /// Bound on the error of `clock_offset_ms`, in ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_uncertainty_ms: Option<f64>,
}

/// Accept (H,W) or (N,H,W) frames (casts to uint8) → (flat bytes, N,H,W).
///
/// The bytes copy runs **without the GIL** when the input array is contiguous,
/// so concurrent producers calling into the streamer don't serialize on this
/// conversion. The numpy buffer is pinned by `PyReadonlyArrayDyn` for the
/// duration of the function, so the raw pointer stays valid across the
/// `allow_threads` boundary.
pub fn numpy_frames_to_flat(obj: Bound<PyAny>) -> PyResult<(Vec<u8>, usize, usize, usize)> {
    let py = obj.py();

    // cast() returns &Bound<...>; clone it to get Bound<...>.
    let arr_u8: Bound<PyArrayDyn<u8>> = if let Ok(a) = obj.cast::<PyArrayDyn<u8>>() {
        a.clone()
    } else {
        let np = PyModule::import(py, "numpy")?;
        let a = np.getattr("asarray")?.call1((obj,))?;
        let a = a.call_method1("astype", ("uint8",))?;
        a.cast::<PyArrayDyn<u8>>()?.clone()
    };

    let ro: PyReadonlyArrayDyn<u8> = arr_u8.readonly();
    let view = ro.as_array();
    let shape = view.shape();

    let (n, h, w) = match shape.len() {
        2 => (1usize, shape[0], shape[1]),
        3 => (shape[0], shape[1], shape[2]),
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "frames must be (H,W) or (N,H,W)",
            ))
        }
    };

    let nbytes = n
        .checked_mul(h)
        .and_then(|x| x.checked_mul(w))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("N*H*W overflow"))?;

    // Fast path: contiguous standard-layout array → memcpy off-GIL.
    // `as usize` round-trip is the standard pyo3 idiom for moving a non-Send
    // raw pointer into an `allow_threads` closure.
    let flat = if let Some(slice) = view.as_slice() {
        let src_addr = slice.as_ptr() as usize;
        py.detach(|| {
            let mut buf: Vec<u8> = Vec::with_capacity(nbytes);
            unsafe {
                std::ptr::copy_nonoverlapping(src_addr as *const u8, buf.as_mut_ptr(), nbytes);
                buf.set_len(nbytes);
            }
            buf
        })
    } else {
        // Non-contiguous fallback (e.g. transposed views): GIL-held copy via
        // ndarray. Rare in practice; preserves prior semantics.
        let (flat, off) = view.to_owned().into_raw_vec_and_offset();
        assert!(off == Some(0));
        flat
    };

    Ok((flat, n, h, w))
}

/// Build one outer chunk (N,H,W), shard into (1,H,W), return encoded bytes of chunk [0,0,0].
pub fn package_to_zarr_bytes(frames_flat: &[u8], n: usize, h: usize, w: usize) -> Result<Vec<u8>> {
    let need = n
        .checked_mul(h)
        .and_then(|x| x.checked_mul(w))
        .ok_or_else(|| anyhow!("N*H*W overflow"))?;
    if frames_flat.len() != need {
        return Err(anyhow!("flat len {} != N*H*W {}", frames_flat.len(), need));
    }

    // zarrs 0.23 chunk shapes are `Vec<NonZeroU64>`. All dims are > 0 here (a
    // zero dim would force `need == 0` and mismatch a non-empty buffer above),
    // but convert defensively rather than unwrap.
    let to_nonzero = |x: usize| {
        NonZeroU64::new(x as u64).ok_or_else(|| anyhow!("zarr chunk dimension must be non-zero"))
    };
    let subchunk_shape: Vec<NonZeroU64> = vec![to_nonzero(1)?, to_nonzero(h)?, to_nonzero(w)?];

    let store = Arc::new(MemoryStore::new());
    // ArrayBuilder::new arg order in 0.23: (shape, chunk_grid, data_type, fill_value).
    let arr = ArrayBuilder::new(
        vec![n as u64, h as u64, w as u64],
        vec![n as u64, h as u64, w as u64],
        data_type::uint8(),
        0u8,
    )
    .array_to_bytes_codec(
        // Lower compression for quick tests; bump to 9 if desired.
        ShardingCodecBuilder::new(subchunk_shape, &data_type::uint8())
            .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(3, false))])
            .build_arc(),
    )
    .build(store, "/frames")?;

    arr.store_metadata()?;
    arr.store_chunk(&[0, 0, 0], frames_flat)?;
    arr.retrieve_encoded_chunk(&[0, 0, 0])?
        .ok_or_else(|| anyhow!("missing encoded chunk"))
}

pub async fn post_for_presigned(
    client: &Client,
    url: &str,
    original_filename: &str,
    chunk_metadata: &FrameChunkMetadata,
    api_key: &str,
) -> Result<String> {
    debug!("[rheed_stream] presign: POST {}", url);

    // Serialize once, then stringify numbers/bools for APIs that expect all strings.
    let mut meta = serde_json::to_value(chunk_metadata)?;
    if let serde_json::Value::Object(map) = &mut meta {
        for v in map.values_mut() {
            if matches!(v, serde_json::Value::Number(_) | serde_json::Value::Bool(_)) {
                *v = serde_json::Value::String(v.to_string());
            }
        }
    }
    debug!(
        "[presign] metadata json:\n{}",
        serde_json::to_string_pretty(&meta).unwrap_or_default()
    );

    let req = client
        .post(url)
        .header("X-API-KEY", api_key)
        .query(&[
            ("original_filename", original_filename),
            ("num_parts", "1"),
            ("staging_type", "stream"),
        ])
        .json(&meta);

    let resp = req.send().await?;
    let status = resp.status();
    let final_url = resp.url().clone();
    debug!("[presign] -> {} {}", status, final_url);

    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(HttpStatusError { status, body: text }).context("presign");
    }

    let v: serde_json::Value = serde_json::from_str(&text)?;

    // NEW: handle array response like: [{ "upload_id": "...", "url": "...", "part": 0, ... }]
    if let Some(arr) = v.as_array() {
        if let Some(u) = arr
            .first()
            .and_then(|o| o.get("url"))
            .and_then(|x| x.as_str())
        {
            return Ok(u.to_string());
        }
        return Err(anyhow::anyhow!("missing 'url' in array response: {v}"));
    }

    Ok(v.get("url")
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing 'url'"))?
        .to_string())
}

/// How long one shard PUT may take: a fixed allowance plus the time to send the shard at the
/// slowest uplink worth supporting.
#[derive(Clone, Copy, Debug)]
pub struct PutTimeout {
    pub base: Duration,
    pub min_bytes_per_sec: u64,
}

impl Default for PutTimeout {
    /// 60 s plus 100 kB/s: 440 s for a 38 MB shard (240 frames of 640×480).
    fn default() -> Self {
        Self {
            base: Duration::from_secs(60),
            min_bytes_per_sec: 100_000,
        }
    }
}

impl PutTimeout {
    pub fn for_len(&self, len: usize) -> Duration {
        self.base + Duration::from_secs_f64(len as f64 / self.min_bytes_per_sec.max(1) as f64)
    }
}

/// PUT bytes to the presigned URL (async). `timeout` replaces the client's own for this request,
/// which is too short for a large shard on a slow uplink.
pub async fn put_bytes_presigned(
    client: &Client,
    url: &str,
    bytes: Bytes,
    timeout: Duration,
) -> Result<()> {
    debug!(
        "[rheed_stream] put: PUT {} ({} bytes, timeout {timeout:?})",
        url,
        bytes.len()
    );

    let req = client
        .put(url)
        .header("content-type", "")
        .timeout(timeout)
        .body(bytes);

    let resp = req.send().await?;
    let status = resp.status();
    let final_url = resp.url().clone();
    let resp_headers = resp.headers().clone();

    debug!("[put] -> {} {}", status, final_url);

    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        debug!("[put] resp headers: {:#?}", resp_headers);
        return Err(HttpStatusError { status, body: text }).context("put presigned");
    }

    if let Some(etag) = resp_headers.get("etag").and_then(|v| v.to_str().ok()) {
        debug!("[put] ETag: {}", etag);
    }

    Ok(())
}

/// Uploads spooled RHEED shards: a presigned URL from the backend, then a PUT of the shard to it.
///
/// Every attempt asks for a fresh URL, so one that expired while the upload waited (or while the
/// process was down) is never reused. A shard always lands at the same key for its chunk, so an
/// attempt that repeats one that already succeeded overwrites it with the same bytes.
pub struct ShardUploader {
    pub client: Client,
    pub api_key: String,
    /// `{endpoint}/data_entries/raw_data/staged/upload_urls/`.
    pub presign_url: String,
    /// The presign request keeps the client's timeout; the PUT gets one sized to the shard.
    pub put_timeout: PutTimeout,
}

impl Uploader for ShardUploader {
    fn upload<'a>(&'a self, entry: &'a EntryMeta, shard: Bytes) -> UploadFuture<'a> {
        Box::pin(async move {
            let shard_key = format!("frames.zarr/frames/c/{}/0/0", entry.chunk_idx);
            let url = post_for_presigned(
                &self.client,
                &self.presign_url,
                &shard_key,
                &entry.metadata,
                &self.api_key,
            )
            .await
            .context("presigned URL request failed")
            .map_err(|error| {
                // The backend refusing the chunk outright will not change on a retry; anything
                // else (no connection, 5xx, 429, a garbled reply) might.
                if is_rejection(&error) {
                    UploadError::GiveUp(error)
                } else {
                    UploadError::Retry(error)
                }
            })?;
            // A failed PUT is always worth another try with a fresh URL: S3 answers 400
            // RequestTimeout when the body stalls mid-upload and 403 once the URL has expired.
            let timeout = self.put_timeout.for_len(shard.len());
            put_bytes_presigned(&self.client, &url, shard, timeout)
                .await
                .context("PUT bytes failed")
                .map_err(UploadError::Retry)
        })
    }
}
