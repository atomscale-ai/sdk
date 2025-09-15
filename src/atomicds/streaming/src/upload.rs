use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use reqwest::Client;
use serde::Serialize;
use std::sync::Arc;
use zarrs::{
    array::{
        codec::{array_to_bytes::sharding::ShardingCodecBuilder, bytes_to_bytes::zstd::ZstdCodec},
        ArrayBuilder, DataType, FillValue,
    },
    storage::store::MemoryStore,
};

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use tracing::debug;

#[derive(Serialize, Debug)]
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
}

/// Accept (H,W) or (N,H,W) frames (casts to uint8) → (flat bytes, N,H,W).
pub fn numpy_frames_to_flat(obj: Bound<PyAny>) -> PyResult<(Vec<u8>, usize, usize, usize)> {
    // downcast() returns &Bound<...>; clone it to get Bound<...>.
    let arr_u8: Bound<PyArrayDyn<u8>> = if let Ok(a) = obj.downcast::<PyArrayDyn<u8>>() {
        a.clone()
    } else {
        let np = PyModule::import(obj.py(), "numpy")?;
        let a = np.getattr("asarray")?.call1((obj,))?;
        let a = a.call_method1("astype", ("uint8",))?;
        a.downcast::<PyArrayDyn<u8>>()?.clone()
    };

    let ro: PyReadonlyArrayDyn<u8> = arr_u8.readonly();
    let v = ro.as_array();
    let s = v.shape();

    match s.len() {
        2 => {
            let (h, w) = (s[0], s[1]);
            let (flat, off) = v.to_owned().into_raw_vec_and_offset();
            assert!(off == Some(0));
            Ok((flat, 1, h, w))
        }
        3 => {
            let (n, h, w) = (s[0], s[1], s[2]);
            let (flat, off) = v.to_owned().into_raw_vec_and_offset();
            assert!(off == Some(0));
            Ok((flat, n, h, w))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "frames must be (H,W) or (N,H,W)",
        )),
    }
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

    let store = Arc::new(MemoryStore::new());
    let arr = ArrayBuilder::new(
        vec![n as u64, h as u64, w as u64],
        DataType::UInt8,
        vec![n as u64, h as u64, w as u64]
            .try_into()
            .context("chunk grid")?,
        FillValue::from(0u8),
    )
    .array_to_bytes_codec(
        // Lower compression for quick tests; bump to 9 if desired.
        ShardingCodecBuilder::new(vec![1u64, h as u64, w as u64].try_into()?)
            .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(3, false))])
            .build_arc(),
    )
    .build(store, "/frames")?;

    arr.store_metadata()?;
    arr.store_chunk_elements(&[0, 0, 0], frames_flat)?;
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
        return Err(anyhow::anyhow!("presign {}: {}", status, text));
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

/// PUT bytes to the presigned URL (async).
pub async fn put_bytes_presigned(client: &Client, url: &str, bytes: &[u8]) -> Result<()> {
    debug!("[rheed_stream] put: PUT {} ({} bytes)", url, bytes.len());

    let req = client
        .put(url)
        .header("content-type", "")
        .body(Bytes::copy_from_slice(bytes));

    let resp = req.send().await?;
    let status = resp.status();
    let final_url = resp.url().clone();
    let resp_headers = resp.headers().clone();

    debug!("[put] -> {} {}", status, final_url);

    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        debug!("[put] resp headers: {:#?}", resp_headers);
        debug!("[put] resp body: {}", text);
        return Err(anyhow::anyhow!("put presigned {}: {}", status, text));
    }

    if let Some(etag) = resp_headers.get("etag").and_then(|v| v.to_str().ok()) {
        debug!("[put] ETag: {}", etag);
    }

    Ok(())
}
