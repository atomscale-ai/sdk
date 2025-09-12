use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use reqwest::Client;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use zarrs::{
    array::{
        codec::{array_to_bytes::sharding::ShardingCodecBuilder, bytes_to_bytes::zstd::ZstdCodec},
        ArrayBuilder, DataType, FillValue,
    },
    storage::store::MemoryStore,
};

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")] // Ensures JSON fields are snake_case (e.g., data_id)
pub struct FrameChunkMetadata {
    data_id: String,
    data_stream: String,
    is_stream: u8,
    is_rotating: u8,
    raw_frame_rate: f64,
    avg_frame_rate: f64,
    chunk_size: usize,
    dims: String,
    start_unix_ms_utc: u64,
    end_unix_ms_utc: u64,
}

/// Accept (H,W) or (N,H,W) frames (casts to uint8) → (flat bytes, N,H,W).
pub fn numpy_frames_to_flat(obj: Bound<PyAny>) -> PyResult<(Vec<u8>, usize, usize, usize)> {
    use numpy::{PyArrayDyn, PyReadonlyArrayDyn};

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

/// POST for a presigned URL (async). Returns the "url" string.
pub async fn post_for_presigned(
    client: &Client,
    url: &str,
    original_filename: &str,
    chunk_metadata: &FrameChunkMetadata,
    api_key: &str,
) -> Result<String> {
    let req = client
        .post(url)
        .header("X-API-KEY", api_key)
        .query(&[
            ("original_filename", original_filename),
            ("num_parts", "1"),
            ("staging_type", "stream"),
        ])
        .json(chunk_metadata);
    let v: Value = req.send().await?.error_for_status()?.json().await?;
    Ok(v.get("url")
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow!("missing 'url'"))?
        .to_string())
}

/// PUT bytes to the presigned URL (async).
pub async fn put_bytes_presigned(
    client: &Client,
    url: &str,
    bytes: &[u8],
    hdrs: &HashMap<String, String>,
) -> Result<()> {
    // Use Bytes to avoid an extra copy inside reqwest
    let mut req = client
        .put(url)
        .header("content-type", "application/octet-stream")
        .header("Connection", "close")
        .body(Bytes::copy_from_slice(bytes));

    for (k, v) in hdrs {
        req = req.header(k, v);
    }

    req.send().await?.error_for_status()?;
    Ok(())
}
