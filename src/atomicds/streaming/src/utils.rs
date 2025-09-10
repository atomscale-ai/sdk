use anyhow::{anyhow, Context, Result};
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyModule};
use reqwest::Client;
use serde_json::{Map as JsonMap, Value};
use std::collections::HashMap;
use std::sync::Arc;
use zarrs::{
    array::{
        codec::{array_to_bytes::sharding::ShardingCodecBuilder, bytes_to_bytes::zstd::ZstdCodec},
        ArrayBuilder, DataType, FillValue,
    },
    storage::store::MemoryStore,
};

/// PyDict → JSON (common primitives, lists, dicts; fallback to str()).
pub fn pydict_to_json(d: Option<Bound<PyDict>>) -> Result<Value> {
    fn to_value(obj: &Bound<PyAny>) -> Result<Value> {
        if let Ok(s) = obj.extract::<String>() {
            return Ok(Value::String(s));
        }
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(i.into());
        }
        if let Ok(f) = obj.extract::<f64>() {
            return Ok(serde_json::json!(f));
        }
        if let Ok(b) = obj.extract::<bool>() {
            return Ok(Value::Bool(b));
        }
        if let Ok(d) = obj.downcast::<PyDict>() {
            let mut m = JsonMap::new();
            for (k, v) in d.iter() {
                m.insert(k.str()?.to_str()?.to_string(), to_value(&v)?);
            }
            return Ok(Value::Object(m));
        }
        if let Ok(lst) = obj.downcast::<PyList>() {
            let mut a = Vec::with_capacity(lst.len());
            for it in lst.iter() {
                a.push(to_value(&it)?);
            }
            return Ok(Value::Array(a));
        }
        Ok(Value::String(obj.str()?.to_str()?.to_string()))
    }

    if let Some(d) = d {
        let mut m = JsonMap::new();
        for (k, v) in d.iter() {
            m.insert(k.str()?.to_str()?.to_string(), to_value(&v)?);
        }
        Ok(Value::Object(m))
    } else {
        Ok(Value::Object(JsonMap::new()))
    }
}

/// PyDict → header map (String → String).
pub fn pydict_to_headers(d: Option<Bound<PyDict>>) -> Result<HashMap<String, String>> {
    let mut out = HashMap::new();
    if let Some(d) = d {
        for (k, v) in d.iter() {
            out.insert(
                k.str()?.to_str()?.to_string(),
                v.str()?.to_str()?.to_string(),
            );
        }
    }
    Ok(out)
}

/// Accept (H,W) or (N,H,W) frames (casts to uint8) → (flat bytes, N,H,W).
pub fn numpy_frames_to_flat(obj: Bound<PyAny>) -> PyResult<(Vec<u8>, usize, usize, usize)> {
    use numpy::{PyArrayDyn, PyReadonlyArrayDyn};

    // NOTE: downcast() returns &Bound<...>; clone it to get Bound<...>.
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
        ShardingCodecBuilder::new(vec![1u64, h as u64, w as u64].try_into()?)
            .bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(9, false))])
            .build_arc(),
    )
    .build(store, "/frames")?;

    arr.store_metadata()?;
    arr.store_chunk_elements(&[0, 0, 0], frames_flat)?;
    arr.retrieve_encoded_chunk(&[0, 0, 0])?
        .ok_or_else(|| anyhow!("missing encoded chunk"))
}

/// POST for a presigned URL; return "url".
pub async fn post_for_presigned(
    client: &Client,
    url: &str,
    body: Value,
    hdrs: &HashMap<String, String>,
) -> Result<String> {
    let mut req = client.post(url).json(&body);
    for (k, v) in hdrs {
        req = req.header(k, v);
    }
    let v: Value = req.send().await?.error_for_status()?.json().await?;
    Ok(v.get("url")
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow!("missing 'url'"))?
        .to_string())
}

/// PUT bytes to the presigned URL.
pub async fn put_bytes_presigned(
    client: &Client,
    url: &str,
    bytes: &[u8],
    hdrs: &HashMap<String, String>,
) -> Result<()> {
    let mut req = client
        .put(url)
        .header("content-type", "application/octet-stream")
        .body(bytes.to_vec());
    for (k, v) in hdrs {
        req = req.header(k, v);
    }
    req.send().await?.error_for_status()?;
    Ok(())
}
