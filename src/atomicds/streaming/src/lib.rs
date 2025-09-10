use anyhow::{anyhow, Context, Result};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule};
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;
use tokio::runtime::Runtime;

mod utils;
use utils::{numpy_frames_to_flat, package_to_zarr_bytes, pydict_to_headers, pydict_to_json};

use crate::utils::post_for_presigned;

#[pyclass]
pub struct ConcurrentStreamer {
    client: Client, // unused in this mode, ok to keep
    post_url: String,
    params: Value,
    headers: HashMap<String, String>,
    rt: Runtime, // ✅ bring back a Tokio runtime to run tasks concurrently
}

#[pymethods]
impl ConcurrentStreamer {
    #[new]
    fn new(
        post_url: String,
        base_params: Option<Py<PyDict>>,
        headers: Option<Py<PyDict>>,
        py: Python,
    ) -> PyResult<Self> {
        let params = pydict_to_json(base_params.as_ref().map(|p| p.bind(py)).cloned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let headers = pydict_to_headers(headers.as_ref().map(|p| p.bind(py)).cloned())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        eprintln!("[rheed_stream] init: post_url={}", post_url);
        Ok(Self {
            client,
            post_url,
            params,
            headers,
            rt,
        })
    }

    /// Concurrent mode:
    /// - Main (GIL) thread: iterate generator, do NumPy→Vec<u8> prep, print quick stats
    /// - For each chunk: spawn a task that does the **package** step on a blocking worker
    ///   and prints timing + shard preview. All tasks run in parallel.
    fn run(&self, py: Python, frames_iter: Bound<PyAny>) -> PyResult<()> {
        eprintln!("[rheed_stream] run: starting (concurrent: prepare→spawn package tasks)");
        let iter = PyIterator::from_object(&frames_iter)?;
        let mut handles = Vec::new();

        let t_total0 = Instant::now();

        for (idx, it) in iter.enumerate() {
            let t0 = Instant::now();
            let obj = it?; // next yielded item

            // Prepare (under GIL): convert to (flat bytes, N,H,W)
            let (flat, n, h, w) = numpy_frames_to_flat(obj)?;
            let flat_len = flat.len();
            let flat_checksum: u64 = flat.iter().take(1024).fold(0u64, |acc, &b| acc + b as u64);
            let flat_preview = {
                let take = flat
                    .iter()
                    .take(16)
                    .map(|b| format!("{:02x}", b))
                    .collect::<Vec<_>>()
                    .join(" ");
                if flat_len > 16 {
                    format!("{take} …")
                } else {
                    take
                }
            };

            eprintln!(
                "[rheed_stream] item#{idx}: prepared in {:.2?} → N,H,W=({n},{h},{w}), flat_len={}, checksum={}, preview=[{}]; spawning task…",
                t0.elapsed(), flat_len, flat_checksum, flat_preview
            );

            // Move data into the task; package on a blocking worker thread
            let task_id = idx;
            let flat_task = flat; // move
            let client    = self.client.clone();
            let post_url  = self.post_url.clone();
            let params    = self.params.clone();
            let headers   = self.headers.clone();
            let handle = self.rt.spawn(async move {
                eprintln!("[rheed_stream] task#{task_id}: package start (N,H,W=({n},{h},{w}), flat_len={flat_len})");
                let t_pack0 = Instant::now();
                let url_fut = post_for_presigned(&client, &post_url, params, &headers);
                let shard_handle = tokio::task::spawn_blocking(move || package_to_zarr_bytes(&flat_task, n, h, w));
                let (url, shard): (String, Vec<u8>) = tokio::try_join!(
                    async {
                            url_fut 
                            .await
                            .with_context(|| format!("task#{task_id}/url request failed"))
                    },
                    async {
                        let bytes_res = shard_handle
                            .await
                            .map_err(|e| anyhow!("task#{task_id}/shard join error: {e}"))?;
                        let bytes = bytes_res
                            .with_context(|| format!("task#{task_id}/shard worker failed"))?;
                        Ok::<_, anyhow::Error>(bytes)
                    }
                )?;
                
                let pack_dur = t_pack0.elapsed();
                let shard_len = shard.len();
                let shard_checksum: u64 = shard.iter().take(1024).fold(0u64, |acc, &b| acc + b as u64);
                let shard_preview = {
                    let take = shard.iter().take(16).map(|b| format!("{:02x}", b)).collect::<Vec<_>>().join(" ");
                    if shard_len > 16 { format!("{take} …") } else { take }
                };

                eprintln!(
                    "[rheed_stream] task#{task_id}: packaged in {:.2?} → shard_len={}, checksum={}, preview=[{}], url={}",
                    pack_dur, shard_len, shard_checksum, shard_preview, url
                );
                Ok::<(), anyhow::Error>(())
            });

            handles.push(handle);
        }

        eprintln!(
            "[rheed_stream] run: spawned {} packaging task(s); awaiting…",
            handles.len()
        );

        // Wait for all tasks to finish; print errors if any
        let res = self.rt.block_on(async {
            for (i, h) in handles.into_iter().enumerate() {
                match h.await {
                    Ok(Ok(())) => eprintln!("[rheed_stream] task#{i}: completed"),
                    Ok(Err(e)) => {
                        eprintln!("[rheed_stream] task#{i}: ERROR: {e}");
                        return Err(e);
                    }
                    Err(e) => {
                        eprintln!("[rheed_stream] task#{i}: JOIN ERROR: {e}");
                        return Err(anyhow::anyhow!(e));
                    }
                }
            }
            Ok::<(), anyhow::Error>(())
        });

        if let Err(e) = res {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string()));
        }

        let total_dur = t_total0.elapsed();
        eprintln!("[rheed_stream] run: all tasks done in {:.2?}", total_dur);
        Ok(())
    }
}

#[pymodule]
fn rheed_stream(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<ConcurrentStreamer>()?;
    Ok(())
}
