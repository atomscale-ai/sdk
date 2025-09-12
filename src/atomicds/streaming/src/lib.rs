use anyhow::{anyhow, Context};
use chrono::{Local, Utc};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyModule};
use reqwest::Client;
use std::time::Instant;
use tokio::runtime::Runtime;

mod initialize;
use initialize::{post_for_initialization, RHEEDStreamSettings};

mod upload;
use upload::{numpy_frames_to_flat, package_to_zarr_bytes};

use crate::upload::{post_for_presigned, put_bytes_presigned, FrameChunkMetadata};

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
    #[new]
    fn new(api_key: String, endpoint: Option<String>) -> PyResult<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let endpoint = endpoint.unwrap_or("https://api.atomicdatasciences.com".to_string());

        eprintln!("[rheed_stream] init: base_url={}", endpoint);

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
    fn initialize(
        &mut self,
        stream_name: Option<String>,
        fps: f64,
        rotations_per_min: f64,
        chunk_size: usize,
    ) -> PyResult<String> {
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
            rotations_per_min: rotations_per_min,
            fps_capture_rate: fps,
        };

        let base_endpoint = self.endpoint.clone();
        let post_url = format!("{base_endpoint}/rheed/stream/");
        let init_fut = post_for_initialization(&self.client, &post_url, &settings, &self.api_key);

        let data_id = self
            .rt
            .block_on(init_fut)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        self.fps = Some(fps);
        self.rotating = Some(rotations_per_min > 0.0);
        self.chunk_size = Some(chunk_size);

        Ok(data_id)
    }

    /// Concurrent mode:
    /// - Main (GIL) thread: iterate generator, do NumPy→Vec<u8> prep, print quick stats
    /// - For each chunk: spawn a task that does the **package** step on a blocking worker
    ///   and prints timing + shard preview. All tasks run in parallel.
    fn run(&self, data_id: String, frames_iter: Bound<PyAny>) -> PyResult<()> {
        let rotating = self.rotating.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("rotating is not set; call initialize(...).")
        })?;

        let fps = self.fps.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("fps is not set; call initialize(...).")
        })?;

        let chunk_size = self.chunk_size.ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "chunk size is not set; call initialize(...).",
            )
        })?;

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

            eprintln!(
                "[rheed_stream] item#{idx}: prepared in {:.2?} → N,H,W=({n},{h},{w}), flat_len={}; spawning task…",
                t0.elapsed(), flat_len
            );

            // Move data into the task; package on a blocking worker thread
            let task_id = idx;
            let flat_task = flat; // move

            // Get request related items
            let client = self.client.clone();
            let base_endpoint = self.endpoint.clone();
            let post_url = format!("{base_endpoint}/data_entries/raw_data/staged/upload_urls/");
            let api_key = self.api_key.clone();

            // Populate metadata for S3
            let now_ms_utc = Utc::now().timestamp_millis();

            #[allow(clippy::redundant_field_names)]
            let metadata = FrameChunkMetadata {
                data_id: data_id.clone(),
                data_stream: "rheed".to_string(),
                is_stream: 1,
                is_rotating: rotating as u8,
                raw_frame_rate: fps,
                avg_frame_rate: fps,
                chunk_size: chunk_size,
                dims: format!("{n},{h},{w}"),
                start_unix_ms_utc: now_ms_utc,
                end_unix_ms_utc: now_ms_utc + (((n as f64 / fps) * 1000.0) as i64),
            };

            // This is the object key we will PUT to S3 (single shard file)
            let zarr_shard_key = format!("frames.zarr/frames/c/{idx}/0/0");

            // let metadata = FrameChunkMetadata{data_id: data_id, };
            let handle = self.rt.spawn(async move {
                // eprintln!("[rheed_stream] task#{task_id}: package start (N,H,W=({n},{h},{w}), flat_len={flat_len})");
                let t_pack0 = Instant::now();

                let url_fut =
                    post_for_presigned(&client, &post_url, &zarr_shard_key, &metadata, &api_key);
                let shard_handle =
                    tokio::task::spawn_blocking(move || package_to_zarr_bytes(&flat_task, n, h, w));
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

                eprintln!(
                    "[rheed_stream] task#{task_id}: packaged in {:.2?} → uploading with url={}",
                    pack_dur, url
                );

                // PUT request to upload the byte data
                put_bytes_presigned(&client, &url, &shard)
                    .await
                    .with_context(|| format!("task#{task_id}/put bytes request failed"))?;

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
    m.add_class::<RHEEDStreamer>()?;
    Ok(())
}
