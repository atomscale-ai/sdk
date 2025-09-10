use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule};
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use tokio::runtime::Runtime;

mod utils;
use utils::{
    numpy_frames_to_flat, package_to_zarr_bytes, post_for_presigned, put_bytes_presigned,
    pydict_to_headers, pydict_to_json,
};

#[pyclass]
pub struct ConcurrentStreamer {
    client: Client,
    post_url: String,
    params: Value,
    headers: HashMap<String, String>,
    rt: Runtime,
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

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            client: Client::new(),
            post_url,
            params,
            headers,
            rt,
        })
    }

    /// For each item from `frames_iter`, spawn: (POST presign ⟂ package) → PUT upload.
    fn run(&self, _py: Python, frames_iter: Bound<PyAny>) -> PyResult<()> {
        let iter = PyIterator::from_object(&frames_iter)?;
        let mut handles = Vec::new();

        for it in iter {
            let obj = it?;
            let (flat, n, h, w) = numpy_frames_to_flat(obj)?;

            let client = self.client.clone();
            let post_url = self.post_url.clone();
            let params = self.params.clone();
            let headers = self.headers.clone();

            handles.push(self.rt.spawn(async move {
                let (url, shard) = tokio::try_join!(
                    post_for_presigned(&client, &post_url, params, &headers),
                    async { package_to_zarr_bytes(&flat, n, h, w) }
                )?;
                put_bytes_presigned(&client, &url, &shard, &headers).await
            }));
        }

        self.rt
            .block_on(async {
                for h in handles {
                    h.await??;
                }
                Ok::<(), anyhow::Error>(())
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }
}

#[pymodule]
fn rheed_stream(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<ConcurrentStreamer>()?;
    Ok(())
}
