use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::Serialize;
use serde_json::Value;

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")] // Ensures JSON fields are snake_case (e.g., data_id)
pub struct RHEEDStreamSettings {
    data_item_name: String,
    rotational_period: f64,
    rotations_per_min: usize,
    fps_capture_rate: f64,
}

/// POST request to initialize a RHEED stream
pub async fn post_for_initialization(
    client: &Client,
    url: &str,
    stream_settings: &RHEEDStreamSettings,
    api_key: &str,
) -> Result<String> {
    let req = client
        .post(url)
        .header("X-API-KEY", api_key)
        .json(stream_settings);

    let v: Value = req.send().await?.error_for_status()?.json().await?;

    Ok(v.get("url")
        .and_then(|x| x.as_str())
        .ok_or_else(|| anyhow!("missing 'url'"))?
        .to_string())
}
