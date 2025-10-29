use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")] // Ensures JSON fields are snake_case (e.g., data_id)
pub struct RHEEDStreamSettings {
    pub data_item_name: String,
    pub rotational_period: f64,
    pub rotations_per_min: f64,
    pub fps_capture_rate: f64,
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

    let v: String = req.send().await?.error_for_status()?.json().await?;

    Ok(v)
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
struct PhysicalSampleSummary {
    id: String,
    name: String,
}

#[derive(Serialize)]
struct CreatePhysicalSampleRequest<'a> {
    name: &'a str,
}

#[derive(Deserialize, Debug)]
struct CreatePhysicalSampleResponse {
    #[serde(alias = "id")]
    physical_sample_id: String,
}

#[derive(Serialize)]
struct LinkPhysicalSampleRequest {
    data_ids: Vec<String>,
    physical_sample_id: String,
}

pub async fn ensure_physical_sample_link(
    client: &Client,
    base_endpoint: &str,
    api_key: &str,
    data_id: &str,
    sample_name: &str,
) -> Result<()> {
    let sample_name = sample_name.trim();
    if sample_name.is_empty() {
        return Ok(());
    }

    let list_url = format!("{base_endpoint}/physical_samples/");
    let existing_samples: Vec<PhysicalSampleSummary> = client
        .get(&list_url)
        .header("X-API-KEY", api_key)
        .send()
        .await
        .context("failed to request physical samples")?
        .error_for_status()
        .context("physical sample list returned error status")?
        .json()
        .await
        .context("failed to deserialize physical sample list")?;

    let sample_id = if let Some(sample) = existing_samples
        .into_iter()
        .find(|sample| sample.name.eq_ignore_ascii_case(sample_name))
    {
        sample.id
    } else {
        let create_body = CreatePhysicalSampleRequest { name: sample_name };
        let created: CreatePhysicalSampleResponse = client
            .post(&list_url)
            .header("X-API-KEY", api_key)
            .json(&create_body)
            .send()
            .await
            .context("failed to create physical sample")?
            .error_for_status()
            .context("physical sample creation returned error status")?
            .json()
            .await
            .context("failed to deserialize physical sample creation response")?;
        created.physical_sample_id
    };

    let link_url = format!("{base_endpoint}/data_entries/physical_sample");
    let link_body = LinkPhysicalSampleRequest {
        data_ids: vec![data_id.to_string()],
        physical_sample_id: sample_id,
    };

    client
        .post(&link_url)
        .header("X-API-KEY", api_key)
        .json(&link_body)
        .send()
        .await
        .context("failed to link physical sample to data item")?
        .error_for_status()
        .context("physical sample link returned error status")?;

    Ok(())
}
