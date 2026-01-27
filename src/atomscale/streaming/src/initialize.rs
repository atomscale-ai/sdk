use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")] // Ensures JSON fields are snake_case (e.g., data_id)
pub struct RHEEDStreamSettings {
    pub data_item_name: String,
    pub rotational_period: f64,
    pub rotations_per_min: f64,
    pub fps_capture_rate: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
}

// Project-related structs for GET /projects/ response
#[derive(Deserialize, Debug)]
struct DataConfiguration {
    api_configuration: Option<Value>,
}

#[derive(Deserialize, Debug)]
struct ProjectSummary {
    id: String,
    configuration: Option<DataConfiguration>,
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
) -> Result<String> {
    let sample_name = sample_name.trim();
    if sample_name.is_empty() {
        anyhow::bail!("sample_name cannot be empty");
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
        physical_sample_id: sample_id.clone(),
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

    Ok(sample_id)
}

/// Updates the project's tracking_physical_sample_id in its configuration.
/// Fetches current configuration, updates the tracking sample, and POSTs back.
pub async fn update_project_tracking_sample(
    client: &Client,
    base_endpoint: &str,
    api_key: &str,
    project_id: &str,
    physical_sample_id: &str,
) -> Result<()> {
    // GET /projects/ to find the project and its current configuration
    let projects_url = format!("{base_endpoint}/projects/");
    let projects: Vec<ProjectSummary> = client
        .get(&projects_url)
        .header("X-API-KEY", api_key)
        .send()
        .await
        .context("failed to request projects")?
        .error_for_status()
        .context("projects list returned error status")?
        .json()
        .await
        .context("failed to deserialize projects list")?;

    // Find the project by ID
    let project = projects
        .into_iter()
        .find(|p| p.id == project_id)
        .ok_or_else(|| anyhow::anyhow!("project with id {} not found", project_id))?;

    // Build updated configuration, preserving existing fields
    let mut config = match project.configuration {
        Some(data_config) => data_config.api_configuration.unwrap_or_else(|| Value::Object(Default::default())),
        None => Value::Object(Default::default()),
    };

    // Update tracking_physical_sample_id in the configuration
    if let Value::Object(ref mut map) = config {
        map.insert(
            "tracking_physical_sample_id".to_string(),
            Value::String(physical_sample_id.to_string()),
        );
    }

    // POST /projects/{project_id}/configuration with the updated config
    let config_url = format!("{base_endpoint}/projects/{project_id}/configuration");
    client
        .post(&config_url)
        .header("X-API-KEY", api_key)
        .json(&config)
        .send()
        .await
        .context("failed to update project configuration")?
        .error_for_status()
        .context("project configuration update returned error status")?;

    Ok(())
}
