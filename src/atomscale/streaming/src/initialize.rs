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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
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

/// Check if a string looks like a UUID (e.g., "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").
fn looks_like_uuid(s: &str) -> bool {
    // UUID format: 8-4-4-4-12 hex digits with dashes (36 chars total)
    if s.len() != 36 {
        return false;
    }
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let expected_lens = [8, 4, 4, 4, 12];
    for (part, &expected_len) in parts.iter().zip(&expected_lens) {
        if part.len() != expected_len || !part.chars().all(|c| c.is_ascii_hexdigit()) {
            return false;
        }
    }
    true
}

pub async fn ensure_physical_sample_link(
    client: &Client,
    base_endpoint: &str,
    api_key: &str,
    data_id: &str,
    sample_name_or_id: &str,
) -> Result<String> {
    let sample_name_or_id = sample_name_or_id.trim();
    if sample_name_or_id.is_empty() {
        anyhow::bail!("physical_sample cannot be empty");
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

    // Determine if input looks like a UUID or a name
    let sample_id = if looks_like_uuid(sample_name_or_id) {
        // UUID provided: look up by ID, error if not found
        existing_samples
            .into_iter()
            .find(|sample| sample.id == sample_name_or_id)
            .map(|sample| sample.id)
            .ok_or_else(|| anyhow::anyhow!(
                "physical sample with id '{}' not found",
                sample_name_or_id
            ))?
    } else {
        // Name provided: look up by name (case-insensitive), create if not found
        if let Some(sample) = existing_samples
            .into_iter()
            .find(|sample| sample.name.eq_ignore_ascii_case(sample_name_or_id))
        {
            sample.id
        } else {
            let create_body = CreatePhysicalSampleRequest { name: sample_name_or_id };
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
        }
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

/// Adds a physical sample to a project's tracking list (and project membership).
///
/// Uses `POST /projects/{project_id}/configuration/tracking_samples`, a
/// targeted endpoint that patches only `tracking_physical_sample_ids` and
/// (optionally) `active_tracking_physical_sample_id`, and creates the
/// `ProjectPhysicalSampleAssociation` row, all without re-validating the rest
/// of the project's `GrowthMonitoringConfiguration`.
///
/// This replaces the older flow that did GET `/projects/` followed by
/// POST `/projects/{id}/configuration` to set `tracking_physical_sample_id`.
/// The older endpoint strictly re-validated the entire configuration and
/// rejected on any pre-existing config quirk (references to deleted samples,
/// fields from older schema versions, samples from another org) — so the
/// streamer would surface "project configuration update returned error
/// status" with no useful detail.
#[derive(Serialize)]
struct AddTrackingSampleBody<'a> {
    physical_sample_id: &'a str,
    set_active: bool,
}

pub async fn add_sample_to_project(
    client: &Client,
    base_endpoint: &str,
    api_key: &str,
    project_id: &str,
    physical_sample_id: &str,
) -> Result<()> {
    let url = format!("{base_endpoint}/projects/{project_id}/configuration/tracking_samples");
    let body = AddTrackingSampleBody {
        physical_sample_id,
        set_active: true,
    };

    client
        .post(&url)
        .header("X-API-KEY", api_key)
        .json(&body)
        .send()
        .await
        .context("failed to add tracking sample to project")?
        .error_for_status()
        .context("project tracking-sample update returned error status")?;

    Ok(())
}

#[derive(Deserialize, Debug)]
struct TagSummary {
    id: String,
    name: String,
}

#[derive(Serialize)]
struct CreateTagRequest<'a> {
    name: &'a str,
}

#[derive(Deserialize, Debug)]
struct CreateTagResponse {
    id: String,
}

#[derive(Serialize)]
struct AttachTagsRequest {
    data_ids: Vec<String>,
    tag_ids: Vec<String>,
}

/// Resolves tag inputs to tag IDs and attaches them to a data item.
///
/// For each input string:
///   - If it looks like a UUID, looks up the tag by id (errors if not found).
///   - Otherwise, finds an existing tag by case-insensitive name match, or
///     creates a new tag using the input's exact casing.
///
/// Inputs are trimmed; empty entries are dropped; remaining entries are
/// deduplicated case-insensitively (preserving first-seen casing) before
/// resolution. If no inputs remain, no requests are made and an empty
/// vector is returned.
///
/// On success, issues a single bulk `POST /tags/data-items/` to attach all
/// resolved tag IDs to the given `data_id`.
pub async fn ensure_tags_attached(
    client: &Client,
    base_endpoint: &str,
    api_key: &str,
    data_id: &str,
    tag_inputs: &[String],
) -> Result<Vec<String>> {
    // Trim, drop empties, dedupe case-insensitively (preserve first-seen casing).
    let mut seen_lower: Vec<String> = Vec::new();
    let mut cleaned: Vec<String> = Vec::new();
    for raw in tag_inputs {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_ascii_lowercase();
        if seen_lower.iter().any(|s| s == &lower) {
            continue;
        }
        seen_lower.push(lower);
        cleaned.push(trimmed.to_string());
    }
    if cleaned.is_empty() {
        return Ok(Vec::new());
    }

    let list_url = format!("{base_endpoint}/tags/");
    let existing_tags: Vec<TagSummary> = client
        .get(&list_url)
        .header("X-API-KEY", api_key)
        .send()
        .await
        .context("failed to request tags")?
        .error_for_status()
        .context("tag list returned error status")?
        .json()
        .await
        .context("failed to deserialize tag list")?;

    let mut tag_ids: Vec<String> = Vec::with_capacity(cleaned.len());
    for input in &cleaned {
        let id = if looks_like_uuid(input) {
            existing_tags
                .iter()
                .find(|t| t.id == *input)
                .map(|t| t.id.clone())
                .ok_or_else(|| anyhow::anyhow!("tag with id '{}' not found", input))?
        } else if let Some(tag) = existing_tags
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(input))
        {
            tag.id.clone()
        } else {
            let body = CreateTagRequest { name: input };
            let created: CreateTagResponse = client
                .post(&list_url)
                .header("X-API-KEY", api_key)
                .json(&body)
                .send()
                .await
                .context("failed to create tag")?
                .error_for_status()
                .context("tag creation returned error status")?
                .json()
                .await
                .context("failed to deserialize tag creation response")?;
            created.id
        };
        tag_ids.push(id);
    }

    let attach_url = format!("{base_endpoint}/tags/data-items/");
    let attach_body = AttachTagsRequest {
        data_ids: vec![data_id.to_string()],
        tag_ids: tag_ids.clone(),
    };

    client
        .post(&attach_url)
        .header("X-API-KEY", api_key)
        .json(&attach_body)
        .send()
        .await
        .context("failed to attach tags to data item")?
        .error_for_status()
        .context("tag attach returned error status")?;

    Ok(tag_ids)
}
