use anyhow::Result;
use reqwest::Client;
use serde_json::Value;
use std::sync::OnceLock;
use tracing_subscriber::{fmt, layer::SubscriberExt, reload, EnvFilter, Registry};

type ReloadHandle = reload::Handle<EnvFilter, Registry>; // <-- match the actual subscriber

static RELOAD: OnceLock<ReloadHandle> = OnceLock::new();

fn level_str(v: u8) -> &'static str {
    match v {
        0 => "off",
        1 => "error",
        2 => "info",
        3 => "debug",
        _ => "trace",
    }
}

pub fn init_tracing_once(default_level: u8) {
    if RELOAD.get().is_some() {
        return; // already initialized
    }

    let filter = EnvFilter::try_new(format!(
        "{}={}",
        env!("CARGO_PKG_NAME"),
        level_str(default_level)
    ))
    .unwrap_or_else(|_| EnvFilter::new("off"));

    // Create a reloadable filter layer and keep its handle.
    let (filter_layer, handle) = reload::Layer::new(filter);

    // Build the subscriber on a Registry (not fmt::Subscriber).
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_level(true)
        .with_ansi(false); // stdout from Python looks cleaner without ANSI

    let subscriber = Registry::default().with(filter_layer).with(fmt_layer);

    let _ = tracing::subscriber::set_global_default(subscriber);
    let _ = RELOAD.set(handle);
}

pub async fn generic_post(client: &Client, url: &str, api_key: &str) -> Result<Value> {
    let req = client.post(url).header("X-API-KEY", api_key);

    let v: Value = req.send().await?.error_for_status()?.json().await?;

    Ok(v)
}
