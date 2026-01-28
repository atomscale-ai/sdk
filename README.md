<p align="center">
  <img src="docs/_static/AtomscaleLogoFull.png" alt="Atomscale" width="300">
</p>

<h1 align="center">Python SDK</h1>

<p align="center">
  <a href="https://github.com/atomscale-ai/sdk/actions/workflows/testing.yml?query=branch%3Amain"><img src="https://github.com/atomscale-ai/sdk/actions/workflows/testing.yml/badge.svg?branch=main" alt="Testing"></a>
  <a href="https://github.com/atomscale-ai/sdk/releases/"><img src="https://img.shields.io/github/tag/atomscale-ai/sdk?include_prereleases=&sort=semver&color=blue" alt="GitHub tag"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python">
  <a href="#license"><img src="https://img.shields.io/badge/License-GPLv3-blue" alt="License"></a>
</p>

<p align="center">
  <a href="https://atomscale-ai.github.io/sdk/"><img src="https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge" alt="Documentation"></a>
</p>

---

The official Python SDK for the [Atomscale](https://www.atomscale.ai) platform. Upload RHEED videos, stream live data, search the catalogue, and retrieve analysis results programmatically.

## Features

- **Unified Client** – Single interface for uploads, search, and downloads
- **Live Streaming** – Push or pull RHEED frames in real-time
- **Flexible Search** – Filter by keywords, data type, status, or time bounds
- **Rich Results** – Access timeseries, diffraction graphs, and processed videos
- **Polling Utilities** – Sync, async, and threaded options for monitoring updates

## Installation

```bash
pip install atomscale
```

> **Note:** The package was renamed from `atomicds`. Importing `atomicds` still works but emits a `DeprecationWarning`.

## Quick Start

```python
from atomscale import Client

# Create a client (reads AS_API_KEY from environment)
client = Client()

# Upload files
client.upload(files=["rheed_video.mp4"])

# Search the catalogue
results = client.search(keywords=["GaN"], status="success")

# Fetch analysis results
items = client.get(results["Data ID"].to_list())
for item in items:
    print(item.timeseries_data.tail())
```

## Documentation

Full documentation is available at **[atomscale-ai.github.io/sdk](https://atomscale-ai.github.io/sdk/)**.

- [Quickstart Guide](https://atomscale-ai.github.io/sdk/guides/quickstart.html)
- [Upload Data](https://atomscale-ai.github.io/sdk/guides/upload-data.html)
- [Search the Catalogue](https://atomscale-ai.github.io/sdk/guides/search-data.html)
- [Stream RHEED Video](https://atomscale-ai.github.io/sdk/guides/stream-rheed.html)
- [API Reference](https://atomscale-ai.github.io/sdk/modules.html)

## License

This project is licensed under the [GPLv3 License](LICENSE).
