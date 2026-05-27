"""FilmSense FS-1 adapter, exposed through the generic adapter host."""

from __future__ import annotations

import threading
from typing import Any

from atomscale.adapters.base import Adapter, StatusEmitter
from atomscale.adapters.filmsense.config import AdapterConfig
from atomscale.adapters.filmsense.service import run_filmsense

# Operator-facing config. Mirrors the dict consumed by ``AdapterConfig.from_dict``.
# The streamer API key is NOT here — it is supplied via the ``AS_API_KEY``
# environment variable so the host never persists a second copy of the secret.
_CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FilmSense FS-1 adapter",
    "type": "object",
    "properties": {
        "fs1": {
            "type": "object",
            "title": "FS-1 instrument",
            "properties": {
                "host": {
                    "type": "string",
                    "title": "FS-1 host / IP",
                    "default": "169.254.1.1",
                },
                "port": {"type": "integer", "title": "TCP port", "default": 4000},
                "acquisition_seconds": {
                    "type": "number",
                    "title": "Acquisition time (s)",
                    "default": 0.5,
                    "minimum": 0.1,
                },
                "model_index": {
                    "type": ["integer", "null"],
                    "title": "Model index (optional)",
                    "default": None,
                },
            },
            "additionalProperties": False,
        },
        "streamer": {
            "type": "object",
            "title": "atomscale streaming",
            "description": "API key comes from the AS_API_KEY environment variable.",
            "properties": {
                "endpoint": {
                    "type": ["string", "null"],
                    "title": "API endpoint (optional)",
                    "default": None,
                },
                "points_per_chunk": {
                    "type": "integer",
                    "default": 10,
                    "minimum": 1,
                },
                "push_interval_seconds": {
                    "type": "number",
                    "title": "Push interval (s)",
                    "default": 5.0,
                    "minimum": 0.5,
                },
            },
            "additionalProperties": False,
        },
        "lifecycle": {
            "type": "object",
            "title": "Run lifecycle (sentinel files)",
            "properties": {
                "watch_dir": {
                    "type": "string",
                    "title": "Sentinel watch directory",
                },
                "poll_interval_seconds": {
                    "type": "number",
                    "default": 1.0,
                    "minimum": 0.1,
                },
                "idle_timeout_seconds": {"type": "number", "default": 300.0},
            },
            "required": ["watch_dir"],
            "additionalProperties": False,
        },
        "wal": {
            "type": "object",
            "title": "Durability (write-ahead log)",
            "properties": {
                "path": {"type": "string", "title": "WAL database path"},
                "retention_hours": {"type": "number", "default": 24.0},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "archive": {
            "type": "object",
            "title": "FS-1 native archive",
            "properties": {
                "enabled": {"type": "boolean", "default": False},
                "folder": {"type": "string", "default": "Default"},
            },
            "additionalProperties": False,
        },
        "dry_run": {
            "type": "boolean",
            "title": "Dry run (parse + map, never push)",
            "default": False,
        },
    },
    "required": ["lifecycle", "wal"],
    "additionalProperties": False,
}


class FilmSenseAdapter(Adapter):
    """Streams FilmSense FS-1 ellipsometry into atomscale in real time."""

    id = "filmsense"
    version = "1"

    def config_schema(self) -> dict[str, Any]:
        return _CONFIG_SCHEMA

    def run(
        self,
        config: dict[str, Any],
        emit: StatusEmitter,
        stop: threading.Event,
    ) -> None:
        adapter_config = AdapterConfig.from_dict(config)
        run_filmsense(adapter_config, stop, emit)
