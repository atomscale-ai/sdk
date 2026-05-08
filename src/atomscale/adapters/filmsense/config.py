"""TOML-backed configuration for the FilmSense adapter."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_tomllib():
    """Lazy import so callers using only ``from_dict`` don't need tomli on 3.10."""
    if sys.version_info >= (3, 11):
        import tomllib  # noqa: PLC0415

        return tomllib
    try:
        import tomli  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        msg = (
            "Python 3.10 requires the 'tomli' package to read TOML config files. "
            "Install it with `pip install tomli`."
        )
        raise ImportError(msg) from e
    return tomli


@dataclass(frozen=True)
class FS1Config:
    """Connection settings for the FilmSense FS-1 instrument."""

    host: str = "169.254.1.1"
    port: int = 4000
    acquisition_seconds: float = 0.5
    """Per-measurement acquisition time. FS-1 default is 0.4-1.0 s."""
    model_index: int | None = None
    """Optional ``SetModel`` index. None = leave whatever model is loaded."""


@dataclass(frozen=True)
class StreamerConfig:
    """Settings for the atomscale ``TimeseriesStreamer``."""

    api_key: str
    endpoint: str | None = None
    points_per_chunk: int = 10
    push_interval_seconds: float = 5.0
    """Wall-clock cadence for flushing buffered samples to atomscale."""


@dataclass(frozen=True)
class LifecycleConfig:
    """Settings for the run-start/run-end sentinel-file watcher."""

    watch_dir: Path
    poll_interval_seconds: float = 1.0
    idle_timeout_seconds: float = 300.0
    """Auto-finalize a run after this many seconds without new samples."""


@dataclass(frozen=True)
class WalConfig:
    """Settings for the local SQLite write-ahead log."""

    path: Path
    retention_hours: float = 24.0


@dataclass(frozen=True)
class ArchiveConfig:
    """Settings for the FS-1 native binary archive (file-watcher sidecar)."""

    enabled: bool = False
    folder: str = "Default"
    """Folder name on the FS-1 (passed to SaveDynamicMeasurements)."""


@dataclass(frozen=True)
class AdapterConfig:
    """Top-level adapter configuration."""

    fs1: FS1Config
    streamer: StreamerConfig
    lifecycle: LifecycleConfig
    wal: WalConfig
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    dry_run: bool = False
    """If True, parse and map but never call ``streamer.push``."""

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> AdapterConfig:
        tomllib = _load_tomllib()
        with Path(path).open("rb") as fh:
            data = tomllib.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdapterConfig:
        fs1_data = data.get("fs1", {})
        streamer_data = data.get("streamer", {})
        lifecycle_data = data.get("lifecycle", {})
        wal_data = data.get("wal", {})
        archive_data = data.get("archive", {})

        api_key = streamer_data.get("api_key") or os.environ.get("AS_API_KEY")
        if not api_key:
            msg = (
                "streamer.api_key is required (set in TOML or via AS_API_KEY env var)"
            )
            raise ValueError(msg)

        return cls(
            fs1=FS1Config(**fs1_data),
            streamer=StreamerConfig(
                api_key=api_key,
                endpoint=streamer_data.get("endpoint"),
                points_per_chunk=streamer_data.get("points_per_chunk", 10),
                push_interval_seconds=streamer_data.get("push_interval_seconds", 5.0),
            ),
            lifecycle=LifecycleConfig(
                watch_dir=Path(lifecycle_data["watch_dir"]).expanduser(),
                poll_interval_seconds=lifecycle_data.get("poll_interval_seconds", 1.0),
                idle_timeout_seconds=lifecycle_data.get("idle_timeout_seconds", 300.0),
            ),
            wal=WalConfig(
                path=Path(wal_data["path"]).expanduser(),
                retention_hours=wal_data.get("retention_hours", 24.0),
            ),
            archive=ArchiveConfig(
                enabled=archive_data.get("enabled", False),
                folder=archive_data.get("folder", "Default"),
            ),
            dry_run=data.get("dry_run", False),
        )
