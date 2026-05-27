"""Tests for FilmSenseAdapter and the shared ``run_filmsense`` service wiring."""

from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from atomscale.adapters.base import StatusEmitter
from atomscale.adapters.filmsense.adapter import FilmSenseAdapter
from atomscale.adapters.filmsense.config import (
    AdapterConfig,
    ArchiveConfig,
    FS1Config,
    LifecycleConfig,
    StreamerConfig,
    WalConfig,
)
from atomscale.adapters.filmsense.service import run_filmsense


@dataclass
class StubStreamer:
    """Minimal StreamerProtocol implementation that records calls."""

    pushes: list[dict[str, Any]] = field(default_factory=list)
    finalized: list[str] = field(default_factory=list)

    def initialize(
        self,
        stream_name: str | None = None,
        synth_source_id: int | None = None,
        physical_sample: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        return "data-id-stub"

    def push_multi(
        self,
        data_id: str,
        chunk_index: int,
        channels: dict[str, dict[str, Any]],
    ) -> None:
        self.pushes.append({"chunk_index": chunk_index, "channels": dict(channels)})

    def finalize(self, data_id: str) -> None:
        self.finalized.append(data_id)


def _config(tmp_path, host: str, port: int) -> AdapterConfig:
    return AdapterConfig(
        fs1=FS1Config(host=host, port=port, acquisition_seconds=0.01),
        streamer=StreamerConfig(
            api_key="test-key", push_interval_seconds=0.05, points_per_chunk=10
        ),
        lifecycle=LifecycleConfig(
            watch_dir=tmp_path / "watch", poll_interval_seconds=0.05
        ),
        wal=WalConfig(path=tmp_path / "wal.sqlite"),
        archive=ArchiveConfig(enabled=False),
    )


def test_manifest_shape():
    manifest = FilmSenseAdapter().manifest()
    assert manifest["id"] == "filmsense"
    assert manifest["version"] == "1"
    schema = manifest["config_schema"]
    assert schema["type"] == "object"
    assert set(schema["required"]) >= {"lifecycle", "wal"}


def test_config_from_schema_dict(tmp_path, monkeypatch):
    """A schema-shaped config dict (api key from env) builds an AdapterConfig."""
    monkeypatch.setenv("AS_API_KEY", "env-key")
    raw = {
        "fs1": {"host": "127.0.0.1", "port": 4000, "acquisition_seconds": 0.01},
        "streamer": {"points_per_chunk": 10, "push_interval_seconds": 0.05},
        "lifecycle": {"watch_dir": str(tmp_path / "watch")},
        "wal": {"path": str(tmp_path / "wal.sqlite")},
    }
    config = AdapterConfig.from_dict(raw)
    assert config.streamer.api_key == "env-key"
    assert config.fs1.host == "127.0.0.1"


def test_run_filmsense_streams_and_emits_status(mock_fs1, tmp_path):
    parms = [("Psi_465", 21.5), ("Thickness", 12.3), ("MSE", 0.002)]
    for _ in range(500):
        mock_fs1.state.queue_parms(list(parms))

    config = _config(tmp_path, mock_fs1.host, mock_fs1.port)
    config.lifecycle.watch_dir.mkdir(parents=True, exist_ok=True)

    streamer = StubStreamer()
    buf = io.StringIO()
    emit = StatusEmitter(stream=buf)
    stop = threading.Event()

    thread = threading.Thread(
        target=run_filmsense,
        args=(config, stop, emit),
        kwargs={"streamer": streamer},
        daemon=True,
    )
    thread.start()
    try:
        # A run-start sentinel triggers a streaming session.
        (config.lifecycle.watch_dir / "run1.run-start.json").write_text(
            json.dumps({"recipe": "A", "physical_sample": "wafer-1"})
        )
        _wait_until(lambda: bool(streamer.pushes), timeout=5)
        assert streamer.pushes, "expected at least one chunk pushed"

        # A run-end sentinel stops the session.
        (config.lifecycle.watch_dir / "run1.run-end.json").write_text(
            json.dumps({"recipe": "A"})
        )
        _wait_until(lambda: '"run_end"' in buf.getvalue(), timeout=5)
    finally:
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

    events = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    kinds = {e["event"] for e in events}
    assert {"watching", "run_start", "session_initialized", "chunk", "run_end"} <= kinds

    chunk = next(e for e in events if e["event"] == "chunk")
    assert chunk["points"] >= 1
    assert "psi_465" in chunk["channels"]
    assert streamer.finalized == ["data-id-stub"]

    # Every line is a well-formed, versioned status event.
    assert all(e["protocol_version"] == 1 for e in events)


def _wait_until(predicate, *, timeout: float, interval: float = 0.05) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
