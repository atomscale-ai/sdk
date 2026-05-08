from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from atomscale.adapters.filmsense.client import FilmSenseClient
from atomscale.adapters.filmsense.config import (
    AdapterConfig,
    ArchiveConfig,
    FS1Config,
    LifecycleConfig,
    StreamerConfig,
    WalConfig,
)
from atomscale.adapters.filmsense.runner import FilmSenseRunner, RunMetadata


@dataclass
class StubStreamer:
    """Implements StreamerProtocol; records every call for assertions."""

    init_args: dict[str, Any] | None = None
    pushes: list[dict[str, Any]] = field(default_factory=list)
    finalized_data_ids: list[str] = field(default_factory=list)
    fail_push: bool = False

    def initialize(
        self,
        stream_name: str | None = None,
        synth_source_id: int | None = None,
        physical_sample: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        self.init_args = {
            "stream_name": stream_name,
            "synth_source_id": synth_source_id,
            "physical_sample": physical_sample,
            "project_id": project_id,
            "tags": tags,
        }
        return "data-id-stub"

    def push_multi(
        self,
        data_id: str,
        chunk_index: int,
        channels: dict[str, dict[str, Any]],
    ) -> None:
        if self.fail_push:
            msg = "stub push failure"
            raise RuntimeError(msg)
        self.pushes.append(
            {
                "data_id": data_id,
                "chunk_index": chunk_index,
                "channels": {k: dict(v) for k, v in channels.items()},
            }
        )

    def finalize(self, data_id: str) -> None:
        self.finalized_data_ids.append(data_id)


def _make_config(tmp_path, host: str, port: int, *, push_interval=0.05, dry_run=False, archive=False):
    return AdapterConfig(
        fs1=FS1Config(host=host, port=port, acquisition_seconds=0.01),
        streamer=StreamerConfig(
            api_key="test-key",
            push_interval_seconds=push_interval,
            points_per_chunk=10,
        ),
        lifecycle=LifecycleConfig(watch_dir=tmp_path / "watch"),
        wal=WalConfig(path=tmp_path / "wal.sqlite"),
        archive=ArchiveConfig(enabled=archive),
        dry_run=dry_run,
    )


def _drive_runner(runner: FilmSenseRunner, metadata: RunMetadata, *, run_for: float = 0.3):
    stop_event = threading.Event()
    result: dict[str, Any] = {}

    def _go():
        try:
            result["data_id"] = runner.run_session(metadata, stop_event)
        except BaseException as e:  # noqa: BLE001
            result["error"] = e

    thread = threading.Thread(target=_go, daemon=True)
    thread.start()
    time.sleep(run_for)
    stop_event.set()
    thread.join(timeout=5)
    assert not thread.is_alive(), "runner thread did not stop"
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result["data_id"]


@pytest.fixture
def populated_fs1(mock_fs1):
    """Mock FS-1 that returns the same parameter snapshot for every GetParms call."""
    parms = [
        ("Psi_465", 21.5),
        ("Delta_465", 170.25),
        ("Thickness", 12.34),
        ("MSE", 0.0023),
    ]
    # Pre-load enough snapshots for many polls; queue refills if exhausted.
    for _ in range(200):
        mock_fs1.state.queue_parms(list(parms))
    return mock_fs1


def test_run_session_initializes_with_metadata(populated_fs1, tmp_path):
    config = _make_config(tmp_path, populated_fs1.host, populated_fs1.port)
    streamer = StubStreamer()
    runner = FilmSenseRunner(config, streamer)

    metadata = RunMetadata(
        stream_name="Run-2026-05-07",
        physical_sample="wafer-001",
        project_id="proj-uuid",
        tags=["customer:hinkle", "tool:fs1-bay3"],
    )
    data_id = _drive_runner(runner, metadata)

    assert data_id == "data-id-stub"
    assert streamer.init_args == {
        "stream_name": "Run-2026-05-07",
        "synth_source_id": None,
        "physical_sample": "wafer-001",
        "project_id": "proj-uuid",
        "tags": ["customer:hinkle", "tool:fs1-bay3"],
    }


def test_run_session_pushes_normalized_channels(populated_fs1, tmp_path):
    config = _make_config(tmp_path, populated_fs1.host, populated_fs1.port)
    streamer = StubStreamer()
    runner = FilmSenseRunner(config, streamer)
    _drive_runner(runner, RunMetadata(stream_name="run"))

    assert streamer.pushes, "expected at least one push"

    # Inspect the first push: channels should be normalized names with units.
    first = streamer.pushes[0]
    assert first["data_id"] == "data-id-stub"
    assert first["chunk_index"] == 0
    channels = first["channels"]

    assert "psi_465" in channels
    assert channels["psi_465"]["units"] == "deg"
    assert channels["delta_465"]["units"] == "deg"
    assert "thickness" in channels and channels["thickness"]["units"] == "nm"
    assert "mse_fit" in channels  # MSE → mse_fit, no units

    # Each channel should have at least 1 sample with parallel timestamps/values.
    for payload in channels.values():
        assert len(payload["timestamps"]) == len(payload["values"])
        assert len(payload["timestamps"]) >= 1
        # Timestamps must be Unix seconds (≥ 2020 epoch)
        assert all(t > 1_577_836_800 for t in payload["timestamps"])


def test_run_session_finalizes_even_after_push_error(populated_fs1, tmp_path):
    config = _make_config(tmp_path, populated_fs1.host, populated_fs1.port)
    streamer = StubStreamer(fail_push=True)
    runner = FilmSenseRunner(config, streamer)
    data_id = _drive_runner(runner, RunMetadata())

    assert streamer.finalized_data_ids == [data_id]


def test_dry_run_skips_push_but_still_finalizes(populated_fs1, tmp_path):
    config = _make_config(tmp_path, populated_fs1.host, populated_fs1.port, dry_run=True)
    streamer = StubStreamer()
    runner = FilmSenseRunner(config, streamer)
    _drive_runner(runner, RunMetadata())

    assert streamer.pushes == []
    assert streamer.finalized_data_ids == ["data-id-stub"]


def test_archive_triggers_save_dynamic_measurements(populated_fs1, tmp_path):
    config = _make_config(
        tmp_path, populated_fs1.host, populated_fs1.port, archive=True
    )
    streamer = StubStreamer()
    runner = FilmSenseRunner(config, streamer)
    _drive_runner(runner, RunMetadata(stream_name="archive-run"))

    # SaveDynamicMeasurements should have been issued with the configured folder.
    assert populated_fs1.state.saved == ("Default", "archive-run")


def test_chunk_indexes_are_monotonic(populated_fs1, tmp_path):
    config = _make_config(
        tmp_path, populated_fs1.host, populated_fs1.port, push_interval=0.02
    )
    streamer = StubStreamer()
    runner = FilmSenseRunner(config, streamer)
    _drive_runner(runner, RunMetadata(), run_for=0.4)

    indexes = [p["chunk_index"] for p in streamer.pushes]
    assert indexes == sorted(indexes)
    assert indexes[0] == 0
    if len(indexes) > 1:
        # Strictly increasing, no duplicates
        assert all(b > a for a, b in zip(indexes, indexes[1:]))


def test_acq_time_is_set_on_fs1(populated_fs1, tmp_path):
    config = _make_config(tmp_path, populated_fs1.host, populated_fs1.port)
    streamer = StubStreamer()
    runner = FilmSenseRunner(config, streamer)
    _drive_runner(runner, RunMetadata())

    # FS1Config.acquisition_seconds = 0.01 in _make_config
    assert populated_fs1.state.acq_time == pytest.approx(0.01, abs=1e-6)


def test_wal_records_pushed_chunks(populated_fs1, tmp_path):
    config = _make_config(tmp_path, populated_fs1.host, populated_fs1.port)
    streamer = StubStreamer()
    runner = FilmSenseRunner(config, streamer)
    _drive_runner(runner, RunMetadata())

    # All successfully-pushed chunks should be acknowledged in the WAL.
    pending = runner._wal.pending("data-id-stub")
    assert pending == [], f"expected zero pending WAL chunks, got {len(pending)}"
