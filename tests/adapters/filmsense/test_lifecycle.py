from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from atomscale.adapters.filmsense.config import LifecycleConfig
from atomscale.adapters.filmsense.lifecycle import (
    SentinelEvent,
    SentinelWatcher,
    parse_sentinel,
)


def _write_sentinel(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def test_parse_sentinel_basic(tmp_path: Path):
    p = tmp_path / "myrun.run-start.json"
    _write_sentinel(p, {
        "recipe": "Al2O3",
        "physical_sample": "wafer-1",
        "project_id": "proj-uuid",
        "tags": ["env:prod"],
    })
    md = parse_sentinel(p)
    assert md.stream_name == "Al2O3"
    assert md.physical_sample == "wafer-1"
    assert md.project_id == "proj-uuid"
    assert "env:prod" in md.tags
    assert "instrument:filmsense_fs1" in md.tags
    assert "model:Al2O3" in md.tags


def test_parse_sentinel_missing_recipe_uses_filename(tmp_path: Path):
    p = tmp_path / "lot42.run-start.json"
    _write_sentinel(p, {"physical_sample": "wafer-1"})
    md = parse_sentinel(p)
    assert md.stream_name == "lot42"


def test_parse_sentinel_invalid_tags_rejected(tmp_path: Path):
    p = tmp_path / "bad.run-start.json"
    _write_sentinel(p, {"tags": "not-a-list"})
    with pytest.raises(ValueError, match="tags"):
        parse_sentinel(p)


def test_watcher_fires_callbacks_in_order(tmp_path: Path):
    config = LifecycleConfig(watch_dir=tmp_path / "watch", poll_interval_seconds=0.05)
    config.watch_dir.mkdir(parents=True, exist_ok=True)

    started: list[SentinelEvent] = []
    ended: list[SentinelEvent] = []
    done = threading.Event()

    def on_start(e: SentinelEvent) -> None:
        started.append(e)

    def on_end(e: SentinelEvent) -> None:
        ended.append(e)
        done.set()

    watcher = SentinelWatcher(config, on_start, on_end)
    watcher.start()
    try:
        _write_sentinel(
            config.watch_dir / "run1.run-start.json",
            {"recipe": "A", "physical_sample": "wafer-1"},
        )
        # Tiny gap so mtime-ordering is unambiguous on filesystems with low
        # resolution timestamps.
        time.sleep(0.05)
        _write_sentinel(
            config.watch_dir / "run1.run-end.json",
            {"recipe": "A", "physical_sample": "wafer-1"},
        )
        assert done.wait(2.0), "watcher did not fire on_end within 2s"
    finally:
        watcher.stop()

    assert [e.metadata.stream_name for e in started] == ["A"]
    assert [e.metadata.stream_name for e in ended] == ["A"]


def test_watcher_archives_processed_files(tmp_path: Path):
    config = LifecycleConfig(watch_dir=tmp_path / "watch", poll_interval_seconds=0.05)
    config.watch_dir.mkdir(parents=True, exist_ok=True)

    fired = threading.Event()

    def on_start(_e: SentinelEvent) -> None:
        fired.set()

    watcher = SentinelWatcher(config, on_start, lambda _e: None)
    watcher.start()
    try:
        _write_sentinel(
            config.watch_dir / "x.run-start.json", {"physical_sample": "wafer-1"}
        )
        assert fired.wait(2.0)
        # Allow the watcher's post-callback move to land
        time.sleep(0.2)
    finally:
        watcher.stop()

    # The original sentinel is gone; an archived copy exists in processed/
    assert not (config.watch_dir / "x.run-start.json").exists()
    archived = list((config.watch_dir / "processed").iterdir())
    assert len(archived) == 1
    assert archived[0].name.endswith("x.run-start.json")


def test_watcher_handles_invalid_json(tmp_path: Path):
    config = LifecycleConfig(watch_dir=tmp_path / "watch", poll_interval_seconds=0.05)
    config.watch_dir.mkdir(parents=True, exist_ok=True)

    fired = threading.Event()

    def on_start(_e: SentinelEvent) -> None:
        fired.set()

    watcher = SentinelWatcher(config, on_start, lambda _e: None)
    watcher.start()
    try:
        # Bad JSON
        (config.watch_dir / "broken.run-start.json").write_text("not json")
        # Then a valid one to confirm the watcher recovered
        time.sleep(0.1)
        _write_sentinel(
            config.watch_dir / "good.run-start.json", {"recipe": "B"}
        )
        assert fired.wait(2.0), "watcher did not recover after bad sentinel"
    finally:
        watcher.stop()

    # Both files were archived (bad sentinel still moved out of the way)
    archived = list((config.watch_dir / "processed").iterdir())
    assert len(archived) == 2
