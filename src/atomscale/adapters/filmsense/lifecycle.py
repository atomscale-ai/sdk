"""Sentinel-file lifecycle watcher.

The adapter cooperates with whatever chamber-control / MES system Hinkle
already runs by watching a configured directory for ``*.run-start`` and
``*.run-end`` JSON files. Each ``*.run-start`` file kicks off a session and is
moved to a sibling ``processed/`` directory once consumed.

Sentinel-file format (``*.run-start.json`` / ``*.run-end.json``)::

    {
      "recipe": "Al2O3-ALD-process-A",
      "physical_sample": "wafer-2026-05-07-001",
      "project_id": "uuid-or-null",
      "tags": ["customer:hinkle", "tool:fs1-bay3"]
    }

Implemented with stdlib polling so the MVP has zero new runtime
dependencies. ``watchdog`` would give us OS-level events; trade-off is more
deps for marginally lower latency on file appearance.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from atomscale.adapters.filmsense.config import LifecycleConfig
from atomscale.adapters.filmsense.runner import RunMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentinelEvent:
    kind: str  # "start" or "end"
    path: Path
    metadata: RunMetadata


def parse_sentinel(path: Path) -> RunMetadata:
    """Parse a ``*.run-start`` / ``*.run-end`` JSON file."""
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        msg = f"sentinel {path}: expected JSON object, got {type(data).__name__}"
        raise ValueError(msg)

    tags = data.get("tags") or []
    if not isinstance(tags, list):
        msg = f"sentinel {path}: 'tags' must be a list of strings"
        raise ValueError(msg)
    tags = [str(t) for t in tags]
    # Always tag the run with our identification so atomscale queries can find it.
    if "instrument:filmsense_fs1" not in tags:
        tags.append("instrument:filmsense_fs1")
    if data.get("recipe"):
        tags.append(f"model:{data['recipe']}")

    return RunMetadata(
        stream_name=data.get("recipe") or path.stem.replace(".run-start", ""),
        physical_sample=data.get("physical_sample"),
        project_id=data.get("project_id"),
        tags=tags,
    )


class SentinelWatcher:
    """Polling watcher that fires callbacks on run-start / run-end files.

    Files are moved to ``<watch_dir>/processed/`` after being consumed so the
    same sentinel never fires twice.
    """

    START_SUFFIX = ".run-start"
    END_SUFFIX = ".run-end"

    def __init__(
        self,
        config: LifecycleConfig,
        on_start: Callable[[SentinelEvent], None],
        on_end: Callable[[SentinelEvent], None],
    ) -> None:
        self.config = config
        self.on_start = on_start
        self.on_end = on_end
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.config.watch_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir = self.config.watch_dir / "processed"
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # ----------------------------------------------------------- internals

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan_once()
            except Exception:
                logger.exception("sentinel scan tick failed")
            self._stop.wait(self.config.poll_interval_seconds)

    def _scan_once(self) -> None:
        # Sort by mtime so a paired start/end fires in order.
        candidates: list[tuple[float, Path]] = []
        for entry in self.config.watch_dir.iterdir():
            if not entry.is_file():
                continue
            if entry.suffix == ".json" and (
                entry.stem.endswith(self.START_SUFFIX)
                or entry.stem.endswith(self.END_SUFFIX)
            ):
                try:
                    candidates.append((entry.stat().st_mtime, entry))
                except FileNotFoundError:
                    continue
        candidates.sort()

        for _, path in candidates:
            try:
                metadata = parse_sentinel(path)
            except (ValueError, json.JSONDecodeError) as e:
                logger.error("invalid sentinel %s: %s", path, e)
                self._move_to_processed(path)
                continue

            kind = "start" if path.stem.endswith(self.START_SUFFIX) else "end"
            event = SentinelEvent(kind=kind, path=path, metadata=metadata)
            try:
                if kind == "start":
                    self.on_start(event)
                else:
                    self.on_end(event)
            finally:
                self._move_to_processed(path)

    def _move_to_processed(self, path: Path) -> None:
        try:
            target = self._processed_dir / f"{int(time.time() * 1000)}_{path.name}"
            path.rename(target)
        except OSError:
            logger.exception("failed to archive sentinel %s", path)
