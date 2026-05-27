"""Shared FilmSense run wiring, used by both the adapter host and legacy CLI.

Builds the streamer + runner, wires the sentinel-file run lifecycle, and runs
until a stop event is set. Run start/stop is driven by ``*.run-start`` /
``*.run-end`` sentinel files that the chamber-control / MES system writes; the
caller's ``stop`` event controls the *process*, not an individual run.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from atomscale.adapters.base import StatusEmitter
from atomscale.adapters.filmsense.config import AdapterConfig
from atomscale.adapters.filmsense.lifecycle import SentinelEvent, SentinelWatcher
from atomscale.adapters.filmsense.runner import FilmSenseRunner, StreamerProtocol

logger = logging.getLogger(__name__)

_STOP_POLL_SECONDS = 0.5
_HEARTBEAT_SECONDS = 15.0


def build_streamer(config: AdapterConfig) -> StreamerProtocol:
    """Construct the production atomscale ``TimeseriesStreamer``.

    Imported lazily so callers that inject a streamer (tests, or a dry run
    against a stub) don't trigger the Rust extension load.
    """
    from atomscale.streaming import TimeseriesStreamer  # noqa: PLC0415

    return TimeseriesStreamer(
        api_key=config.streamer.api_key,
        endpoint=config.streamer.endpoint,
        points_per_chunk=config.streamer.points_per_chunk,
    )


def _runner_event_sink(
    emit: StatusEmitter | None,
) -> Any:
    """Adapt a ``StatusEmitter`` to the runner's ``(name, fields)`` callback."""
    if emit is None:
        return None

    def _sink(name: str, fields: dict[str, Any]) -> None:
        emit.emit(name, **fields)

    return _sink


def run_filmsense(
    config: AdapterConfig,
    stop: threading.Event,
    emit: StatusEmitter | None = None,
    *,
    streamer: StreamerProtocol | None = None,
    runner: FilmSenseRunner | None = None,
) -> None:
    """Run the FilmSense adapter until ``stop`` is set.

    ``streamer`` / ``runner`` are injection points for testing; production
    callers pass neither and a real streamer is built from ``config``.
    """
    if streamer is None:
        streamer = build_streamer(config)
    if runner is None:
        runner = FilmSenseRunner(config, streamer, events=_runner_event_sink(emit))

    state_lock = threading.Lock()
    current: dict[str, Any] = {}  # {"stop": Event, "thread": Thread} while a run is live

    def on_start(event: SentinelEvent) -> None:
        with state_lock:
            if current.get("stop") is not None:
                logger.warning(
                    "ignoring run-start while a session is in flight (run %r)",
                    event.metadata.stream_name,
                )
                if emit is not None:
                    emit.warning(
                        "ignoring run-start; a session is already in flight",
                        stream_name=event.metadata.stream_name,
                    )
                return
            session_stop = threading.Event()
            current["stop"] = session_stop
        logger.info("starting session: %s", event.metadata.stream_name)
        if emit is not None:
            emit.run_start(
                stream_name=event.metadata.stream_name,
                physical_sample=event.metadata.physical_sample,
            )

        def _do_run() -> None:
            try:
                runner.run_session(event.metadata, session_stop)
            except Exception:
                logger.exception("session crashed")
                if emit is not None:
                    emit.error(
                        "session crashed", stream_name=event.metadata.stream_name
                    )
            finally:
                with state_lock:
                    current.clear()
                if emit is not None:
                    emit.run_end(stream_name=event.metadata.stream_name)

        thread = threading.Thread(target=_do_run, daemon=True)
        with state_lock:
            current["thread"] = thread
        thread.start()

    def on_end(event: SentinelEvent) -> None:
        with state_lock:
            session_stop = current.get("stop")
        if session_stop is None:
            logger.warning(
                "run-end seen with no active session: %s", event.metadata.stream_name
            )
            if emit is not None:
                emit.warning(
                    "run-end with no active session",
                    stream_name=event.metadata.stream_name,
                )
            return
        logger.info("stopping session: %s", event.metadata.stream_name)
        session_stop.set()

    watcher = SentinelWatcher(config.lifecycle, on_start, on_end)
    watcher.start()
    logger.info(
        "filmsense adapter ready; watching %s (dry_run=%s)",
        config.lifecycle.watch_dir,
        config.dry_run,
    )
    if emit is not None:
        emit.emit(
            "watching",
            watch_dir=str(config.lifecycle.watch_dir),
            dry_run=config.dry_run,
        )

    try:
        last_heartbeat = time.monotonic()
        while not stop.wait(_STOP_POLL_SECONDS):
            now = time.monotonic()
            if emit is not None and now - last_heartbeat >= _HEARTBEAT_SECONDS:
                with state_lock:
                    session_active = current.get("stop") is not None
                emit.heartbeat(session_active=session_active)
                last_heartbeat = now
    finally:
        with state_lock:
            session_stop = current.get("stop")
            session_thread = current.get("thread")
        if session_stop is not None:
            session_stop.set()
        if session_thread is not None:
            session_thread.join(timeout=10)
        watcher.stop()
