"""CLI entry point for the FilmSense adapter.

Run as::

    python -m atomscale.adapters.filmsense --config /etc/filmsense.toml

Or, when shipped as a Windows service, the service shim invokes ``main()``
directly. The CLI wires the sentinel-file lifecycle watcher to a single
``FilmSenseRunner`` so that ``*.run-start`` files spawn one in-flight
session and ``*.run-end`` (or idle timeout) finalizes it.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from dataclasses import replace
from pathlib import Path

from atomscale.adapters.filmsense.config import AdapterConfig
from atomscale.adapters.filmsense.lifecycle import SentinelEvent, SentinelWatcher
from atomscale.adapters.filmsense.runner import FilmSenseRunner

logger = logging.getLogger(__name__)


def _build_streamer(config: AdapterConfig):
    """Construct the production atomscale TimeseriesStreamer.

    Imported lazily so unit tests don't trigger the Rust extension load.
    """
    from atomscale.streaming import TimeseriesStreamer  # noqa: PLC0415

    return TimeseriesStreamer(
        api_key=config.streamer.api_key,
        endpoint=config.streamer.endpoint,
        points_per_chunk=config.streamer.points_per_chunk,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="atomscale.adapters.filmsense",
        description="FilmSense FS-1 → atomscale real-time ingestion adapter",
    )
    parser.add_argument("--config", type=Path, required=True, help="TOML config path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse FS-1 output and log chunks but never call streamer.push",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = AdapterConfig.from_toml(args.config)
    if args.dry_run:
        config = replace(config, dry_run=True)

    streamer = _build_streamer(config)
    runner = FilmSenseRunner(config, streamer)

    session_lock = threading.Lock()
    active_session: dict[str, threading.Event] = {}
    shutdown = threading.Event()

    def on_start(event: SentinelEvent) -> None:
        with session_lock:
            if active_session:
                logger.warning(
                    "ignoring run-start while a session is in flight (run %r)",
                    event.metadata.stream_name,
                )
                return
            stop_event = threading.Event()
            active_session["stop"] = stop_event
            logger.info("starting session: %s", event.metadata.stream_name)

        def _do_run():
            try:
                runner.run_session(event.metadata, stop_event)
            except Exception:
                logger.exception("session crashed")
            finally:
                with session_lock:
                    active_session.clear()

        threading.Thread(target=_do_run, daemon=True).start()

    def on_end(event: SentinelEvent) -> None:
        with session_lock:
            stop_event = active_session.get("stop")
        if stop_event is None:
            logger.warning(
                "run-end seen with no active session: %s",
                event.metadata.stream_name,
            )
            return
        logger.info("stopping session: %s", event.metadata.stream_name)
        stop_event.set()

    watcher = SentinelWatcher(config.lifecycle, on_start, on_end)
    watcher.start()

    def _signal_handler(signum, _frame):
        logger.info("received signal %s; shutting down", signum)
        shutdown.set()

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    logger.info(
        "filmsense adapter ready; watching %s (dry_run=%s)",
        config.lifecycle.watch_dir,
        config.dry_run,
    )

    try:
        shutdown.wait()
    finally:
        # Cleanly stop in-flight session if any
        with session_lock:
            stop_event = active_session.get("stop")
        if stop_event is not None:
            stop_event.set()
        watcher.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
