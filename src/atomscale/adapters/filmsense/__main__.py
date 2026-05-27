"""CLI entry point for the FilmSense adapter (TOML config).

Run as::

    python -m atomscale.adapters.filmsense --config /etc/filmsense.toml

Thin convenience wrapper around the shared run wiring in
``service.run_filmsense`` — the generic adapter host
(``python -m atomscale.adapters run filmsense``) drives the same code path.
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
from atomscale.adapters.filmsense.service import run_filmsense

logger = logging.getLogger(__name__)


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
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = AdapterConfig.from_toml(args.config)
    if args.dry_run:
        config = replace(config, dry_run=True)

    stop = threading.Event()

    def _signal_handler(signum, _frame):
        logger.info("received signal %s; shutting down", signum)
        stop.set()

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    run_filmsense(config, stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
