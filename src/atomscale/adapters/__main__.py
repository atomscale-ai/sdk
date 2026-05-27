"""atomscale ingestion-adapter host.

Usage::

    python -m atomscale.adapters list
    python -m atomscale.adapters run <adapter_id> [--config PATH|-] [--dry-run]

``list`` prints a JSON array of adapter manifests (id, version, config_schema)
to stdout — the host GUI renders a settings form from each schema.

``run`` launches one adapter and streams a JSON-line status feed on stdout until
SIGINT/SIGTERM or a ``stop`` line arrives on stdin. The streamer API key is read
from the ``AS_API_KEY`` environment variable (never passed in the config).

stdout is reserved for machine-readable output (manifests / status events);
all logging goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from pathlib import Path

from atomscale.adapters import registry
from atomscale.adapters.base import StatusEmitter

logger = logging.getLogger(__name__)


def _cmd_list() -> int:
    manifests = [adapter.manifest() for adapter in registry.available().values()]
    sys.stdout.write(json.dumps(manifests, indent=2, default=str) + "\n")
    return 0


def _load_config(source: str | None) -> dict:
    """Load config JSON from a file path, or stdin when ``source`` is ``None``/``-``."""
    raw = sys.stdin.read() if source in (None, "-") else Path(source).read_text("utf-8")
    return json.loads(raw) if raw.strip() else {}


def _watch_stdin_for_stop(stop: threading.Event, emit: StatusEmitter) -> None:
    """Set ``stop`` when a ``stop`` line arrives on stdin (host control channel)."""
    try:
        for line in sys.stdin:
            if line.strip().lower() == "stop":
                emit.warning("received 'stop' on stdin; stopping")
                stop.set()
                return
    except (OSError, ValueError):
        return


def _cmd_run(adapter_id: str, config_source: str | None, *, dry_run: bool) -> int:
    emit = StatusEmitter()

    try:
        adapter = registry.get(adapter_id)
    except KeyError as e:
        emit.error(str(e), fatal=True)
        return 2

    try:
        config = _load_config(config_source)
    except (OSError, json.JSONDecodeError) as e:
        emit.error(f"failed to load config: {e}", fatal=True)
        return 2
    if dry_run:
        config["dry_run"] = True

    stop = threading.Event()

    def _signal_handler(signum, _frame):
        emit.warning(f"received signal {signum}; stopping")
        stop.set()

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    # stdin doubles as the host's control channel only when config came from a file.
    if config_source not in (None, "-"):
        threading.Thread(
            target=_watch_stdin_for_stop, args=(stop, emit), daemon=True
        ).start()

    emit.ready(adapter=adapter_id, version=adapter.version, dry_run=dry_run)
    try:
        adapter.run(config, emit, stop)
    except Exception as e:
        logger.exception("adapter %s crashed", adapter_id)
        emit.error(f"{type(e).__name__}: {e}", fatal=True)
        return 1
    emit.emit("stopped")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomscale.adapters",
        description="atomscale ingestion-adapter host",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list available adapters and their config schemas")
    run_p = sub.add_parser("run", help="run an adapter until stopped")
    run_p.add_argument("adapter_id", help="adapter id (see `list`)")
    run_p.add_argument(
        "--config",
        default=None,
        help="config JSON path, or '-'/omitted to read from stdin",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and map but never push to atomscale",
    )
    args = parser.parse_args(argv)

    # Logs to STDERR so STDOUT stays a clean machine-readable channel.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "list":
        return _cmd_list()
    return _cmd_run(args.adapter_id, args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
