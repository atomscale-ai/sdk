"""Generic ingestion-adapter contract for the atomscale adapter host.

An *adapter* bridges one instrument or data source to atomscale. The host
(``python -m atomscale.adapters``) discovers adapters, exposes their config
schema, launches them, and monitors a JSON-line status stream.

Two pieces make up the contract:

- :class:`Adapter` — implement ``id`` / ``version`` / :meth:`Adapter.config_schema`
  / :meth:`Adapter.run`.
- :class:`StatusEmitter` — how a running adapter reports progress and health
  back to its host.

Adapters stream **directly** to atomscale via the SDK; the host is a control
plane only — data never flows through it. Status events are written to
**stdout** as one JSON object per line; ordinary logging must go to **stderr**
so the two streams never interleave.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import IO, Any

# Bumped only on a backwards-incompatible change to the status-event shape.
# Host and adapters ship in one installer and update together, so a single
# guard suffices — no cross-version negotiation.
PROTOCOL_VERSION = 1


class StatusEmitter:
    """Writes versioned JSON-line status events to a stream (stdout by default).

    Each event is one JSON object on its own line::

        {"protocol_version": 1, "event": "chunk", "ts": 1.7e9, "points": 40, ...}

    Thread-safe: an adapter may emit from worker threads while the host reads.
    """

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        """Emit one status event with arbitrary JSON-serializable fields."""
        record = {
            "protocol_version": PROTOCOL_VERSION,
            "event": event,
            "ts": time.time(),
            **fields,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    # -- Documented event vocabulary (thin wrappers over ``emit``) ------------

    def ready(self, **fields: Any) -> None:
        """Adapter process started and about to begin work."""
        self.emit("ready", **fields)

    def run_start(self, **fields: Any) -> None:
        """A measurement run has begun (e.g. a sentinel ``run-start``)."""
        self.emit("run_start", **fields)

    def run_end(self, **fields: Any) -> None:
        """A measurement run has ended."""
        self.emit("run_end", **fields)

    def chunk(self, **fields: Any) -> None:
        """A chunk of samples was flushed (throughput signal)."""
        self.emit("chunk", **fields)

    def warning(self, message: str, **fields: Any) -> None:
        """A non-fatal problem the operator should see."""
        self.emit("warning", message=message, **fields)

    def error(self, message: str, *, fatal: bool = False, **fields: Any) -> None:
        """An error. ``fatal=True`` means the adapter is exiting."""
        self.emit("error", message=message, fatal=fatal, **fields)

    def heartbeat(self, **fields: Any) -> None:
        """Periodic liveness signal."""
        self.emit("heartbeat", **fields)


class Adapter(ABC):
    """Base class for atomscale ingestion adapters.

    Subclasses set ``id`` and ``version`` as class attributes and implement
    :meth:`config_schema` and :meth:`run`.
    """

    id: str
    version: str

    @abstractmethod
    def config_schema(self) -> dict[str, Any]:
        """Return a JSON Schema for this adapter's operator-facing config.

        The host renders a settings form from this schema, so it must describe
        every field an operator can set. Secrets (API keys) are supplied via the
        environment, never the config, and must not appear here.
        """

    @abstractmethod
    def run(
        self,
        config: dict[str, Any],
        emit: StatusEmitter,
        stop: threading.Event,
    ) -> None:
        """Run the adapter until ``stop`` is set.

        ``config`` is a dict validated against :meth:`config_schema`. Stream
        results directly to atomscale and report progress through ``emit``.
        Return promptly once ``stop`` is set.
        """

    def manifest(self) -> dict[str, Any]:
        """Discovery descriptor: id, version, and config schema."""
        return {
            "id": self.id,
            "version": self.version,
            "config_schema": self.config_schema(),
        }
