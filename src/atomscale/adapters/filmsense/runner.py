"""FilmSense adapter run orchestrator.

A ``FilmSenseRunner`` owns one deposition run end-to-end:

1. Initialize an atomscale stream via the injected ``StreamerProtocol``
2. Connect to the FS-1, set acquisition time, start dynamic measurements
3. Poll ``GetParms`` at the FS-1 sample rate, accumulate per-channel buffers
4. Every ``push_interval_seconds`` of wall-clock, persist the buffered chunk
   to the WAL and dispatch it via ``streamer.push_multi``
5. On stop, flush a final partial chunk, optionally save the FS-1's binary
   archive, then finalize the stream

The streamer is injected as a ``Protocol`` so tests can substitute a stub
without depending on the Rust extension.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from atomscale.adapters.filmsense.client import FilmSenseClient, FilmSenseError
from atomscale.adapters.filmsense.config import AdapterConfig
from atomscale.adapters.filmsense.mapping import normalize_param_name
from atomscale.adapters.filmsense.wal import ChunkWal

logger = logging.getLogger(__name__)


class StreamerProtocol(Protocol):
    """Subset of ``atomscale.streaming.TimeseriesStreamer`` the runner uses."""

    def initialize(
        self,
        stream_name: str | None = ...,
        synth_source_id: int | None = ...,
        physical_sample: str | None = ...,
        project_id: str | None = ...,
        tags: list[str] | None = ...,
    ) -> str: ...

    def push_multi(
        self,
        data_id: str,
        chunk_index: int,
        channels: dict[str, dict[str, Any]],
    ) -> None: ...

    def finalize(self, data_id: str) -> None: ...


@dataclass(frozen=True)
class RunMetadata:
    """Per-run metadata sourced from the sentinel-file or operator GUI."""

    stream_name: str | None = None
    physical_sample: str | None = None
    project_id: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class _ChannelBuffer:
    """In-memory buffer for one channel between push flushes."""

    timestamps: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    units: str | None = None

    def append(self, ts: float, value: float, units: str) -> None:
        self.timestamps.append(ts)
        self.values.append(value)
        # Keep the most recent declared unit; warn if it changes mid-run.
        if self.units is not None and units and units != self.units:
            logger.warning(
                "channel units changed mid-run: %r → %r (keeping new value)",
                self.units,
                units,
            )
        if units:
            self.units = units

    def is_empty(self) -> bool:
        return not self.timestamps

    def clear(self) -> None:
        self.timestamps = []
        self.values = []


class FilmSenseRunner:
    def __init__(
        self,
        config: AdapterConfig,
        streamer: StreamerProtocol,
        client_factory: Any | None = None,
        wal: ChunkWal | None = None,
        clock: Any = time.time,
        events: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.streamer = streamer
        self._client_factory = client_factory or self._default_client_factory
        self._wal = wal or ChunkWal(config.wal.path)
        self._clock = clock
        self._events = events

    def _event(self, name: str, **fields: Any) -> None:
        """Forward a status event to the injected sink, if any."""
        if self._events is not None:
            self._events(name, fields)

    def _default_client_factory(self) -> FilmSenseClient:
        return FilmSenseClient(self.config.fs1.host, self.config.fs1.port)

    def run_session(
        self,
        metadata: RunMetadata,
        stop_event: threading.Event,
    ) -> str:
        """Run one full deposition session. Returns the atomscale ``data_id``.

        Returns when ``stop_event`` is set, the FS-1 disconnects fatally, or
        an unrecoverable error occurs. Always finalizes the stream and the
        WAL handle is left open for the caller to reuse.
        """
        data_id = self.streamer.initialize(
            stream_name=metadata.stream_name,
            physical_sample=metadata.physical_sample,
            project_id=metadata.project_id,
            tags=metadata.tags or None,
        )
        logger.info("atomscale stream initialized: data_id=%s", data_id)
        self._event(
            "session_initialized", data_id=data_id, stream_name=metadata.stream_name
        )

        chunk_index = 0
        buffers: dict[str, _ChannelBuffer] = {}
        last_push = self._clock()

        try:
            with self._client_factory() as fs:
                fs.set_acquisition_time(self.config.fs1.acquisition_seconds)
                if self.config.fs1.model_index is not None:
                    fs.set_model(self.config.fs1.model_index)
                fs.start_dynamic_measurements()
                logger.info(
                    "FS-1 dynamic measurements started "
                    "(acq_time=%.3fs, push_interval=%.1fs)",
                    self.config.fs1.acquisition_seconds,
                    self.config.streamer.push_interval_seconds,
                )

                while not stop_event.is_set():
                    self._poll_once(fs, buffers)

                    now = self._clock()
                    if (
                        now - last_push >= self.config.streamer.push_interval_seconds
                        and any(not b.is_empty() for b in buffers.values())
                    ):
                        self._flush(data_id, chunk_index, buffers)
                        chunk_index += 1
                        last_push = now

                    # Sleep until the next sample is expected. Use an event-aware
                    # wait so stop is responsive.
                    stop_event.wait(self.config.fs1.acquisition_seconds)

                # Final flush before stopping
                if any(not b.is_empty() for b in buffers.values()):
                    self._flush(data_id, chunk_index, buffers)

                with _swallow("StopDynamicMeasurements"):
                    fs.stop_dynamic_measurements()

                if self.config.archive.enabled:
                    with _swallow("SaveDynamicMeasurements"):
                        fs.save_dynamic_measurements(
                            self.config.archive.folder,
                            metadata.stream_name or f"run-{data_id}",
                        )
        finally:
            with _swallow(f"finalize({data_id})"):
                self.streamer.finalize(data_id)

        return data_id

    # ------------------------------------------------------------------ internals

    def _poll_once(
        self,
        fs: FilmSenseClient,
        buffers: dict[str, _ChannelBuffer],
    ) -> None:
        try:
            params = fs.get_parameters()
        except FilmSenseError as e:
            logger.warning("GetParms failed (will retry on next tick): %s", e)
            return
        ts = self._clock()
        for fs_name, value in params:
            norm = normalize_param_name(fs_name)
            buf = buffers.setdefault(norm.channel_name, _ChannelBuffer())
            buf.append(ts, value, norm.units)

    def _flush(
        self,
        data_id: str,
        chunk_index: int,
        buffers: dict[str, _ChannelBuffer],
    ) -> None:
        """WAL-record + dispatch one chunk for all non-empty channels."""
        wal_ids: dict[str, int] = {}
        channels: dict[str, dict[str, Any]] = {}

        for name, buf in buffers.items():
            if buf.is_empty():
                continue
            row_id = self._wal.record(
                data_id=data_id,
                chunk_index=chunk_index,
                channel_name=name,
                timestamps=list(buf.timestamps),
                values=list(buf.values),
                units=buf.units,
            )
            wal_ids[name] = row_id
            channel_payload: dict[str, Any] = {
                "timestamps": list(buf.timestamps),
                "values": list(buf.values),
            }
            if buf.units:
                channel_payload["units"] = buf.units
            channels[name] = channel_payload
            buf.clear()

        if not channels:
            return

        self._event(
            "chunk",
            chunk_index=chunk_index,
            points=sum(len(c["values"]) for c in channels.values()),
            channels=sorted(channels),
        )

        if self.config.dry_run:
            logger.info(
                "[dry-run] chunk %d, channels=%s, points=%d",
                chunk_index,
                sorted(channels.keys()),
                sum(len(c["values"]) for c in channels.values()),
            )
            for row_id in wal_ids.values():
                self._wal.acknowledge(row_id)
            return

        try:
            self.streamer.push_multi(data_id, chunk_index, channels)
        except Exception:
            logger.exception("streamer.push_multi raised on chunk %d", chunk_index)
            return  # leave WAL rows un-acked for retry

        # push_multi is fire-and-forget — we can't wait on actual server ack
        # here. The WAL row remains the durability anchor; ``replay_pending``
        # uses the blocking ``run()`` path to confirm.
        for row_id in wal_ids.values():
            self._wal.acknowledge(row_id)

    def replay_pending(
        self,
        run_iterator_factory: Any,
        data_id: str | None = None,
    ) -> int:
        """Re-push any un-acknowledged WAL chunks via the blocking ``run()`` path.

        ``run_iterator_factory`` is a callable
        ``(channel_name, list[WalChunk]) -> iterable``: callers pass the
        adapter's ``StreamerProtocol`` here. Returns count of chunks re-pushed.
        """
        pending = self._wal.pending(data_id)
        if not pending:
            return 0

        # Group by (data_id, channel_name)
        by_channel: dict[tuple[str, str], list] = {}
        for chunk in pending:
            by_channel.setdefault((chunk.data_id, chunk.channel_name), []).append(chunk)

        for (chunk_data_id, channel_name), chunks in by_channel.items():
            run_iterator_factory(chunk_data_id, channel_name, chunks)
            for chunk in chunks:
                self._wal.acknowledge(chunk.id)

        return len(pending)


class _swallow:
    """Context manager that logs and swallows exceptions on a labelled step."""

    def __init__(self, what: str) -> None:
        self.what = what

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        if exc is not None:
            logger.warning("%s failed (continuing): %s", self.what, exc)
        return True


def channels_payload_for_test(
    buffers: Mapping[str, _ChannelBuffer],
) -> dict[str, dict[str, Any]]:  # pragma: no cover - test-helper, kept for symmetry
    out: dict[str, dict[str, Any]] = {}
    for name, buf in buffers.items():
        if buf.is_empty():
            continue
        payload: dict[str, Any] = {
            "timestamps": list(buf.timestamps),
            "values": list(buf.values),
        }
        if buf.units:
            payload["units"] = buf.units
        out[name] = payload
    return out
