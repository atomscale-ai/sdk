from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from typing import Any

import numpy as np
from numpy.typing import NDArray

class RHEEDStreamer:
    """High-performance RHEED frame streaming client.

    Bridges Python to a Rust/PyO3 backend for efficient, concurrent packaging
    and upload of uint8 grayscale frames to the Atomscale platform.

    ``push()`` and ``run()`` write each chunk's encoded shard to a disk spool
    and return; background workers upload the spool oldest first. A slow or
    failing network never stalls the caller, so it never delays chunk
    timestamps. See ``spool_status()``.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str | None = None,
        verbosity: int | None = None,
        spool_dir: str | PathLike[str] | None = None,
        spool_max_bytes: int | None = None,
        spool_min_free_bytes: int | None = None,
        spool_retention_s: float | None = None,
    ) -> None:
        """Create a streaming client.

        Chunks an earlier run of this endpoint left in the spool (after a
        crash, say) are uploaded in the background if their stream is still
        open, and discarded otherwise.

        Args:
            api_key: Your Atomscale API key.
            endpoint: Base API URL. Defaults to production.
            verbosity: Logging verbosity (0-4). Higher = more detail.
            spool_dir: Where chunks wait to upload, one subfolder per stream.
                Defaults to ``$ATOMSCALE_SPOOL_DIR``, else
                ``<user cache dir>/atomscale/rheed-spool``.
            spool_max_bytes: Cap on spooled data. Past it, new chunks are
                dropped (and counted) instead of blocking. Defaults to
                min(20 GB, 10% of the disk's free space).
            spool_min_free_bytes: Free disk space the spool always leaves.
                Defaults to 1 GB.
            spool_retention_s: Age past which a chunk that still has not
                uploaded is discarded. Defaults to 24 hours.

        Raises:
            RuntimeError: If the spool directory cannot be created.
        """
    def initialize(
        self,
        fps: float,
        rotations_per_min: float,
        chunk_size: int,
        stream_name: str | None = None,
        physical_sample: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
        azimuth: str | None = None,
    ) -> str:
        """Initialize a new RHEED stream.

        Creates the server-side data entries and returns a data_id to use
        for subsequent push/run/finalize calls.

        Args:
            fps: Capture rate in frames per second.
            rotations_per_min: Wafer/crystal rotations per minute. Use 0.0
                for stationary operation.
            chunk_size: Number of frames per chunk. Must be >= 2 * fps.
            stream_name: Human-readable name for the stream. Defaults to
                "RHEED Stream @ <time>".
            physical_sample: Name or UUID of a physical sample to associate
                with the data item. If a UUID is provided, it must match an
                existing sample. If a name is provided, it is matched
                case-insensitively against existing samples, or a new sample
                is created if no match is found.
            project_id: UUID of the associated project. When provided along
                with physical_sample, the sample is added to the project's
                tracking list and membership via
                POST /projects/{id}/configuration/tracking_samples. The sample
                is also marked as the project's active tracking sample for
                growth monitoring.
            tags: List of tag names or UUIDs to attach to the data item.
                Names are matched case-insensitively against existing
                organization tags; unknown names are created. UUIDs must
                reference existing tags. Tags do not set the stream's
                azimuth; use ``azimuth``.
            azimuth: The azimuth a parked stream is recorded at, e.g. "100".
                Its views are labelled with it in place of the automatic
                azimuth classifier until a user edits the view. Only valid
                with ``rotations_per_min=0``.

        Returns:
            The data_id for this stream.

        Raises:
            ValueError: If ``azimuth`` is given for a rotating stream.
        """
    def run(
        self,
        data_id: str,
        frames_iter: Iterable[NDArray[np.uint8]],
    ) -> None:
        """Spool every chunk an iterator yields, then wait for the uploads.

        Each item is spooled as it is yielded; the iterator is never held up
        by uploads. Once it is exhausted, waits up to 10 minutes for this
        stream's spool to drain.

        Raises:
            RuntimeError: If a chunk cannot be packaged, or if any chunk of
                this run was lost (dropped because the spool was full or the
                disk low, or refused by the backend). Chunks still uploading
                when the wait ends only warn; ``finalize()`` keeps draining.
        """
    def push(
        self,
        data_id: str,
        chunk_idx: int,
        frames: NDArray[np.uint8],
        capture_start_ms_utc: int | None = None,
    ) -> None:
        """Stamp, package and spool one chunk of frames for upload.

        Returns once the chunk is on disk; never waits on uploads. If the
        spool is at its byte cap or the disk at its free-space floor, the
        chunk is dropped instead (logged, and counted in ``spool_status()``
        and ``finalize()``), so the backend sees a missing shard rather than
        your loop stalling.

        Args:
            data_id: Stream identifier from initialize().
            chunk_idx: Zero-based chunk index for ordering.
            frames: Grayscale uint8 array, shape (N, H, W) or (H, W).
            capture_start_ms_utc: Optional capture-time timestamp (ms since
                UNIX epoch, UTC) for the first frame in the chunk. Pass it
                whenever you have a hardware/trigger timestamp aligned with
                frame capture. When omitted, the chunk is taken to have just
                finished capturing: ``Utc::now()`` at push entry minus the
                chunk's span (``N / fps``). Chunks are always stamped with
                capture time, never push time.
        """
    def finalize(
        self, data_id: str, drain_timeout_s: float | None = None
    ) -> dict[str, Any]:
        """Wait for the stream's spool to drain, then close the stream.

        Whatever is still spooled after ``drain_timeout_s`` (default 600) is
        discarded. If any chunk was never uploaded, one ``RuntimeWarning``
        gives the count; the stream is still closed. Also accepts a stream an
        earlier run left in the spool.

        Returns:
            ``chunks_spooled``, ``chunks_uploaded``, ``chunks_dropped`` (spool
            full, disk low or write failed), ``chunks_failed`` (refused by the
            backend), ``chunks_purged`` (discarded unsent),
            ``chunks_not_uploaded`` (the last three summed), ``chunks_pending``,
            ``drained`` (whether the wait finished in time), ``closed`` and
            ``lost``, a list of ``(chunk_idx, reason)``.

        Raises:
            RuntimeError: If the finalization POST fails.
        """
    def spool_status(self) -> dict[str, Any]:
        """The disk spool chunks wait in before uploading.

        Returns:
            ``dir``, ``bytes`` (spooled now), ``max_bytes``, ``min_free_bytes``,
            ``free_bytes`` (``None`` if unknown), ``chunks_pending``, the
            cumulative ``chunks_spooled``, ``chunks_uploaded``,
            ``chunks_dropped``, ``chunks_failed`` and ``chunks_purged``, and
            ``streams``: the same counts per ``data_id``.
        """
    def clock_status(self) -> dict[str, float] | None:
        """This computer's clock measured against AtomCloud's reference clock.

        Every timestamp is read as this computer's clock and sent as reference UTC,
        like the gateway: the offset comes from an NTP-style four-timestamp exchange
        with ``GET /gateway/time``, so it accounts for the transit time in both
        directions. Returns ``None`` when correction is off (set the
        ``ATOMSCALE_SKIP_CLOCK_SYNC`` environment variable) or no measurement has
        succeeded yet; then timestamps are sent as read.

        Returns:
            ``offset_ms`` (reference UTC minus this computer's clock),
            ``uncertainty_ms`` (bound on its error: half the best round trip) and
            ``measured_at_unix_ms``.
        """

class TimeseriesStreamer:
    """High-performance instrument timeseries streaming client.

    Streams scalar timeseries data (temperature, pressure, power, etc.) from
    instruments to the Atomscale platform. Uses a Rust/PyO3 backend for
    efficient, concurrent chunk uploads.

    Example:
        >>> streamer = TimeseriesStreamer(api_key="YOUR_API_KEY")
        >>> data_id = streamer.initialize(stream_name="Growth Run 1")
        >>> streamer.push(data_id, 0, "temperature", [0.0, 0.1], [25.0, 25.1])
        >>> streamer.finalize(data_id)
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str | None = None,
        points_per_chunk: int = 100,
        verbosity: int | None = None,
    ) -> None:
        """Create a new timeseries streamer.

        Args:
            api_key: Your Atomscale API key.
            endpoint: Base API URL. Defaults to production.
            points_per_chunk: Expected points per chunk for array positioning.
            verbosity: Logging verbosity (0-4). Higher = more detail.
        """

    def clock_status(self) -> dict[str, float] | None:
        """This computer's clock measured against AtomCloud's reference clock.

        Every timestamp is read as this computer's clock and sent as reference UTC,
        like the gateway: the offset comes from an NTP-style four-timestamp exchange
        with ``GET /gateway/time``, so it accounts for the transit time in both
        directions. Returns ``None`` when correction is off (set the
        ``ATOMSCALE_SKIP_CLOCK_SYNC`` environment variable) or no measurement has
        succeeded yet; then timestamps are sent as read.

        Returns:
            ``offset_ms`` (reference UTC minus this computer's clock),
            ``uncertainty_ms`` (bound on its error: half the best round trip) and
            ``measured_at_unix_ms``.
        """

    def initialize(
        self,
        stream_name: str | None = None,
        synth_source_id: int | None = None,
        physical_sample: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Initialize a new timeseries stream.

        Creates the server-side data entries and returns a data_id to use
        for subsequent push/run/finalize calls.

        Args:
            stream_name: Human-readable name for the stream.
            synth_source_id: Growth instrument ID to link. Must belong to your
                organization. Use list_instruments() to see available instruments.
            physical_sample: Name or UUID of a physical sample to associate
                with the data item. If a UUID is provided, it must match an
                existing sample. If a name is provided, it is matched
                case-insensitively against existing samples, or a new sample
                is created if no match is found.
            project_id: UUID of the associated project. When provided along
                with physical_sample, the sample is added to the project's
                tracking list and membership via
                POST /projects/{id}/configuration/tracking_samples. The sample
                is also marked as the project's active tracking sample for
                growth monitoring.

        Returns:
            The data_id for this stream.
        """

    def push(
        self,
        data_id: str,
        chunk_index: int,
        channel_name: str,
        timestamps: list[float],
        values: list[float],
        units: str | None = None,
    ) -> None:
        """Push a single chunk of timeseries data (fire-and-forget).

        Spawns an async upload task and returns immediately.

        Args:
            data_id: Stream identifier from initialize().
            chunk_index: Zero-based chunk index for ordering.
            channel_name: Name of the channel (e.g., "temperature").
            timestamps: Unix epoch timestamps in seconds.
            values: Measured values corresponding to timestamps.
            units: Optional units for the values (e.g., "C", "mbar").
        """

    def push_multi(
        self,
        data_id: str,
        chunk_index: int,
        channels: dict[str, dict],
    ) -> None:
        """Push data for multiple channels at once (fire-and-forget).

        All channels of the chunk go in one request, so the server writes the shared
        time axis once per chunk. Prefer this over one push() per channel. Channels
        with no points are left out. Network errors, 5xx and 429 responses are retried
        up to 3 times; a chunk that still fails is logged at warning level.

        Args:
            data_id: Stream identifier from initialize().
            chunk_index: Zero-based chunk index for ordering.
            channels: Mapping of channel_name to channel data dict.
                Each dict should have "timestamps", "values", and optional "units".

        Example:
            >>> streamer.push_multi(data_id, 0, {
            ...     "temperature": {"timestamps": [0.0, 0.1], "values": [25.0, 25.1], "units": "C"},
            ...     "pressure": {"timestamps": [0.0, 0.1], "values": [1.0, 1.1]},
            ... })
        """

    def run(
        self,
        data_id: str,
        channel_name: str,
        data_iter: Iterable[tuple[list[float], list[float]]],
        units: str | None = None,
    ) -> None:
        """Stream timeseries data from an iterator (blocking).

        Iterates over chunks, uploading each one. Blocks until all uploads
        complete.

        Args:
            data_id: Stream identifier from initialize().
            channel_name: Name of the channel (e.g., "temperature").
            data_iter: Iterator yielding (timestamps, values) tuples.
            units: Optional units for the values.
        """

    def finalize(self, data_id: str) -> None:
        """Finalize the stream, marking it as complete.

        Call this after all data has been pushed to signal the stream is
        finished. This allows the platform to begin processing.

        Args:
            data_id: Stream identifier from initialize().
        """
