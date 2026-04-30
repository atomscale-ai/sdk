from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

class RHEEDStreamer:
    """High-performance RHEED frame streaming client.

    Bridges Python to a Rust/PyO3 backend for efficient, concurrent packaging
    and upload of uint8 grayscale frames to the Atomscale platform.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str | None = None,
        verbosity: int | None = None,
    ) -> None: ...
    def initialize(
        self,
        fps: float,
        rotations_per_min: float,
        chunk_size: int,
        stream_name: str | None = None,
        physical_sample: str | None = None,
        project_id: str | None = None,
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

        Returns:
            The data_id for this stream.
        """
    def run(
        self,
        data_id: str,
        frames_iter: Iterable[NDArray[np.uint8]],
    ) -> None: ...
    def push(
        self,
        data_id: str,
        chunk_idx: int,
        frames: NDArray[np.uint8],
    ) -> None: ...
    def finalize(self, data_id: str) -> None: ...

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

    def initialize(
        self,
        stream_name: str | None = None,
        synth_source_id: int | None = None,
        physical_sample: str | None = None,
        project_id: str | None = None,
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
