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
    ) -> str: ...
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
