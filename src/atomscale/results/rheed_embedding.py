from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import numpy as np
from monty.json import MSONable
from numpy.typing import NDArray
from pandas import DataFrame, concat


class RHEEDEmbeddingResult(MSONable):
    """Raw Chronos timeseries embedding vectors for a single RHEED data item.

    These are the per-window or clustered-prototype vectors the similarity
    embed-store persisted to the S3 Vectors index — the *inputs* to the
    similarity comparison, distinct from
    :class:`~atomscale.results.SimilarityTrajectoryResult`, which carries the
    derived similarity-vs-time curve (the *output*).

    Args:
        data_id: The data ID the embeddings belong to.
        workflow: Similarity workflow name (e.g. ``"rheed_stationary"``).
        window_span: Embedding window span in seconds the vectors were built at.
        kind: ``"window"`` (one vector per sliding window, time-resolved) or
            ``"prototype"`` (clustered prototype vectors).
        vectors: ``(N, D)`` float32 array of embedding vectors.
        indices: Length-``N`` window indices (``kind="window"``) or prototype
            indices (``kind="prototype"``).
        real_time_seconds: Length-``N`` window times in seconds (window kind only).
        unix_time_ms: Length-``N`` absolute window times in ms (window kind only).
        cluster_sizes: Length-``N`` prototype cluster sizes (prototype kind only).
        count: Total vectors available server-side before any offset/limit slice.
        truncated: True when the server hit its per-item window cap (the series
            may be incomplete).
    """

    def __init__(
        self,
        data_id: UUID | str,
        workflow: str,
        window_span: float,
        kind: str,
        vectors: NDArray | Sequence[Sequence[float]],
        indices: Sequence[int],
        real_time_seconds: NDArray | None = None,
        unix_time_ms: NDArray | None = None,
        cluster_sizes: NDArray | None = None,
        count: int | None = None,
        truncated: bool = False,
    ):
        self.data_id = data_id
        self.workflow = workflow
        self.window_span = window_span
        self.kind = kind
        arr = np.asarray(vectors, dtype=np.float32)
        # Normalize an empty/degenerate array to a clean (0, 0) shape.
        self.vectors: NDArray = arr if arr.ndim == 2 and arr.size else np.zeros((0, 0), np.float32)
        self.indices: list[int] = list(indices)
        self.real_time_seconds = (
            np.asarray(real_time_seconds, dtype=np.float64)
            if real_time_seconds is not None
            else None
        )
        self.unix_time_ms = (
            np.asarray(unix_time_ms, dtype=np.float64) if unix_time_ms is not None else None
        )
        # float64 (not int64) so a partial/missing cluster_size (None) coerces to
        # NaN rather than raising at array construction.
        self.cluster_sizes = (
            np.asarray(cluster_sizes, dtype=np.float64) if cluster_sizes is not None else None
        )
        self.count = count if count is not None else int(self.vectors.shape[0])
        self.truncated = truncated

    @property
    def dimension(self) -> int | None:
        """Embedding dimensionality D, or None when there are no vectors."""
        if self.vectors.ndim == 2 and self.vectors.shape[0] > 0:
            return int(self.vectors.shape[1])
        return None

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def to_dataframe(self) -> DataFrame:
        """Return one row per vector: locus/metadata columns + ``v0..v{D-1}``,
        indexed by the window/prototype index.
        """
        cols: dict[str, object] = {"index": self.indices}
        if self.real_time_seconds is not None:
            cols["real_time_seconds"] = list(self.real_time_seconds)
        if self.unix_time_ms is not None:
            cols["unix_time_ms"] = list(self.unix_time_ms)
        if self.cluster_sizes is not None:
            cols["cluster_size"] = list(self.cluster_sizes)
        df = DataFrame(cols)

        dim = self.dimension
        if len(self) and dim:
            vec_df = DataFrame(self.vectors, columns=[f"v{i}" for i in range(dim)])
            df = concat([df, vec_df], axis=1)

        if "index" in df.columns:
            df = df.set_index("index")
        return df
