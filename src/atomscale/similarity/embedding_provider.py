"""Provider for RHEED timeseries embedding vectors and k-NN neighbor queries.

Wraps the read-only embedding endpoints the backend exposes under
``/similarity/{workflow}/{data_id}/embeddings/`` (raw vectors) and
``.../embeddings/neighbors/`` (k-NN "find similar"). Mirrors the fetch/build
split of :class:`~atomscale.similarity.provider.SimilarityTrajectoryProvider`,
but returns a numpy-backed :class:`RHEEDEmbeddingResult` rather than a
timeseries DataFrame, so it is intentionally not a ``TimeseriesProvider``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pandas import DataFrame

from atomscale.core import BaseClient
from atomscale.results.rheed_embedding import RHEEDEmbeddingResult

_NEIGHBOR_COLUMNS = [
    "data_id",
    "similarity",
    "source_index",
    "neighbor_index",
    "real_time_seconds",
    "unix_time_ms",
]


class RHEEDEmbeddingProvider:
    TYPE = "rheed_embeddings"

    def fetch_raw(self, client: BaseClient, data_id: str, **kwargs: Any) -> Any:
        """Fetch stored embedding vectors for a data_id.

        Args:
            client: The API client.
            data_id: The data ID to fetch embeddings for.
            **kwargs: Must include ``workflow``. Optional: ``window_span``,
                ``kind`` ("window"|"prototype"), ``offset``, ``limit``.
        """
        workflow = kwargs.pop("workflow")
        return client._get(
            sub_url=f"similarity/{workflow}/{data_id}/embeddings/",
            params=kwargs,
        )

    def fetch_neighbors_raw(self, client: BaseClient, data_id: str, **kwargs: Any) -> Any:
        """Fetch k-NN neighbors ("find similar") for a data_id.

        Args:
            client: The API client.
            data_id: The data ID whose vectors seed the query.
            **kwargs: Must include ``workflow``. Optional: ``window_span``,
                ``kind`` ("prototype"|"window"), ``top_k``.
        """
        workflow = kwargs.pop("workflow")
        return client._get(
            sub_url=f"similarity/{workflow}/{data_id}/embeddings/neighbors/",
            params=kwargs,
        )

    def to_result(self, raw: Any) -> RHEEDEmbeddingResult:
        """Build a RHEEDEmbeddingResult from a fetch_raw payload."""
        raw = raw or {}
        points = raw.get("points", []) or []
        kind = raw.get("kind", "window")

        if points:
            vectors = np.asarray([p.get("vector", []) for p in points], dtype=np.float32)
        else:
            vectors = np.zeros((0, 0), dtype=np.float32)
        indices = [p.get("index") for p in points]

        real_time = None
        unix_time = None
        cluster_sizes = None
        if kind == "window":
            if any(p.get("real_time_seconds") is not None for p in points):
                real_time = [p.get("real_time_seconds") for p in points]
            if any(p.get("unix_time_ms") is not None for p in points):
                unix_time = [p.get("unix_time_ms") for p in points]
        elif any(p.get("cluster_size") is not None for p in points):
            cluster_sizes = [p.get("cluster_size") for p in points]

        return RHEEDEmbeddingResult(
            data_id=raw.get("data_id"),
            workflow=raw.get("workflow", ""),
            window_span=raw.get("window_span", 0.0),
            kind=kind,
            vectors=vectors,
            indices=indices,
            real_time_seconds=real_time,
            unix_time_ms=unix_time,
            cluster_sizes=cluster_sizes,
            count=raw.get("count"),
            truncated=bool(raw.get("truncated", False)),
        )

    @staticmethod
    def neighbors_to_dataframe(raw: Any) -> DataFrame:
        """Build a tidy neighbors DataFrame from a fetch_neighbors_raw payload."""
        neighbors = (raw or {}).get("neighbors", []) or []
        if not neighbors:
            return DataFrame(columns=_NEIGHBOR_COLUMNS)
        df = DataFrame(neighbors)
        ordered = [c for c in _NEIGHBOR_COLUMNS if c in df.columns]
        extra = [c for c in df.columns if c not in _NEIGHBOR_COLUMNS]
        return df[ordered + extra]
