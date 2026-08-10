"""Provider for RHEED embedding-vector k-NN neighbor queries.

Wraps the read-only embedding-neighbors endpoint the backend exposes under
``/similarity/{workflow}/{data_id}/embeddings/neighbors/`` (k-NN "find similar").
Mirrors the fetch/build split of
:class:`~atomscale.similarity.provider.SimilarityTrajectoryProvider`, returning a
tidy neighbors DataFrame. Raw embedding *vectors* are fetched separately via
:meth:`atomscale.Client.get_embeddings`.
"""

from __future__ import annotations

from typing import Any

from pandas import DataFrame

from atomscale.core import BaseClient

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

    def fetch_neighbors_raw(
        self, client: BaseClient, data_id: str, **kwargs: Any
    ) -> Any:
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

    @staticmethod
    def neighbors_to_dataframe(raw: Any) -> DataFrame:
        """Build a tidy neighbors DataFrame from a fetch_neighbors_raw payload."""
        neighbors = (raw or {}).get("neighbors", []) or []
        if not neighbors:
            return DataFrame(columns=_NEIGHBOR_COLUMNS)
        frame = DataFrame(neighbors)
        ordered = [c for c in _NEIGHBOR_COLUMNS if c in frame.columns]
        extra = [c for c in frame.columns if c not in _NEIGHBOR_COLUMNS]
        return frame[ordered + extra]
