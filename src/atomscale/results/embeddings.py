"""Result object for similarity embedding vectors."""

from __future__ import annotations

import warnings
from typing import Any
from uuid import UUID

import numpy as np
from monty.json import MSONable
from numpy.typing import NDArray


class EmbeddingsResult(MSONable):
    """Embedding vectors for a single data entry.

    Returned by :meth:`atomscale.Client.get_embeddings`. The vectors are held as
    a dense ``(n_returned, dimension)`` float array in :attr:`vectors`, with
    parallel metadata arrays. The two ``kind`` variants carry different metadata:

    - ``kind="window"``: one time-resolved vector per window. :attr:`real_times`
      (relative seconds) and :attr:`unix_times_ms` (absolute milliseconds) give
      the point in time each vector corresponds to.
    - ``kind="prototype"``: a small set of representative vectors.
      :attr:`cluster_sizes` gives how many windows each one summarizes.

    Metadata arrays not relevant to the returned ``kind`` are ``None``.

    Attributes:
        data_id (UUID | str): Data ID the embeddings were computed for.
        workflow (str): Similarity workflow name (e.g. ``"rheed_stationary"``).
        kind (str): ``"window"`` or ``"prototype"``.
        window_span (float): Window span (seconds) the vectors were computed at.
        vectors (NDArray): ``(n_returned, dimension)`` array of embedding vectors,
            where ``n_returned == len(vectors)``.
        dimension (int): Length of each embedding vector (``0`` when the result is empty).
        count (int): Total vectors available for this ``data_id`` *before*
            ``offset``/``limit`` — may exceed ``len(vectors)``. The number
            actually returned is ``len(vectors)``.
        offset (int): Number of leading vectors skipped (window kind).
        truncated (bool): True when more vectors are available than were returned,
            so the result is incomplete.
        real_times (NDArray | None): ``(n_returned,)`` relative time in seconds (window kind).
        unix_times_ms (NDArray | None): ``(n_returned,)`` absolute unix time in ms (window kind).
        cluster_sizes (NDArray | None): ``(n_returned,)`` windows summarized per vector (prototype kind).
    """

    def __init__(
        self,
        data_id: UUID | str,
        workflow: str,
        kind: str,
        window_span: float,
        vectors: NDArray,
        dimension: int,
        count: int,
        truncated: bool,
        offset: int = 0,
        real_times: NDArray | None = None,
        unix_times_ms: NDArray | None = None,
        cluster_sizes: NDArray | None = None,
    ):
        self.data_id = data_id
        self.workflow = workflow
        self.kind = kind
        self.window_span = window_span
        self.vectors = vectors
        self.dimension = dimension
        self.count = count
        self.offset = offset
        self.truncated = truncated
        self.real_times = real_times
        self.unix_times_ms = unix_times_ms
        self.cluster_sizes = cluster_sizes

    def __repr__(self) -> str:
        return (
            f"EmbeddingsResult(data_id={self.data_id!r}, kind={self.kind!r}, "
            f"returned={len(self.vectors)}, count={self.count}, "
            f"dimension={self.dimension}, truncated={self.truncated})"
        )

    @staticmethod
    def _point_column(
        points: list[dict[str, Any]], key: str, *, integer: bool
    ) -> NDArray | None:
        """Collect ``point[key]`` across ``points`` into an array (``None`` if all absent).

        Returns an ``int64`` array for integer columns when every value is
        present; otherwise a ``float64`` array with ``NaN`` for missing entries
        (``int64`` cannot represent a null). Used for the per-kind metadata
        columns (``real_time_seconds``/``unix_time_ms``/``cluster_size``), which
        the endpoint only populates for the matching ``kind``.
        """
        values = [p.get(key) for p in points]
        if all(v is None for v in values):
            return None
        if integer and all(v is not None for v in values):
            return np.asarray(values, dtype=np.int64)
        return np.asarray(
            [np.nan if v is None else v for v in values], dtype=np.float64
        )

    @classmethod
    def from_api(
        cls,
        payload: dict[str, Any] | None,
        *,
        data_id: UUID | str,
        workflow: str,
        kind: str,
        window_span: float,
    ) -> EmbeddingsResult:
        """Build an :class:`EmbeddingsResult` from a raw endpoint payload.

        When no embeddings are available for the given workflow / window span,
        this emits a :class:`UserWarning` and returns an empty result so loops
        over many IDs don't crash.
        """
        payload = payload or {}
        points: list[dict[str, Any]] = payload.get("points") or []

        # The endpoint echoes kind/window_span; prefer them, fall back to request.
        resolved_kind = payload.get("kind", kind)
        resolved_span = payload.get("window_span", window_span)

        if not points:
            warnings.warn(
                f"No embeddings returned for data_id {data_id!r} (workflow "
                f"{workflow!r}, kind {resolved_kind!r}, window_span "
                f"{resolved_span}). The entry may not be embedded for this "
                "workflow/window span.",
                stacklevel=2,
            )
            return cls(
                data_id=data_id,
                workflow=workflow,
                kind=resolved_kind,
                window_span=resolved_span,
                vectors=np.empty((0, 0), dtype=np.float64),
                dimension=int(payload.get("dimension") or 0),
                count=int(payload.get("count", 0)),
                offset=int(payload.get("offset", 0)),
                truncated=bool(payload.get("truncated", False)),
            )

        try:
            vectors = np.asarray([p["vector"] for p in points], dtype=np.float64)
            if vectors.ndim != 2:
                raise ValueError(f"expected 2-D vectors, got shape {vectors.shape}")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Malformed embeddings payload for data_id {data_id!r} "
                f"(workflow {workflow!r}, kind {resolved_kind!r}): could not build "
                f"a numeric (count, dimension) array from the point vectors "
                f"(likely ragged or non-numeric). {exc}"
            ) from exc

        declared_dim = payload.get("dimension")
        dimension = int(declared_dim) if declared_dim else int(vectors.shape[1])
        count = int(payload.get("count", vectors.shape[0]))

        return cls(
            data_id=data_id,
            workflow=workflow,
            kind=resolved_kind,
            window_span=resolved_span,
            vectors=vectors,
            dimension=dimension,
            count=count,
            offset=int(payload.get("offset", 0)),
            truncated=bool(payload.get("truncated", False)),
            real_times=cls._point_column(points, "real_time_seconds", integer=False),
            unix_times_ms=cls._point_column(points, "unix_time_ms", integer=True),
            cluster_sizes=cls._point_column(points, "cluster_size", integer=True),
        )
