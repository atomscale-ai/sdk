"""Provider classes for similarity trajectory data.

Note on unix-time units: the similarity API emits ``unix_times`` in
**seconds** (raw DB values), unlike the metrology / optical /
ellipsometry endpoints which ship ``unix_timestamp_ms`` in milliseconds.
The column is renamed to ``UNIX Timestamp`` (unit-less display name);
the magnitude-based fallback in :func:`align._infer_absolute_time`
correctly detects seconds for these values. In typed Python surfaces
the canonical type for individual unix_times values is
:class:`decimal.Decimal`; pandas columns stay numeric for performance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from pandas import DataFrame, concat

from atomscale.core import BaseClient
from atomscale.results.similarity_trajectory import SimilarityTrajectoryResult
from atomscale.timeseries.provider import TimeseriesProvider, finalize_dataframe


class SimilarityTrajectoryProvider(TimeseriesProvider[SimilarityTrajectoryResult]):
    TYPE = "similarity_trajectory"

    RENAME_MAP: Mapping[str, str] = {
        "reference_id": "Reference ID",
        "reference_item_name": "Reference Name",
        "real_time_seconds": "Time",
        "similarity_values": "Similarity",
        "unix_times": "UNIX Timestamp",
        "is_active": "Active",
        "averaged_count": "Averaged Count",
    }
    # Index columns as raw API names; resolved to display labels when display=True.
    INDEX_COLS: Sequence[str] = ["reference_id", "real_time_seconds"]

    def fetch_raw(self, client: BaseClient, data_id: str, **kwargs: Any) -> Any:
        """Fetch similarity trajectory data from the API.

        Args:
            client: The API client.
            data_id: The source ID for the similarity query.
            **kwargs: Must include 'workflow' (required). Optional parameters:
                window_span, reference_ids, softmax_mode, reference_n_values.

        Returns:
            Raw API response payload.

        Raises:
            KeyError: If 'workflow' is not provided in kwargs.
        """
        workflow = kwargs.pop("workflow")
        return client._get(
            sub_url=f"similarity/{workflow}/{data_id}/trajectory/",
            params=kwargs,
        )

    def to_dataframe(self, raw: Any, *, display: bool = True) -> DataFrame:
        if not raw:
            return DataFrame(None)

        trajectories = raw.get("trajectories", [])
        if not trajectories:
            return DataFrame(None)

        frames: list[DataFrame] = []
        for traj in trajectories:
            ref_id = traj.get("reference_id")
            ref_name = traj.get("reference_item_name")
            similarity_values = traj.get("similarity_values", [])
            real_time_seconds = traj.get("real_time_seconds", [])
            unix_times = traj.get("unix_times", [])
            is_active = traj.get("is_active")
            averaged_count = traj.get("averaged_count")

            if not similarity_values:
                continue

            # Build dataframe from columnar data
            traj_df = DataFrame(
                {
                    "reference_id": ref_id,
                    "reference_item_name": ref_name,
                    "similarity_values": similarity_values,
                    "real_time_seconds": real_time_seconds,
                    "unix_times": unix_times,
                    "is_active": is_active,
                    "averaged_count": averaged_count,
                }
            )
            frames.append(traj_df)

        if not frames:
            return DataFrame(None)

        df_all = concat(frames, axis=0, ignore_index=True)

        return finalize_dataframe(
            df_all, self.RENAME_MAP, display=display, index_cols=self.INDEX_COLS
        )

    def build_result(
        self,
        client: BaseClient,  # noqa: ARG002
        data_id: str,
        data_type: str,  # noqa: ARG002
        ts_df: DataFrame,
        *,
        workflow: str = "",
        window_span: float = 0.0,
        source_data_ids: Sequence[UUID | str] | None = None,
    ) -> SimilarityTrajectoryResult:
        return SimilarityTrajectoryResult(
            source_id=data_id,
            workflow=workflow,
            window_span=window_span,
            timeseries_data=ts_df,
            source_data_ids=source_data_ids,
        )
