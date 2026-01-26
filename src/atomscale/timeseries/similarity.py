from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from pandas import DataFrame, concat

from atomscale.core import BaseClient
from atomscale.results.similarity_trajectory import SimilarityTrajectoryResult
from atomscale.timeseries.provider import TimeseriesProvider


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
    INDEX_COLS: Sequence[str] = ["Reference ID", "Time"]

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

    def to_dataframe(self, raw: Any) -> DataFrame:
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
        df_all = df_all.rename(columns=self.RENAME_MAP)

        idx_cols = [c for c in self.INDEX_COLS if c in df_all.columns]
        if idx_cols:
            df_all = df_all.set_index(idx_cols)

        return df_all

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
