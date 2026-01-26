from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from monty.json import MSONable
from pandas import DataFrame


class SimilarityTrajectoryResult(MSONable):
    def __init__(
        self,
        source_id: UUID | str,
        workflow: str,
        window_span: float,
        timeseries_data: DataFrame,
        source_data_ids: Sequence[UUID | str] | None = None,
    ):
        """Similarity trajectory result

        Args:
            source_id (UUID | str): Source ID for the similarity trajectory query.
            workflow (str): Workflow name used for the similarity analysis.
            window_span (float): Window span parameter used for the trajectory.
            timeseries_data (DataFrame): Pandas DataFrame with similarity trajectory data.
            source_data_ids (Sequence[UUID | str] | None): Sequence of source data IDs included in the trajectory.
        """
        self.source_id = source_id
        self.workflow = workflow
        self.window_span = window_span
        self.timeseries_data = timeseries_data
        self.source_data_ids: list[UUID | str] = (
            list(source_data_ids) if source_data_ids else []
        )
