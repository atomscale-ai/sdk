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
            window_span (float): Length of the time window, in seconds, used when computing the similarity trajectory.
            timeseries_data (DataFrame): Similarity over time, indexed by ("Reference ID", "Time")
                with columns "Similarity", "Reference Name", "UNIX Timestamp", "Active", and "Averaged Count".
            source_data_ids (Sequence[UUID | str] | None): Sequence of source data IDs included in the trajectory.
        """
        self.source_id = source_id
        self.workflow = workflow
        self.window_span = window_span
        self.timeseries_data = timeseries_data
        self.source_data_ids: list[UUID | str] = (
            list(source_data_ids) if source_data_ids else []
        )
