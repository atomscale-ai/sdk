from __future__ import annotations

from uuid import UUID

from monty.json import MSONable
from pandas import DataFrame


class ToolStateResult(MSONable):
    """Processed tool-state timeseries returned by the Atomscale API."""

    def __init__(
        self,
        data_id: UUID | str,
        timeseries_data: DataFrame,
        collected_datetime: str | None = None,
    ):
        """Initialize a tool-state result.

        Args:
            data_id: Data ID for the entry in the data catalogue.
            timeseries_data: Sensor readback timeseries for the processed tool-state data.
            collected_datetime: Datetime when the data was collected.
        """
        self.data_id = data_id
        self.timeseries_data = timeseries_data
        self.collected_datetime = collected_datetime
