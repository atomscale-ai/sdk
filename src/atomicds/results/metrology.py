from __future__ import annotations

from uuid import UUID

from monty.json import MSONable
from pandas import DataFrame


class MetrologyResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        timeseries_data: DataFrame,
    ):
        """Optical result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            timeseries_data (DataFrame): Pandas DataFrame with timeseries data associated with the ingested metrology data.
        """
        self.data_id = data_id
        self.timeseries_data = timeseries_data
