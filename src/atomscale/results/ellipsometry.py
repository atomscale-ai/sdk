from __future__ import annotations

from uuid import UUID

from monty.json import MSONable
from pandas import DataFrame


class EllipsometryResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        timeseries_data: DataFrame,
        collected_datetime: str | None = None,
    ):
        """Ellipsometry result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            timeseries_data (DataFrame): Pandas DataFrame with timeseries data associated with
                the ellipsometry measurement. Columns include per-wavelength channels
                (e.g. ``psi_<λ>``, ``delta_<λ>``, ``depol_<λ>``, ``incidentI_<λ>``) and
                scalar fits such as ``thickness``.
            collected_datetime (str | None): Datetime when the data was collected.
        """
        self.data_id = data_id
        self.timeseries_data = timeseries_data
        self.collected_datetime = collected_datetime
