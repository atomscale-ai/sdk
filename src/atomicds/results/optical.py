from __future__ import annotations

from uuid import UUID

from monty.json import MSONable
from pandas import DataFrame
from PIL.Image import Image


class OpticalImageResult:
    def __init__(self, data_id: str, processed_image: Image):
        self.data_id = data_id
        self.processed_image = processed_image


class OpticalResult(MSONable):
    def __init__(
        self,
        data_id: UUID | str,
        timeseries_data: DataFrame,
        snapshot_image_data: list[OpticalImageResult] | None,
    ):
        """Optical result

        Args:
            data_id (UUID | str): Data ID for the entry in the data catalogue.
            timeseries_data (DataFrame): Pandas DataFrame with timeseries data associated with the video.
        """
        self.data_id = data_id
        self.timeseries_data = timeseries_data
        self.snapshot_image_data = snapshot_image_data
