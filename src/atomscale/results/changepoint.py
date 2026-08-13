"""Result object for changepoint detection records."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from monty.json import MSONable


class ChangepointResult(MSONable):
    def __init__(
        self,
        id: UUID | str,
        data_id: UUID | str,
        data_modality: str,
        property_name: str,
        severity: str,
        score: float,
        window_start_elapsed: float,
        window_end_elapsed: float,
        detection_method: str,
        detail: dict[str, Any],
        label: str | None = None,
    ):
        """Changepoint detection result

        Args:
            id (UUID | str): Unique ID of the changepoint record.
            data_id (UUID | str): Data ID in the catalogue this changepoint was detected on.
            data_modality (str): The data modality the changepoint was detected in (e.g. "rheed_stationary").
            property_name (str): Property/channel on which the changepoint was detected.
            severity (str): One of "info", "warning", "critical".
            score (float): Normalized changepoint score in [0, 1].
            window_start_elapsed (float): Start of the changepoint window, seconds from timeseries start.
            window_end_elapsed (float): End of the changepoint window, seconds from timeseries start.
            detection_method (str): One of "forecasting", "clustering", "intensity_profile".
            detail (dict): Additional details about the changepoint; the available keys depend on the detection_method used.
            label (str | None): Applied category label, if any.
        """
        self.id = id
        self.data_id = data_id
        self.data_modality = data_modality
        self.property_name = property_name
        self.severity = severity
        self.score = score
        self.window_start_elapsed = window_start_elapsed
        self.window_end_elapsed = window_end_elapsed
        self.detection_method = detection_method
        self.detail = detail
        self.label = label

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> ChangepointResult:
        return cls(
            id=payload["id"],
            data_id=payload["data_id"],
            data_modality=payload["data_modality"],
            property_name=payload["property_name"],
            severity=payload["severity"],
            score=payload["score"],
            window_start_elapsed=payload["window_start_elapsed"],
            window_end_elapsed=payload["window_end_elapsed"],
            detection_method=payload["detection_method"],
            detail=payload.get("detail") or {},
            label=payload.get("label_category"),
        )
