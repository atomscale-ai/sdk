from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Generic, TypeVar

from pandas import DataFrame

from atomscale.core import BaseClient

R = TypeVar("R")  # the result type this provider returns


_STAT_SUFFIX_LABELS: Mapping[str, str] = {
    "_z_score": "Z Score",
    "_ema_z_score": "EMA Z Score",
    "_anomaly_probability": "Anomaly Probability",
}


def extend_with_statistics(rename_map: Mapping[str, str]) -> dict[str, str]:
    """Add suffix-based statistical labels to a rename map."""
    extended = dict(rename_map)
    for source_name, label in rename_map.items():
        for suffix, suffix_label in _STAT_SUFFIX_LABELS.items():
            extended[f"{source_name}{suffix}"] = f"{label} {suffix_label}"
    return extended


class TimeseriesProvider(ABC, Generic[R]):
    """Strategy interface for parsing timeseries by domain."""

    # canonical domain name used as a key in the registry
    TYPE: ClassVar[str]

    @abstractmethod
    def fetch_raw(self, client: BaseClient, data_id: str) -> Any:
        """Perform the HTTP GET(s) to retrieve raw payload(s)."""

    @abstractmethod
    def to_dataframe(self, raw: Any) -> DataFrame:
        """Convert raw payload to a tidy DataFrame with domain-specific renames/index."""

    @abstractmethod
    def build_result(
        self,
        client: BaseClient,
        data_id: str,
        data_type: str,
        ts_df: DataFrame,
    ) -> R:
        """Build time series result object"""

    # Optional override points
    def snapshot_url(self, data_id: str) -> str:  # noqa: ARG002
        """API endpoint that exposes extracted/snapshot frames."""
        return ""

    def snapshot_image_uuids(
        self,
        frames_payload: dict[str, Any],  # noqa: ARG002
    ) -> list[dict]:
        """Extract requests from frames payload. Default: no snapshots."""
        return []

    def fetch_snapshot(
        self,
        client: BaseClient,  # noqa: ARG002
        req: dict,  # noqa: ARG002
    ) -> Any | None:
        """Resolve one snapshot request → domain-specific ImageResult (or None)."""
        return None
