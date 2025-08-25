from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pandas import DataFrame

from atomicds.core import BaseClient


class TimeseriesProvider(ABC):
    """Strategy interface for parsing timeseries by domain."""

    # canonical domain name used as a key in the registry
    TYPE: ClassVar[str]

    @abstractmethod
    def fetch_raw(self, client: BaseClient, data_id: str) -> Any:
        """Perform the HTTP GET(s) to retrieve raw payload(s)."""

    @abstractmethod
    def to_dataframe(self, raw: Any) -> DataFrame:
        """Convert raw payload to a tidy DataFrame with domain-specific renames/index."""

    # Optional override points
    def supports_snapshots(self) -> bool:
        """Whether this domain exposes extracted/snapshot frames."""
        return False

    def snapshot_image_uuids(self, raw_frames_payload: dict[str, Any]) -> list[dict]:  # noqa: ARG002
        """Given the payload from `data_entries/video_single_frames/{data_id}`, return
        a list of dicts like {'image_uuid': ..., 'metadata': {...}} to be resolved."""
        return []
