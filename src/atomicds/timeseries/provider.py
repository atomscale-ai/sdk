from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pandas import DataFrame

from atomicds.client import Client
from atomicds.core import BaseClient

R = TypeVar("R")  # the result type this provider returns


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

    def build_result(
        self,
        client: Client,
        data_id: str,
        df: DataFrame,
        *,
        context: dict | None = None,
    ) -> R: ...

    # Optional override points
    def snapshot_url(self, data_id: str) -> str:  # noqa: ARG002
        """API endpoint that exposes extracted/snapshot frames."""
        return ""

    def snapshot_image_uuids(self, raw_frames_payload: dict[str, Any]) -> list[dict]:  # noqa: ARG002
        """Given the payload from `snapshot_url`, return
        a list of dicts like {'image_uuid': ..., 'metadata': {...}} to be resolved."""
        return []
