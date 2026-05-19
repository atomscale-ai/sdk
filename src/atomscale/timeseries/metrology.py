from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pandas import DataFrame

from atomscale.core import BaseClient
from atomscale.results.metrology import MetrologyResult
from atomscale.timeseries.provider import (
    TimeseriesProvider,
    properties_payload_to_dataframe,
)


class MetrologyProvider(TimeseriesProvider[MetrologyResult]):
    TYPE = "metrology"

    # The property-centric payload returns property values keyed by their
    # human-readable API names (preserved as-is). Only the time columns
    # are renamed to user-facing display labels.
    RENAME_MAP: Mapping[str, str] = {
        "relative_time_seconds": "Time",
        "unix_timestamp_ms": "UNIX Timestamp",
    }

    def fetch_raw(self, client: BaseClient, data_id: str) -> Any:
        return client._get(sub_url=f"metrology/{data_id}/timeseries/")

    def to_dataframe(self, raw: Any) -> DataFrame:
        if not raw:
            return DataFrame()
        if not isinstance(raw, dict) or "properties" not in raw:
            got = list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
            raise ValueError(
                f"{type(self).__name__} payload missing 'properties' key. "
                f"Got: {got}. The legacy 'series' shape is no longer supported."
            )
        properties_df = properties_payload_to_dataframe(raw["properties"])
        return properties_df.rename(columns=self.RENAME_MAP)

    def build_result(
        self,
        client: BaseClient,  # noqa: ARG002
        data_id: str,
        data_type: str,  # noqa: ARG002
        ts_df: DataFrame,
    ) -> MetrologyResult:
        return MetrologyResult(
            data_id=data_id,
            timeseries_data=ts_df,
        )
