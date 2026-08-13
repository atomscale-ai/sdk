from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pandas import DataFrame

from atomscale.core import BaseClient
from atomscale.results.tool_state import ToolStateResult
from atomscale.timeseries.provider import (
    TimeseriesProvider,
    properties_payload_to_dataframe,
    series_payload_to_dataframe,
)


class ToolStateProvider(TimeseriesProvider[ToolStateResult]):
    """Fetch and parse observed tool-state sensor readbacks."""

    TYPE = "tool_state"

    # Property values retain their human-readable API names. Only the time
    # columns are renamed to the SDK's user-facing display labels.
    RENAME_MAP: Mapping[str, str] = {
        "relative_time_seconds": "Time",
        "unix_timestamp_ms": "UNIX Timestamp",
    }

    def fetch_raw(self, client: BaseClient, data_id: str) -> Any:
        return client._get(sub_url=f"tool-state/{data_id}/timeseries/")

    def to_dataframe(self, raw: Any) -> DataFrame:
        if not raw:
            return DataFrame()
        if not isinstance(raw, dict):
            raise ValueError(
                f"{type(self).__name__} payload must be a dict; got "
                f"{type(raw).__name__}."
            )
        if "properties" in raw:
            parsed = properties_payload_to_dataframe(raw["properties"])
        elif "series" in raw:
            parsed = series_payload_to_dataframe(raw["series"])
        else:
            raise ValueError(
                f"{type(self).__name__} payload missing both 'properties' and "
                f"'series' keys. Got: {list(raw.keys())}."
            )
        return parsed.rename(columns=self.RENAME_MAP)

    def build_result(
        self,
        client: BaseClient,  # noqa: ARG002
        data_id: str,
        data_type: str,  # noqa: ARG002
        ts_df: DataFrame,
    ) -> ToolStateResult:
        return ToolStateResult(data_id=data_id, timeseries_data=ts_df)
