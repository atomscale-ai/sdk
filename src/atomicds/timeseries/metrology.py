from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pandas import DataFrame

from atomicds.core import BaseClient
from atomicds.results.metrology import MetrologyResult
from atomicds.timeseries.provider import TimeseriesProvider


class MetrologyProvider(TimeseriesProvider[MetrologyResult]):
    TYPE = "metrology"

    RENAME_MAP: Mapping[str, str] = {
        "relative_time_seconds": "Time",
        "frame_number": "Frame Number",
        "unix_timestamp_ms": "UNIX Timestamp",
        "ratio_pyrometer": "Ratio Pyrometer",
        "sc_pyrometer": "SC Pyrometer",
        "decay_constant_minutes": "Decay Constant",
        "median_period": "Median Period",
        "median_period_seconds": "Median Period",
    }
    INDEX_COLS: Sequence[str] = ["Frame Number"]

    def fetch_raw(self, client: BaseClient, data_id: str) -> Any:
        return client._get(sub_url=f"metrology/{data_id}/timeseries/")

    def to_dataframe(self, raw: Any) -> DataFrame:
        if not raw:
            return DataFrame(None)
        series = raw.get("series") if isinstance(raw, dict) else raw
        series_df = DataFrame(series or None).rename(columns=self.RENAME_MAP)
        idx_cols = [c for c in self.INDEX_COLS if c in series_df.columns]
        if idx_cols:
            series_df = series_df.set_index(idx_cols)
        return series_df

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
