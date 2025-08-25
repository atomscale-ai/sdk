from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pandas import DataFrame

from atomicds.core import BaseClient
from atomicds.timeseries.provider import TimeseriesProvider


class OpticalProvider(TimeseriesProvider):
    TYPE = "optical"

    RENAME_MAP: Mapping[str, str] = {
        "time_seconds": "Time",
        "perimeter_px": "Edge Perimeter",
        "circularity": "Edge Circularity",
        "edge_roughness": "Edge Roughness",
        "hausdorff_px": "Hausdorff Similarity",
    }
    INDEX_COLS: Sequence[str] = ["Time"]

    def snapshot_url(self, data_id: str) -> str:
        return f"optical/frame/video_single_frames/{data_id}"

    def fetch_raw(self, client: BaseClient, data_id: str) -> Any:
        return client._get(sub_url=f"optical/timeseries/{data_id}/")

    def to_dataframe(self, raw: Any) -> DataFrame:
        if not raw:
            return DataFrame(None)

        # Handle both {"series": [...]} or raw list
        series = raw.get("series") if isinstance(raw, dict) else raw
        series_df = DataFrame(series or None).rename(columns=self.RENAME_MAP)
        idx_cols = [c for c in self.INDEX_COLS if c in series_df.columns]
        if idx_cols:
            series_df = series_df.set_index(idx_cols)
        return series_df
