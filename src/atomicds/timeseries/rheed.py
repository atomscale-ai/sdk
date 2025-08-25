from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pandas import DataFrame, concat

from atomicds.core import BaseClient
from atomicds.timeseries.provider import TimeseriesProvider


class RHEEDProvider(TimeseriesProvider):
    TYPE = "rheed"

    # Mapping from API fields → user-facing column names
    RENAME_MAP: Mapping[str, str] = {
        "time_seconds": "Time",
        "frame_number": "Frame Number",
        "cluster_id": "Cluster ID",
        "cluster_std": "Cluster ID Uncertainty",
        "referenced_strain": "Strain",
        "nearest_neighbor_strain": "Cumulative Strain",
        "oscillation_period": "Oscillation Period",
        "spot_count": "Diffraction Spot Count",
        "first_order_intensity": "First Order Intensity",
        "half_order_intensity": "Half Order Intensity",
        "specular_intensity": "Specular Intensity",
        "reconstruction_intensity": "Reconstruction Intensity",
        "specular_fwhm_1": "Specular FWHM",
        "first_order_fwhm_1": "First Order FWHM",
        "lattice_spacing": "Lattice Spacing",
        "tar_metric": "TAR Metric",
    }
    DROP_IF_ALL_NA: Sequence[str] = ["reconstruction_intensity", "tar_metric"]
    INDEX_COLS: Sequence[str] = ["Angle", "Frame Number"]

    def fetch_raw(self, client: BaseClient, data_id: str) -> Any:
        return client._get(sub_url=f"rheed/timeseries/{data_id}/")

    def to_dataframe(self, raw: Any) -> DataFrame:
        if not raw:
            return DataFrame(None)

        frames: list[DataFrame] = []
        # payload shape: {"series_by_angle": [{"angle": <deg>, "series": [...]}, ...]}
        for angle_block in raw.get("series_by_angle", []):
            angle_df = DataFrame(angle_block["series"])
            angle_df["Angle"] = angle_block["angle"]
            frames.append(angle_df)

        if not frames:
            return DataFrame(None)

        df_all = concat(frames, axis=0, ignore_index=True)

        # drop confusing all-NA metrics
        for col in self.DROP_IF_ALL_NA:
            if col in df_all and df_all[col].isna().all():
                df_all = df_all.drop(columns=[col])

        df_all = df_all.rename(columns=self.RENAME_MAP)

        # Ensure index exists even if Angle/Frame Number are missing
        idx_cols = [c for c in self.INDEX_COLS if c in df_all.columns]
        if idx_cols:
            df_all = df_all.set_index(idx_cols)

        return df_all

    def snapshot_url(self, data_id: str) -> str:
        return f"data_entries/video_single_frames/{data_id}"

    def snapshot_image_uuids(self, raw_frames_payload: dict[str, Any]) -> list[dict]:
        # payload shape: {"frames": [{"image_uuid": "...", "timestamp_seconds": ...}, ...]}
        out = []
        for frame in (raw_frames_payload or {}).get("frames", []):
            meta = {k: v for k, v in frame.items() if k in {"timestamp_seconds"}}
            out.append({"image_uuid": frame["image_uuid"], "metadata": meta})
        return out
