from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pandas import DataFrame, concat

from atomicds.core import BaseClient
from atomicds.results import RHEEDImageResult, RHEEDVideoResult, _get_rheed_image_result
from atomicds.timeseries.provider import TimeseriesProvider


class RHEEDProvider(TimeseriesProvider[RHEEDVideoResult]):
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
        "first_order_intensity_l": "First Order Intensity L",
        "first_order_intensity_r": "First Order Intensity R",
        "half_order_intensity": "Half Order Intensity",
        "half_order_intensity_l": "Half Order Intensity L",
        "half_order_intensity_r": "Half Order Intensity R",
        "specular_intensity": "Specular Intensity",
        "reconstruction_intensity": "Reconstruction Intensity",
        "specular_fwhm_1": "Specular FWHM",
        "first_order_fwhm_1": "First Order FWHM",
        "lattice_spacing": "Lattice Spacing",
        "tar_metric": "TAR Metric",
    }
    DROP_IF_ALL_NA: Sequence[str] = ["reconstruction_intensity", "tar_metric"]
    INDEX_COLS: Sequence[str] = ["Angle", "Frame Number"]

    def fetch_raw(self, client: BaseClient, data_id: str, **kwargs) -> Any:
        return client._get(sub_url=f"rheed/timeseries/{data_id}/", params=kwargs)

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

    def snapshot_image_uuids(self, frames_payload: dict[str, Any]) -> list[dict]:
        # payload shape: {"frames": [{"image_uuid": "...", "timestamp_seconds": ...}, ...]}
        out = []
        for frame in (frames_payload or {}).get("frames", []):
            meta = {k: v for k, v in frame.items() if k in {"timestamp_seconds"}}
            out.append({"image_uuid": frame["image_uuid"], "metadata": meta})
        return out

    def fetch_snapshot(self, client: BaseClient, req: dict) -> RHEEDImageResult | None:
        img_uuid = req.get("image_uuid")
        if not img_uuid:
            return None
        # Reuse the client helper to build a RHEEDImageResult (graph, mask, etc.)
        return _get_rheed_image_result(
            client=client, data_id=img_uuid, metadata=req.get("metadata", {})
        )

    def build_result(
        self, client: BaseClient, data_id: str, data_type: str, ts_df: DataFrame
    ) -> RHEEDVideoResult:
        extracted = None
        idx_url = self.snapshot_url(data_id)
        if idx_url:
            frames_payload: dict | None = client._get(sub_url=idx_url)  # type: ignore[assignment]
            if frames_payload:
                reqs = self.snapshot_image_uuids(frames_payload)
                extracted = [
                    res
                    for res in client._multi_thread(
                        self.fetch_snapshot,
                        [{"client": client, "req": r} for r in reqs],
                    )
                    if res
                ]
        return RHEEDVideoResult(
            data_id=data_id,
            timeseries_data=ts_df,
            snapshot_image_data=extracted,
            rotating=(data_type == "rheed_rotating"),
        )
