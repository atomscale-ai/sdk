from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pandas import DataFrame, concat, json_normalize

from atomscale.core import BaseClient
from atomscale.results import (
    RHEEDImageResult,
    RHEEDVideoResult,
    _get_rheed_image_result,
)
from atomscale.timeseries.provider import TimeseriesProvider


class RHEEDProvider(TimeseriesProvider[RHEEDVideoResult]):
    TYPE = "rheed"

    # Mapping from API fields → user-facing column names
    RENAME_MAP: Mapping[str, str] = {
        "time_seconds": "Time",
        "relative_time_seconds": "Relative Time",
        "unix_timestamp_ms": "UNIX Timestamp",
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
        "composition_metric": "Composition Metric",
    }
    DROP_IF_ALL_NA: Sequence[str] = [
        "reconstruction_intensity",
        "tar_metric",
        "composition_metric",
    ]
    INDEX_COLS: Sequence[str] = ["Angle", "Frame Number"]

    def fetch_raw(self, client: BaseClient, data_id: str, **kwargs) -> Any:
        return client._get(sub_url=f"rheed/timeseries/{data_id}/", params=kwargs)

    def to_dataframe(self, raw: Any) -> DataFrame:
        if not raw:
            return DataFrame(None)

        frames: list[DataFrame] = []
        # payload shape: {"series_by_angle": [{"angle": <deg>, "series": [...]}, ...]}
        for angle_block in raw.get("series_by_angle", []):
            series = angle_block.get("series") or []
            if not series:
                continue
            # Flatten the nested low_level_features dict (present only when the
            # request set include_low_level_features) into one column per feature
            # BEFORE the all-NA drop, so empty low-level columns get pruned too.
            angle_df = self._expand_low_level_features(DataFrame(series))
            # Drop columns that are all-NA within this angle block before concat;
            # otherwise pandas issues a FutureWarning about empty/all-NA entries
            # widening the result dtype.
            angle_df = angle_df.dropna(axis=1, how="all")
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

    @staticmethod
    def _expand_low_level_features(angle_df: DataFrame) -> DataFrame:
        """Flatten the nested ``low_level_features`` column into one column per
        feature.

        The backend attaches a ``low_level_features`` dict (e.g.
        ``{"area_0": ..., "eccentricity_0": ...}``) to each point only when the
        request set ``include_low_level_features``. Expand it so each feature
        becomes its own column; missing keys/points become NaN. These columns
        are not in ``RENAME_MAP``, so they keep their raw backend names.
        """
        if "low_level_features" not in angle_df.columns:
            return angle_df
        nested = angle_df["low_level_features"]
        expanded = json_normalize(
            [v if isinstance(v, dict) else {} for v in nested], max_level=1
        )
        expanded.index = angle_df.index
        angle_df = angle_df.drop(columns=["low_level_features"])
        # Defensive: never clobber an existing top-level column.
        new_cols = [c for c in expanded.columns if c not in angle_df.columns]
        if new_cols:
            angle_df = concat([angle_df, expanded[new_cols]], axis=1)
        return angle_df

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
