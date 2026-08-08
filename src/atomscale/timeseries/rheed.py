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
    # Columns added to the timeseries DataFrame when per-frame masks are attached.
    MASK_COLS: Sequence[str] = ["mask_rle", "mask_height", "mask_width"]

    def fetch_raw(self, client: BaseClient, data_id: str, **kwargs) -> Any:
        return client._get(sub_url=f"rheed/timeseries/{data_id}/", params=kwargs)

    @staticmethod
    def _flatten_low_level_features(df: DataFrame) -> DataFrame:
        """Expand a ``low_level_features`` column of per-point dicts into columns.

        Each point in the raw series may carry a ``low_level_features`` mapping
        (present only when the timeseries was fetched with
        ``include_low_level_features=True`` via
        :meth:`atomscale.Client.get_rheed_timeseries`). This lifts those keys to
        top-level columns — nested keys flattened with a dotted path — and drops
        the original nested column. Points missing the mapping contribute NA for
        those columns. Existing top-level columns win on name collision so known
        metrics are never clobbered by a low-level feature of the same name.
        """
        normalized = df["low_level_features"].apply(
            lambda v: v if isinstance(v, dict) else {}
        )
        expanded = json_normalize(normalized.tolist())
        expanded.index = df.index

        collisions = [c for c in expanded.columns if c in df.columns]
        if collisions:
            expanded = expanded.drop(columns=collisions)

        return df.drop(columns=["low_level_features"]).join(expanded)

    def to_dataframe(self, raw: Any) -> DataFrame:
        if not raw:
            return DataFrame(None)

        frames: list[DataFrame] = []
        # payload shape: {"series_by_angle": [{"angle": <deg>, "series": [...]}, ...]}
        for angle_block in raw.get("series_by_angle", []):
            series = angle_block.get("series") or []
            if not series:
                continue
            angle_df = DataFrame(series)
            # Flatten per-point low-level feature dicts into their own columns
            # before the all-NA prune below so the pruning applies to them too.
            if "low_level_features" in angle_df.columns:
                angle_df = self._flatten_low_level_features(angle_df)
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

    @classmethod
    def attach_frame_masks(
        cls, df: DataFrame, mask_rows: Sequence[Mapping[str, Any]]
    ) -> DataFrame:
        """Attach per-frame RLE segmentation masks to a RHEED timeseries DataFrame.

        Joins each mask row onto the timeseries row(s) with the matching absolute
        frame number, adding ``mask_rle`` / ``mask_height`` / ``mask_width`` columns
        (see :data:`MASK_COLS`). ``mask_rows`` are the raw rows returned by
        :meth:`atomscale.Client.get_frame_masks` (each with ``frame_number``,
        ``mask_rle``, ``mask_height``, ``mask_width``).

        Coverage is sparse — masks exist only for featurized frames — so timeseries
        rows whose frame has no mask get NA in the mask columns. When ``mask_rows``
        is empty (no mask artifact for the video), the columns are still added and
        are all-NA, so a caller that asked for masks always gets the columns.

        Returns ``df`` unchanged (no mask columns) when it has no ``Frame Number``
        axis to key on, since masks cannot be aligned without it.
        """
        has_frame_axis = "Frame Number" in (df.index.names or []) or (
            "Frame Number" in df.columns
        )
        if df.empty or not has_frame_axis:
            return df

        cols = ["frame_number", *cls.MASK_COLS]
        mask_df = (
            DataFrame(list(mask_rows), columns=cols)
            .rename(columns={"frame_number": "Frame Number"})
            .set_index("Frame Number")
        )

        # Drop any pre-existing mask columns so a re-attach doesn't create
        # duplicate/suffixed columns via the join.
        clashing = [c for c in cls.MASK_COLS if c in df.columns]
        base = df.drop(columns=clashing) if clashing else df

        return base.join(mask_df, on="Frame Number")

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
