from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any

from pandas import DataFrame
from PIL import Image

from atomicds.core import BaseClient
from atomicds.results.optical import OpticalImageResult, OpticalResult
from atomicds.timeseries.provider import TimeseriesProvider


class OpticalProvider(TimeseriesProvider):
    TYPE = "optical"

    RENAME_MAP: Mapping[str, str] = {
        "relative_time_seconds": "Time",
        "frame_number": "Frame Number",
        "unix_timestamp_ms": "UNIX Timestamp",
        "perimeter_px": "Edge Perimeter",
        "circularity": "Edge Circularity",
        "edge_roughness": "Edge Roughness",
        "hausdorff_px": "Hausdorff Similarity",
    }
    INDEX_COLS: Sequence[str] = ["Frame Number"]

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

    def snapshot_image_uuids(self, frames_payload: dict[str, Any]) -> list[dict]:
        out: list[dict] = []
        frames = (frames_payload or {}).get("frames", [])
        for frame in frames:
            # Optical may emit different id keys; support both
            uuid = frame.get("image_uuid")

            if not uuid:
                continue

            meta = {
                k: v
                for k, v in frame.items()
                if k in {"time_seconds", "timestamp_seconds"}
            }
            out.append({"uuid": uuid, "metadata": meta})
        return out

    def _download_image_from_presigned(
        self, client: BaseClient, url: str
    ) -> Image.Image:
        img_bytes: bytes = client._get(base_override=url, sub_url="", deserialize=False)  # type: ignore[arg-type]
        return Image.open(BytesIO(img_bytes))

    def fetch_snapshot(
        self, client: BaseClient, req: dict
    ) -> OpticalImageResult | None:
        uuid = req.get("uuid") or req.get("image_uuid")

        if not uuid:
            return None

        meta = client._get(
            sub_url=f"data_entries/processed_data/{uuid}",
            params={"return_as": "url-download"},
        )  # type: ignore[assignment]
        if not meta:
            return None

        img = self._download_image_from_presigned(client, meta["url"])  # type: ignore  # noqa: PGH003

        return OpticalImageResult(
            data_id=uuid,
            processed_image=img,
        )

    def build_result(
        self,
        client: BaseClient,
        data_id: str,
        data_type: str,  # noqa: ARG002
        ts_df: DataFrame,
    ) -> OpticalResult:
        snapshots = None
        idx_url = self.snapshot_url(data_id)
        if idx_url:
            frames_payload: dict | None = client._get(sub_url=idx_url)  # type: ignore[assignment]
            if frames_payload:
                reqs = self.snapshot_image_uuids(frames_payload)
                snapshots = [
                    res
                    for res in client._multi_thread(
                        lambda req: self.fetch_snapshot(client, req),
                        kwargs_list=[{"req": r} for r in reqs],
                    )
                    if res
                ]
        return OpticalResult(
            data_id=data_id,
            timeseries_data=ts_df,
            snapshot_image_data=snapshots,
        )
