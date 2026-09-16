"""Read effective azimuth annotations for parked and rotating RHEED views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from pandas import DataFrame

# One row per (data item, seed frame).
RHEED_AZIMUTH_COLUMNS: tuple[str, ...] = (
    "data_id",
    "seed_frame",
    "view_id",
    "interval_id",
    "processed_data_id",
    "start_frame",
    "end_frame",
    "start_unix_ms",
    "end_unix_ms",
    "rpm",
    "azimuth_source",
    "azimuth_label",
    "label_confidence",
    "crystal_system",
    "surface_miller",
)


def _empty() -> DataFrame:
    frame = DataFrame({c: [] for c in RHEED_AZIMUTH_COLUMNS})
    frame["seed_frame"] = frame["seed_frame"].astype("int64")
    frame["label_confidence"] = frame["label_confidence"].astype("float64")
    return frame


def rheed_azimuths_to_dataframe(payload: Sequence[Mapping[str, Any]]) -> DataFrame:
    """One row per RHEED view, retaining identity even when labels repeat."""
    rows = []
    for view in payload:
        automatic = (view.get("annotation") or {}).get("automatic") or {}
        surface = automatic.get("surface_miller")
        rows.append({
            **{key: view.get(key) for key in RHEED_AZIMUTH_COLUMNS},
            "data_id": str(view["data_id"]),
            "view_id": str(view["view_id"]),
            "interval_id": str(view["interval_id"]),
            "seed_frame": int(view["seed_frame"]),
            "label_confidence": view.get("confidence"),
            "crystal_system": automatic.get("crystal_system"),
            "surface_miller": "".join(str(i) for i in surface) if surface else None,
        })
    if not rows:
        return _empty()
    return DataFrame(rows)[list(RHEED_AZIMUTH_COLUMNS)].sort_values(
        ["data_id", "start_frame", "seed_frame"], kind="stable",
    ).reset_index(drop=True)


def azimuth_label_by_seed_frame(
    azimuths: DataFrame, data_id: str
) -> dict[int, str | None]:
    """``{seed_frame: azimuth_label}`` for one data item, for joining onto series.

    Args:
        azimuths: A frame from :func:`rheed_azimuths_to_dataframe`.
        data_id: The data item to select.

    Returns:
        dict: Seed frame -> label, including seed frames whose label is ``None``.
    """
    if not len(azimuths):
        return {}
    mine = azimuths[azimuths["data_id"].astype(str) == str(data_id)]
    return {
        int(row.seed_frame): (None if pd.isna(row.azimuth_label) else row.azimuth_label)
        for row in mine.itertuples()
    }
