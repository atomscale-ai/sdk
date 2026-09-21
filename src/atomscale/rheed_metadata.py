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


# The rotation rate each pre-unification catalogue type implies. A backend older
# than the views endpoint serves no rpm at all, so the type name is the only
# evidence left: "rheed_stationary" means a parked stage, 0 by definition, while
# "rheed_rotating" means the stage turned at a rate that backend never recorded
# — reported as unknown rather than invented.
LEGACY_TYPE_RPM: dict[str, float] = {
    "rheed_stationary": 0.0,
    "rheed_rotating": float("nan"),
}


def legacy_views_frame(data_type: str) -> DataFrame:
    """One placeholder view for a backend with no ``/azimuths`` endpoint.

    Keeps ``RHEEDVideoResult.views`` the same shape whichever backend answered,
    so callers read ``rpm`` without first working out which one they are on. It
    carries no view identity: a backend that cannot serve views has none to give.
    """
    if data_type not in LEGACY_TYPE_RPM:
        return _empty()
    row: dict[str, Any] = dict.fromkeys(RHEED_AZIMUTH_COLUMNS)
    row["rpm"] = LEGACY_TYPE_RPM[data_type]
    return DataFrame([row])[list(RHEED_AZIMUTH_COLUMNS)]


def rheed_azimuths_to_dataframe(payload: Sequence[Mapping[str, Any]]) -> DataFrame:
    """One row per RHEED view, retaining identity even when labels repeat."""
    rows = []
    for view in payload:
        automatic = (view.get("annotation") or {}).get("automatic") or {}
        surface = automatic.get("surface_miller")
        rows.append(
            {
                **{key: view.get(key) for key in RHEED_AZIMUTH_COLUMNS},
                "data_id": str(view["data_id"]),
                "view_id": str(view["view_id"]),
                "interval_id": str(view["interval_id"]),
                "seed_frame": int(view["seed_frame"]),
                "label_confidence": view.get("confidence"),
                "crystal_system": automatic.get("crystal_system"),
                "surface_miller": "".join(str(i) for i in surface) if surface else None,
            }
        )
    if not rows:
        return _empty()
    return (
        DataFrame(rows)[list(RHEED_AZIMUTH_COLUMNS)]
        .sort_values(
            ["data_id", "start_frame", "seed_frame"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


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
