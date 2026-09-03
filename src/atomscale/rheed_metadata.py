"""Parse per-azimuth RHEED metadata for a rotating-substrate recording.

``GET /configuration/rheed/video/`` carries, for each seed frame of a rotating
video, both the rotation angle the frame sits at (``seed_frame_angles``) and the
crystallographic azimuth the production classifier assigned it
(``api_configuration.azimuth_labels`` — one of ``"100"`` / ``"110"`` / ``"210"``,
or ``None`` when it could not be called), plus the fit detail behind that call in
``azimuth_label_meta``.

The distinction matters when comparing recordings: the rotation angle is a
property of how the substrate happened to be parked, so the *same* azimuth turns
up at different angles in different recordings of the same sample. The
crystallographic label is the stable identity, and therefore the correct key for
aligning or concatenating series across recordings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from pandas import DataFrame

# One row per (data item, seed frame).
RHEED_AZIMUTH_COLUMNS: tuple[str, ...] = (
    "data_id",
    "seed_frame",
    "angle_degrees",
    "azimuth_label",
    "label_confidence",
    "crystal_system",
    "surface_miller",
)


def _empty() -> DataFrame:
    frame = DataFrame({c: [] for c in RHEED_AZIMUTH_COLUMNS})
    frame["seed_frame"] = frame["seed_frame"].astype("int64")
    frame["angle_degrees"] = frame["angle_degrees"].astype("float64")
    frame["label_confidence"] = frame["label_confidence"].astype("float64")
    return frame


def rheed_azimuths_to_dataframe(payload: Sequence[Mapping[str, Any]]) -> DataFrame:
    """Convert a ``list[RheedVideoConfigurationResponse]`` to one row per azimuth.

    Args:
        payload: Decoded response body — one entry per requested data id, each
            with ``data_id``, ``seed_frame_angles`` (seed frame -> degrees) and an
            ``api_configuration`` carrying ``azimuth_labels`` (seed frame ->
            label) and ``azimuth_label_meta`` (seed frame -> fit detail).

    Returns:
        DataFrame: The columns in :data:`RHEED_AZIMUTH_COLUMNS`, sorted by
        ``data_id`` then ``seed_frame``. ``azimuth_label`` is ``None`` for a seed
        frame the classifier could not label, so callers can fall back to the
        angle rather than inventing an identity. Non-rotating or unconfigured
        entries contribute no rows; an empty payload yields an empty frame.
    """
    rows: list[dict[str, Any]] = []
    for entry in payload or []:
        data_id = entry.get("data_id")
        angles = entry.get("seed_frame_angles") or {}
        api_configuration = entry.get("api_configuration") or {}
        labels = api_configuration.get("azimuth_labels") or {}
        meta = api_configuration.get("azimuth_label_meta") or {}

        # Seed frames come from the angle map: it is the endpoint's own record of
        # which frames this recording actually has, whereas the label map may be
        # partial (an unlabelled azimuth is simply absent).
        for seed_frame, angle in angles.items():
            detail = meta.get(seed_frame) or {}
            surface = detail.get("surface")
            rows.append(
                {
                    "data_id": str(data_id),
                    "seed_frame": int(seed_frame),
                    "angle_degrees": float(angle)
                    if angle is not None
                    else float("nan"),
                    "azimuth_label": labels.get(seed_frame),
                    "label_confidence": (
                        float(detail["confidence"])
                        if detail.get("confidence") is not None
                        else float("nan")
                    ),
                    "crystal_system": detail.get("system"),
                    # A Miller index arrives as a list; join it so the column stays
                    # scalar and groupable.
                    "surface_miller": (
                        "".join(str(_i) for _i in surface)
                        if isinstance(surface, list | tuple)
                        else surface
                    ),
                }
            )

    if not rows:
        return _empty()

    return (
        DataFrame(rows)[list(RHEED_AZIMUTH_COLUMNS)]
        .sort_values(["data_id", "seed_frame"], kind="stable")
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
