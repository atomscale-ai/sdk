"""Parse sample-scoped computed timeseries results.

The backend endpoint ``GET /physical_samples/{id}/timeseries/`` returns the
*current* row per computed property that per-sample DBOS workflows produce —
today ``rheed_quality`` and ``composition_metric``. Each property is a compiled
series content-addressed by its constituent ``data_id`` set; the endpoint
returns the latest row for each property.

This module converts that payload into a long-form DataFrame. Long form (rather
than a wide join on ``real_time_seconds``) because distinct properties can carry
*different* axes — different ``result_id`` / constituent-data-id sets — so a
naive wide join would silently mis-align them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

# Long-form output schema (scalar columns). Per-property list-valued provenance
# (``constituent_data_ids``) is carried in ``DataFrame.attrs`` instead of a
# list-in-cell column so the frame stays flat and groupby/filter-friendly.
PHYSICAL_SAMPLE_TS_COLUMNS: tuple[str, ...] = (
    "property_name",
    "real_time_seconds",
    "value",
    "result_id",
    "last_updated",
    "generating_dbos_workflow_id",
)


def _float_array(values: Sequence[Any]) -> np.ndarray:
    """Coerce a sequence to ``float64``, mapping ``None`` (JSON null) to ``NaN``."""
    return np.array(
        [np.nan if v is None else float(v) for v in values], dtype="float64"
    )


def _empty_frame() -> DataFrame:
    """Empty long-form frame with the standard columns and dtypes."""
    frame = DataFrame({c: [] for c in PHYSICAL_SAMPLE_TS_COLUMNS})
    frame["real_time_seconds"] = frame["real_time_seconds"].astype("float64")
    frame["value"] = frame["value"].astype("float64")
    frame["last_updated"] = pd.to_datetime(frame["last_updated"])
    frame.attrs["constituent_data_ids"] = {}
    return frame


def physical_sample_timeseries_to_dataframe(
    payload: Mapping[str, Any],
    *,
    property_names: Sequence[str] | None = None,
) -> DataFrame:
    """Convert a ``PhysicalSampleTimeseriesResponse`` payload to long form.

    Args:
        payload: The decoded response body, with a ``"properties"`` list whose
            entries carry ``property_name``, ``property_values`` (JSON ``null``
            for gaps), ``real_time_seconds`` (1:1 with values), and provenance
            (``result_id``, ``last_updated``, ``constituent_data_ids``,
            ``generating_dbos_workflow_id``). If ``unix_times`` is present (a
            backend follow-up), it is passed through as a column.
        property_names: If given, keep only these properties (client-side; the
            endpoint has no filter param yet). ``None`` keeps all.

    Returns:
        DataFrame: One row per (property, sample-point) with the columns in
        :data:`PHYSICAL_SAMPLE_TS_COLUMNS` (plus ``unix_times`` when the payload
        provides it). ``value`` and ``real_time_seconds`` are ``float64`` with
        ``NaN`` for null gaps; ``last_updated`` is datetime. Per-property
        ``constituent_data_ids`` are stored in ``df.attrs["constituent_data_ids"]``
        keyed by property name. Empty / fully-filtered payloads yield an empty
        frame (never an error).

    Raises:
        ValueError: If a property's ``property_values`` and ``real_time_seconds``
            differ in length (malformed backend response).
    """
    properties = list(payload.get("properties") or [])

    if property_names is not None:
        wanted = set(property_names)
        properties = [p for p in properties if p.get("property_name") in wanted]

    if not properties:
        return _empty_frame()

    has_unix = any(p.get("unix_times") for p in properties)

    frames: list[DataFrame] = []
    constituent: dict[str, list] = {}
    for prop in properties:
        name = prop.get("property_name")
        values = list(prop.get("property_values") or [])
        axis = list(prop.get("real_time_seconds") or [])
        if len(values) != len(axis):
            raise ValueError(
                f"Malformed physical-sample timeseries for property {name!r}: "
                f"{len(values)} value(s) vs {len(axis)} time point(s)."
            )

        axis_arr = _float_array(axis)
        data: dict[str, Any] = {
            # Scalars broadcast to the axis length (0 rows when the axis is empty).
            "property_name": name,
            "real_time_seconds": axis_arr,
            "value": _float_array(values),
            "result_id": prop.get("result_id"),
            "last_updated": prop.get("last_updated"),
            "generating_dbos_workflow_id": prop.get("generating_dbos_workflow_id"),
        }
        if has_unix:
            unix = list(prop.get("unix_times") or [])
            data["unix_times"] = (
                _float_array(unix)
                if len(unix) == len(axis_arr)
                else np.full(len(axis_arr), np.nan, dtype="float64")
            )
        frames.append(DataFrame(data))
        constituent[name] = list(prop.get("constituent_data_ids") or [])

    out = pd.concat(frames, ignore_index=True)

    ordered = ["property_name", "real_time_seconds"]
    if has_unix:
        ordered.append("unix_times")
    ordered += ["value", "result_id", "last_updated", "generating_dbos_workflow_id"]
    out = out[ordered]

    out["last_updated"] = pd.to_datetime(out["last_updated"], errors="coerce")
    out.attrs["constituent_data_ids"] = constituent
    return out
