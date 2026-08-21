"""Parse the sample-scoped payloads that are *not* timeseries.

Three read endpoints describe a physical sample without returning a series:

* ``GET /physical_samples/{id}/process_steps`` — the sample's temporal phases
  (one growth/anneal/cooldown segment each), derived by clustering its data
  items' acquisition windows.
* ``GET /physical_samples/{id}/annotations/`` — spatially-resolved property
  values pinned to surface coordinates (e.g. an XPS ``sr_ti_ratio`` measured
  every 22.5 mm across a wafer).
* ``GET /summary/{data_id}`` — the agent-written summary of one data item,
  together with the status of the summarization workflow.

Each helper converts one decoded response body into a flat DataFrame. Sample
timeseries live in :mod:`atomscale.timeseries.physical_sample` instead; these
payloads are metadata and event records, so they share no axis and are kept
apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from pandas import DataFrame

# Process steps: one row per step, ordered by ``step_index``.
#
# ``data_ids`` stays in-frame as a tuple rather than moving to ``DataFrame.attrs``
# (the convention :mod:`atomscale.timeseries.physical_sample` uses for its
# provenance ids) because step membership *is* the payload here — callers fetch
# steps precisely to learn which data items belong to which phase — and a step
# with no members has to survive as a row.
PROCESS_STEP_COLUMNS: tuple[str, ...] = (
    "step_index",
    "process_step_id",
    "name",
    "start_unix_ms_utc",
    "end_unix_ms_utc",
    "duration_seconds",
    "start_datetime",
    "end_datetime",
    "n_data_ids",
    "data_ids",
    "last_updated",
)

# Spatial annotations: one row per annotated point.
SPATIAL_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "annotation_id",
    "data_id",
    "char_source_type",
    "property_name",
    "property_value",
    "property_unit",
    "coord_x",
    "coord_y",
    "coord_z",
    "coord_ref_frame",
    "coord_units",
    "metadata",
    "last_updated",
)

# AI summaries: one row per *requested* data id, so a data item with no summary
# is visible as a ``task_status`` row rather than a missing row.
AI_SUMMARY_COLUMNS: tuple[str, ...] = (
    "data_id",
    "task_status",
    "summary_id",
    "content",
    "structured_content",
    "generation_model",
    "generation_timestamp",
    "user_feedback",
    "workflow_id",
)

_FLOAT_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "process_steps": (
        "start_unix_ms_utc",
        "end_unix_ms_utc",
        "duration_seconds",
    ),
    "annotations": ("property_value", "coord_x", "coord_y", "coord_z"),
}


def _empty(columns: Sequence[str], float_columns: Sequence[str] = ()) -> DataFrame:
    """Empty frame carrying the full column set, with float columns typed."""
    frame = DataFrame({c: [] for c in columns})
    for column in float_columns:
        frame[column] = frame[column].astype("float64")
    return frame


def _as_float(value: Any) -> float:
    """Coerce a JSON number (or ``None``) to ``float``, ``None`` -> ``NaN``."""
    return float("nan") if value is None else float(value)


def process_steps_to_dataframe(payload: Mapping[str, Any]) -> DataFrame:
    """Convert a ``ProcessStepsResponse`` payload to one row per process step.

    Args:
        payload: Decoded response body, with a ``"process_steps"`` list whose
            entries carry ``id``, ``step_index``, ``name``,
            ``start_unix_ms_utc`` / ``end_unix_ms_utc`` (float epoch
            milliseconds, UTC), ``data_ids``, and ``last_updated``.

    Returns:
        DataFrame: The columns in :data:`PROCESS_STEP_COLUMNS`, sorted by
        ``step_index``. ``duration_seconds`` and the tz-aware
        ``start_datetime`` / ``end_datetime`` are derived from the epoch-ms
        bounds for convenience; the raw bounds are kept so callers aligning
        against absolute-time series (RHEED shutter times, tool state) never
        have to round-trip through a datetime. A sample with no steps yields an
        empty frame, never an error.
    """
    steps = list(payload.get("process_steps") or [])
    if not steps:
        return _empty(PROCESS_STEP_COLUMNS, _FLOAT_COLUMNS["process_steps"])

    rows = []
    for step in steps:
        start = _as_float(step.get("start_unix_ms_utc"))
        end = _as_float(step.get("end_unix_ms_utc"))
        data_ids = tuple(step.get("data_ids") or ())
        rows.append(
            {
                "step_index": step.get("step_index"),
                "process_step_id": step.get("id"),
                "name": step.get("name"),
                "start_unix_ms_utc": start,
                "end_unix_ms_utc": end,
                "duration_seconds": (end - start) / 1000.0,
                "n_data_ids": len(data_ids),
                "data_ids": data_ids,
                "last_updated": step.get("last_updated"),
            }
        )

    frame = DataFrame(rows)
    frame["start_datetime"] = pd.to_datetime(
        frame["start_unix_ms_utc"], unit="ms", utc=True
    )
    frame["end_datetime"] = pd.to_datetime(
        frame["end_unix_ms_utc"], unit="ms", utc=True
    )
    frame["last_updated"] = pd.to_datetime(frame["last_updated"], errors="coerce")
    return (
        frame[list(PROCESS_STEP_COLUMNS)]
        .sort_values("step_index", kind="stable")
        .reset_index(drop=True)
    )


def spatial_annotations_to_dataframe(payload: Sequence[Mapping[str, Any]]) -> DataFrame:
    """Convert a ``list[SpatialAnnotationRead]`` payload to one row per point.

    Args:
        payload: Decoded response body — a list of annotation objects, each with
            ``id``, ``property_name``, ``property_value``, optional
            ``property_unit`` / ``char_source_type`` / ``data_id``, the
            ``coord_x`` / ``coord_y`` / ``coord_z`` surface position with its
            ``coord_ref_frame`` and ``coord_units``, and ``metadata``.

    Returns:
        DataFrame: The columns in :data:`SPATIAL_ANNOTATION_COLUMNS`, sorted by
        ``property_name`` then position so a scan reads left-to-right. The
        server-side ``id`` is exposed as ``annotation_id`` to keep it distinct
        from ``data_id``. No annotations yields an empty frame, never an error.
    """
    records = list(payload or [])
    if not records:
        return _empty(SPATIAL_ANNOTATION_COLUMNS, _FLOAT_COLUMNS["annotations"])

    rows = [
        {
            "annotation_id": record.get("id"),
            "data_id": record.get("data_id"),
            "char_source_type": record.get("char_source_type"),
            "property_name": record.get("property_name"),
            "property_value": _as_float(record.get("property_value")),
            "property_unit": record.get("property_unit"),
            "coord_x": _as_float(record.get("coord_x")),
            "coord_y": _as_float(record.get("coord_y")),
            "coord_z": _as_float(record.get("coord_z")),
            "coord_ref_frame": record.get("coord_ref_frame"),
            "coord_units": record.get("coord_units"),
            "metadata": record.get("metadata"),
            "last_updated": record.get("last_updated"),
        }
        for record in records
    ]

    frame = DataFrame(rows)
    frame["last_updated"] = pd.to_datetime(frame["last_updated"], errors="coerce")
    return (
        frame[list(SPATIAL_ANNOTATION_COLUMNS)]
        .sort_values(
            ["property_name", "coord_x", "coord_y"], kind="stable", na_position="last"
        )
        .reset_index(drop=True)
    )


def ai_summary_row(data_id: str, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten one ``GetSummaryResponse`` into an :data:`AI_SUMMARY_COLUMNS` row.

    Args:
        data_id: The data id the response was fetched for; carried through so a
            batch of rows stays attributable.
        payload: Decoded response body with ``summary`` (the note, or ``None``),
            ``task_status`` (``"available"`` / ``"pending"`` / ``"not_found"``),
            and ``workflow_id``. ``None`` — a 404 from the endpoint — is
            reported as ``task_status="not_found"``, matching how the endpoint
            reports an un-summarized data item that does exist.

    Returns:
        dict: One flat row; summary-note fields are ``None`` unless the note is
        present.
    """
    body = payload or {}
    summary = body.get("summary") or {}
    return {
        "data_id": data_id,
        "task_status": body.get("task_status") or "not_found",
        "summary_id": summary.get("id"),
        "content": summary.get("content"),
        "structured_content": summary.get("structured_content"),
        "generation_model": summary.get("generation_model"),
        "generation_timestamp": summary.get("generation_timestamp"),
        "user_feedback": summary.get("user_feedback"),
        "workflow_id": body.get("workflow_id"),
    }


def ai_summaries_to_dataframe(rows: Sequence[Mapping[str, Any]]) -> DataFrame:
    """Assemble :func:`ai_summary_row` outputs into a frame.

    Args:
        rows: One row per requested data id, in the caller's request order.

    Returns:
        DataFrame: The columns in :data:`AI_SUMMARY_COLUMNS`, request order
        preserved, with ``generation_timestamp`` parsed to datetime. An empty
        ``rows`` yields an empty frame.
    """
    if not rows:
        return _empty(AI_SUMMARY_COLUMNS)

    frame = DataFrame(list(rows))[list(AI_SUMMARY_COLUMNS)]
    frame["generation_timestamp"] = pd.to_datetime(
        frame["generation_timestamp"], errors="coerce"
    )
    return frame
