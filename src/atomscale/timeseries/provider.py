"""Provider classes for accessing timeseries data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
import pandas as pd
from pandas import DataFrame

from atomscale.core import BaseClient

R = TypeVar("R")  # the result type this provider returns


_STAT_SUFFIX_LABELS: Mapping[str, str] = {
    "_z_score": "Z Score",
    "_ema_z_score": "EMA Z Score",
    "_anomaly_probability": "Anomaly Probability",
}


def extend_with_statistics(rename_map: Mapping[str, str]) -> dict[str, str]:
    """Add suffix-based statistical labels to a rename map."""
    extended = dict(rename_map)
    for source_name, label in rename_map.items():
        for suffix, suffix_label in _STAT_SUFFIX_LABELS.items():
            extended[f"{source_name}{suffix}"] = f"{label} {suffix_label}"
    return extended


def _to_int64_ms(values: Sequence[Any]) -> np.ndarray:
    """Coerce a sequence of unix-ms values (Decimal | float | int | str) to int64.

    Whole-millisecond values fit losslessly in int64 well past year 2286.
    Decimals with sub-millisecond fractional parts are truncated; this is
    correct for the property-centric API where ms are emitted as whole
    numbers (DECIMAL(13,3) seconds * 1000 -> integer ms).
    """
    out = np.empty(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        if isinstance(v, Decimal | int | np.integer | float):
            out[i] = int(v)
        else:
            out[i] = int(Decimal(str(v)))
    return out


def properties_payload_to_dataframe(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    display: bool = True,
) -> DataFrame:
    """Merge a property-centric payload into a wide DataFrame.

    The payload shape is::

        {
          "<property_name>": {
            "relative_time_seconds": [...],
            "unix_timestamp_ms":     [...],
            "values":                [...],
            "units": "..."
          },
          ...
        }

    Each property carries its own ``unix_timestamp_ms`` and
    ``relative_time_seconds`` arrays (potentially different sample rates).
    Values are outer-joined on the union of ``unix_timestamp_ms`` across
    all properties and forward-filled onto that index.

    Forward-fill (rather than numerical interpolation) is correct for
    setpoints and shutter states, which are step functions that hold
    constant between samples. For dense continuous sensors (pyrometers
    etc.) the error between samples is bounded. Forward-fill preserves
    real measurements rather than manufacturing synthetic values.

    Returned columns:
      - ``UNIX Timestamp`` (int64, milliseconds)
      - ``Time`` (float64, relative seconds; anchored to the longest
        property's t=0 via piecewise-linear extrapolation)
      - one column per property keyed by its API name (case preserved —
        these are setpoint/shutter/etc. names users grep for)

    When ``display`` is False the two time columns are emitted under their
    raw API names (``unix_timestamp_ms`` / ``relative_time_seconds``)
    instead of the user-facing ``UNIX Timestamp`` / ``Time`` labels. The
    per-property columns are already raw API names, so they are unaffected.

    Index is row number. Empty input → empty DataFrame.

    Unix timestamps are accepted as ``Decimal``, ``float``, ``int``, or
    numeric strings; they are cast to int64 for the DataFrame column.
    """
    if not properties:
        return DataFrame()

    series_by_name: dict[str, pd.Series] = {}
    longest_name: str | None = None
    longest_len = -1
    longest_ts_ms: np.ndarray = np.array([], dtype=np.int64)
    longest_rel: np.ndarray = np.array([], dtype=np.float64)

    for name, prop in properties.items():
        ts_raw = prop.get("unix_timestamp_ms") or []
        values = prop.get("values") or []
        rel_raw = prop.get("relative_time_seconds") or []

        # Drop entries where the timestamp is null.
        kept_ts: list[Any] = []
        kept_values: list[Any] = []
        kept_rel: list[Any] = []
        for j, t in enumerate(ts_raw):
            if t is None:
                continue
            kept_ts.append(t)
            if j < len(values):
                kept_values.append(values[j])
            else:
                kept_values.append(None)
            if j < len(rel_raw):
                kept_rel.append(rel_raw[j])
            else:
                kept_rel.append(None)

        if not kept_ts:
            continue

        ts_ms = _to_int64_ms(kept_ts)
        # Stable order is enforced by the merged index sort below.
        series_by_name[name] = pd.Series(kept_values, index=ts_ms, name=name)

        if len(ts_ms) > longest_len:
            longest_len = len(ts_ms)
            longest_name = name
            longest_ts_ms = ts_ms
            longest_rel = np.asarray(
                [float(r) if r is not None else np.nan for r in kept_rel],
                dtype=np.float64,
            )

    if not series_by_name:
        return DataFrame()

    merged = pd.concat(series_by_name.values(), axis=1, join="outer")
    merged = merged.sort_index()
    merged = merged.ffill()

    unix_ms = np.asarray(merged.index, dtype=np.int64)

    # Build a Time (relative seconds) column anchored to the longest
    # property's t=0 via piecewise-linear extrapolation on its
    # (unix_ms, relative_seconds) pairs.
    if longest_name is not None and len(longest_ts_ms) >= 1:
        # np.interp clamps outside-range values to the nearest endpoint.
        # We want extrapolation: rel(unix_ms) ≈ rel0 + (unix_ms - ts0)/1000.
        if len(longest_ts_ms) >= 2:
            time_col = np.interp(unix_ms, longest_ts_ms, longest_rel).astype(np.float64)
            # Fix up extrapolation past either edge.
            below = unix_ms < longest_ts_ms[0]
            above = unix_ms > longest_ts_ms[-1]
            time_col[below] = (
                longest_rel[0] + (unix_ms[below] - longest_ts_ms[0]) / 1000.0
            )
            time_col[above] = (
                longest_rel[-1] + (unix_ms[above] - longest_ts_ms[-1]) / 1000.0
            )
        else:
            # One sample: extrapolate purely from the (anchor_ts, anchor_rel) pair.
            anchor_ts = float(longest_ts_ms[0])
            anchor_rel = float(longest_rel[0]) if not np.isnan(longest_rel[0]) else 0.0
            time_col = anchor_rel + (unix_ms - anchor_ts) / 1000.0
    else:
        time_col = np.zeros(len(unix_ms), dtype=np.float64)

    out = merged.reset_index(drop=True)
    out.insert(0, "UNIX Timestamp" if display else "unix_timestamp_ms", unix_ms)
    out.insert(1, "Time" if display else "relative_time_seconds", time_col)
    return out


def series_payload_to_dataframe(
    series: Sequence[Mapping[str, Any]],
) -> DataFrame:
    """Parse the legacy row-oriented ``series`` payload into a wide DataFrame.

    Each row carries ``unix_timestamp_ms``, ``relative_time_seconds``, and one
    key per property. This shape predates the property-centric payload and is
    still emitted by un-migrated API deployments. Output schema mirrors
    :func:`properties_payload_to_dataframe` so downstream code is shape-stable
    across both API versions.
    """
    if not series:
        return DataFrame()

    rows = DataFrame(list(series))

    if "unix_timestamp_ms" in rows.columns:
        rows["unix_timestamp_ms"] = _to_int64_ms(rows["unix_timestamp_ms"].tolist())
    if "relative_time_seconds" in rows.columns:
        rows["relative_time_seconds"] = rows["relative_time_seconds"].astype("float64")

    leading = [
        c for c in ("unix_timestamp_ms", "relative_time_seconds") if c in rows.columns
    ]
    remaining = [c for c in rows.columns if c not in leading]
    return rows[leading + remaining]


def finalize_dataframe(
    df: DataFrame,
    rename_map: Mapping[str, str],
    *,
    display: bool = True,
    index_cols: Sequence[str] = (),
) -> DataFrame:
    """Apply user-facing column names and set the domain index.

    ``index_cols`` are given as raw (API/source) column names. When
    ``display`` is True the frame's columns are renamed to their display
    labels via ``rename_map`` and the index names are resolved through the
    same map; when False the raw snake_case API names are preserved on both
    the columns and the index. Index columns absent from the frame are
    skipped so the result stays shape-stable across payload variants.
    """
    if display:
        df = df.rename(columns=rename_map)
    resolved = [rename_map.get(c, c) if display else c for c in index_cols]
    present = [c for c in resolved if c in df.columns]
    if present:
        df = df.set_index(present)
    return df


class TimeseriesProvider(ABC, Generic[R]):
    """Strategy interface for parsing timeseries by domain."""

    # canonical domain name used as a key in the registry
    TYPE: ClassVar[str]

    @abstractmethod
    def fetch_raw(self, client: BaseClient, data_id: str) -> Any:
        """Perform the HTTP GET(s) to retrieve raw payload(s)."""

    @abstractmethod
    def to_dataframe(self, raw: Any, *, display: bool = True) -> DataFrame:
        """Convert raw payload to a tidy DataFrame with domain-specific renames/index.

        When ``display`` is True (default) columns carry user-facing display
        labels; when False they keep their raw snake_case API names.
        """

    @abstractmethod
    def build_result(
        self,
        client: BaseClient,
        data_id: str,
        data_type: str,
        ts_df: DataFrame,
    ) -> R:
        """Build time series result object"""

    # Optional override points
    def snapshot_url(self, data_id: str) -> str:  # noqa: ARG002
        """API endpoint that exposes extracted/snapshot frames."""
        return ""

    def snapshot_image_uuids(
        self,
        frames_payload: dict[str, Any],  # noqa: ARG002
    ) -> list[dict]:
        """Extract requests from frames payload. Default: no snapshots."""
        return []

    def fetch_snapshot(
        self,
        client: BaseClient,  # noqa: ARG002
        req: dict,  # noqa: ARG002
    ) -> Any | None:
        """Resolve one snapshot request → domain-specific ImageResult (or None)."""
        return None
