from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from atomscale.results import (
    EllipsometryResult,
    OpticalResult,
    RHEEDVideoResult,
    ToolStateResult,
)

# Column names whose suffix encodes the unit. Order matters: more specific
# names (with explicit unit) take priority over unit-less display names.
_UNIT_BY_NAME: dict[str, str] = {
    "unix_timestamp_ns": "ns",
    "timestamp_ns": "ns",
    "unix_timestamp_us": "us",
    "timestamp_us": "us",
    "unix_timestamp_ms": "ms",
    "timestamp_ms": "ms",
    "unix_timestamp_seconds": "s",
    "timestamp_seconds": "s",
    "unix_timestamp_s": "s",
    "timestamp_s": "s",
}

# Boundaries used to infer which unit a column's magnitude actually suggests.
# Values above each threshold indicate the corresponding unit (or larger).
_MAGNITUDE_THRESHOLDS: list[tuple[float, str]] = [
    (1e18, "ns"),
    (1e15, "us"),
    (1e12, "ms"),
    (1e9, "s"),
]


def _unit_suggested_by_magnitude(max_val: float) -> str:
    """Best-guess unit for an absolute timestamp based on its max magnitude.

    >>> _unit_suggested_by_magnitude(1.7e12)  # ms-scale
    'ms'
    >>> _unit_suggested_by_magnitude(1.7e9)   # s-scale
    's'
    """
    for threshold, unit in _MAGNITUDE_THRESHOLDS:
        if max_val >= threshold:
            return unit
    # Below year ~2001; ambiguous but assume seconds (the smallest unit).
    return "s"


# Display names that don't carry a unit. The magnitude heuristic is used
# for these (preserving the prior behavior where users render with
# "UNIX Timestamp" without committing to a unit suffix).
ABS_TIME_COLS_UNITLESS = (
    "UNIX Timestamp",
    "Unix Timestamp",
    "unix_timestamp",
    "timestamp",
)

# Backwards-compatible export for any consumer importing this constant.
ABS_TIME_COLS = tuple(_UNIT_BY_NAME.keys()) + ABS_TIME_COLS_UNITLESS

REL_TIME_COLS = (
    "time_seconds",
    "relative_time_seconds",
    "Relative Time",
    "Time",
)


def _infer_absolute_time(df: pd.DataFrame) -> pd.Series | None:
    """Return UTC datetime index from absolute (epoch-based) columns.

    A column whose name encodes the unit (``unix_timestamp_ms``,
    ``timestamp_seconds``, etc.) is treated as authoritative for the
    unit. If the values' magnitude is implausibly small for that unit
    (e.g. a ``_ms`` column carrying values < 1e10), raise rather than
    silently mis-parse.

    Unit-less names (``UNIX Timestamp``, ``timestamp``, etc.) fall through
    to magnitude-based heuristic detection.
    """
    for col, declared_unit in _UNIT_BY_NAME.items():
        if col not in df.columns:
            continue
        target = pd.to_numeric(df[col], errors="coerce")
        max_val = target.max(skipna=True)
        if pd.notna(max_val):
            suggested = _unit_suggested_by_magnitude(float(max_val))
            if suggested != declared_unit:
                raise ValueError(
                    f"Column {col!r} declares unit {declared_unit!r} but the "
                    f"value magnitude (max={float(max_val):g}) suggests unit "
                    f"{suggested!r}. The column appears to be filled with "
                    "values in a different unit."
                )
        return pd.to_datetime(target, unit=declared_unit, errors="coerce", utc=True)

    for col in ABS_TIME_COLS_UNITLESS:
        if col not in df.columns:
            continue
        series = df[col]

        numeric_series = pd.to_numeric(series, errors="coerce")
        has_numeric = numeric_series.notna().any()
        target = numeric_series if has_numeric else series

        if pd.api.types.is_integer_dtype(target) or has_numeric:
            max_val = target.max(skipna=True)
            if max_val >= 1e18:
                unit = "ns"
            elif max_val >= 1e15:
                unit = "us"
            elif max_val >= 1e12:
                unit = "ms"
            else:
                unit = "s"
            return pd.to_datetime(target, unit=unit, errors="coerce", utc=True)

        if pd.api.types.is_float_dtype(target):
            return pd.to_datetime(target, unit="s", errors="coerce", utc=True)

        # assume already datetime-like strings
        return pd.to_datetime(target, errors="coerce", utc=True)

    return None


def _infer_relative_time(df: pd.DataFrame) -> pd.Series | None:
    """Return timedelta series from relative time columns (seconds)."""
    for col in REL_TIME_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        return pd.to_timedelta(series, unit="s", errors="coerce")
    return None


def _extract_timeseries(result):
    """Return (data_id, domain, df_with_timeindex) or None for non-timeseries."""
    if isinstance(result, RHEEDVideoResult):
        domain = "rheed"
        timeseries = result.timeseries_data
    elif isinstance(result, OpticalResult):
        domain = "optical"
        timeseries = result.timeseries_data
    elif isinstance(result, ToolStateResult):
        domain = "tool_state"
        timeseries = result.timeseries_data
    elif isinstance(result, EllipsometryResult):
        domain = "ellipsometry"
        timeseries = result.timeseries_data
    else:
        return None

    if timeseries is None or timeseries.empty:
        return None

    # Build time index: prefer absolute epochs; fall back to upload_datetime + relative offsets.
    upload_dt = getattr(result, "upload_datetime", None)

    time_index = _infer_absolute_time(timeseries)
    if time_index is None and upload_dt is not None:
        base = pd.to_datetime(upload_dt, utc=True, errors="coerce")
        rel = _infer_relative_time(timeseries)
        if base is not pd.NaT and rel is not None:
            time_index = base + rel

    if time_index is None:
        return None

    valid_mask = time_index.notna()
    if not valid_mask.any():
        return None

    indexed = timeseries.loc[valid_mask].copy(deep=False)
    indexed.index = pd.Index(time_index[valid_mask], name="time")
    indexed = indexed.sort_index()

    if not indexed.index.is_unique:
        indexed = indexed[~indexed.index.duplicated(keep="first")]

    return str(result.data_id), domain, indexed


def _infer_resample_freq(indices: list[pd.DatetimeIndex]) -> pd.Timedelta | None:
    """Infer a reasonable base frequency from multiple datetime indices."""
    deltas: list[pd.Timedelta] = []
    for idx in indices:
        if idx.size < 2:
            continue
        # use numpy diff to avoid Series construction overhead
        diffs = idx.view("int64")[1:] - idx.view("int64")[:-1]
        if diffs.size:
            median_ns = pd.Series(diffs).median()  # nan-safe median
            if pd.notna(median_ns) and median_ns > 0:
                deltas.append(pd.Timedelta(median_ns, unit="ns"))

    if not deltas:
        return None

    # Use the median of medians to avoid over-densifying to the smallest step
    median_delta = pd.Series(deltas).median()
    if median_delta <= pd.Timedelta(0):
        return None

    # Enforce a floor to avoid overly dense grids
    floor = pd.Timedelta(seconds=0.5)
    if median_delta < floor:
        median_delta = floor

    # Clamp frequency to keep total grid size reasonable (performance guard)
    min_ts = min(idx[0] for idx in indices if len(idx))
    max_ts = max(idx[-1] for idx in indices if len(idx))
    span = max_ts - min_ts
    if span <= pd.Timedelta(0):
        return median_delta

    est_points = span / median_delta
    max_points = 1_000_000  # tighter cap for performance
    if est_points > max_points:
        median_delta = span / max_points

    return median_delta


def align_timeseries(
    results: Iterable,
    *,
    how: str = "outer",
) -> pd.DataFrame | None:
    """Align timeseries results by time index.

    Args:
        results: Iterable of result objects (RHEEDVideoResult, OpticalResult, ToolStateResult).
        how: Join strategy for the outer alignment. Defaults to "outer".

    Returns:
        DataFrame | None: Aligned DataFrame with MultiIndex columns (data_id, domain, metric).
    """
    frames: list[pd.DataFrame] = []
    indices: list[pd.DatetimeIndex] = []
    for item in results:
        extracted = _extract_timeseries(item)
        if not extracted:
            continue

        data_id, domain, frame = extracted
        frame = frame.copy(deep=False)
        frame.columns = pd.MultiIndex.from_product([[data_id], [domain], frame.columns])
        frames.append(frame)
        indices.append(frame.index)

    if not frames:
        return pd.DataFrame()

    join_how = how if how in {"outer", "inner"} else "outer"
    aligned = pd.concat(frames, axis=1, join=join_how, copy=False, sort=False)
    aligned = aligned.sort_index()

    if how == "left":
        aligned = aligned.reindex(frames[0].index)
    elif how == "right":
        aligned = aligned.reindex(frames[-1].index)

    freq = _infer_resample_freq(indices)
    if freq:
        aligned = aligned.resample(freq).mean(numeric_only=True)

    # Drop raw time columns now that the index is the aligned time base.
    if isinstance(aligned.columns, pd.MultiIndex):
        time_metrics = {
            "Time",
            "UNIX Timestamp",
            "Relative Time",
            "timestamp",
            "timestamp_ms",
            "timestamp_seconds",
        }
        aligned = aligned.loc[
            :, [c for c in aligned.columns if c[-1] not in time_metrics]
        ]

    # Merge compatible metrics across items: if multiple columns share (domain, metric)
    # and never conflict where they overlap, collapse into (shared, domain, metric).
    def _merge_compatible_metrics(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data.columns, pd.MultiIndex):
            return data
        domains = data.columns.get_level_values(1)
        metrics = data.columns.get_level_values(2)
        new_cols: dict = {}
        drop_cols: list = []

        for domain in domains.unique():
            for metric in metrics.unique():
                cols = [
                    c
                    for c in data.columns
                    if c[1] == domain and c[2] == metric and c[0] != "shared"
                ]
                if len(cols) <= 1:
                    continue

                merged = data[cols[0]]
                conflict = False
                for c in cols[1:]:
                    other = data[c]
                    overlap_mask = merged.notna() & other.notna()
                    if (merged[overlap_mask] != other[overlap_mask]).any():
                        conflict = True
                        break
                    merged = merged.combine_first(other)

                if conflict:
                    continue

                new_col = ("shared", domain, metric)
                new_cols[new_col] = merged
                drop_cols.extend(cols)

        if new_cols:
            data = data.drop(columns=drop_cols)
            for col, series in new_cols.items():
                data[col] = series
            data = data.sort_index(axis=1)
        return data

    return _merge_compatible_metrics(aligned)
