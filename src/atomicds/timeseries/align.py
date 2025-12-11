from __future__ import annotations

from collections.abc import Iterable
from functools import reduce

import pandas as pd

from atomicds.results import MetrologyResult, OpticalResult, RHEEDVideoResult


def _infer_time_column(df: pd.DataFrame) -> pd.Series:
    """Return a datetime-like series suitable for alignment."""
    candidates = [
        "UNIX Timestamp",
        "unix_timestamp_ms",
        "timestamp_ms",
        "timestamp",
        "time_seconds",
        "relative_time_seconds",
        "Time",
    ]
    for col in candidates:
        if col in df.columns:
            series = df[col]
            # Heuristically pick unit for integer epoch values.
            if pd.api.types.is_integer_dtype(series):
                max_val = series.max(skipna=True)
                if max_val > 1e12:
                    dt = pd.to_datetime(series, unit="ns", errors="coerce")
                elif max_val > 1e10:
                    dt = pd.to_datetime(series, unit="ms", errors="coerce")
                else:
                    dt = pd.to_datetime(series, unit="s", errors="coerce")
            elif pd.api.types.is_float_dtype(series):
                dt = pd.to_datetime(series, unit="s", errors="coerce")
            else:
                dt = pd.to_datetime(series, errors="coerce")
            return dt
    # Fallback to existing index if no candidate column is present.
    if df.index.nlevels == 1:
        idx = df.index
        if pd.api.types.is_datetime64_any_dtype(idx):
            return pd.Series(idx, index=df.index)
    # Final fallback: monotonically increasing integers.
    return pd.Series(range(len(df)), index=df.index).astype("int64")


def _extract_timeseries(result):
    """Return (data_id, domain, df_with_timeindex) or None for non-timeseries."""
    if isinstance(result, RHEEDVideoResult):
        domain = "rheed"
        df = result.timeseries_data
    elif isinstance(result, OpticalResult):
        domain = "optical"
        df = result.timeseries_data
    elif isinstance(result, MetrologyResult):
        domain = "metrology"
        df = result.timeseries_data
    else:
        return None

    if df is None or df.empty:
        return None

    flat_df = df.copy().reset_index()
    flat_df["__time__"] = _infer_time_column(flat_df)
    flat_df = flat_df.set_index("__time__")
    flat_df.index.name = "time"
    flat_df = flat_df.sort_index()

    return str(result.data_id), domain, flat_df


def align_timeseries(
    results: Iterable,
    *,
    how: str = "outer",
    resample: str | None = None,
) -> pd.DataFrame | None:
    """Align timeseries results by time index.

    Args:
        results: Iterable of result objects (RHEEDVideoResult, OpticalResult, MetrologyResult).
        how: Join strategy for the outer alignment. Defaults to "outer".
        resample: Optional pandas resample rule (e.g., "1s").

    Returns:
        DataFrame | None: Aligned DataFrame with MultiIndex columns (data_id, domain, metric).
    """
    frames = []
    for item in results:
        extracted = _extract_timeseries(item)
        if not extracted:
            continue
        data_id, domain, df = extracted
        df = df.copy()
        df.columns = pd.MultiIndex.from_product([[data_id], [domain], df.columns])
        frames.append(df)

    if not frames:
        return None

    aligned = reduce(lambda a, b: a.join(b, how=how), frames)
    if resample:
        aligned = aligned.resample(resample).mean()
    return aligned
