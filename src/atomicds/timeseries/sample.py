from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from pandas import DataFrame

_TIME_COL_CANDIDATES: tuple[str, ...] = (
    "UNIX Timestamp",
    "Unix Timestamp",
    "unix_timestamp_ms",
    "unix_timestamp",
    "timestamp",
    "Time",
    "time_seconds",
)


def _extract_time_index(ts_df: DataFrame) -> tuple[pd.Index, DataFrame] | None:
    """Return a time-indexed copy of the frame.

    Args:
        ts_df (DataFrame): Timeseries data frame containing a time-like column.

    Returns:
        tuple[pd.Index, DataFrame] | None: Time index and value frame without the
        time column, or ``None`` if a usable time column is missing.
    """

    if ts_df is None or ts_df.empty:
        return None

    reset_df = ts_df.reset_index(drop=False)
    time_col = next((col for col in _TIME_COL_CANDIDATES if col in reset_df), None)

    if time_col is None:
        return None

    time_series = reset_df.pop(time_col)

    if "timestamp" in time_col.lower():
        if pd.api.types.is_datetime64_any_dtype(time_series):
            time_index: pd.Index = pd.Index(pd.to_datetime(time_series))
        else:
            time_index = pd.Index(
                pd.to_datetime(time_series, unit="ms", errors="coerce")
            )
    else:
        time_index = pd.Index(pd.to_timedelta(time_series, unit="s"))

    # Drop obvious bookkeeping columns that are not signals
    drop_cols = {"Frame Number"}
    reset_df = reset_df.drop(columns=[c for c in drop_cols if c in reset_df])

    reset_df.index = time_index
    return time_index, reset_df


def align_timeseries_frames(
    frames: Mapping[str, DataFrame],
    *,
    freq: str = "1s",
    interpolation_window: str | None = "15s",
) -> DataFrame:
    """Combine and align multiple timeseries frames on a common time grid.

    Each input frame must contain a time-like column (for example ``UNIX Timestamp``
    or ``Time`` in seconds). Columns are prefixed with their key to avoid name
    collisions before concatenation. The result is optionally resampled to the
    requested frequency and linearly interpolated across short gaps.

    Args:
        frames (Mapping[str, DataFrame]): Mapping from a user-supplied label to a
            timeseries DataFrame.
        freq (str, optional): Resampling frequency understood by pandas. Defaults
            to ``"1s"``.
        interpolation_window (str | None, optional): Maximum gap to fill using
            linear interpolation. Provided as a pandas-parsable duration. Set to
            ``None`` to disable interpolation. Defaults to ``"15s"``.

    Returns:
        DataFrame: A wide, time-indexed DataFrame containing aligned columns. The
        index type is datetime-like when UNIX timestamps are present, otherwise a
        timedelta index.
    """

    prepared: list[DataFrame] = []

    for label, frame in frames.items():
        extracted = _extract_time_index(frame)
        if extracted is None:
            continue

        _, value_frame = extracted
        value_frame = value_frame.add_prefix(f"{label} | ")
        prepared.append(value_frame)

    if not prepared:
        return DataFrame()

    aligned = pd.concat(prepared, axis=1).sort_index()

    if freq:
        aligned = aligned.resample(freq).mean()

        if interpolation_window:
            freq_td = pd.Timedelta(freq)
            window_td = pd.Timedelta(interpolation_window)
            limit: int | None = (
                int(window_td / freq_td) if freq_td and window_td else None
            )
            aligned = aligned.interpolate(method="time", limit=limit)

    return aligned
