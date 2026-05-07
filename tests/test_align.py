import pandas as pd
import pytest

from atomscale.results import MetrologyResult, OpticalResult
from atomscale.timeseries.align import _infer_absolute_time, align_timeseries
from atomscale.timeseries.metrology import MetrologyProvider
from atomscale.timeseries.optical import OpticalProvider


def test_unit_name_takes_priority_over_magnitude():
    # `timestamp_seconds` declares unit 's', but the values look like ms (~1.7e12).
    # Magnitude alone would silently parse as ms and produce a 2023 datetime;
    # name-priority wins and raises instead.
    df = pd.DataFrame({"timestamp_seconds": [1_700_000_000_000, 1_700_000_001_000]})
    with pytest.raises(ValueError, match="declares unit 's'"):
        _infer_absolute_time(df)


def test_unit_name_ms_parses_correctly():
    df = pd.DataFrame(
        {"unix_timestamp_ms": [1_700_000_000_000, 1_700_000_001_000]}
    )
    out = _infer_absolute_time(df)
    assert out is not None
    assert out.iloc[0].year == 2023


def test_unit_name_seconds_parses_correctly():
    df = pd.DataFrame(
        {"unix_timestamp_seconds": [1_700_000_000.0, 1_700_000_001.0]}
    )
    out = _infer_absolute_time(df)
    assert out is not None
    assert out.iloc[0].year == 2023


def test_unitless_name_falls_through_to_magnitude_ms():
    df = pd.DataFrame(
        {"UNIX Timestamp": [1_700_000_000_000, 1_700_000_001_000]}
    )
    out = _infer_absolute_time(df)
    assert out is not None
    assert out.iloc[0].year == 2023


def test_unitless_name_falls_through_to_magnitude_seconds():
    df = pd.DataFrame({"timestamp": [1_700_000_000.0, 1_700_000_001.0]})
    out = _infer_absolute_time(df)
    assert out is not None
    assert out.iloc[0].year == 2023


def test_no_time_column_returns_none():
    df = pd.DataFrame({"some_value": [1.0, 2.0]})
    assert _infer_absolute_time(df) is None


def _metrology_payload():
    return {
        "properties": {
            "Sub T setpoint": {
                "relative_time_seconds": [0.0, 1.0, 2.0, 3.0],
                "unix_timestamp_ms": [
                    1_700_000_000_000,
                    1_700_000_001_000,
                    1_700_000_002_000,
                    1_700_000_003_000,
                ],
                "values": [400.0, 400.0, 450.0, 450.0],
                "units": "C",
            }
        },
        "series_max_time": 3.0,
    }


def _optical_payload():
    return {
        "properties": {
            "perimeter_px": {
                "relative_time_seconds": [0.0, 1.0, 2.0, 3.0],
                "unix_timestamp_ms": [
                    1_700_000_000_500,
                    1_700_000_001_500,
                    1_700_000_002_500,
                    1_700_000_003_500,
                ],
                "values": [10.0, 10.5, 11.0, 11.4],
                "units": "px",
            }
        },
        "series_max_time": 3.0,
    }


def test_align_outer_produces_metrology_and_optical_columns():
    """End-to-end: property-centric DataFrames flow through align_timeseries
    and the aligned MultiIndex carries (data_id, domain, metric) entries
    for both metrology and optical."""
    metro_df = MetrologyProvider().to_dataframe(_metrology_payload())
    optical_df = OpticalProvider().to_dataframe(_optical_payload())

    metro_result = MetrologyResult(data_id="metro-1", timeseries_data=metro_df)
    optical_result = OpticalResult(
        data_id="optical-1", timeseries_data=optical_df, snapshot_image_data=None
    )

    aligned = align_timeseries([metro_result, optical_result], how="outer")

    assert aligned is not None
    assert not aligned.empty
    assert isinstance(aligned.columns, pd.MultiIndex)

    domains = {col[1] for col in aligned.columns}
    assert "metrology" in domains
    assert "optical" in domains

    metro_metrics = {col[2] for col in aligned.columns if col[1] == "metrology"}
    optical_metrics = {col[2] for col in aligned.columns if col[1] == "optical"}
    assert "Sub T setpoint" in metro_metrics
    assert "perimeter_px" in optical_metrics
