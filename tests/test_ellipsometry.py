import pytest
from pandas import DataFrame

from atomscale.timeseries.ellipsometry import EllipsometryProvider

PROP_PAYLOAD = {
    "properties": {
        "psi": {
            "relative_time_seconds": [0.0, 1.0, 2.0],
            "unix_timestamp_ms": [
                1_700_000_000_000,
                1_700_000_001_000,
                1_700_000_002_000,
            ],
            "values": [20.1, 20.3, 20.5],
            "units": "deg",
        },
        "delta": {
            "relative_time_seconds": [0.5, 1.5],
            "unix_timestamp_ms": [
                1_700_000_000_500,
                1_700_000_001_500,
            ],
            "values": [170.0, 169.5],
            "units": "deg",
        },
    },
    "series_max_time": 2.0,
}


def test_property_centric_parse():
    df = EllipsometryProvider().to_dataframe(PROP_PAYLOAD)
    assert isinstance(df, DataFrame)
    assert "psi" in df.columns
    assert "delta" in df.columns
    assert {"UNIX Timestamp", "Time"} <= set(df.columns)
    assert df["UNIX Timestamp"].is_monotonic_increasing
    assert df["Time"].is_monotonic_increasing


def test_legacy_series_payload_rejected():
    with pytest.raises(ValueError, match="properties"):
        EllipsometryProvider().to_dataframe(
            {"series": [{"unix_timestamp_ms": 1, "psi": 1.0}]}
        )


def test_empty_payload_returns_empty_df():
    assert EllipsometryProvider().to_dataframe({}).empty
    assert EllipsometryProvider().to_dataframe({"properties": {}}).empty
    assert EllipsometryProvider().to_dataframe(None).empty
