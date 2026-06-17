from decimal import Decimal

import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.results import MetrologyResult
from atomscale.timeseries.metrology import MetrologyProvider

from .conftest import ResultIDs

PROP_PAYLOAD = {
    "properties": {
        "Sub T setpoint": {
            "relative_time_seconds": [0.0, 1.0, 2.0, 3.0, 4.0],
            "unix_timestamp_ms": [
                1_700_000_000_000,
                1_700_000_001_000,
                1_700_000_002_000,
                1_700_000_003_000,
                1_700_000_004_000,
            ],
            "values": [400.0, 400.0, 450.0, 450.0, 500.0],
            "units": "C",
        },
        "Ratio Pyrometer": {
            "relative_time_seconds": [0.5, 1.5, 2.5, 3.5],
            "unix_timestamp_ms": [
                1_700_000_000_500,
                1_700_000_001_500,
                1_700_000_002_500,
                1_700_000_003_500,
            ],
            "values": [395.2, 425.7, 451.3, 480.1],
            "units": "C",
        },
    },
    "series_max_time": 4.0,
}


@pytest.fixture
def client():
    return Client()


def test_property_centric_parse():
    df = MetrologyProvider().to_dataframe(PROP_PAYLOAD)
    assert isinstance(df, DataFrame)
    # Property columns preserve their API names (case-sensitive).
    assert "Sub T setpoint" in df.columns
    assert "Ratio Pyrometer" in df.columns
    # Time columns are present and named.
    assert {"UNIX Timestamp", "Time"} <= set(df.columns)
    # Time columns are monotonic.
    assert df["UNIX Timestamp"].is_monotonic_increasing
    assert df["Time"].is_monotonic_increasing
    # Index is row number (RangeIndex), not the timestamp.
    assert df.index.tolist() == list(range(len(df)))
    # Forward-fill: at t=1_700_000_000_500 (first pyrometer sample),
    # the setpoint is forward-filled from the t=0 sample (400.0).
    row = df.loc[df["UNIX Timestamp"] == 1_700_000_000_500].iloc[0]
    assert row["Sub T setpoint"] == 400.0
    assert row["Ratio Pyrometer"] == pytest.approx(395.2)


def test_property_centric_columns_are_int64_ms():
    df = MetrologyProvider().to_dataframe(PROP_PAYLOAD)
    assert str(df["UNIX Timestamp"].dtype) == "int64"


def test_decimal_unix_ms_input_preserved_to_int64():
    payload = {
        "properties": {
            "P": {
                "relative_time_seconds": [0.0],
                "unix_timestamp_ms": [Decimal("1700000000123")],
                "values": [1.0],
                "units": None,
            }
        },
        "series_max_time": 0.0,
    }
    df = MetrologyProvider().to_dataframe(payload)
    assert str(df["UNIX Timestamp"].dtype) == "int64"
    assert int(df["UNIX Timestamp"].iloc[0]) == 1_700_000_000_123


def test_legacy_series_payload_parses():
    df = MetrologyProvider().to_dataframe(
        {
            "series": [
                {
                    "unix_timestamp_ms": 1_700_000_000_000,
                    "relative_time_seconds": 0.0,
                    "ratio_pyrometer": 1.0,
                },
                {
                    "unix_timestamp_ms": 1_700_000_000_500,
                    "relative_time_seconds": 0.5,
                    "ratio_pyrometer": 1.5,
                },
            ]
        }
    )
    assert {"UNIX Timestamp", "Time", "ratio_pyrometer"} <= set(df.columns)
    assert str(df["UNIX Timestamp"].dtype) == "int64"


def test_duplicate_timestamps_within_property_do_not_crash():
    # A property emitting two samples at the same unix-ms used to crash the
    # outer-join with "cannot reindex on an axis with duplicate labels"
    # whenever a second property was present (regression test).
    payload = {
        "properties": {
            "Sub T setpoint": {
                "relative_time_seconds": [0.0, 0.0, 1.0],
                "unix_timestamp_ms": [
                    1_700_000_000_000,
                    1_700_000_000_000,  # duplicate instant
                    1_700_000_001_000,
                ],
                "values": [400.0, 410.0, 450.0],
                "units": "C",
            },
            "Ratio Pyrometer": {
                "relative_time_seconds": [0.0, 1.0],
                "unix_timestamp_ms": [
                    1_700_000_000_000,
                    1_700_000_001_000,
                ],
                "values": [395.2, 451.3],
                "units": "C",
            },
        }
    }
    df = MetrologyProvider().to_dataframe(payload)
    # Duplicate instant collapsed to a single row; index stays unique.
    assert df["UNIX Timestamp"].is_unique
    assert df["UNIX Timestamp"].is_monotonic_increasing
    # "keep last" wins at the duplicated instant.
    row = df.loc[df["UNIX Timestamp"] == 1_700_000_000_000].iloc[0]
    assert row["Sub T setpoint"] == 410.0


def test_empty_payload_returns_empty_df():
    assert MetrologyProvider().to_dataframe({}).empty
    assert MetrologyProvider().to_dataframe({"properties": {}}).empty
    assert MetrologyProvider().to_dataframe(None).empty


# -----------------------------------------------------------------------------
# Live data path (skipped unless ResultIDs.metrology is populated)
# -----------------------------------------------------------------------------


@pytest.fixture
def result(client: Client):
    if not ResultIDs.metrology:
        pytest.skip("No metrology data available")

    results = client.get(data_ids=ResultIDs.metrology)
    return results[0]


def test_live_get_dataframe(result: MetrologyResult):
    df = result.timeseries_data
    assert isinstance(df, DataFrame)
    if not df.empty:
        assert "UNIX Timestamp" in df.columns
        assert "Time" in df.columns
