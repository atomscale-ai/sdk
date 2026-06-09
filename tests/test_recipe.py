from decimal import Decimal

import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.results import RecipeResult
from atomscale.timeseries.recipe import RecipeProvider

from .conftest import ResultIDs

# Recipe payloads share tool-state's property-centric shape (setpoints expanded
# to a time series), so parsing is exercised identically.
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
        "Pressure setpoint": {
            "relative_time_seconds": [0.5, 1.5, 2.5, 3.5],
            "unix_timestamp_ms": [
                1_700_000_000_500,
                1_700_000_001_500,
                1_700_000_002_500,
                1_700_000_003_500,
            ],
            "values": [1.0e-6, 2.0e-6, 1.5e-6, 1.0e-6],
            "units": "Torr",
        },
    },
    "series_max_time": 4.0,
}


@pytest.fixture
def client():
    return Client()


def test_property_centric_parse():
    df = RecipeProvider().to_dataframe(PROP_PAYLOAD)
    assert isinstance(df, DataFrame)
    # Property columns preserve their API names (case-sensitive).
    assert "Sub T setpoint" in df.columns
    assert "Pressure setpoint" in df.columns
    # Time columns are present and named.
    assert {"UNIX Timestamp", "Time"} <= set(df.columns)
    # Time columns are monotonic.
    assert df["UNIX Timestamp"].is_monotonic_increasing
    assert df["Time"].is_monotonic_increasing
    # Index is row number (RangeIndex), not the timestamp.
    assert df.index.tolist() == list(range(len(df)))
    # Forward-fill: at t=1_700_000_000_500 (first pressure sample), the setpoint
    # is forward-filled from the t=0 sample (400.0).
    row = df.loc[df["UNIX Timestamp"] == 1_700_000_000_500].iloc[0]
    assert row["Sub T setpoint"] == 400.0
    assert row["Pressure setpoint"] == pytest.approx(1.0e-6)


def test_property_centric_columns_are_int64_ms():
    df = RecipeProvider().to_dataframe(PROP_PAYLOAD)
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
    df = RecipeProvider().to_dataframe(payload)
    assert str(df["UNIX Timestamp"].dtype) == "int64"
    assert int(df["UNIX Timestamp"].iloc[0]) == 1_700_000_000_123


def test_legacy_series_payload_parses():
    df = RecipeProvider().to_dataframe(
        {
            "series": [
                {
                    "unix_timestamp_ms": 1_700_000_000_000,
                    "relative_time_seconds": 0.0,
                    "sub_t_setpoint": 400.0,
                },
                {
                    "unix_timestamp_ms": 1_700_000_000_500,
                    "relative_time_seconds": 0.5,
                    "sub_t_setpoint": 450.0,
                },
            ]
        }
    )
    assert {"UNIX Timestamp", "Time", "sub_t_setpoint"} <= set(df.columns)
    assert str(df["UNIX Timestamp"].dtype) == "int64"


def test_empty_payload_returns_empty_df():
    assert RecipeProvider().to_dataframe({}).empty
    assert RecipeProvider().to_dataframe({"properties": {}}).empty
    assert RecipeProvider().to_dataframe(None).empty


def test_recipe_provider_fetches_recipe_endpoint():
    # Recipe must hit its own /recipe/ endpoint, not tool-state — the two are
    # distinct concerns (plan vs observed readback).
    captured: dict[str, str] = {}

    class _FakeClient:
        def _get(self, sub_url: str):
            captured["sub_url"] = sub_url
            return {}

    RecipeProvider().fetch_raw(_FakeClient(), "abc-123")
    assert captured["sub_url"] == "recipe/abc-123/timeseries/"


# -----------------------------------------------------------------------------
# Live data path (skipped unless ResultIDs.recipe is populated)
# -----------------------------------------------------------------------------


@pytest.fixture
def result(client: Client):
    if not ResultIDs.recipe:
        pytest.skip("No recipe data available")

    results = client.get(data_ids=ResultIDs.recipe)
    return results[0]


def test_live_get_dataframe(result: RecipeResult):
    df = result.timeseries_data
    assert isinstance(df, DataFrame)
    if not df.empty:
        assert "UNIX Timestamp" in df.columns
        assert "Time" in df.columns
