import pytest
from pandas import DataFrame
from PIL.Image import Image

from atomscale import Client
from atomscale.results import OpticalResult
from atomscale.timeseries.optical import OpticalProvider

from .conftest import ResultIDs

PROP_PAYLOAD = {
    "properties": {
        "perimeter_px": {
            "relative_time_seconds": [0.0, 1.0, 2.0, 3.0],
            "unix_timestamp_ms": [
                1_700_000_000_000,
                1_700_000_001_000,
                1_700_000_002_000,
                1_700_000_003_000,
            ],
            "values": [10.5, 11.0, 11.2, 11.4],
            "units": "px",
        },
        "circularity": {
            "relative_time_seconds": [0.5, 2.5],
            "unix_timestamp_ms": [
                1_700_000_000_500,
                1_700_000_002_500,
            ],
            "values": [0.92, 0.93],
            "units": None,
        },
    },
    "series_max_time": 3.0,
}


@pytest.fixture
def client():
    return Client()


def test_property_centric_parse():
    df = OpticalProvider().to_dataframe(PROP_PAYLOAD)
    assert isinstance(df, DataFrame)
    assert "perimeter_px" in df.columns
    assert "circularity" in df.columns
    assert {"UNIX Timestamp", "Time"} <= set(df.columns)
    assert df["UNIX Timestamp"].is_monotonic_increasing
    assert df["Time"].is_monotonic_increasing


def test_legacy_series_payload_rejected():
    with pytest.raises(ValueError, match="properties"):
        OpticalProvider().to_dataframe(
            {"series": [{"unix_timestamp_ms": 1, "perimeter_px": 1.0}]}
        )


def test_empty_payload_returns_empty_df():
    assert OpticalProvider().to_dataframe({}).empty
    assert OpticalProvider().to_dataframe({"properties": {}}).empty
    assert OpticalProvider().to_dataframe(None).empty


# -----------------------------------------------------------------------------
# Live data path (skipped unless ResultIDs.optical is populated)
# -----------------------------------------------------------------------------


@pytest.fixture
def result(client: Client):
    if not ResultIDs.optical:
        pytest.skip("No optical data available")

    results = client.get(data_ids=ResultIDs.optical)
    return results[0]


def test_live_get_dataframe(result: OpticalResult):
    df = result.timeseries_data
    assert isinstance(df, DataFrame)
    if not df.empty:
        assert "UNIX Timestamp" in df.columns
        assert "Time" in df.columns


def test_snapshot_images(result: OpticalResult):
    snapshots = result.snapshot_image_data
    if not snapshots:
        pytest.skip("No optical snapshot images available")

    assert isinstance(snapshots[0].processed_image, Image)
