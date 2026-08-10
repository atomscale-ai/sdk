import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.results import RHEEDVideoResult
from atomscale.timeseries.rheed import RHEEDProvider

from .conftest import ResultIDs


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def result(client: Client):
    results = client.get(data_ids=ResultIDs.rheed_rotating)
    return results[0]


# def test_get_plot(result: RHEEDVideoResult):
#     plot = result.get_plot()
#     assert isinstance(plot, Figure)


def test_get_dataframe(result: RHEEDVideoResult):
    # Core columns that should always be present
    column_names = set(
        [
            "Strain",
            "Cumulative Strain",
            "Lattice Spacing",
            "Diffraction Spot Count",
            "Oscillation Period",
            "Specular Intensity",
            "First Order Intensity",
            "First Order Intensity L",
            "First Order Intensity R",
            "Half Order Intensity",
            "Half Order Intensity L",
            "Half Order Intensity R",
            "Specular FWHM",
            "First Order FWHM",
            "UNIX Timestamp",
            "Relative Time",
            # Optional columns (included if data exists)
            "Time",
            "TAR Metric",
            "Composition Metric",
        ]
    )

    assert isinstance(result.timeseries_data, DataFrame)
    assert not len(set(result.timeseries_data.keys().values) - column_names)
    assert result.timeseries_data.index.names == ["Angle", "Frame Number"]


def test_to_dataframe_flattens_low_level_features():
    """When include_low_level_features is set, the backend nests a
    low_level_features dict per point; the provider flattens it into raw-named
    columns (no RENAME_MAP entry) and drops the nested column."""
    raw = {
        "series_by_angle": [
            {
                "angle": "0",
                "series": [
                    {
                        "frame_number": 0,
                        "relative_time_seconds": 0.0,
                        "unix_timestamp_ms": 0.0,
                        "specular_intensity": 5.0,
                        "low_level_features": {"area_0": 1.2, "eccentricity_0": 0.3},
                    },
                    {
                        "frame_number": 1,
                        "relative_time_seconds": 0.1,
                        "unix_timestamp_ms": 100.0,
                        "specular_intensity": 6.0,
                        "low_level_features": {"area_0": 1.5, "eccentricity_0": 0.4},
                    },
                ],
            }
        ]
    }
    df = RHEEDProvider().to_dataframe(raw)
    assert "area_0" in df.columns
    assert "eccentricity_0" in df.columns
    assert "low_level_features" not in df.columns
    assert "Specular Intensity" in df.columns  # RENAME_MAP still applied
    assert df["area_0"].tolist() == [1.2, 1.5]


def test_to_dataframe_without_low_level_features():
    """No low_level_features key (flag off) -> no extra columns."""
    raw = {
        "series_by_angle": [
            {
                "angle": "0",
                "series": [
                    {
                        "frame_number": 0,
                        "relative_time_seconds": 0.0,
                        "unix_timestamp_ms": 0.0,
                        "specular_intensity": 5.0,
                    }
                ],
            }
        ]
    }
    df = RHEEDProvider().to_dataframe(raw)
    assert "area_0" not in df.columns
    assert "low_level_features" not in df.columns
