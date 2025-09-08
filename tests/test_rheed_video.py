import pytest
from atomicds import Client
from .conftest import ResultIDs
from atomicds.results import RHEEDVideoResult
from pandas import DataFrame


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
            "Time",
        ]
    )

    assert isinstance(result.timeseries_data, DataFrame)
    assert not len(set(result.timeseries_data.keys().values) - column_names)
    assert result.timeseries_data.index.names == ["Angle", "Frame Number"]
