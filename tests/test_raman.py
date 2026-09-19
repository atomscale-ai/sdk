import pytest
from matplotlib.figure import Figure

from atomscale import Client
from atomscale.results import RamanResult

from .conftest import ResultIDs


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def result(client: Client):
    if not ResultIDs.raman:
        pytest.skip("No Raman data available")

    results = client.get(data_ids=ResultIDs.raman)
    return results[0]


def test_get_plot(result: RamanResult):
    plot = result.get_plot()
    assert isinstance(plot, Figure)


def test_data_structure(result: RamanResult):
    assert isinstance(result.raman_shift, list)
    assert isinstance(result.intensities, list)
    assert len(result.raman_shift) == len(result.intensities)
    # The API stores detected peaks as a list of per-peak dicts.
    assert isinstance(result.detected_peaks, list)
    assert all(isinstance(peak, dict) for peak in result.detected_peaks)
