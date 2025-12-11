import pytest
from atomicds import Client
from atomicds.results import XPSResult
from matplotlib.figure import Figure

from .conftest import ResultIDs


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def result(client: Client):
    if not ResultIDs.xps:
        pytest.skip("No XPS data available")

    results = client.get(data_ids=ResultIDs.xps)
    return results[0]


def test_get_plot(result: XPSResult):
    plot = result.get_plot()
    assert isinstance(plot, Figure)


def test_data_structure(result: XPSResult):
    assert isinstance(result.binding_energies, list)
    assert isinstance(result.intensities, list)
    assert len(result.binding_energies) == len(result.intensities)
    assert isinstance(result.predicted_composition, dict)
    assert isinstance(result.detected_peaks, list)
    assert isinstance(result.elements_manually_set, bool)
