import pytest
from atomicds import Client
from atomicds.results import PhotoluminescenceResult
from matplotlib.figure import Figure

from .conftest import ResultIDs


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def result(client: Client):
    if not ResultIDs.photoluminescence:
        pytest.skip("No photoluminescence data available")

    results = client.get(data_ids=ResultIDs.photoluminescence)
    return results[0]


def test_get_plot(result: PhotoluminescenceResult):
    plot = result.get_plot()
    assert isinstance(plot, Figure)


def test_data_structure(result: PhotoluminescenceResult):
    assert isinstance(result.energies, list)
    assert isinstance(result.intensities, list)
    assert len(result.energies) == len(result.intensities)
    assert isinstance(result.detected_peaks, dict)
