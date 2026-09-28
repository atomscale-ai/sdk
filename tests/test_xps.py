import pytest
from matplotlib.figure import Figure

from atomscale import Client
from atomscale.results import XPSResult


@pytest.fixture
def result(client: Client, result_ids):
    results = client.get(data_ids=result_ids.xps)
    assert len(results) == 1
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
