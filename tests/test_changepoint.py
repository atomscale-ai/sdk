import pytest
from atomscale import Client
from atomscale.results import ChangepointResult
from pandas import DataFrame

from .conftest import ResultIDs


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def result(client: Client):
    if not ResultIDs.changepoint:
        pytest.skip("No changepoint data available")

    return client.get_changepoints(data_ids=ResultIDs.changepoint)


def test_data_structure(result: DataFrame):
    assert isinstance(result, DataFrame)
    expected_cols = {
        "id",
        "data_id",
        "data_modality",
        "property_name",
        "severity",
        "score",
        "window_start_elapsed",
        "window_end_elapsed",
        "detection_method",
    }
    if result.empty:
        return
    assert expected_cols.issubset(set(result.columns))
    assert (result["detection_method"] == "intensity_profile").all()
    assert (result["severity"] == "critical").all()


def test_as_objects(client: Client):
    if not ResultIDs.changepoint:
        pytest.skip("No changepoint data available")

    results = client.get_changepoints(
        data_ids=ResultIDs.changepoint, as_dataframe=False
    )
    assert isinstance(results, list)
    for cp in results:
        assert isinstance(cp, ChangepointResult)
