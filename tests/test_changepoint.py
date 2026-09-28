import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.results import ChangepointResult


@pytest.fixture
def result(client: Client, result_ids):
    return client.get_changepoints(data_ids=result_ids.changepoint)


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
    assert not result.empty
    assert expected_cols.issubset(set(result.columns))
    assert (result["detection_method"] == "intensity_profile").all()
    assert (result["severity"] == "critical").all()


def test_as_objects(client: Client, result_ids):
    results = client.get_changepoints(
        data_ids=result_ids.changepoint, as_dataframe=False
    )
    assert isinstance(results, list)
    assert len(results) == 1
    for cp in results:
        assert isinstance(cp, ChangepointResult)
