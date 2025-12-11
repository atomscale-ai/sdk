import pytest
from atomicds import Client
from atomicds.results import MetrologyResult
from atomicds.timeseries.metrology import MetrologyProvider
from pandas import DataFrame

from .conftest import ResultIDs


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def result(client: Client):
    if not ResultIDs.metrology:
        pytest.skip("No metrology data available")

    results = client.get(data_ids=ResultIDs.metrology)
    return results[0]


def test_get_dataframe(result: MetrologyResult):
    column_names = set(MetrologyProvider.RENAME_MAP.values())
    df = result.timeseries_data

    assert isinstance(df, DataFrame)
    # assert not set(df.keys().values) - column_names
    if df.index.names != [None]:
        assert df.index.names == ["Frame Number"]
