import pytest
from atomscale import Client
from atomscale.results import OpticalResult
from atomscale.timeseries.optical import OpticalProvider
from pandas import DataFrame
from PIL.Image import Image

from .conftest import ResultIDs


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def result(client: Client):
    if not ResultIDs.optical:
        pytest.skip("No optical data available")

    results = client.get(data_ids=ResultIDs.optical)
    return results[0]


def test_get_dataframe(result: OpticalResult):
    column_names = set(OpticalProvider.RENAME_MAP.values())
    df = result.timeseries_data

    assert isinstance(df, DataFrame)
    assert not set(df.keys().values) - column_names
    if df.index.names != [None]:
        assert df.index.names == ["Frame Number"]


def test_snapshot_images(result: OpticalResult):
    snapshots = result.snapshot_image_data
    if not snapshots:
        pytest.skip("No optical snapshot images available")

    assert isinstance(snapshots[0].processed_image, Image)
