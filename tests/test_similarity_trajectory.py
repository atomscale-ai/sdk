import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.results import SimilarityTrajectoryResult
from atomscale.similarity import SimilarityTrajectoryProvider

from .conftest import ResultIDs


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def provider():
    return SimilarityTrajectoryProvider()


@pytest.fixture
def raw_data(client: Client, provider: SimilarityTrajectoryProvider):
    if not ResultIDs.similarity_workflow or not ResultIDs.similarity_source_id:
        pytest.skip("No similarity trajectory data available")

    data = provider.fetch_raw(
        client,
        ResultIDs.similarity_source_id,
        workflow=ResultIDs.similarity_workflow,
    )

    if not data or not data.get("trajectories"):
        pytest.skip("No trajectory data returned from API")

    return data


@pytest.fixture
def result(
    client: Client, provider: SimilarityTrajectoryProvider, raw_data: dict
) -> SimilarityTrajectoryResult:
    df = provider.to_dataframe(raw_data)
    return provider.build_result(
        client=client,
        data_id=ResultIDs.similarity_source_id,
        data_type="similarity_trajectory",
        ts_df=df,
        workflow=ResultIDs.similarity_workflow,
    )


def test_type_constant():
    """Verify TYPE constant is set correctly."""
    assert SimilarityTrajectoryProvider.TYPE == "similarity_trajectory"


def test_fetch_raw(raw_data: dict):
    """Verify raw data is fetched from API."""
    assert raw_data is not None
    assert "trajectories" in raw_data


def test_to_dataframe(provider: SimilarityTrajectoryProvider, raw_data: dict):
    """Verify dataframe conversion."""
    df = provider.to_dataframe(raw_data)

    assert isinstance(df, DataFrame)
    assert not df.empty


def test_to_dataframe_columns(provider: SimilarityTrajectoryProvider, raw_data: dict):
    """Verify column names and index."""
    df = provider.to_dataframe(raw_data)

    # Check index names
    assert df.index.names == ["Reference ID", "Time"]

    # Check expected columns exist
    expected_columns = {"Similarity", "Reference Name", "UNIX Timestamp", "Active", "Averaged Count"}
    assert expected_columns == set(df.columns)


def test_build_result(result: SimilarityTrajectoryResult):
    """Verify result object construction."""
    assert isinstance(result, SimilarityTrajectoryResult)
    assert result.source_id == ResultIDs.similarity_source_id
    assert result.workflow == ResultIDs.similarity_workflow
    assert isinstance(result.timeseries_data, DataFrame)


def test_result_dataframe(result: SimilarityTrajectoryResult):
    """Verify result contains valid timeseries data."""
    df = result.timeseries_data

    assert isinstance(df, DataFrame)
    if df.index.names != [None]:
        assert df.index.names == ["Reference ID", "Time"]
