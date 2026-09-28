import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.results import SimilarityTrajectoryResult
from atomscale.similarity import SimilarityTrajectoryProvider


@pytest.fixture
def provider():
    return SimilarityTrajectoryProvider()


@pytest.fixture
def raw_data(client: Client, provider: SimilarityTrajectoryProvider, result_ids):
    data = provider.fetch_raw(
        client,
        result_ids.similarity_source_id,
        workflow=result_ids.similarity_workflow,
    )

    assert data and data["trajectories"]

    return data


@pytest.fixture
def result(
    client: Client, provider: SimilarityTrajectoryProvider, raw_data: dict, result_ids
) -> SimilarityTrajectoryResult:
    df = provider.to_dataframe(raw_data)
    return provider.build_result(
        client=client,
        data_id=result_ids.similarity_source_id,
        data_type="similarity_trajectory",
        ts_df=df,
        workflow=result_ids.similarity_workflow,
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
    expected_columns = {
        "Similarity",
        "Reference Name",
        "UNIX Timestamp",
        "Active",
        "Averaged Count",
    }
    assert expected_columns == set(df.columns)


def test_build_result(result: SimilarityTrajectoryResult, result_ids):
    """Verify result object construction."""
    assert isinstance(result, SimilarityTrajectoryResult)
    assert result.source_id == result_ids.similarity_source_id
    assert result.workflow == result_ids.similarity_workflow
    assert isinstance(result.timeseries_data, DataFrame)


def test_result_dataframe(result: SimilarityTrajectoryResult):
    """Verify result contains valid timeseries data."""
    df = result.timeseries_data

    assert isinstance(df, DataFrame)
    if df.index.names != [None]:
        assert df.index.names == ["Reference ID", "Time"]
