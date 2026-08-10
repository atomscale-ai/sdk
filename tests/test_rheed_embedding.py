import os

import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.similarity.embedding_provider import RHEEDEmbeddingProvider

from .conftest import ResultIDs

# --------------------------- pure unit tests (no network) ---------------------------


def test_type_constant():
    assert RHEEDEmbeddingProvider.TYPE == "rheed_embeddings"


def test_neighbors_to_dataframe():
    raw = {
        "neighbors": [
            {"data_id": "x", "similarity": 0.9, "source_index": 0, "neighbor_index": 3,
             "real_time_seconds": None, "unix_time_ms": None},
            {"data_id": "y", "similarity": 0.7, "source_index": 1, "neighbor_index": 0,
             "real_time_seconds": None, "unix_time_ms": None},
        ]
    }
    df = RHEEDEmbeddingProvider().neighbors_to_dataframe(raw)
    assert list(df.columns)[:2] == ["data_id", "similarity"]
    assert df.iloc[0]["data_id"] == "x"
    assert df.iloc[0]["similarity"] == 0.9


def test_neighbors_to_dataframe_empty():
    df = RHEEDEmbeddingProvider().neighbors_to_dataframe({"neighbors": []})
    assert isinstance(df, DataFrame)
    assert list(df.columns) == [
        "data_id", "similarity", "source_index", "neighbor_index",
        "real_time_seconds", "unix_time_ms",
    ]
    assert df.empty


# --------------------------- live-API tests (gated on creds) ---------------------------


def _skip_without_api():
    if not os.getenv("AS_API_KEY") and not os.getenv("ATOMSCALE_API_KEY"):
        pytest.skip("No API key configured for live embedding tests")
    if not ResultIDs.similarity_source_id or not ResultIDs.similarity_workflow:
        pytest.skip("No similarity source configured")


def test_query_rheed_embeddings_live():
    _skip_without_api()
    df = Client().query_rheed_embeddings(
        ResultIDs.similarity_source_id,
        workflow=ResultIDs.similarity_workflow,
        window_span=60.0,
        kind="prototype",
        top_k=5,
    )
    assert isinstance(df, DataFrame)
    # the source item must never be returned as its own neighbor
    if not df.empty:
        assert str(ResultIDs.similarity_source_id) not in set(df["data_id"].astype(str))
