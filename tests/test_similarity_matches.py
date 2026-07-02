"""Unit tests for Client.get_similarity_matches."""

import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.client import _DEFAULT_SIMILARITY_METRIC


@pytest.fixture
def client():
    return Client(api_key="key_test", endpoint="http://example.com/")


def test_matches_sends_camelcase_chamfer_params(client, monkeypatch):
    """The simplified method fixes metric/usePrototypes and sends camelCase params."""
    captured: dict = {}

    def fake_get(sub_url, params=None, **kwargs):
        captured["sub_url"] = sub_url
        captured["params"] = params
        return []

    monkeypatch.setattr(client, "_get", fake_get)

    client.get_similarity_matches(
        "source-1",
        workflow="rheed_stationary",
        window_span=45.0,
        live_comparison=True,
        limit=25,
    )

    assert captured["sub_url"] == "similarity/rheed_stationary/source-1/matches/"
    assert captured["params"] == {
        "metric": _DEFAULT_SIMILARITY_METRIC,
        "windowSpan": 45.0,
        "usePrototypes": True,
        "liveComparison": True,
        "limit": 25,
    }
    # Removed knobs must not leak into the query.
    assert "refine" not in captured["params"]
    assert "returnNullIf404" not in captured["params"]


def test_matches_defaults(client, monkeypatch):
    """Defaults hit the chamfer path: prototypes on, window_span 60, fixed metric."""
    captured: dict = {}

    def fake_get(sub_url, params=None, **kwargs):
        captured["params"] = params
        return []

    monkeypatch.setattr(client, "_get", fake_get)
    client.get_similarity_matches("source-1")

    assert captured["params"]["usePrototypes"] is True
    assert captured["params"]["windowSpan"] == 60.0
    assert captured["params"]["metric"] == _DEFAULT_SIMILARITY_METRIC
    assert captured["params"]["liveComparison"] is False


def test_matches_dataframe_shape(client, monkeypatch):
    # Backend serializes SimilarityMatchResponse by alias (camelCase on the wire),
    # and includes extra fields (workflow, profile, image) the SDK drops.
    payload = [
        {
            "workflow": "rheed_stationary",
            "dataId": "a",
            "itemName": "Sample A",
            "similarity": 0.95,
            "profile": [0.1],
            "image": None,
        },
        {
            "workflow": "rheed_stationary",
            "dataId": "b",
            "itemName": "Sample B",
            "similarity": 0.80,
            "profile": [0.2],
            "image": None,
        },
    ]
    monkeypatch.setattr(client, "_get", lambda *a, **k: payload)

    df = client.get_similarity_matches("source-1")

    assert isinstance(df, DataFrame)
    assert list(df.columns) == ["data_id", "item_name", "similarity"]
    assert len(df) == 2
    assert df.iloc[0]["data_id"] == "a"
    assert df.iloc[0]["item_name"] == "Sample A"
    assert df.iloc[0]["similarity"] == 0.95


def test_matches_empty_returns_empty_frame_with_columns(client, monkeypatch):
    monkeypatch.setattr(client, "_get", lambda *a, **k: None)

    df = client.get_similarity_matches("source-1")

    assert isinstance(df, DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["data_id", "item_name", "similarity"]


def test_matches_tolerates_wrapped_payload(client, monkeypatch):
    payload = {"matches": [{"dataId": "a", "itemName": "A", "similarity": 0.5}]}
    monkeypatch.setattr(client, "_get", lambda *a, **k: payload)

    df = client.get_similarity_matches("source-1")

    assert len(df) == 1
    assert df.iloc[0]["data_id"] == "a"


def test_matches_enforces_exact_three_columns(client, monkeypatch):
    """camelCase aliases are renamed, extra fields dropped, column order enforced."""
    payload = [
        # camelCase wire keys, out of canonical order, with extra backend fields
        {
            "similarity": 0.9,
            "profile": [0.1],
            "workflow": "wf",
            "itemName": "A",
            "dataId": "a",
        },
    ]
    monkeypatch.setattr(client, "_get", lambda *a, **k: payload)

    df = client.get_similarity_matches("source-1")

    assert list(df.columns) == ["data_id", "item_name", "similarity"]
    assert df.iloc[0]["data_id"] == "a"
    assert df.iloc[0]["item_name"] == "A"
    assert df.iloc[0]["similarity"] == 0.9


def test_matches_missing_item_name_filled_na(client, monkeypatch):
    """A match without itemName still yields all three columns (item_name NA)."""
    payload = [{"dataId": "a", "similarity": 0.9}]  # itemName omitted
    monkeypatch.setattr(client, "_get", lambda *a, **k: payload)

    df = client.get_similarity_matches("source-1")

    assert list(df.columns) == ["data_id", "item_name", "similarity"]
    assert df.iloc[0]["data_id"] == "a"
    assert df["item_name"].isna().all()


def test_matches_tolerates_snake_case_keys(client, monkeypatch):
    """If the backend ever stops aliasing, snake_case keys pass through unchanged."""
    payload = [{"data_id": "a", "item_name": "A", "similarity": 0.9}]
    monkeypatch.setattr(client, "_get", lambda *a, **k: payload)

    df = client.get_similarity_matches("source-1")

    assert list(df.columns) == ["data_id", "item_name", "similarity"]
    assert df.iloc[0]["data_id"] == "a"


def test_matches_rejects_removed_kwargs(client):
    """The removed knobs are no longer accepted."""
    for bad_kwarg in ("metric", "use_prototypes", "refine", "return_null_if_404"):
        with pytest.raises(TypeError):
            client.get_similarity_matches("source-1", **{bad_kwarg: True})  # type: ignore[arg-type]
