"""Unit tests for Client.get_embeddings and EmbeddingsResult.

Payloads mirror the backend ``EmbeddingVectorsResponse`` / ``EmbeddingVectorPoint``
models: a ``points`` list where each point carries a singular ``vector`` plus
per-kind metadata (``real_time_seconds`` / ``unix_time_ms`` for window,
``cluster_size`` for prototype), and top-level ``dimension`` / ``count`` /
``offset`` / ``truncated``.
"""

import numpy as np
import pytest

from atomscale import Client
from atomscale.results import EmbeddingsResult


@pytest.fixture
def client():
    return Client(api_key="key_test", endpoint="http://example.com/")


def _capturing_get(captured, payload):
    def fake_get(sub_url, params=None, **kwargs):
        captured["sub_url"] = sub_url
        captured["params"] = params
        return payload

    return fake_get


def _window_payload():
    return {
        "data_id": "data-1",
        "workflow": "rheed_stationary",
        "window_span": 60.0,
        "kind": "window",
        "dimension": 3,
        "count": 2,
        "offset": 0,
        "truncated": True,
        "points": [
            {
                "index": 0,
                "vector": [1.0, 2.0, 3.0],
                "real_time_seconds": 0.0,
                "unix_time_ms": 1000.0,
            },
            {
                "index": 1,
                "vector": [4.0, 5.0, 6.0],
                "real_time_seconds": 60.0,
                "unix_time_ms": 61000.0,
            },
        ],
    }


def _prototype_payload():
    return {
        "data_id": "data-1",
        "workflow": "rheed_stationary",
        "window_span": 30.0,
        "kind": "prototype",
        "dimension": 2,
        "count": 3,
        "offset": 0,
        "truncated": False,
        "points": [
            {"index": 0, "vector": [1.0, 2.0], "cluster_size": 10},
            {"index": 1, "vector": [3.0, 4.0], "cluster_size": 5},
            {"index": 2, "vector": [5.0, 6.0], "cluster_size": 2},
        ],
    }


def test_get_embeddings_window(client, monkeypatch):
    monkeypatch.setattr(client, "_get", _capturing_get({}, _window_payload()))

    result = client.get_embeddings("data-1")

    assert isinstance(result, EmbeddingsResult)
    assert result.kind == "window"
    assert result.workflow == "rheed_stationary"
    assert result.vectors.shape == (2, 3)
    assert result.dimension == 3
    assert result.count == 2
    assert result.truncated is True
    np.testing.assert_array_equal(result.real_times, [0.0, 60.0])
    np.testing.assert_array_equal(result.unix_times_ms, [1000, 61000])
    assert result.unix_times_ms.dtype == np.int64
    # window kind carries no cluster sizes
    assert result.cluster_sizes is None


def test_get_embeddings_prototype(client, monkeypatch):
    monkeypatch.setattr(client, "_get", _capturing_get({}, _prototype_payload()))

    result = client.get_embeddings("data-1", kind="prototype", window_span=30.0)

    assert result.kind == "prototype"
    assert result.vectors.shape == (3, 2)
    np.testing.assert_array_equal(result.cluster_sizes, [10, 5, 2])
    assert result.cluster_sizes.dtype == np.int64
    # prototype kind carries no per-window times
    assert result.real_times is None
    assert result.unix_times_ms is None


def test_get_embeddings_param_mapping(client, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(client, "_get", _capturing_get(captured, _window_payload()))

    client.get_embeddings(
        "data-99",
        workflow="custom_wf",
        window_span=45.0,
        kind="prototype",
        offset=5,
        limit=100,
    )

    assert captured["sub_url"] == "similarity/custom_wf/data-99/embeddings/"
    assert captured["params"] == {
        "window_span": 45.0,
        "kind": "prototype",
        "offset": 5,
        "limit": 100,
    }


def test_get_embeddings_default_window_span_is_60(client, monkeypatch):
    """SDK default is 60s (endpoint default is 30) and must be sent explicitly."""
    captured: dict = {}
    monkeypatch.setattr(client, "_get", _capturing_get(captured, _window_payload()))

    client.get_embeddings("data-1")

    assert captured["params"]["window_span"] == 60.0


def test_get_embeddings_empty_warns(client, monkeypatch):
    """A structured empty response (points=[]) means the entry isn't embedded."""
    empty_payload = {
        "data_id": "unembedded-id",
        "workflow": "rheed_stationary",
        "window_span": 60.0,
        "kind": "window",
        "dimension": None,
        "count": 0,
        "offset": 0,
        "truncated": False,
        "points": [],
    }
    monkeypatch.setattr(client, "_get", lambda *a, **k: empty_payload)

    with pytest.warns(UserWarning, match="No embeddings returned"):
        result = client.get_embeddings("unembedded-id")

    assert result.count == 0
    assert len(result.vectors) == 0
    assert result.vectors.shape == (0, 0)
    assert result.truncated is False


def test_get_embeddings_none_response_warns(client, monkeypatch):
    """A None response (404/empty body) is handled defensively without crashing."""
    monkeypatch.setattr(client, "_get", lambda *a, **k: None)

    with pytest.warns(UserWarning, match="No embeddings returned"):
        result = client.get_embeddings("missing-id")

    assert result.count == 0
    assert len(result.vectors) == 0


def test_count_is_total_available_and_offset_preserved():
    """count reflects total-before-offset/limit; len(vectors) is what's returned."""
    payload = {
        "kind": "window",
        "dimension": 2,
        "count": 100,  # total available
        "offset": 10,
        "truncated": True,
        "points": [
            {
                "index": 10,
                "vector": [1.0, 2.0],
                "real_time_seconds": 5.0,
                "unix_time_ms": 5000.0,
            },
            {
                "index": 11,
                "vector": [3.0, 4.0],
                "real_time_seconds": 6.0,
                "unix_time_ms": 6000.0,
            },
        ],
    }
    result = EmbeddingsResult.from_api(
        payload, data_id="d", workflow="w", kind="window", window_span=60.0
    )
    assert result.count == 100
    assert result.offset == 10
    assert len(result.vectors) == 2  # actually returned


def test_from_api_infers_dimension_when_absent():
    """dimension falls back to the vectors array shape when the server omits it."""
    payload = {
        "kind": "window",
        "points": [
            {
                "index": 0,
                "vector": [1.0, 2.0, 3.0],
                "real_time_seconds": 0.0,
                "unix_time_ms": 0.0,
            },
        ],
    }
    result = EmbeddingsResult.from_api(
        payload, data_id="d", workflow="w", kind="window", window_span=60.0
    )
    assert result.dimension == 3


def test_from_api_reads_window_span_and_kind_from_payload():
    """The result reflects the echoed kind/window_span from the payload."""
    payload = {
        "kind": "prototype",
        "window_span": 30.0,
        "dimension": 1,
        "count": 1,
        "points": [{"index": 0, "vector": [1.0], "cluster_size": 7}],
    }
    result = EmbeddingsResult.from_api(
        payload, data_id="d", workflow="w", kind="window", window_span=60.0
    )
    # Payload's authoritative values win over the request-time args.
    assert result.kind == "prototype"
    assert result.window_span == 30.0


def test_from_api_empty_reads_metadata():
    """Empty points with a populated body still surfaces dimension/truncated."""
    payload = {
        "kind": "window",
        "dimension": None,
        "count": 0,
        "truncated": True,
        "points": [],
    }
    with pytest.warns(UserWarning, match="No embeddings returned"):
        result = EmbeddingsResult.from_api(
            payload, data_id="d", workflow="w", kind="window", window_span=60.0
        )
    assert result.count == 0
    assert result.dimension == 0
    assert result.truncated is True


def test_from_api_ragged_vectors_raises_clear_error():
    payload = {
        "kind": "window",
        "points": [
            {"index": 0, "vector": [1.0, 2.0, 3.0]},
            {"index": 1, "vector": [4.0, 5.0]},  # ragged
        ],
    }
    with pytest.raises(ValueError, match="Malformed embeddings payload"):
        EmbeddingsResult.from_api(
            payload, data_id="bad-id", workflow="w", kind="window", window_span=60.0
        )
