"""Unit tests for Client.get_physical_sample_timeseries.

Payload mirrors the backend ``PhysicalSampleTimeseriesResponse`` /
``PhysicalSampleTimeseriesProperty`` models: a top-level ``physical_sample_id`` plus a
``properties`` list where each property carries ``property_values`` aligned 1:1 with
``real_time_seconds`` (latest result per property).
"""

import pytest

from atomscale import Client


@pytest.fixture
def client():
    return Client(api_key="key_test", endpoint="http://example.com/")


def _capturing_get(captured, payload):
    def fake_get(sub_url, params=None, **kwargs):
        captured["sub_url"] = sub_url
        captured["params"] = params
        return payload

    return fake_get


def _payload():
    return {
        "physical_sample_id": "ps-1",
        "properties": [
            {
                "property_name": "composition_metric",
                "property_values": [0.10, 0.20, 0.30],
                "real_time_seconds": [0.0, 60.0, 120.0],
                "result_id": "r-1",
                "last_updated": "2026-07-03T00:00:00",
            },
            {
                "property_name": "other_metric",
                "property_values": [1.0, 2.0],
                "real_time_seconds": [0.0, 60.0],
                "result_id": "r-2",
                "last_updated": "2026-07-03T00:00:00",
            },
        ],
    }


_COLS = ["property_name", "real_time_seconds", "value", "result_id", "last_updated"]


def test_all_properties_long_dataframe(client, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(client, "_get", _capturing_get(captured, _payload()))

    df = client.get_physical_sample_timeseries("ps-1")

    assert captured["sub_url"] == "physical_samples/ps-1/timeseries/"
    assert list(df.columns) == _COLS
    assert len(df) == 5  # 3 composition + 2 other
    assert set(df["property_name"]) == {"composition_metric", "other_metric"}


def test_property_name_filter(client, monkeypatch):
    monkeypatch.setattr(client, "_get", _capturing_get({}, _payload()))

    df = client.get_physical_sample_timeseries(
        "ps-1", property_name="composition_metric"
    )

    assert (df["property_name"] == "composition_metric").all()
    assert list(df["real_time_seconds"]) == [0.0, 60.0, 120.0]
    assert list(df["value"]) == [0.10, 0.20, 0.30]


def test_as_dataframe_false_returns_raw_props(client, monkeypatch):
    monkeypatch.setattr(client, "_get", _capturing_get({}, _payload()))

    props = client.get_physical_sample_timeseries(
        "ps-1", property_name="composition_metric", as_dataframe=False
    )

    assert isinstance(props, list)
    assert len(props) == 1
    assert props[0]["property_name"] == "composition_metric"
    assert props[0]["property_values"] == [0.10, 0.20, 0.30]


def test_empty_payload_returns_typed_empty_frame(client, monkeypatch):
    monkeypatch.setattr(client, "_get", _capturing_get({}, None))

    df = client.get_physical_sample_timeseries("ps-1", property_name="composition_metric")

    assert df.empty
    assert list(df.columns) == _COLS
