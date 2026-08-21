"""Tests for sample-scoped computed timeseries results.

Covers the parse helper, ``Client.get_physical_sample_timeseries`` (mocked
``_get``), and the ``PhysicalSampleResult.sample_metrics`` passthrough, plus one
integration test against a real BTO sample exposing ``rheed_quality``.
"""

import numpy as np
import pytest
from pandas import DataFrame, isna

from atomscale import Client
from atomscale.core import ClientError
from atomscale.timeseries.physical_sample import (
    PHYSICAL_SAMPLE_TS_COLUMNS,
    physical_sample_timeseries_to_dataframe,
)

PSID = "af9eeb92-0000-0000-0000-000000000000"


@pytest.fixture
def client():
    return Client(api_key="key_test", endpoint="http://example.com/")


def _payload():
    """A response body matching the §3 schema: two properties, a null gap."""
    return {
        "physical_sample_id": PSID,
        "properties": [
            {
                "property_name": "rheed_quality",
                "property_values": [0.83, 0.84, None, 0.82],
                "real_time_seconds": [0.0, 30.0, 60.0, 90.0],
                "result_id": "rq-result-1",
                "constituent_data_ids": ["d1", "d2", "d3"],
                "last_updated": "2026-08-11T07:06:47.750195",
                "generating_dbos_workflow_id": "rheed-quality-metric-physical-sample-x",
            },
            {
                "property_name": "composition_metric",
                "property_values": [1.0, 1.1],
                "real_time_seconds": [0.0, 45.0],
                "result_id": "cm-result-1",
                "constituent_data_ids": ["d1", "d2"],
                "last_updated": "2026-08-11T07:07:00.000000",
                "generating_dbos_workflow_id": "composition-metric-physical-sample-x",
            },
        ],
    }


# --------------------------------------------------------------------------
# Parse helper.
# --------------------------------------------------------------------------


def test_parse_long_form_null_to_nan_and_shape():
    df = physical_sample_timeseries_to_dataframe(_payload())

    assert list(df.columns) == list(PHYSICAL_SAMPLE_TS_COLUMNS)
    # 4 rheed_quality points + 2 composition_metric points.
    assert len(df) == 6

    rq = df[df.property_name == "rheed_quality"]
    # JSON null -> NaN; per property len(value) == len(real_time_seconds).
    assert isna(rq["value"].iloc[2])
    assert rq["value"].tolist()[:2] == [0.83, 0.84]
    assert len(rq) == len(rq["real_time_seconds"])
    assert df["value"].dtype == "float64"
    assert df["real_time_seconds"].dtype == "float64"

    # Provenance reachable.
    assert set(rq["result_id"]) == {"rq-result-1"}
    assert str(rq["last_updated"].iloc[0]).startswith("2026-08-11")
    assert set(rq["generating_dbos_workflow_id"]) == {
        "rheed-quality-metric-physical-sample-x"
    }
    assert df.attrs["constituent_data_ids"]["rheed_quality"] == ["d1", "d2", "d3"]


def test_parse_property_names_filter():
    df = physical_sample_timeseries_to_dataframe(
        _payload(), property_names=["rheed_quality"]
    )
    assert set(df.property_name) == {"rheed_quality"}
    assert len(df) == 4


def test_parse_empty_properties_returns_empty_frame():
    df = physical_sample_timeseries_to_dataframe(
        {"physical_sample_id": PSID, "properties": []}
    )
    assert isinstance(df, DataFrame)
    assert df.empty
    assert list(df.columns) == list(PHYSICAL_SAMPLE_TS_COLUMNS)


def test_parse_filter_to_absent_property_is_empty():
    df = physical_sample_timeseries_to_dataframe(
        _payload(), property_names=["does_not_exist"]
    )
    assert df.empty
    assert list(df.columns) == list(PHYSICAL_SAMPLE_TS_COLUMNS)


def test_parse_length_mismatch_raises():
    bad = {
        "properties": [
            {
                "property_name": "rheed_quality",
                "property_values": [0.1, 0.2, 0.3],
                "real_time_seconds": [0.0, 30.0],  # mismatched length
            }
        ]
    }
    with pytest.raises(ValueError, match="value\\(s\\) vs"):
        physical_sample_timeseries_to_dataframe(bad)


def test_parse_unix_times_passed_through_when_present():
    """Forward-compat with the §7 backend follow-up: pass unix_times through."""
    payload = {
        "properties": [
            {
                "property_name": "rheed_quality",
                "property_values": [0.83, 0.84],
                "real_time_seconds": [0.0, 30.0],
                "unix_times": [1.0e9, 1.0e9 + 30.0],
                "result_id": "rq-result-1",
                "constituent_data_ids": ["d1"],
                "last_updated": "2026-08-11T07:06:47.750195",
                "generating_dbos_workflow_id": "wf-x",
            }
        ]
    }
    df = physical_sample_timeseries_to_dataframe(payload)
    assert "unix_times" in df.columns
    assert df["unix_times"].tolist() == [1.0e9, 1.0e9 + 30.0]


# --------------------------------------------------------------------------
# Client.get_physical_sample_timeseries (mocked _get).
# --------------------------------------------------------------------------


def test_get_physical_sample_timeseries_hits_endpoint(client, monkeypatch):
    captured: dict = {}

    def fake_get(sub_url, params=None, **kwargs):
        captured["sub_url"] = sub_url
        return _payload()

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_physical_sample_timeseries(PSID)

    assert captured["sub_url"] == f"physical_samples/{PSID}/timeseries/"
    assert set(df.property_name) == {"rheed_quality", "composition_metric"}


def test_get_physical_sample_timeseries_filters(client, monkeypatch):
    monkeypatch.setattr(client, "_get", lambda sub_url, **kwargs: _payload())
    df = client.get_physical_sample_timeseries(PSID, property_names=["rheed_quality"])
    assert set(df.property_name) == {"rheed_quality"}


def test_get_physical_sample_timeseries_404_raises(client, monkeypatch):
    # ``_get`` returns None on 404; the method promotes that to ClientError.
    monkeypatch.setattr(client, "_get", lambda sub_url, **kwargs: None)
    with pytest.raises(ClientError) as exc:
        client.get_physical_sample_timeseries("missing-sample")
    assert exc.value.status_code == 404


def test_get_physical_sample_timeseries_no_metrics_returns_empty(client, monkeypatch):
    monkeypatch.setattr(
        client, "_get", lambda sub_url, **kwargs: {"properties": []}
    )
    df = client.get_physical_sample_timeseries(PSID)
    assert df.empty


# --------------------------------------------------------------------------
# PhysicalSampleResult.sample_metrics passthrough via get_physical_sample.
# --------------------------------------------------------------------------


def test_get_physical_sample_populates_sample_metrics(client, monkeypatch):
    def fake_get(sub_url, params=None, **kwargs):
        if sub_url == "physical_samples/":
            return [{"id": PSID, "name": "BTO-1"}]
        if sub_url == "data_entries/":
            return []  # no data items -> no self.get() call
        if sub_url == f"physical_samples/{PSID}/timeseries/":
            return _payload()
        return None

    monkeypatch.setattr(client, "_get", fake_get)
    result = client.get_physical_sample(PSID)

    assert isinstance(result.sample_metrics, DataFrame)
    assert "rheed_quality" in set(result.sample_metrics.property_name)


def test_get_physical_sample_survives_metrics_error(client, monkeypatch):
    """A non-404 error from the metrics endpoint must not abort the sample fetch."""

    def fake_get(sub_url, params=None, **kwargs):
        if sub_url == "physical_samples/":
            return [{"id": PSID, "name": "BTO-1"}]
        if sub_url == "data_entries/":
            return []
        if sub_url == f"physical_samples/{PSID}/timeseries/":
            raise ClientError("boom", status_code=500)
        return None

    monkeypatch.setattr(client, "_get", fake_get)
    with pytest.warns(UserWarning, match="Could not fetch sample metrics"):
        result = client.get_physical_sample(PSID)

    # Sample still returned; metrics simply absent.
    assert result.physical_sample_id == PSID
    assert result.sample_metrics is None


def test_get_physical_sample_can_skip_sample_metrics(client, monkeypatch):
    seen: list[str] = []

    def fake_get(sub_url, params=None, **kwargs):
        seen.append(sub_url)
        if sub_url == "physical_samples/":
            return [{"id": PSID, "name": "BTO-1"}]
        return []

    monkeypatch.setattr(client, "_get", fake_get)
    result = client.get_physical_sample(PSID, include_sample_metrics=False)

    assert result.sample_metrics is None
    assert f"physical_samples/{PSID}/timeseries/" not in seen


# --------------------------------------------------------------------------
# Integration: real BTO sample exposing rheed_quality.
# --------------------------------------------------------------------------


@pytest.fixture
def live_client():
    try:
        return Client()
    except ValueError:
        pytest.skip("No Atomscale API key available for integration test")


def test_get_physical_sample_timeseries_integration(live_client):
    samples = live_client.list_physical_samples()
    if not len(samples):
        pytest.skip("No physical samples available")

    # Probe BTO-looking samples first, then fall back to any sample; stop at the
    # first one exposing rheed_quality. Bounded so the scan stays cheap.
    #
    # Work on one frame filtered to rows that actually have an id, so the name
    # mask can't misalign against it. Names are coerced with astype(str) rather
    # than fillna("") because the live catalogue carries non-string names (None,
    # NaN, even dicts); .str.contains would yield NA on those and pandas refuses
    # to index with a non-boolean mask.
    have_id = samples[samples["Physical Sample ID"].notna()]
    ids = have_id["Physical Sample ID"].astype(str)
    if "Physical Sample Name" in have_id.columns:
        looks_bto = (
            have_id["Physical Sample Name"]
            .astype(str)
            .str.contains("bto", case=False, na=False)
            .fillna(False)
            .astype(bool)
        )
        ordered = ids[looks_bto].tolist() + ids[~looks_bto].tolist()
    else:
        ordered = ids.tolist()

    found = None
    for sid in ordered[:25]:
        df = live_client.get_physical_sample_timeseries(
            sid, property_names=["rheed_quality"]
        )
        if not df.empty and df["value"].notna().any():
            found = df
            break

    if found is None:
        pytest.skip("No accessible sample exposes rheed_quality")

    q = found.loc[found.property_name == "rheed_quality", "value"].dropna()
    assert len(q) > 0
    assert np.isfinite(q.to_numpy()).all()
