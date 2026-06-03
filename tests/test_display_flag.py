"""Tests for the ``display`` flag that toggles user-facing display column
labels (default) versus raw snake_case API names (``display=False``)."""

import types

import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.similarity import SimilarityTrajectoryProvider
from atomscale.similarity.polling import _default_trajectory_until
from atomscale.timeseries.ellipsometry import EllipsometryProvider
from atomscale.timeseries.metrology import MetrologyProvider
from atomscale.timeseries.optical import OpticalProvider
from atomscale.timeseries.provider import (
    finalize_dataframe,
    properties_payload_to_dataframe,
)
from atomscale.timeseries.rheed import RHEEDProvider

PROP_PAYLOAD = {
    "properties": {
        "Sub T setpoint": {
            "relative_time_seconds": [0.0, 1.0, 2.0],
            "unix_timestamp_ms": [
                1_700_000_000_000,
                1_700_000_001_000,
                1_700_000_002_000,
            ],
            "values": [400.0, 450.0, 500.0],
            "units": "C",
        },
    },
    "series_max_time": 2.0,
}

SERIES_PAYLOAD = {
    "series": [
        {
            "unix_timestamp_ms": 1_700_000_000_000,
            "relative_time_seconds": 0.0,
            "ratio_pyrometer": 1.0,
        },
        {
            "unix_timestamp_ms": 1_700_000_000_500,
            "relative_time_seconds": 0.5,
            "ratio_pyrometer": 1.5,
        },
    ]
}

RHEED_PAYLOAD = {
    "series_by_angle": [
        {
            "angle": 0.0,
            "series": [
                {
                    "frame_number": 0,
                    "referenced_strain": 0.0,
                    "relative_time_seconds": 0.0,
                    "unix_timestamp_ms": 1_700_000_000_000,
                    "specular_intensity": 10.0,
                },
                {
                    "frame_number": 1,
                    "referenced_strain": 0.1,
                    "relative_time_seconds": 1.0,
                    "unix_timestamp_ms": 1_700_000_001_000,
                    "specular_intensity": 11.0,
                },
            ],
        }
    ]
}

SIM_PAYLOAD = {
    "trajectories": [
        {
            "reference_id": "ref-1",
            "reference_item_name": "Ref One",
            "similarity_values": [0.1, 0.2],
            "real_time_seconds": [0.0, 1.0],
            "unix_times": [1_700_000_000, 1_700_000_001],
            "is_active": True,
            "averaged_count": 5,
        }
    ]
}


# -----------------------------------------------------------------------------
# finalize_dataframe helper
# -----------------------------------------------------------------------------


def test_finalize_dataframe_display_true_renames_and_indexes():
    df = DataFrame({"frame_number": [0, 1], "referenced_strain": [0.0, 0.1]})
    rename_map = {"frame_number": "Frame Number", "referenced_strain": "Strain"}
    out = finalize_dataframe(df, rename_map, display=True, index_cols=["frame_number"])
    assert out.index.name == "Frame Number"
    assert "Strain" in out.columns
    assert "referenced_strain" not in out.columns


def test_finalize_dataframe_display_false_keeps_snake_case():
    df = DataFrame({"frame_number": [0, 1], "referenced_strain": [0.0, 0.1]})
    rename_map = {"frame_number": "Frame Number", "referenced_strain": "Strain"}
    out = finalize_dataframe(df, rename_map, display=False, index_cols=["frame_number"])
    assert out.index.name == "frame_number"
    assert "referenced_strain" in out.columns
    assert "Strain" not in out.columns


def test_finalize_dataframe_skips_absent_index_cols():
    df = DataFrame({"a": [1, 2]})
    out = finalize_dataframe(df, {}, display=False, index_cols=["missing"])
    # No index set when the column is absent; stays a default RangeIndex.
    assert out.index.tolist() == [0, 1]


# -----------------------------------------------------------------------------
# properties_payload_to_dataframe
# -----------------------------------------------------------------------------


def test_properties_payload_display_true_uses_display_time_labels():
    df = properties_payload_to_dataframe(PROP_PAYLOAD["properties"], display=True)
    assert {"UNIX Timestamp", "Time"} <= set(df.columns)
    assert "unix_timestamp_ms" not in df.columns


def test_properties_payload_display_false_uses_snake_time_labels():
    df = properties_payload_to_dataframe(PROP_PAYLOAD["properties"], display=False)
    assert {"unix_timestamp_ms", "relative_time_seconds"} <= set(df.columns)
    assert "UNIX Timestamp" not in df.columns
    assert "Time" not in df.columns
    # Property columns are already raw API names and are unaffected.
    assert "Sub T setpoint" in df.columns


# -----------------------------------------------------------------------------
# Provider.to_dataframe — property/series payload domains
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls", [MetrologyProvider, OpticalProvider, EllipsometryProvider]
)
def test_property_provider_display_false_snake_case(provider_cls):
    df = provider_cls().to_dataframe(PROP_PAYLOAD, display=False)
    assert {"unix_timestamp_ms", "relative_time_seconds"} <= set(df.columns)
    assert "UNIX Timestamp" not in df.columns
    # Default (display=True) keeps the display labels.
    df_display = provider_cls().to_dataframe(PROP_PAYLOAD)
    assert {"UNIX Timestamp", "Time"} <= set(df_display.columns)


@pytest.mark.parametrize(
    "provider_cls", [MetrologyProvider, OpticalProvider, EllipsometryProvider]
)
def test_property_provider_series_path_display_false(provider_cls):
    df = provider_cls().to_dataframe(SERIES_PAYLOAD, display=False)
    assert {"unix_timestamp_ms", "relative_time_seconds", "ratio_pyrometer"} <= set(
        df.columns
    )
    assert "UNIX Timestamp" not in df.columns


# -----------------------------------------------------------------------------
# RHEED provider
# -----------------------------------------------------------------------------


def test_rheed_display_true_uses_display_names_and_index():
    df = RHEEDProvider().to_dataframe(RHEED_PAYLOAD)
    assert list(df.index.names) == ["Angle", "Frame Number"]
    assert "Strain" in df.columns
    assert "Specular Intensity" in df.columns
    assert "referenced_strain" not in df.columns


def test_rheed_display_false_uses_snake_case_and_index():
    df = RHEEDProvider().to_dataframe(RHEED_PAYLOAD, display=False)
    assert list(df.index.names) == ["angle", "frame_number"]
    assert "referenced_strain" in df.columns
    assert "specular_intensity" in df.columns
    assert "Strain" not in df.columns
    assert "Angle" not in df.columns


# -----------------------------------------------------------------------------
# Similarity trajectory provider
# -----------------------------------------------------------------------------


def test_similarity_display_true_uses_display_names_and_index():
    df = SimilarityTrajectoryProvider().to_dataframe(SIM_PAYLOAD)
    assert list(df.index.names) == ["Reference ID", "Time"]
    assert "Similarity" in df.columns
    assert "Active" in df.columns


def test_similarity_display_false_uses_snake_case_and_index():
    df = SimilarityTrajectoryProvider().to_dataframe(SIM_PAYLOAD, display=False)
    assert list(df.index.names) == ["reference_id", "real_time_seconds"]
    assert "similarity_values" in df.columns
    assert "is_active" in df.columns
    assert "Similarity" not in df.columns


def test_default_trajectory_until_handles_both_label_modes():
    display_df = SimilarityTrajectoryProvider().to_dataframe(SIM_PAYLOAD)
    snake_df = SimilarityTrajectoryProvider().to_dataframe(SIM_PAYLOAD, display=False)
    # is_active is True for the sole trajectory -> not done yet in either mode.
    assert _default_trajectory_until(display_df) is False
    assert _default_trajectory_until(snake_df) is False


# -----------------------------------------------------------------------------
# Client method forwarding / catalogue-style DataFrames
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    return Client(api_key="key_test", endpoint="http://example.com/")


def test_get_forwards_display_to_result_data(mock_client, monkeypatch):
    captured = {}

    monkeypatch.setattr(
        mock_client,
        "_get",
        lambda sub_url, params=None: [
            {"data_id": "abc", "char_source_type": "metrology"}
        ],
    )
    monkeypatch.setattr(
        mock_client,
        "_multi_thread",
        lambda func, kwargs_list, *a, **k: [func(**kw) for kw in kwargs_list],
    )

    def fake_result_data(data_id, data_type, catalogue_entry=None, *, display=True):
        captured["display"] = display
        return "result"

    monkeypatch.setattr(mock_client, "_get_result_data", fake_result_data)

    mock_client.get("abc", display=False)
    assert captured["display"] is False


def test_get_result_data_forwards_display_to_provider(mock_client, monkeypatch):
    captured = {}

    class FakeProvider:
        def fetch_raw(self, client, data_id):
            return {"properties": {}}

        def to_dataframe(self, raw, *, display=True):
            captured["display"] = display
            return DataFrame()

        def build_result(self, client, data_id, data_type, ts_df):
            return types.SimpleNamespace()

    monkeypatch.setattr("atomscale.client.get_provider", lambda t: FakeProvider())

    mock_client._get_result_data(
        "abc", "metrology", catalogue_entry=None, display=False
    )
    assert captured["display"] is False


def test_get_similarity_trajectory_forwards_display(mock_client, monkeypatch):
    captured = {}

    class FakeProvider:
        def fetch_raw(self, client, source_id, **kwargs):
            return {"trajectories": []}

        def to_dataframe(self, raw, *, display=True):
            captured["display"] = display
            return DataFrame()

        def build_result(self, *args, **kwargs):
            return types.SimpleNamespace()

    monkeypatch.setattr("atomscale.client.get_provider", lambda t: FakeProvider())

    mock_client.get_similarity_trajectory("src", display=False)
    assert captured["display"] is False


def test_fetch_result_forwards_display(mock_client, monkeypatch):
    from atomscale.timeseries import polling

    captured = {}

    class FakeProvider:
        def fetch_raw(self, client, data_id, **kwargs):
            return {"x": 1}

        def to_dataframe(self, raw, *, display=True):
            captured["display"] = display
            return DataFrame()

    monkeypatch.setattr(polling, "get_provider", lambda name: FakeProvider())

    polling._fetch_result(mock_client, "abc", None, display=False)
    assert captured["display"] is False


def test_fetch_trajectory_result_forwards_display(mock_client, monkeypatch):
    from atomscale.similarity import polling

    captured = {}

    class FakeProvider:
        def fetch_raw(self, client, source_id, **kwargs):
            return {"trajectories": []}

        def to_dataframe(self, raw, *, display=True):
            captured["display"] = display
            return DataFrame()

    monkeypatch.setattr(polling, "get_provider", lambda name: FakeProvider())

    polling._fetch_trajectory_result(mock_client, "src", None, display=False)
    assert captured["display"] is False


def test_search_display_false_returns_snake_case(mock_client, monkeypatch):
    entry = {
        "data_id": "abc",
        "raw_name": "file.dat",
        "char_source_type": "xps",
        "pipeline_status": "success",
    }
    monkeypatch.setattr(mock_client, "_get", lambda sub_url, params=None: [entry])

    snake = mock_client.search(display=False)
    assert {"data_id", "raw_name", "char_source_type"} <= set(snake.columns)
    assert "Data ID" not in snake.columns

    display = mock_client.search()
    assert "Data ID" in display.columns
    assert "data_id" not in display.columns


def test_list_physical_samples_display_false_returns_snake_case(
    mock_client, monkeypatch
):
    monkeypatch.setattr(
        mock_client,
        "_get",
        lambda sub_url, params=None: [{"id": "s1", "name": "Sample One"}],
    )

    snake = mock_client.list_physical_samples(display=False)
    assert {"id", "name"} <= set(snake.columns)
    assert "Physical Sample ID" not in snake.columns

    display = mock_client.list_physical_samples()
    assert "Physical Sample ID" in display.columns


def test_list_projects_display_false_returns_snake_case(mock_client, monkeypatch):
    monkeypatch.setattr(
        mock_client,
        "_get",
        lambda sub_url, params=None: [{"id": "p1", "name": "Project One"}],
    )

    snake = mock_client.list_projects(display=False)
    assert {"id", "name"} <= set(snake.columns)
    assert "Project ID" not in snake.columns

    display = mock_client.list_projects()
    assert "Project ID" in display.columns
