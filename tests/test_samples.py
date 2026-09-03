"""Tests for the sample-scoped non-timeseries accessors.

Covers the three parse helpers in :mod:`atomscale.samples` and the
``Client.get_process_steps`` / ``get_spatial_annotations`` / ``get_ai_summaries``
methods against a mocked ``_get``.
"""

import pandas as pd
import pytest
from pandas import DataFrame, isna

from atomscale import Client
from atomscale.core import ClientError
from atomscale.samples import (
    AI_SUMMARY_COLUMNS,
    PROCESS_STEP_COLUMNS,
    SPATIAL_ANNOTATION_COLUMNS,
    ai_summaries_to_dataframe,
    ai_summary_row,
    process_steps_to_dataframe,
    spatial_annotations_to_dataframe,
)

PSID = "287badaa-0000-0000-0000-000000000000"
DID_A = "d661bcdc-0000-0000-0000-000000000000"
DID_B = "5896d610-0000-0000-0000-000000000000"


@pytest.fixture
def client():
    return Client(api_key="key_test", endpoint="http://example.com/")


def _process_steps_payload():
    """Two steps, deliberately out of ``step_index`` order."""
    return {
        "physical_sample_id": PSID,
        "process_steps": [
            {
                "id": "649ec4d9-0000-0000-0000-000000000000",
                "physical_sample_id": PSID,
                "step_index": 1,
                "name": "STO",
                "start_unix_ms_utc": 1786735668791.987,
                "end_unix_ms_utc": 1786735758830.9885,
                "data_ids": [DID_B],
                "last_updated": "2026-08-17T03:23:05.765064",
            },
            {
                "id": "01b5e3cf-0000-0000-0000-000000000000",
                "physical_sample_id": PSID,
                "step_index": 0,
                "name": None,
                "start_unix_ms_utc": 1734041680727.122,
                "end_unix_ms_utc": 1734041770766.1235,
                "data_ids": [DID_A],
                "last_updated": "2026-08-17T03:23:05.764993",
            },
        ],
    }


def _annotations_payload():
    """Three XPS points on one line scan, plus one from another property."""
    return [
        {
            "id": "b9fa5538-0000-0000-0000-000000000000",
            "physical_sample_id": PSID,
            "data_id": None,
            "char_source_type": "xps",
            "property_name": "sr_ti_ratio",
            "property_value": 0.754963,
            "property_unit": None,
            "coord_x": -135.0,
            "coord_y": 0.0,
            "coord_z": None,
            "coord_ref_frame": "sample_relative",
            "coord_units": "mm",
            "metadata": None,
            "last_updated": "2026-04-03T14:36:51.973648",
        },
        {
            "id": "afed4025-0000-0000-0000-000000000000",
            "physical_sample_id": PSID,
            "data_id": DID_A,
            "char_source_type": "xps",
            "property_name": "sr_ti_ratio",
            "property_value": 0.741318,
            "property_unit": None,
            "coord_x": -90.0,
            "coord_y": 0.0,
            "coord_z": None,
            "coord_ref_frame": "sample_relative",
            "coord_units": "mm",
            "metadata": {"scan": "line"},
            "last_updated": "2026-04-03T14:36:51.973750",
        },
        {
            "id": "f98efd34-0000-0000-0000-000000000000",
            "physical_sample_id": PSID,
            "data_id": None,
            "char_source_type": "xps",
            "property_name": "sr_ti_ratio",
            "property_value": 0.744831,
            "property_unit": None,
            "coord_x": -112.5,
            "coord_y": 0.0,
            "coord_z": None,
            "coord_ref_frame": "sample_relative",
            "coord_units": "mm",
            "metadata": None,
            "last_updated": "2026-04-03T14:36:51.973719",
        },
        {
            "id": "aaaaaaaa-0000-0000-0000-000000000000",
            "physical_sample_id": PSID,
            "data_id": None,
            "char_source_type": "ellipsometry",
            "property_name": "thickness",
            "property_value": 42.5,
            "property_unit": "nm",
            "coord_x": 0.0,
            "coord_y": 0.0,
            "coord_z": None,
            "coord_ref_frame": "sample_relative",
            "coord_units": "mm",
            "metadata": None,
            "last_updated": "2026-04-03T14:36:52.000000",
        },
    ]


# --------------------------------------------------------------------------
# Process steps.
# --------------------------------------------------------------------------


def test_process_steps_shape_order_and_derived_columns():
    df = process_steps_to_dataframe(_process_steps_payload())

    assert list(df.columns) == list(PROCESS_STEP_COLUMNS)
    # Sorted by step_index even though the payload listed step 1 first.
    assert df["step_index"].tolist() == [0, 1]
    # An unset label is missing-valued; whether that surfaces as None or NaN is
    # a pandas-version detail (3.x infers a str dtype and maps None to NaN).
    assert isna(df["name"].iloc[0])
    assert df["name"].iloc[1] == "STO"
    assert df["data_ids"].tolist() == [(DID_A,), (DID_B,)]
    assert df["n_data_ids"].tolist() == [1, 1]

    # Both steps are ~90 s acquisitions.
    assert df["duration_seconds"].round(3).tolist() == [90.039, 90.039]
    assert str(df["start_datetime"].dt.tz) == "UTC"
    assert df["start_datetime"].iloc[0].year == 2024
    assert df["start_datetime"].iloc[1].year == 2026
    # Offset-free ``last_updated`` is normalized to UTC (not tz-naive) so it stays
    # comparable with the UTC ``start_datetime`` / ``end_datetime`` bounds.
    assert pd.api.types.is_datetime64_any_dtype(df["last_updated"])
    assert str(df["last_updated"].dt.tz) == "UTC"
    # The subtraction the reviewer flagged must not raise on mixed tz-awareness.
    assert (df["last_updated"] - df["start_datetime"]).notna().all()


def test_process_steps_empty_keeps_columns_and_dtypes():
    df = process_steps_to_dataframe({"physical_sample_id": PSID, "process_steps": []})

    assert len(df) == 0
    assert list(df.columns) == list(PROCESS_STEP_COLUMNS)
    assert df["start_unix_ms_utc"].dtype == "float64"
    assert df["duration_seconds"].dtype == "float64"
    # Empty frame shares the populated path's tz-aware datetime dtype.
    for column in ("start_datetime", "end_datetime", "last_updated"):
        assert str(df[column].dt.tz) == "UTC"


def test_process_steps_step_with_no_data_items_survives():
    payload = {
        "physical_sample_id": PSID,
        "process_steps": [
            {
                "id": "01b5e3cf-0000-0000-0000-000000000000",
                "step_index": 0,
                "name": None,
                "start_unix_ms_utc": 1734041680727.0,
                "end_unix_ms_utc": 1734041770766.0,
                "data_ids": [],
                "last_updated": "2026-08-17T03:23:05.764993",
            }
        ],
    }
    df = process_steps_to_dataframe(payload)

    assert len(df) == 1
    assert df["data_ids"].iloc[0] == ()
    assert df["n_data_ids"].iloc[0] == 0


def test_get_process_steps_hits_endpoint(client, monkeypatch):
    seen = {}

    def fake_get(*, sub_url, **kwargs):
        seen["sub_url"] = sub_url
        return _process_steps_payload()

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_process_steps(PSID)

    assert seen["sub_url"] == f"physical_samples/{PSID}/process_steps"
    assert df["step_index"].tolist() == [0, 1]


def test_get_process_steps_missing_sample_raises(client, monkeypatch):
    monkeypatch.setattr(client, "_get", lambda **kwargs: None)

    with pytest.raises(ClientError) as excinfo:
        client.get_process_steps(PSID)
    assert excinfo.value.status_code == 404


# --------------------------------------------------------------------------
# Spatial annotations.
# --------------------------------------------------------------------------


def test_spatial_annotations_shape_and_position_sort():
    df = spatial_annotations_to_dataframe(_annotations_payload())

    assert list(df.columns) == list(SPATIAL_ANNOTATION_COLUMNS)
    assert len(df) == 4
    # Grouped by property, then ascending in x so a line scan reads in order.
    assert df["property_name"].tolist() == ["sr_ti_ratio"] * 3 + ["thickness"]
    assert df["coord_x"].tolist()[:3] == [-135.0, -112.5, -90.0]
    assert df["property_value"].round(4).tolist()[:3] == [0.755, 0.7448, 0.7413]
    # The server-side ``id`` is renamed so it can't be mistaken for a data id.
    assert df["annotation_id"].iloc[0] == "b9fa5538-0000-0000-0000-000000000000"
    assert df["data_id"].tolist().count(DID_A) == 1
    # Offset-free ``last_updated`` is normalized to UTC, not tz-naive.
    assert str(df["last_updated"].dt.tz) == "UTC"


def test_spatial_annotations_missing_coord_becomes_nan():
    df = spatial_annotations_to_dataframe(_annotations_payload())

    assert df["coord_z"].isna().all()
    assert df["coord_z"].dtype == "float64"


def test_spatial_annotations_empty_keeps_columns_and_dtypes():
    df = spatial_annotations_to_dataframe([])

    assert len(df) == 0
    assert list(df.columns) == list(SPATIAL_ANNOTATION_COLUMNS)
    assert df["property_value"].dtype == "float64"
    assert df["coord_x"].dtype == "float64"
    assert str(df["last_updated"].dt.tz) == "UTC"


def test_get_spatial_annotations_forwards_filters(client, monkeypatch):
    seen = {}

    def fake_get(*, sub_url, params=None, **kwargs):
        seen["sub_url"] = sub_url
        seen["params"] = params
        return _annotations_payload()

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_spatial_annotations(
        PSID,
        property_name="sr_ti_ratio",
        char_source_type="xps",
        x_range=(-140.0, -80.0),
        y_range=(None, None),
    )

    assert seen["sub_url"] == f"physical_samples/{PSID}/annotations/"
    assert seen["params"] == {
        "property_name": "sr_ti_ratio",
        "char_source_type": "xps",
        "x_min": -140.0,
        "x_max": -80.0,
        "y_min": None,
        "y_max": None,
    }
    assert len(df) == 4


def test_get_spatial_annotations_empty_list_is_not_a_404(client, monkeypatch):
    monkeypatch.setattr(client, "_get", lambda **kwargs: [])

    df = client.get_spatial_annotations(PSID)
    assert len(df) == 0
    assert list(df.columns) == list(SPATIAL_ANNOTATION_COLUMNS)


def test_get_spatial_annotations_missing_sample_raises(client, monkeypatch):
    monkeypatch.setattr(client, "_get", lambda **kwargs: None)

    with pytest.raises(ClientError) as excinfo:
        client.get_spatial_annotations(PSID)
    assert excinfo.value.status_code == 404


# --------------------------------------------------------------------------
# AI summaries.
# --------------------------------------------------------------------------


def test_ai_summary_row_available():
    payload = {
        "summary": {
            "id": "cccccccc-0000-0000-0000-000000000000",
            "content": "Streaky 2x1 pattern throughout; one reconstruction change.",
            "structured_content": {"quality": "good"},
            "generation_model": "claude-opus-5",
            "generation_timestamp": "2026-08-17T03:23:05.764993",
            "user_feedback": {"thumbs": None},
        },
        "task_status": "available",
        "workflow_id": "generate-data-summary-x-1",
    }
    row = ai_summary_row(DID_A, payload)

    assert row["data_id"] == DID_A
    assert row["task_status"] == "available"
    assert row["content"].startswith("Streaky")
    assert row["structured_content"] == {"quality": "good"}
    assert row["workflow_id"] == "generate-data-summary-x-1"


def test_ai_summary_row_not_found_and_none_payload():
    not_found = ai_summary_row(
        DID_A, {"summary": None, "task_status": "not_found", "workflow_id": None}
    )
    assert not_found["task_status"] == "not_found"
    assert not_found["content"] is None
    assert not_found["summary_id"] is None

    # A 404 reaches the helper as ``None`` and reports the same status.
    assert ai_summary_row(DID_A, None)["task_status"] == "not_found"


def test_ai_summaries_frame_preserves_request_order_and_parses_timestamp():
    rows = [
        ai_summary_row(DID_B, {"summary": None, "task_status": "pending"}),
        ai_summary_row(
            DID_A,
            {
                "summary": {
                    "id": "cccccccc-0000-0000-0000-000000000000",
                    "content": "ok",
                    "structured_content": {},
                    "generation_model": "claude-opus-5",
                    "generation_timestamp": "2026-08-17T03:23:05.764993",
                    "user_feedback": {"thumbs": "up"},
                },
                "task_status": "available",
            },
        ),
    ]
    df = ai_summaries_to_dataframe(rows)

    assert list(df.columns) == list(AI_SUMMARY_COLUMNS)
    assert df["data_id"].tolist() == [DID_B, DID_A]
    assert df["task_status"].tolist() == ["pending", "available"]
    assert isna(df["generation_timestamp"].iloc[0])
    assert df["generation_timestamp"].iloc[1].year == 2026
    # Offset-free ``generation_timestamp`` is normalized to UTC, not tz-naive.
    assert str(df["generation_timestamp"].dt.tz) == "UTC"


def test_ai_summaries_empty_keeps_columns():
    df = ai_summaries_to_dataframe([])
    assert len(df) == 0
    assert list(df.columns) == list(AI_SUMMARY_COLUMNS)
    assert str(df["generation_timestamp"].dt.tz) == "UTC"


def test_get_ai_summaries_one_row_per_requested_id(client, monkeypatch):
    seen = []

    def fake_get(*, sub_url, **kwargs):
        seen.append(sub_url)
        if sub_url.endswith(DID_A):
            return {
                "summary": {
                    "id": "cccccccc-0000-0000-0000-000000000000",
                    "content": "ok",
                    "structured_content": {},
                    "generation_model": "claude-opus-5",
                    "generation_timestamp": "2026-08-17T03:23:05.764993",
                    "user_feedback": {"thumbs": None},
                },
                "task_status": "available",
                "workflow_id": "wf-1",
            }
        return {"summary": None, "task_status": "not_found", "workflow_id": None}

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_ai_summaries([DID_A, DID_B])

    assert sorted(seen) == sorted([f"summary/{DID_A}", f"summary/{DID_B}"])
    # Request order is preserved regardless of which thread finished first.
    assert df["data_id"].tolist() == [DID_A, DID_B]
    assert df["task_status"].tolist() == ["available", "not_found"]


def test_get_ai_summaries_accepts_a_single_id(client, monkeypatch):
    monkeypatch.setattr(
        client,
        "_get",
        lambda **kwargs: {
            "summary": None,
            "task_status": "pending",
            "workflow_id": "w",
        },
    )

    df = client.get_ai_summaries(DID_A)
    assert isinstance(df, DataFrame)
    assert df["data_id"].tolist() == [DID_A]
    assert df["task_status"].tolist() == ["pending"]
