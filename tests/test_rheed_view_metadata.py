"""Unified view labels keep stationary and rotating identities distinct."""

import pytest
from pandas import DataFrame

from atomscale.results import RHEEDVideoResult
from atomscale.rheed_metadata import (
    RHEED_AZIMUTH_COLUMNS,
    legacy_views_frame,
    rheed_azimuths_to_dataframe,
)
from atomscale.timeseries.rheed import RHEEDProvider


def test_repeated_labels_retain_view_identity():
    views = [
        {
            "data_id": "recording",
            "processed_data_id": "derivative",
            "view_id": f"v{i}",
            "interval_id": f"i{i}",
            "seed_frame": i * 10,
            "start_frame": i * 10,
            "end_frame": (i + 1) * 10,
            "rpm": 0 if i == 0 else 12,
            "azimuth_label": "100",
            "azimuth_source": "user",
            "confidence": None,
            "annotation": {},
        }
        for i in range(2)
    ]
    frame = rheed_azimuths_to_dataframe(views)
    assert frame.view_id.tolist() == ["v0", "v1"]
    assert frame.azimuth_label.tolist() == ["100", "100"]
    assert frame.rpm.tolist() == [0, 12]
    payload = {
        "series_by_angle": [
            {
                "view_id": v["view_id"],
                "interval_id": v["interval_id"],
                "angle": "100",
                "azimuth_label": "100",
                "series": [{"frame_number": 0, "specular_intensity": 1}],
            }
            for v in views
        ]
    }
    samples = RHEEDProvider().to_dataframe(payload)
    assert samples.index.is_unique
    assert samples.index.names == ["View ID", "Frame Number"]


def test_legacy_stationary_maps_to_rheed_at_zero_rpm():
    """A pre-unification ``rheed_stationary`` entry is a parked stage: rpm 0."""
    views = legacy_views_frame("rheed_stationary")

    assert len(views) == 1
    assert views["rpm"].tolist() == [0.0]
    # No view identity is invented for a backend that has none to give.
    assert views["view_id"].isna().all()


def test_legacy_rotating_reports_rotation_without_inventing_a_rate():
    views = legacy_views_frame("rheed_rotating")

    assert len(views) == 1
    assert views["rpm"].isna().all()


def test_unified_type_has_no_legacy_placeholder():
    """``rheed`` carries its rotation in the stored views, so there is nothing to infer."""
    assert legacy_views_frame("rheed").empty


@pytest.mark.parametrize(
    ("data_type", "expected"),
    [("rheed_stationary", False), ("rheed_rotating", True)],
)
def test_result_rotating_follows_legacy_type_on_an_older_backend(data_type, expected):
    result = RHEEDVideoResult(
        data_id="d-1",
        timeseries_data=DataFrame(),
        snapshot_image_data=None,
        views=legacy_views_frame(data_type),
    )
    assert result.rotating is expected


@pytest.mark.parametrize(("rpm", "expected"), [(0.0, False), (5.0, True)])
def test_result_rotating_follows_stored_rpm_on_the_unified_backend(rpm, expected):
    row = dict.fromkeys(RHEED_AZIMUTH_COLUMNS)
    row["rpm"] = rpm
    result = RHEEDVideoResult(
        data_id="d-1",
        timeseries_data=DataFrame(),
        snapshot_image_data=None,
        views=DataFrame([row]),
    )
    assert result.rotating is expected
