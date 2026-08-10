"""Unit tests for Client.get_rheed_timeseries and low-level feature flattening."""

import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.timeseries.rheed import RHEEDProvider


@pytest.fixture
def client():
    return Client(api_key="key_test", endpoint="http://example.com/")


@pytest.fixture
def provider():
    return RHEEDProvider()


def test_to_dataframe_flattens_low_level_features(provider):
    raw = {
        "series_by_angle": [
            {
                "angle": 0.0,
                "series": [
                    {
                        "frame_number": 1,
                        "specular_intensity": 100.0,
                        "low_level_features": {"feat_a": 1.1, "feat_b": 2.2},
                    },
                    {
                        "frame_number": 2,
                        "specular_intensity": 110.0,
                        "low_level_features": {"feat_a": 3.3, "feat_b": 4.4},
                    },
                ],
            }
        ]
    }

    df = provider.to_dataframe(raw)

    assert "low_level_features" not in df.columns
    assert "feat_a" in df.columns
    assert "feat_b" in df.columns
    assert df["feat_a"].tolist() == [1.1, 3.3]
    assert df["feat_b"].tolist() == [2.2, 4.4]
    # Known metric still renamed via RENAME_MAP.
    assert "Specular Intensity" in df.columns


def test_to_dataframe_flattens_nested_low_level_features(provider):
    """Nested low-level feature dicts flatten with dotted-path column names."""
    raw = {
        "series_by_angle": [
            {
                "angle": 0.0,
                "series": [
                    {
                        "frame_number": 1,
                        "low_level_features": {"roi": {"x": 1.0, "y": 2.0}},
                    }
                ],
            }
        ]
    }

    df = provider.to_dataframe(raw)

    assert "roi.x" in df.columns
    assert "roi.y" in df.columns
    assert "roi" not in df.columns
    assert df["roi.x"].tolist() == [1.0]
    assert df["roi.y"].tolist() == [2.0]


def test_to_dataframe_low_level_features_missing_points(provider):
    """Points lacking the mapping contribute NA for those columns."""
    raw = {
        "series_by_angle": [
            {
                "angle": 0.0,
                "series": [
                    {"frame_number": 1, "low_level_features": {"feat_a": 1.0}},
                    {"frame_number": 2, "low_level_features": {}},
                ],
            }
        ]
    }

    df = provider.to_dataframe(raw)

    assert "feat_a" in df.columns
    values = df["feat_a"].tolist()
    assert values[0] == 1.0
    assert values[1] != values[1]  # NaN


def test_to_dataframe_without_low_level_features_unchanged(provider):
    raw = {
        "series_by_angle": [
            {"angle": 0.0, "series": [{"frame_number": 1, "specular_intensity": 100.0}]}
        ]
    }

    df = provider.to_dataframe(raw)

    assert "low_level_features" not in df.columns
    assert "Specular Intensity" in df.columns


def test_low_level_feature_does_not_clobber_known_column(provider):
    """A low-level feature colliding with a top-level key must not overwrite it."""
    raw = {
        "series_by_angle": [
            {
                "angle": 0.0,
                "series": [
                    {
                        "frame_number": 1,
                        "specular_intensity": 100.0,
                        "low_level_features": {"specular_intensity": 999.0},
                    }
                ],
            }
        ]
    }

    df = provider.to_dataframe(raw)

    # Top-level value wins; renamed to "Specular Intensity".
    assert df["Specular Intensity"].tolist() == [100.0]


def test_get_rheed_timeseries_forwards_params(client, monkeypatch):
    captured: dict = {}

    def fake_get(sub_url, params=None, **kwargs):
        captured["sub_url"] = sub_url
        captured["params"] = params
        return {
            "series_by_angle": [
                {
                    "angle": 0.0,
                    "series": [
                        {"frame_number": 1, "low_level_features": {"raw_x": 5.0}}
                    ],
                }
            ]
        }

    monkeypatch.setattr(client, "_get", fake_get)

    df = client.get_rheed_timeseries(
        "video-1",
        property_names=["specular_intensity", "spot_count"],
        include_low_level_features=True,
        last_n=50,
        elapsed_seconds=30.0,
    )

    assert captured["sub_url"] == "rheed/timeseries/video-1/"
    assert captured["params"] == {
        "include_low_level_features": True,
        "property_names": ["specular_intensity", "spot_count"],
        "last_n": 50,
        "elapsed_seconds": 30.0,
    }
    assert isinstance(df, DataFrame)
    assert "raw_x" in df.columns


def test_get_rheed_timeseries_defaults(client, monkeypatch):
    captured: dict = {}

    def fake_get(sub_url, params=None, **kwargs):
        captured["params"] = params
        return None

    monkeypatch.setattr(client, "_get", fake_get)
    client.get_rheed_timeseries("video-1")

    assert captured["params"]["include_low_level_features"] is False
    assert captured["params"]["property_names"] is None
