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


# --------------------------------------------------------------------------
# Per-frame mask attachment.
# --------------------------------------------------------------------------


def _ts_df(provider, frame_numbers):
    """Build a small timeseries DataFrame indexed by (Angle, Frame Number)."""
    raw = {
        "series_by_angle": [
            {
                "angle": 0.0,
                "series": [
                    {"frame_number": fn, "specular_intensity": float(fn)}
                    for fn in frame_numbers
                ],
            }
        ]
    }
    return provider.to_dataframe(raw)


def _mask_row(frame_number: int) -> dict:
    return {
        "data_id": "video-1",
        "processed_data_id": "proc-1",
        "frame_number": frame_number,
        "mask_rle": f"rle-{frame_number}",
        "mask_height": 6,
        "mask_width": 5,
    }


def test_attach_frame_masks_sparse_coverage(provider):
    """Masks join on Frame Number; frames without a mask get NA."""
    df = _ts_df(provider, [0, 1, 2, 3])
    out = provider.attach_frame_masks(df, [_mask_row(0), _mask_row(2)])

    assert list(out.index.names) == ["Angle", "Frame Number"]
    assert set(provider.MASK_COLS).issubset(out.columns)
    by_frame = out["mask_rle"].groupby("Frame Number").first()
    assert by_frame[0] == "rle-0"
    assert by_frame[2] == "rle-2"
    assert by_frame[1] != by_frame[1]  # NaN
    assert by_frame[3] != by_frame[3]  # NaN
    # Height/width carried through for the populated frames.
    assert out.xs(0, level="Frame Number")["mask_width"].iloc[0] == 5


def test_attach_frame_masks_empty_adds_all_na_columns(provider):
    """No mask artifact → columns still present, all NA."""
    df = _ts_df(provider, [0, 1])
    out = provider.attach_frame_masks(df, [])

    assert set(provider.MASK_COLS).issubset(out.columns)
    assert out["mask_rle"].isna().all()
    assert list(out.index.names) == ["Angle", "Frame Number"]


def test_attach_frame_masks_no_frame_axis_passthrough(provider):
    """A DataFrame with no Frame Number axis is returned unchanged."""
    df = DataFrame({"a": [1, 2]})
    out = provider.attach_frame_masks(df, [_mask_row(0)])
    assert "mask_rle" not in out.columns
    assert out.equals(df)


def test_attach_frame_masks_reattach_no_duplicate_columns(provider):
    """Re-attaching replaces mask columns rather than suffixing them."""
    df = _ts_df(provider, [0, 1])
    once = provider.attach_frame_masks(df, [_mask_row(0)])
    twice = provider.attach_frame_masks(once, [_mask_row(0), _mask_row(1)])

    assert list(twice.columns).count("mask_rle") == 1
    assert twice["mask_rle"].groupby("Frame Number").first()[1] == "rle-1"


def test_frame_number_bounds(provider):
    assert provider.frame_number_bounds(_ts_df(provider, [3, 7, 5])) == (3, 7)
    # Empty / no-frame-axis DataFrames yield None so callers can fall back.
    assert provider.frame_number_bounds(DataFrame(None)) is None
    assert provider.frame_number_bounds(DataFrame({"a": [1, 2]})) is None


def test_get_rheed_timeseries_include_masks(client, monkeypatch):
    """include_masks=True fetches masks and merges them into the DataFrame."""
    captured: list[str] = []

    def fake_get(sub_url, params=None, **kwargs):
        captured.append(sub_url)
        if sub_url.endswith("/frame_masks"):
            return [_mask_row(1), _mask_row(2)]
        return {
            "series_by_angle": [
                {
                    "angle": 0.0,
                    "series": [
                        {"frame_number": 1, "specular_intensity": 100.0},
                        {"frame_number": 2, "specular_intensity": 110.0},
                    ],
                }
            ]
        }

    monkeypatch.setattr(client, "_get", fake_get)

    df = client.get_rheed_timeseries("video-1", include_masks=True)

    assert "rheed/timeseries/video-1/" in captured
    assert "rheed/images/video-1/frame_masks" in captured
    assert "mask_rle" in df.columns
    assert df["mask_rle"].groupby("Frame Number").first().to_dict() == {
        1: "rle-1",
        2: "rle-2",
    }


def test_get_rheed_timeseries_include_masks_scopes_to_window(client, monkeypatch):
    """Mask fetch is bounded by the (windowed) series' frame range, not the whole video."""
    mask_params: dict = {}

    def fake_get(sub_url, params=None, **kwargs):
        if sub_url.endswith("/frame_masks"):
            mask_params.update(params or {})
            return [_mask_row(100), _mask_row(102)]
        # A last_n-style window: only frames 100..102 come back from the series.
        return {
            "series_by_angle": [
                {
                    "angle": 0.0,
                    "series": [
                        {"frame_number": fn, "specular_intensity": float(fn)}
                        for fn in (100, 101, 102)
                    ],
                }
            ]
        }

    monkeypatch.setattr(client, "_get", fake_get)

    df = client.get_rheed_timeseries("video-1", include_masks=True, last_n=3)

    # from/to are clamped to the frames the series spans, not 0..(all frames).
    assert mask_params == {"from": 100, "to": 102}
    by_frame = df["mask_rle"].groupby("Frame Number").first()
    assert by_frame[100] == "rle-100"
    assert by_frame[102] == "rle-102"
    assert by_frame[101] != by_frame[101]  # NaN — no mask for that frame


def test_get_rheed_timeseries_without_masks_makes_no_mask_call(client, monkeypatch):
    """Default (include_masks=False) must not hit the frame_masks endpoint."""
    seen: list[str] = []

    def fake_get(sub_url, params=None, **kwargs):
        seen.append(sub_url)
        return {
            "series_by_angle": [
                {"angle": 0.0, "series": [{"frame_number": 1, "spot_count": 3}]}
            ]
        }

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_rheed_timeseries("video-1")

    assert not any(s.endswith("/frame_masks") for s in seen)
    assert "mask_rle" not in df.columns
