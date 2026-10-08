from io import BytesIO

import numpy as np
import pytest
from pandas import DataFrame
from PIL import Image as PILImage
from PIL.Image import Image
from pycocotools import mask as mask_util

from atomscale import Client
from atomscale.results import RHEEDImageResult
from atomscale.results.rheed_image import (
    _get_rheed_image_result,
    _get_rheed_view_still,
    decode_mask_rle,
)
from atomscale.rheed_metadata import rheed_azimuths_to_dataframe
from atomscale.timeseries.registry import get_provider

from .conftest import ResultIDs

# Fully-qualified target for the frame image builder as looked up inside the
# RHEED provider (get_frame delegates to provider.fetch_snapshot, which calls
# _get_rheed_image_result imported into the timeseries.rheed module namespace).
_RHEED_IMAGE_BUILDER = "atomscale.timeseries.rheed._get_rheed_image_result"


@pytest.fixture
def client():
    return Client()


# Number of rheed_image catalogue entries to sample when looking for one with a
# populated fingerprint. Some entries legitimately have empty fingerprints, so we
# search up to this many before giving up.
_SAMPLE_LIMIT = 15


def _has_pattern_data(res: RHEEDImageResult | None) -> bool:
    return (
        res is not None
        and res.pattern_graph is not None
        and res.pattern_graph.number_of_nodes() > 0
    )


@pytest.fixture
def result(client: Client):
    # Honour a pinned id if one is configured, otherwise discover candidates.
    if ResultIDs.rheed_image:
        data_ids = [ResultIDs.rheed_image]
    else:
        catalogue = client.search(data_type="rheed_image", status="success")
        data_ids = (
            catalogue["Data ID"].tolist()[:_SAMPLE_LIMIT]
            if "Data ID" in catalogue.columns
            else []
        )

    if not data_ids:
        pytest.skip("No successful rheed_image entries available to test against.")

    # Sample entries until we find one with a populated fingerprint. Empty
    # fingerprints are fine to skip over.
    for data_id in data_ids:
        res = client.get(data_ids=data_id)[0]
        if _has_pattern_data(res):
            return res

    pytest.skip(
        f"No rheed_image entry with a populated fingerprint found in the first "
        f"{len(data_ids)} sampled entries."
    )


def test_get_plot(result: RHEEDImageResult):
    plot = result.get_plot()
    assert isinstance(plot, Image)


def test_get_laue(result: RHEEDImageResult):
    radius, (x, y) = result.get_laue_zero_radius()
    assert isinstance(radius, float)
    assert isinstance(x, float)
    assert isinstance(y, float)


def test_get_dataframe(result: RHEEDImageResult):
    cols = {
        "node_id",
        "centroid_0",
        "centroid_1",
        "specular_origin_0",
        "specular_origin_1",
        "relative_centroid_0",
        "relative_centroid_1",
        "intensity_centroid_0",
        "intensity_centroid_1",
        "relative_intensity_centroid_0",
        "relative_intensity_centroid_1",
        "area",
        "fwhm_0",
        "fwhm_1",
        "mask_rle",
        "bbox_maxc",
        "bbox_maxr",
        "bbox_minc",
        "bbox_minr",
        "distances",
        "spot_area",
        "mask_width",
        "pattern_id",
        "mask_height",
        "streak_area",
        "eccentricity",
        "bbox_intensity",
        "center_distance",
        "roughness_metric",
        "axis_major_length",
        "axis_minor_length",
        "skew_0",
        "kurtosis_0",
        "n_peaks",
        "intensity_axis_major_length",
        "intensity_axis_minor_length",
        "intensity_orientation",
        "participation_ratio",
        "peak_snr",
        "data_id",
        "test",
    }
    df = result.get_pattern_dataframe(extra_data={"test": "test"})

    assert isinstance(df, DataFrame)
    assert not len(set(df.columns) - cols)

    df = result.get_pattern_dataframe(symmetrize=True)
    assert isinstance(df, DataFrame)

    df = result.get_pattern_dataframe(return_as_features=True)
    assert isinstance(df, DataFrame)
    assert len(df) == 1


# --------------------------------------------------------------------------
# Unit tests (no live API) for the missing-mask bugfix and get_frame.
# --------------------------------------------------------------------------


def _png_bytes() -> bytes:
    buf = BytesIO()
    PILImage.new("RGB", (4, 4), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_get_rheed_image_result_handles_missing_mask(monkeypatch):
    """A 404 on the mask endpoint (→ None) must not raise (regression)."""
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    png = _png_bytes()

    def fake_get(sub_url, params=None, deserialize=True, base_override=None):
        if base_override is not None:  # S3 image byte fetch
            return png
        if sub_url.endswith("/fingerprint"):
            return None
        if sub_url.endswith("/mask"):
            return None  # <- previously triggered AttributeError on None.get()
        if sub_url.startswith("data_entries/processed_data/"):
            return {"url": "http://s3.example.com/image.png"}
        return None

    monkeypatch.setattr(unit_client, "_get", fake_get)

    result = _get_rheed_image_result(unit_client, "frame-uuid")

    assert isinstance(result, RHEEDImageResult)
    assert result.mask is None
    assert isinstance(result.processed_image, Image)


def _answer_by_url(monkeypatch, unit_client, routes):
    """Answer ``_get`` per URL as the backend would; any other URL is a 404 (None)."""
    calls = []

    def fake_get(sub_url, params=None, **kwargs):
        calls.append((sub_url, params))
        answer = routes.get(sub_url)
        return answer(params) if callable(answer) else answer

    monkeypatch.setattr(unit_client, "_get", fake_get)
    return calls


def _listing(*frames):
    return {"data_entries/video_single_frames/video-1": {"frames": list(frames)}}


def _two_frame_client(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    _answer_by_url(
        monkeypatch,
        unit_client,
        _listing(
            {"image_uuid": "img-0", "timestamp_seconds": 1.0},
            {"image_uuid": "img-1", "timestamp_seconds": 2.0},
        ),
    )
    return unit_client


def test_get_frame_returns_image_result(monkeypatch):
    unit_client = _two_frame_client(monkeypatch)
    sentinel = object()
    captured: dict = {}

    def fake_image_result(client=None, data_id=None, metadata=None):
        captured["data_id"] = data_id
        captured["metadata"] = metadata
        return sentinel

    monkeypatch.setattr(_RHEED_IMAGE_BUILDER, fake_image_result)

    assert unit_client.get_frame("video-1", frame_index=1) is sentinel
    assert captured["data_id"] == "img-1"
    assert captured["metadata"] == {"timestamp_seconds": 2.0}


def test_get_frame_negative_index_resolves_last_frame(monkeypatch):
    unit_client = _two_frame_client(monkeypatch)
    captured: dict = {}

    def fake_image_result(client=None, data_id=None, metadata=None):
        captured["data_id"] = data_id
        return object()

    monkeypatch.setattr(_RHEED_IMAGE_BUILDER, fake_image_result)

    unit_client.get_frame("video-1", frame_index=-1)
    assert captured["data_id"] == "img-1"


def test_get_frame_missing_image_uuid_returns_none(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    _answer_by_url(
        monkeypatch, unit_client, _listing({"timestamp_seconds": 1.0})
    )  # no image_uuid
    called = {"n": 0}

    def fake_image_result(**kwargs):
        called["n"] += 1
        return object()

    monkeypatch.setattr(_RHEED_IMAGE_BUILDER, fake_image_result)

    assert unit_client.get_frame("video-1", frame_index=0) is None
    assert called["n"] == 0  # builder must not be invoked for an image-less frame


def test_get_frame_out_of_range_returns_none(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    _answer_by_url(monkeypatch, unit_client, _listing({"image_uuid": "img-0"}))
    assert unit_client.get_frame("video-1", frame_index=5) is None


def test_get_frame_no_frames_returns_none(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    monkeypatch.setattr(unit_client, "_get", lambda *a, **k: None)
    assert unit_client.get_frame("video-1") is None


def _view(seed_frame, view_id):
    return {
        "data_id": "video-1",
        "view_id": view_id,
        "interval_id": "interval-1",
        "seed_frame": seed_frame,
        "azimuth_label": f"label-{view_id}",
    }


def _png(height=6, width=8):
    buffer = BytesIO()
    PILImage.new("L", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def _crop_mask_row(frame_number):
    """A mask stored for a 2x3 crop that sits at row 1, column 2 of a 6x8 frame."""
    crop = np.ones((2, 3), dtype=np.uint8)
    counts = mask_util.encode(np.asfortranarray(crop))["counts"].decode("utf-8")
    return {
        "processed_data_id": "proc-1",
        "frame_number": frame_number,
        "mask_rle": counts,
        "mask_height": 2,
        "mask_width": 3,
        "mask_origin_row": 1,
        "mask_origin_col": 2,
        "frame_height": 6,
        "frame_width": 8,
    }


def _still_routes(frame_number):
    return {
        f"rheed/images/video-1/frames/{frame_number}": _png(),
        "rheed/images/video-1/frame_masks": [_crop_mask_row(frame_number)],
    }


def test_an_older_analysis_keeps_its_extracted_seed_frames():
    views = rheed_azimuths_to_dataframe([_view(10, "a"), _view(20, "b")])
    listing = {
        "frames": [
            {"image_uuid": "seed-20", "frame_number": 20},
            {"image_uuid": "seed-10", "frame_number": 10},
        ]
    }

    requests = get_provider("rheed").snapshot_requests("video-1", listing, views)

    assert [req.get("image_uuid") for req in requests] == ["seed-10", "seed-20"]


def test_a_newer_analysis_reads_each_views_frame_from_the_video():
    views = rheed_azimuths_to_dataframe([_view(10, "a"), _view(20, "b")])
    listing = {"frames": [{"image_uuid": "saved-15", "frame_number": 15}]}

    requests = get_provider("rheed").snapshot_requests("video-1", listing, views)

    assert [req["metadata"]["frame_number"] for req in requests] == [10, 15, 20]
    assert [req.get("image_uuid") for req in requests] == [None, "saved-15", None]
    assert requests[0]["metadata"]["view_id"] == "a"
    assert requests[2]["metadata"]["azimuth_label"] == "label-b"


def test_a_view_still_carries_its_mask_on_the_full_frame(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    _answer_by_url(monkeypatch, unit_client, _still_routes(10))

    still = _get_rheed_view_still(unit_client, "video-1", 10, {"view_id": "a"})

    assert isinstance(still, RHEEDImageResult)
    assert still.processed_image.size == (8, 6)
    assert still.pattern_graph is None
    assert still.metadata == {"view_id": "a", "frame_number": 10}
    expected = np.zeros((6, 8), dtype=np.uint8)
    expected[1:3, 2:5] = 1
    assert np.array_equal(still.mask, expected)


def test_a_frame_without_a_kept_image_has_no_still(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    _answer_by_url(monkeypatch, unit_client, {})

    assert _get_rheed_view_still(unit_client, "video-1", 10) is None


def test_get_frame_reads_a_newer_analysis_views_from_the_video(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    _answer_by_url(
        monkeypatch,
        unit_client,
        {
            **_listing(),
            "rheed/video-1/azimuths": [_view(10, "a")],
            **_still_routes(10),
        },
    )

    frame = unit_client.get_frame("video-1", frame_index=0)

    assert isinstance(frame, RHEEDImageResult)
    assert frame.metadata["view_id"] == "a"
    assert frame.metadata["frame_number"] == 10


# --------------------------------------------------------------------------
# Unit tests (no live API) for per-frame mask fetching / decoding.
# --------------------------------------------------------------------------


def _mask_row(frame_number: int, height: int = 6, width: int = 5) -> dict:
    """Build a frame-mask row with a real COCO RLE counts string (as JSON str)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    # A small filled block so the decoded mask is non-trivial and frame-specific.
    mask[1 : 1 + (frame_number % height or 1), 0:2] = 1
    counts = mask_util.encode(np.asfortranarray(mask))["counts"].decode("utf-8")
    return {
        "data_id": "video-1",
        "processed_data_id": "proc-1",
        "frame_number": frame_number,
        "mask_rle": counts,
        "mask_height": height,
        "mask_width": width,
    }


def test_decode_mask_rle_roundtrips():
    row = _mask_row(3)
    mask = decode_mask_rle(row["mask_rle"], row["mask_height"], row["mask_width"])
    assert mask.shape == (row["mask_height"], row["mask_width"])
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask.sum() > 0


# Two views whose last analysed samples are frames 3 and 5.
_LAST_SAMPLES = {
    "rheed/timeseries/video-1/": {
        "series_by_angle": [
            {"series": [{"frame_number": 3}]},
            {"series": [{"frame_number": 5}]},
        ]
    }
}


def test_get_frame_masks_returns_raw_rows(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    rows = [_mask_row(0), _mask_row(5)]
    calls = _answer_by_url(
        monkeypatch,
        unit_client,
        {**_LAST_SAMPLES, "rheed/images/video-1/frame_masks": rows},
    )

    out = unit_client.get_frame_masks("video-1")

    assert out == rows  # raw rows passed through untouched
    # to_frame=None → through the last analysed frame of any view
    assert calls == [
        ("rheed/timeseries/video-1/", {"last_n": 1}),
        ("rheed/images/video-1/frame_masks", {"from": 0, "to": 5}),
    ]


def test_get_frame_masks_fetches_long_spans_window_by_window(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    calls = _answer_by_url(
        monkeypatch,
        unit_client,
        {
            "rheed/images/video-1/frame_masks": lambda params: [
                _mask_row(params["from"])
            ]
        },
    )

    out = unit_client.get_frame_masks("video-1", from_frame=0, to_frame=1200)

    # The server serves at most 500 frames per request.
    assert sorted(params["from"] for _, params in calls) == [0, 500, 1000]
    assert sorted(params["to"] for _, params in calls) == [499, 999, 1200]
    assert [row["frame_number"] for row in out] == [0, 500, 1000]


def test_get_frame_masks_explicit_range(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    captured: dict = {}

    def fake_get(sub_url, params=None, **kwargs):  # noqa: ARG001
        captured["params"] = params
        return []

    monkeypatch.setattr(unit_client, "_get", fake_get)

    unit_client.get_frame_masks("video-1", from_frame=10, to_frame=200)
    assert captured["params"] == {"from": 10, "to": 200}


def test_get_frame_masks_decode(monkeypatch):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    rows = [_mask_row(0), _mask_row(7)]
    _answer_by_url(
        monkeypatch,
        unit_client,
        {
            "rheed/timeseries/video-1/": {
                "series_by_angle": [{"series": [{"frame_number": 7}]}]
            },
            "rheed/images/video-1/frame_masks": rows,
        },
    )

    masks = unit_client.get_frame_masks("video-1", decode=True)

    assert isinstance(masks, dict)
    assert set(masks) == {0, 7}
    for row in rows:
        arr = masks[row["frame_number"]]
        assert arr.shape == (row["mask_height"], row["mask_width"])
        expected = decode_mask_rle(
            row["mask_rle"], row["mask_height"], row["mask_width"]
        )
        assert np.array_equal(arr, expected)


def test_get_frame_masks_no_artifact_returns_empty(monkeypatch):
    """A 404 on the frame-mask endpoint (→ _get returns None) yields empty."""
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    monkeypatch.setattr(unit_client, "_get", lambda *a, **k: None)

    assert unit_client.get_frame_masks("video-1") == []
    assert unit_client.get_frame_masks("video-1", decode=True) == {}


@pytest.mark.parametrize(
    ("from_frame", "to_frame"),
    [(-1, None), (0, -5), (10, 5)],
)
def test_get_frame_masks_invalid_range_raises(monkeypatch, from_frame, to_frame):
    unit_client = Client(api_key="key_test", endpoint="http://example.com/")
    # _get must never be reached when validation fails.
    monkeypatch.setattr(
        unit_client,
        "_get",
        lambda *a, **k: pytest.fail("_get should not be called on invalid range"),
    )
    with pytest.raises(ValueError):  # noqa: PT011
        unit_client.get_frame_masks("video-1", from_frame=from_frame, to_frame=to_frame)
