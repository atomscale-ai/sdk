"""Deterministic client coverage for current and legacy RHEED image responses."""

from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pycocotools import mask as mask_util

from atomscale import Client
from atomscale.results.rheed_image import (
    RHEEDImageCollection,
    _get_rheed_image_result,
)


def _node(node_id, column):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5:8, column - 1 : column + 2] = 1
    return {
        "node_id": node_id,
        "centroid_0": 6.0,
        "centroid_1": float(column),
        "specular_origin_0": 6.0,
        "specular_origin_1": 8.0,
        "relative_centroid_0": 0.0,
        "relative_centroid_1": float(column - 8),
        "intensity_centroid_0": None,
        "intensity_centroid_1": None,
        "relative_intensity_centroid_0": None,
        "relative_intensity_centroid_1": None,
        "area": 9.0,
        "fwhm_0": None,
        "fwhm_1": None,
        "mask_rle": mask_util.encode(np.asfortranarray(mask))["counts"].decode(),
        "mask_height": 16,
        "mask_width": 16,
        "bbox_minr": 5,
        "bbox_maxr": 8,
        "bbox_minc": column - 1,
        "bbox_maxc": column + 2,
        "spot_area": 3,
        "streak_area": 6,
        "pattern_id": 0,
        "eccentricity": 0.0,
        "bbox_intensity": None,
        "center_distance": abs(column - 8),
        "roughness_metric": None,
        "axis_major_length": 3.0,
        "axis_minor_length": 3.0,
        "distances": None,
        "skew_0": None,
        "kurtosis_0": None,
        "n_peaks": None,
        "intensity_axis_major_length": None,
        "intensity_axis_minor_length": None,
        "intensity_orientation": None,
        "participation_ratio": None,
        "peak_snr": None,
        "coherent_intensity": None if node_id == 0 else 0.2,
        "diffuse_intensity": None if node_id == 0 else 0.05,
        "order_label": "specular" if node_id == 0 else "1st",
        "slot_generation": 2,
    }


@pytest.fixture
def load_result(monkeypatch):
    def load(*, legacy=False, empty=False, data_id="image-1"):
        nodes = [] if empty else [_node(0, 8), _node(4, 12)]
        if legacy:
            for node in nodes:
                for field in (
                    "coherent_intensity",
                    "diffuse_intensity",
                    "order_label",
                    "slot_generation",
                ):
                    node.pop(field)
        mask = np.zeros((16, 16), dtype=np.uint8)
        for node in nodes:
            mask |= mask_util.decode({"counts": node["mask_rle"], "size": [16, 16]})
        png = BytesIO()
        Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(
            png, format="PNG"
        )
        metadata = {
            "region_feature_version": 18,
            "image_features": {"feature_pitch": 4.0, "coherent_fraction": None},
        }
        payload = {
            "fingerprint": {
                "nodes": nodes,
                "horizontal_distances": []
                if empty
                else [
                    {
                        "start_id": 0,
                        "end_id": 4,
                        "start_x": 8.0,
                        "end_x": 12.0,
                        "weight": 0.4,
                    }
                ],
            },
            "processed_data_id": "processed-1",
        }
        client = Client(api_key="key_test", endpoint="http://example.com/")

        def get(sub_url, params=None, base_override=None, **kwargs):
            if base_override:
                return png.getvalue()
            if sub_url.endswith("/fingerprint"):
                return payload
            if sub_url.endswith("/mask"):
                return {
                    "mask_rle": mask_util.encode(np.asfortranarray(mask))[
                        "counts"
                    ].decode(),
                    "mask_height": 16,
                    "mask_width": 16,
                }
            if sub_url.startswith("data_entries/processed_data/"):
                return {"url": "http://example.com/image.png", "metadata": metadata}
            raise AssertionError(sub_url)

        monkeypatch.setattr(client, "_get", get)
        return _get_rheed_image_result(client, data_id, {"view_id": "view-1"})

    return load


def test_current_result_preserves_sparse_slots_and_does_not_create_phantom_nodes(
    load_result,
):
    result = load_result()
    assert set(result.pattern_graph) == {0, 4}
    assert set(result.pattern_graph.edges()) == {(0, 4)}
    assert result.pattern_graph.nodes[4]["node_id"] == 4
    assert result.pattern_graph.nodes[4]["order_label"] == "1st"
    assert result.pattern_graph.nodes[4]["slot_generation"] == 2
    assert result.pattern_graph.nodes[4]["coherent_intensity"] == 0.2
    assert result.processed_data_id == "processed-1"


def test_current_result_carries_feature_version_and_frame_measurements(load_result):
    result = load_result()
    assert result.metadata["region_feature_version"] == 18
    assert result.metadata["image_features"] == {
        "feature_pitch": 4.0,
        "coherent_fraction": None,
    }
    assert result.metadata["view_id"] == "view-1"


def test_wide_table_keeps_unavailable_channels_and_current_measurements(load_result):
    result = load_result()
    df = result.get_pattern_dataframe(return_as_features=True)
    assert df.shape[0] == 1
    assert pd.isna(df[("bbox_intensity", 0)].iloc[0])
    assert pd.isna(df[("fwhm_0", 4)].iloc[0])
    assert df[("coherent_intensity", 4)].iloc[0] == 0.2
    assert df[("diffuse_intensity", 4)].iloc[0] == 0.05
    assert set(df.columns.get_level_values(1)) == {0, 4}


def test_raw_and_symmetrized_rows_preserve_current_features(load_result):
    result = load_result()
    for symmetrize in (False, True):
        df = result.get_pattern_dataframe(symmetrize=symmetrize)
        assert {
            "coherent_intensity",
            "diffuse_intensity",
            "order_label",
            "slot_generation",
        } <= set(df)
        assert "1st" in df.order_label.to_list()
        assert df.coherent_intensity.notna().any()


def test_legacy_result_still_builds_numeric_features(load_result):
    result = load_result(legacy=True)
    df = result.get_pattern_dataframe(return_as_features=True)
    assert df.shape[0] == 1
    assert df["coherent_intensity"].isna().all().all()
    assert set(result.pattern_graph) == {0, 4}


def test_collection_keeps_new_channels_and_handles_unavailable_measurements(
    load_result,
):
    collection = RHEEDImageCollection([load_result(), load_result(data_id="image-2")])
    df = collection.get_pattern_dataframe(
        normalize=False, streamline=False, symmetrize=False
    )
    assert len(df) == 2
    assert (df[("coherent_intensity", 4)] == 0.2).all()
    normalized = collection.get_pattern_dataframe()
    assert len(normalized) == 2


def test_empty_detection_remains_plottable_and_reports_no_dataframe(load_result):
    result = load_result(empty=True)
    assert result.pattern_graph.number_of_nodes() == 0
    assert not result.mask.any()
    assert isinstance(result.get_plot(), Image.Image)
    with pytest.raises(ValueError, match="empty fingerprint"):
        result.get_pattern_dataframe()


def test_laue_plot_falls_back_to_geometry_when_intensities_are_unavailable(load_result):
    radius, center = load_result().get_laue_zero_radius()
    assert np.isfinite(radius)
    assert np.isfinite(center).all()
