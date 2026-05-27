import pytest
from pandas import DataFrame
from PIL.Image import Image

from atomscale import Client
from atomscale.results import RHEEDImageResult

from .conftest import ResultIDs


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
