from datetime import datetime
from pathlib import Path
from unittest import mock
from urllib.parse import urljoin

import pytest
from pandas import DataFrame

from atomicds import Client
from atomicds.results import UnknownResult
from atomicds.core import ClientError
from .conftest import ResultIDs


@pytest.fixture
def client():
    return Client()


def test_no_api_key():
    with pytest.raises(ValueError, match="No valid ADS API key supplied"):
        with mock.patch("os.environ.get", return_value=None):
            Client(api_key=None)


def test_generic_search(client: Client):
    orig_data = client.search()
    assert isinstance(orig_data, DataFrame)
    column_names = set(
        [
            "Data ID",
            "Upload Datetime",
            "Last Accessed Datetime",
            "File Metadata",
            "Type",
            "File Name",
            "Status",
            "File Type",
            "Instrument Source",
            "Growth Length",
            "Tags",
            "Owner",
            "Physical Sample ID",
            "Physical Sample Name",
            "Sample Name",
            "Sample Notes",
            "Sample Notes Last Updated",
            "Project ID",
            "Project Name",
        ]
    )
    assert not len(set(orig_data.keys().values) - column_names)


def test_keyword_search(client: Client):
    data = client.search(keywords=".vms")
    assert len(data["Data ID"].values)


def test_include_org_search(client: Client):
    data = client.search(include_organization_data=False)
    assert len(data["Data ID"].values)


def test_data_ids_search(client: Client):
    user_data = client.search(include_organization_data=False)
    data_ids = list(user_data["Data ID"].values)
    data = client.search(data_ids=data_ids)
    assert len(data["Data ID"].values) == len(data_ids)

    data = client.search(data_ids=data_ids[0])
    assert data["Data ID"].values[0] == data_ids[0]


def test_data_type_search(client: Client):
    data_types = ["rheed_image", "rheed_stationary", "rheed_rotating", "xps", "all"]
    for data_type in data_types:
        data = client.search(data_type=data_type)  # type: ignore
        assert len(data["Type"].values)


def test_status_search(client: Client):
    status_values = ["success", "all"]
    for status in status_values:
        data = client.search(status=status)  # type: ignore
        assert len(data["Status"].values)


def test_growth_length_search(client: Client):
    data = client.search(growth_length=(1, None))
    assert len(data["Growth Length"].values)

    data = client.search(growth_length=(None, 1000))
    assert len(data["Growth Length"].values)


def test_upload_datetime_search(client: Client):
    data = client.search(upload_datetime=(None, datetime.utcnow()))
    assert len(data["Upload Datetime"].values)


def test_last_accessed_datetime_search(client: Client):
    data = client.search(last_accessed_datetime=(None, datetime.utcnow()))
    assert len(data["Last Accessed Datetime"].values)


@pytest.mark.order(1)
def test_get(client: Client):
    data_type_aliases = {
        "rheed_image": ["rheed_image"],
        "rheed_stationary": ["rheed_stationary"],
        "rheed_rotating": ["rheed_rotating"],
        "xps": ["xps"],
        "optical": ["optical"],
        "metrology": ["metrology", "recipe"],
        "photoluminescence": ["photoluminescence", "pl"],
        "raman": ["raman"],
    }
    required_types = {
        "rheed_image",
        "rheed_stationary",
        "rheed_rotating",
        "xps",
        # "photoluminescence",
        # "raman",
    }
    data_ids = []

    for result_attr, aliases in data_type_aliases.items():
        data_id = None
        for alias in aliases:
            for include_org in (False, True):
                try:
                    data = client.search(  # type: ignore[arg-type]
                        data_type=alias, include_organization_data=include_org
                    )

                except ClientError:
                    continue

                data_id_values = data["Data ID"].dropna().values if len(data) else []

                if len(data_id_values):
                    data_id = data_id_values[0]
                    break

            if data_id:
                break

        setattr(ResultIDs, result_attr, data_id or "")
        if data_id:
            data_ids.append(data_id)
        elif result_attr in required_types:
            pytest.fail(f"No data_id found for required data type '{result_attr}'")

    results = client.get(data_ids=data_ids)
    data_types = {type(result) for result in results}
    assert len(results) == len(data_ids)
    assert len(data_types) >= 3


def test_get_unknown_type(monkeypatch):
    client = Client(api_key="key_test", endpoint="http://example.com/")
    catalogue_entry = {
        "data_id": "abc",
        "char_source_type": "unknown_type",
        "raw_name": "mystery.dat",
        "pipeline_status": "success",
    }

    def fake_get(sub_url, params=None):
        assert sub_url == "data_entries/"
        return [catalogue_entry]

    def fake_multi(func, kwargs_list, *args, **kwargs):
        return [func(**kw) for kw in kwargs_list]

    monkeypatch.setattr(client, "_get", fake_get)
    monkeypatch.setattr(client, "_multi_thread", fake_multi)

    results = client.get(data_ids=["abc"])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, UnknownResult)
    assert result.data_type == "unknown_type"
    assert result.catalogue_entry.get("raw_name") == "mystery.dat"


def test_list_physical_samples(client: Client):
    samples = client.list_physical_samples()

    assert isinstance(samples, DataFrame)
    if len(samples):
        assert samples["Physical Sample ID"].notna().any()


def test_list_projects(client: Client):
    projects = client.list_projects()

    assert isinstance(projects, DataFrame)
    if len(projects):
        assert projects["Project ID"].notna().any()


def test_get_physical_sample(client: Client):
    samples = client.list_physical_samples()
    if not len(samples):
        pytest.skip("No physical samples available")

    sample_id = samples["Physical Sample ID"].dropna().iloc[0]
    result = client.get_physical_sample(
        sample_id, include_organization_data=False, align=False
    )

    assert result.physical_sample_id == sample_id
    assert isinstance(result.data_results, list)


def test_get_project(client: Client):
    projects = client.list_projects()
    if not len(projects):
        pytest.skip("No projects available")

    project_id = projects["Project ID"].dropna().iloc[0]
    project = client.get_project(
        project_id, include_organization_data=False, align=False
    )

    assert project.project_id == project_id
    assert hasattr(project, "samples")


def test_upload_rejects_missing_file(tmp_path):
    client = Client(api_key="key_test", endpoint="http://example.com/")
    missing_file = tmp_path / "nope.dat"

    with pytest.raises(ClientError, match="does not exist"):
        client.upload(files=[str(missing_file)])


def test_download_videos_missing_metadata(client: Client, tmp_path):
    with pytest.raises(ClientError, match="No processed data found"):
        client.download_videos(
            data_ids="ffffffff-ffff-ffff-ffff-ffffffffffff", dest_dir=tmp_path
        )


# @pytest.mark.order(2)
# @pytest.mark.dependency(name="upload", dependds=["get"])
# def test_upload(client: Client):
#     test_video = str(Path(__file__).parent.absolute()) + "/data/test_rheed.mp4"
#     client.upload(files=[test_video])
#
#
# @pytest.mark.order(3)
# @pytest.mark.dependency(depends=["upload"])
# def test_download(client: Client):
#     # Get data IDs from uploaded test files
#     data = client.search(keywords=["test_rheed"], include_organization_data=False)
#     assert len(data["Data ID"].values)
#
#     data_ids = list(data["Data ID"].values)
#     client.download_videos(data_ids=data_ids, dest_dir="./")
#
#     # Cleanup downloaded files
#     for data_id in data_ids:
#         file_path = Path("./") / f"{data_id}.mp4"
#         if file_path.exists():
#             file_path.unlink()
#
#     response = client.session.delete(
#         url=urljoin(client.endpoint, "/data_entries"),
#         verify=True,
#         params={"data_ids": data_ids},
#     )
#     assert (
#         response.ok
#     ), f"Failed to delete data entries: {response.status_code} - {response.text}"
