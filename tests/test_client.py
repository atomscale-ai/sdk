from pandas import DataFrame
import pytest
from atomicds import Client
from datetime import datetime
from pathlib import Path
from unittest import mock
from urllib.parse import urljoin
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
            "Sample Name",
            "Growth Length",
            "Tags",
            "Owner",
            "Physical Sample ID",
            "Physical Sample Name",
            "Sample Notes",
            "Sample Notes Last Updated",
            "Workspaces",
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
    data_types = ["rheed_image", "rheed_stationary", "rheed_rotating", "xps"]
    data_ids = []

    for data_type in data_types:
        data = client.search(data_type=data_type, include_organization_data=False)  # type: ignore
        data_id = data["Data ID"].values[0] if len(data["Data ID"].values) else None
        data_ids.append(data_id)

        setattr(ResultIDs, data_type, data_id)

    results = client.get(data_ids=data_ids)
    data_types = set([type(result) for result in results])

    assert len(data_types) == 3


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
