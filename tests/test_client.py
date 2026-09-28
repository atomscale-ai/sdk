from datetime import datetime
from unittest import mock

import pytest
from pandas import DataFrame

from atomscale import Client
from atomscale.client import _RETRYABLE_STATUSES, _retry_client_call
from atomscale.core import ClientError
from atomscale.results import UnknownResult


def test_no_api_key():
    with pytest.raises(ValueError, match="No valid Atomscale API key supplied"):
        with mock.patch("os.environ.get", return_value=None):
            Client(api_key=None)


def test_generic_search(client: Client):
    orig_data = client.search()
    assert isinstance(orig_data, DataFrame)
    column_names = set(
        [
            "Data ID",
            "Upload Datetime",
            "Last Updated",
            "File Metadata",
            "Type",
            "File Name",
            "Status",
            "File Type",
            "Instrument Source",
            "Growth Length",
            "Tags",
            "Owner",
            "Workspaces",
            "Physical Sample ID",
            "Physical Sample Name",
            "Sample Name",
            "Sample Notes",
            "Sample Notes Last Updated",
            "Project ID",
            "Project Name",
            "sha256",
            "Collected Datetime",
            "Has Instrument Logs",
        ]
    )
    assert set(orig_data.columns) == column_names


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"keywords": ".vms"}, {"keywords": ".vms"}),
        ({"include_organization_data": False}, {"include_organization_data": False}),
        ({"data_ids": "sandbox-xps"}, {"data_ids": "sandbox-xps"}),
        ({"data_ids": ["sandbox-xps"]}, {"data_ids": ["sandbox-xps"]}),
        *[
            (
                {"data_type": kind},
                {"data_type": "rheed" if kind.startswith("rheed_") else kind},
            )
            for kind in ("rheed_image", "rheed_stationary", "rheed_rotating", "xps")
            if kind != "rheed_image"
        ],
        ({"data_type": "rheed_image"}, {"data_type": "rheed_image"}),
        ({"data_type": "all"}, {"data_type": None}),
        ({"status": "success"}, {"status": "success"}),
        ({"status": "all"}, {"status": "all"}),
        (
            {"growth_length": (1, None)},
            {"growth_length_min": 1, "growth_length_max": None},
        ),
        (
            {"growth_length": (None, 1000)},
            {"growth_length_min": None, "growth_length_max": 1000},
        ),
        (
            {"upload_datetime": (None, datetime(2024, 2, 1))},
            {"upload_datetime_min": None, "upload_datetime_max": datetime(2024, 2, 1)},
        ),
        (
            {"last_updated": (None, datetime(2024, 2, 1))},
            {"last_updated_min": None, "last_updated_max": datetime(2024, 2, 1)},
        ),
    ],
)
def test_search_forwards_filters(client, sandbox, kwargs, expected):
    assert isinstance(client.search(**kwargs), DataFrame)
    path, params, _ = sandbox.calls[-1]
    assert path == "data_entries/"
    assert {key: params[key] for key in expected} == expected


def test_search_empty_catalogue(client, sandbox):
    sandbox.catalogue.clear()
    assert client.search().empty


def test_last_accessed_datetime_alias_forwards_to_last_updated(monkeypatch):
    """Deprecated kwarg must warn and forward to last_updated_min/max params."""
    client = Client(api_key="key_test", endpoint="http://example.com/")

    captured: dict = {}

    def fake_get(sub_url, params=None):
        captured["sub_url"] = sub_url
        captured["params"] = params
        return []

    monkeypatch.setattr(client, "_get", fake_get)

    upper = datetime(2026, 1, 2, 3, 4, 5)
    with pytest.warns(DeprecationWarning, match="last_accessed_datetime"):
        client.search(last_accessed_datetime=(None, upper))

    assert captured["params"]["last_updated_min"] is None
    assert captured["params"]["last_updated_max"] == upper
    assert "last_accessed_datetime_min" not in captured["params"]
    assert "last_accessed_datetime_max" not in captured["params"]


def test_metrology_search_alias_uses_tool_state_enum(monkeypatch):
    client = Client(api_key="key_test", endpoint="http://example.com/")
    captured: dict = {}

    def fake_get(sub_url, params=None):
        captured["sub_url"] = sub_url
        captured["params"] = params
        return []

    monkeypatch.setattr(client, "_get", fake_get)

    client.search(data_type="metrology")

    assert captured["sub_url"] == "data_entries/"
    assert captured["params"]["data_type"] == "tool_state"


@pytest.mark.parametrize(
    "kind, expected_type",
    [
        ("rheed", "RHEEDVideoResult"),
        ("xps", "XPSResult"),
        ("xrd", "XRDResult"),
        ("raman", "RamanResult"),
        ("photoluminescence", "PhotoluminescenceResult"),
        ("optical", "OpticalResult"),
        ("tool_state", "ToolStateResult"),
        ("recipe", "RecipeResult"),
        ("ellipsometry", "EllipsometryResult"),
    ],
)
def test_get(client, result_ids, kind, expected_type):
    data_id = getattr(result_ids, kind)
    results = client.get(data_ids=data_id)
    assert len(results) == 1
    assert type(results[0]).__name__ == expected_type
    assert results[0].data_id == data_id


def test_get_preserves_request_order(client, result_ids):
    ids = [result_ids.recipe, result_ids.xps, result_ids.rheed]
    assert [result.data_id for result in client.get(ids)] == ids


def test_get_missing_id(client):
    assert client.get("missing") == []


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
    assert len(samples) == 1

    sample_id = samples["Physical Sample ID"].dropna().iloc[0]
    result = client.get_physical_sample(
        sample_id, include_organization_data=False, align=False
    )

    assert result.physical_sample_id == sample_id
    assert len(result.data_results) == 9


def test_get_project(client: Client):
    projects = client.list_projects()
    assert len(projects) == 1

    project_id = projects["Project ID"].dropna().iloc[0]
    project = client.get_project(
        project_id, include_organization_data=False, align=False
    )

    assert project.project_id == project_id
    assert len(project.samples) == 1
    assert len(project.samples[0].data_results) == 9


def test_upload_rejects_missing_file(tmp_path):
    client = Client(api_key="key_test", endpoint="http://example.com/")
    missing_file = tmp_path / "nope.dat"

    with pytest.raises(ClientError, match="does not exist"):
        client.upload(files=[str(missing_file)])


def test_retry_client_call_retries_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("atomscale.client.time.sleep", lambda d: sleeps.append(d))

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ClientError("boom", status_code=502, response_text="bad gateway")
        return "ok"

    result = _retry_client_call(flaky, attempts=4, base_delay=0.5, max_delay=10.0)

    assert result == "ok"
    assert calls["n"] == 3
    # Two retries fired -> two sleeps with exponential backoff.
    assert sleeps == [0.5, 1.0]


def test_retry_client_call_does_not_retry_non_retryable_status(monkeypatch):
    monkeypatch.setattr("atomscale.client.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ClientError("nope", status_code=400, response_text="bad request")

    with pytest.raises(ClientError) as exc_info:
        _retry_client_call(fn, attempts=4)

    assert exc_info.value.status_code == 400
    assert calls["n"] == 1


def test_retry_client_call_gives_up_after_attempts(monkeypatch):
    monkeypatch.setattr("atomscale.client.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ClientError("still bad", status_code=502)

    with pytest.raises(ClientError) as exc_info:
        _retry_client_call(fn, attempts=3)

    assert exc_info.value.status_code == 502
    assert calls["n"] == 3


def test_retry_client_call_retries_request_exceptions(monkeypatch):
    from requests.exceptions import ConnectionError as RequestsConnectionError

    monkeypatch.setattr("atomscale.client.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RequestsConnectionError("dropped")
        return "recovered"

    assert _retry_client_call(fn, attempts=3) == "recovered"
    assert calls["n"] == 2


def test_retryable_statuses_include_known_transients():
    assert {429, 500, 502, 503, 504}.issubset(_RETRYABLE_STATUSES)


def test_upload_retries_url_fetch_502(tmp_path, monkeypatch):
    """upload() should retry the upload_urls/ POST on a transient 502."""
    monkeypatch.setattr("atomscale.client.time.sleep", lambda *_: None)

    test_file = tmp_path / "small.bin"
    test_file.write_bytes(b"hello world")

    client = Client(api_key="key_test", endpoint="http://example.com/")

    call_log: list[str] = []
    post_calls = {"n": 0}

    def fake_post_or_put(method, sub_url, **kwargs):
        call_log.append(f"{method} {sub_url}")
        if sub_url == "data_entries/raw_data/staged/upload_urls/":
            post_calls["n"] += 1
            if post_calls["n"] == 1:
                raise ClientError(
                    "Problem sending data to data_entries/raw_data/staged/upload_urls/. HTTP Error 502: bad gateway",
                    status_code=502,
                    response_text="bad gateway",
                )
            return [
                {
                    "part": 1,
                    "url": "http://s3.example.com/chunk1",
                    "data_id": "data-123",
                    "new_filename": "renamed.bin",
                    "upload_id": "upload-abc",
                }
            ]
        if sub_url == "data_entries/raw_data/staged/upload_urls/complete/":
            return {}
        # Chunk PUTs land here with sub_url="" and base_override=<s3 url>.
        return {"ETag": "etag-123"}

    def fake_multi_thread(func, kwargs_list, *args, **kwargs):
        return [func(**kw) for kw in kwargs_list]

    monkeypatch.setattr(client, "_post_or_put", fake_post_or_put)
    monkeypatch.setattr(client, "_multi_thread", fake_multi_thread)

    data_ids = client.upload(files=[str(test_file)])

    assert data_ids == ["data-123"]
    assert post_calls["n"] == 2  # one 502, one success
    # Confirm the complete step ran after the chunk PUTs.
    assert call_log[-1] == "POST data_entries/raw_data/staged/upload_urls/complete/"


def test_upload_surfaces_url_fetch_failure_after_retry_exhaustion(
    tmp_path, monkeypatch
):
    """upload() should give up after retries are exhausted on the upload_urls/ POST."""
    monkeypatch.setattr("atomscale.client.time.sleep", lambda *_: None)

    test_file = tmp_path / "small.bin"
    test_file.write_bytes(b"hello world")

    client = Client(api_key="key_test", endpoint="http://example.com/")
    call_count = {"n": 0}

    def always_502(method, sub_url, **kwargs):
        call_count["n"] += 1
        raise ClientError(
            "Problem sending data to data_entries/raw_data/staged/upload_urls/. HTTP Error 502: bad gateway",
            status_code=502,
            response_text="bad gateway",
        )

    monkeypatch.setattr(client, "_post_or_put", always_502)

    with pytest.raises(ClientError) as exc_info:
        client.upload(files=[str(test_file)])

    assert exc_info.value.status_code == 502
    # Default attempts=4 in _retry_client_call.
    assert call_count["n"] == 4


def test_download_videos_missing_metadata(client: Client, tmp_path):
    with pytest.raises(ClientError, match="No processed data found"):
        client.download_videos(
            data_ids="ffffffff-ffff-ffff-ffff-ffffffffffff", dest_dir=tmp_path
        )


def test_search_falls_back_to_legacy_rheed_types_on_an_older_backend():
    """A pre-unification backend rejects "rheed"; the SDK asks for what it knows.

    The catalogue enum used to carry ``rheed_stationary`` and ``rheed_rotating``
    separately. Requests go out under the canonical ``rheed``, and a backend
    still on the old enum answers 422 — so both legacy names are queried and
    their rows concatenated, giving callers the same result from either backend.
    """
    client = Client(api_key="key_test", endpoint="http://example.com/")
    requested: list[str | None] = []

    def fake_get(**kwargs):
        data_type = (kwargs.get("params") or {}).get("data_type")
        requested.append(data_type)
        if data_type == "rheed":
            raise ClientError("enum", status_code=422, response_text="enum")
        return [{"data_id": f"id-{data_type}", "char_source_type": data_type}]

    client._get = fake_get  # type: ignore[method-assign]
    frame = client.search(data_type="rheed_stationary")

    assert requested == ["rheed", "rheed_stationary", "rheed_rotating"]
    assert frame["Type"].tolist() == ["rheed_stationary", "rheed_rotating"]


def test_search_does_not_retry_when_the_backend_understands_rheed():
    client = Client(api_key="key_test", endpoint="http://example.com/")
    requested: list[str | None] = []

    def fake_get(**kwargs):
        requested.append((kwargs.get("params") or {}).get("data_type"))
        return [{"data_id": "id-1", "char_source_type": "rheed"}]

    client._get = fake_get  # type: ignore[method-assign]
    frame = client.search(data_type="rheed_rotating")

    assert requested == ["rheed"]
    assert frame["Type"].tolist() == ["rheed"]


def test_search_propagates_a_422_that_is_not_about_the_rheed_enum():
    client = Client(api_key="key_test", endpoint="http://example.com/")

    def fake_get(**_kwargs):
        raise ClientError("bad param", status_code=422, response_text="bad param")

    client._get = fake_get  # type: ignore[method-assign]
    with pytest.raises(ClientError):
        client.search(data_type="xps")


def test_get_through_http_transport(httpserver, sandbox):
    """Exercise serialization and hydration together against loopback HTTP."""
    entry = next(row for row in sandbox.catalogue if row["char_source_type"] == "xps")
    data_id = entry["data_id"]
    httpserver.expect_request(
        "/data_entries/",
        query_string={"data_ids": data_id, "include_organization_data": "True"},
        headers={"X-API-KEY": "key_test"},
    ).respond_with_json([entry])
    httpserver.expect_request(f"/xps/{data_id}").respond_with_json(
        sandbox.responses[f"xps/{data_id}"]
    )
    client = Client(
        api_key="key_test", endpoint=httpserver.url_for("/"), mute_bars=True
    )
    try:
        results = client.get(data_id)
        assert len(results) == 1
        assert results[0].binding_energies == [1.0, 2.0, 3.0]
        assert results[0].collected_datetime == entry["collected_datetime"]
        httpserver.check_assertions()
    finally:
        client.session.close()


def test_get_chunks_large_requests(client, sandbox):
    template = next(
        row for row in sandbox.catalogue if row["char_source_type"] == "xps"
    )
    payload = sandbox.responses[f"xps/{template['data_id']}"]
    sandbox.catalogue = [{**template, "data_id": f"entry-{i}"} for i in range(205)]
    ids = [row["data_id"] for row in reversed(sandbox.catalogue)]
    for data_id in ids:
        sandbox.responses[f"xps/{data_id}"] = payload
    assert [result.data_id for result in client.get(ids)] == ids
    requests = [
        params["data_ids"]
        for path, params, _ in sandbox.calls
        if path == "data_entries/"
    ]
    assert [len(chunk) for chunk in requests] == [100, 100, 5]
    assert [data_id for chunk in requests for data_id in chunk] == ids
