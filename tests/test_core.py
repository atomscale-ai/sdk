from pytest_httpserver import HTTPServer
import pytest
from requests import Session

from atomicds.core import BaseClient, _make_progress

from atomicds.core.client import ClientError


def test_core_get_session():
    base_client = BaseClient(api_key="", endpoint="")
    assert isinstance(base_client.session, Session)


def test_core_get_ok(httpserver: HTTPServer):
    httpserver.expect_request("/").respond_with_json({"foo": "bar"})
    httpserver.expect_request(
        "/test",
        headers={"X-API-KEY": "key_test"},
        query_string={"param_foo": "param_bar"},
    ).respond_with_json({"foo_sub": "bar_sub"})

    base_client = BaseClient(api_key="key_test", endpoint=httpserver.url_for("/"))

    response = base_client._get(sub_url="test", params={"param_foo": "param_bar"})
    assert response.get("foo_sub") == "bar_sub"  # type: ignore

    response = base_client._get(
        sub_url="test", params={"param_foo": "param_bar"}, deserialize=False
    )
    assert isinstance(response, bytes)

    response = base_client._get(
        sub_url="",
        base_override=httpserver.url_for("/test"),
        params={"param_foo": "param_bar"},
    )
    assert response.get("foo_sub") == "bar_sub"  # type: ignore


def test_core_get_not_ok(httpserver):
    httpserver.expect_request("/").respond_with_data("Not found", status=404)
    httpserver.expect_request("/bad").respond_with_data("Not found", status=500)

    base_client = BaseClient(api_key="", endpoint=httpserver.url_for("/"))

    response = base_client._get(sub_url="")
    assert response is None

    with pytest.raises(ClientError, match="Problem retrieving data"):
        response = base_client._get(sub_url="bad")


def test_core_multi_thread():
    base_client = BaseClient(api_key="", endpoint="")
    test_func = lambda x: x
    kwargs_list = [{"x": True} for _ in range(8)]
    results = base_client._multi_thread(test_func, kwargs_list)
    assert results == [True] * 8

    # With progress bar
    with _make_progress(False, False) as pbar:
        results = base_client._multi_thread(test_func, kwargs_list, pbar)
    assert results == [True] * 8
