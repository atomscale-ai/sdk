from pytest_httpserver import HTTPServer
import pytest
from requests import Session

from atomscale.core import BaseClient, _make_progress

from atomscale.core.client import ClientError


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


def test_core_get_not_ok(httpserver, monkeypatch):
    # 500 is in the session's status_forcelist; skip the urllib3 backoff sleeps.
    monkeypatch.setattr("urllib3.util.retry.time.sleep", lambda *_: None)

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


def test_client_error_carries_status_and_text(httpserver: HTTPServer, monkeypatch):
    monkeypatch.setattr("urllib3.util.retry.time.sleep", lambda *_: None)

    httpserver.expect_request("/bad").respond_with_data("server boom", status=500)

    base_client = BaseClient(api_key="", endpoint=httpserver.url_for("/"))

    with pytest.raises(ClientError) as exc_info:
        base_client._get(sub_url="bad")

    exc = exc_info.value
    assert exc.status_code == 500
    assert exc.response_text == "server boom"
    # Existing message-based behavior must keep working.
    assert "Problem retrieving data" in str(exc)


def test_client_error_default_args_remain_valid():
    # Validation/usage errors raise ClientError without an HTTP context.
    exc = ClientError("bare message")
    assert exc.status_code is None
    assert exc.response_text is None
    assert str(exc) == "bare message"


def test_get_retries_transient_502(httpserver: HTTPServer, monkeypatch):
    # Sleep would otherwise add ~0.5s + 1s + ... between urllib3 retries.
    monkeypatch.setattr("urllib3.util.retry.time.sleep", lambda *_: None)

    httpserver.expect_ordered_request("/flaky", method="GET").respond_with_data(
        "", status=502
    )
    httpserver.expect_ordered_request("/flaky", method="GET").respond_with_json(
        {"ok": True}
    )

    base_client = BaseClient(api_key="", endpoint=httpserver.url_for("/"))

    response = base_client._get(sub_url="flaky")
    assert response == {"ok": True}


def test_get_surfaces_persistent_502_after_retries(
    httpserver: HTTPServer, monkeypatch
):
    monkeypatch.setattr("urllib3.util.retry.time.sleep", lambda *_: None)

    httpserver.expect_request("/always-bad", method="GET").respond_with_data(
        "still down", status=502
    )

    base_client = BaseClient(api_key="", endpoint=httpserver.url_for("/"))

    with pytest.raises(ClientError) as exc_info:
        base_client._get(sub_url="always-bad")

    assert exc_info.value.status_code == 502
    assert exc_info.value.response_text == "still down"
