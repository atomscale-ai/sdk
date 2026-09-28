"""Regression checks for the isolation boundary shared by the test suite."""

import pytest
import requests

from .sandbox import Sandbox


def test_external_http_is_blocked():
    with pytest.raises(AssertionError, match="External HTTP blocked"):
        requests.get("https://sandbox.invalid/unexpected", timeout=1)


def test_redirect_cannot_escape_loopback(httpserver):
    httpserver.expect_request("/").respond_with_data(
        status=302, headers={"Location": "https://sandbox.invalid/redirect"}
    )
    with pytest.raises(AssertionError, match="External HTTP blocked"):
        requests.get(httpserver.url_for("/"), timeout=1)


def test_unknown_sandbox_route_fails(sandbox):
    with pytest.raises(AssertionError, match="Unexpected sandbox request"):
        sandbox.get("not-a-real-endpoint")


def test_sandbox_payloads_and_instances_are_independent(sandbox):
    sandbox.get("data_entries/")[0]["tags"].append("mutated")
    assert sandbox.catalogue[0]["tags"] == ["sdk-test"]
    sandbox.catalogue.clear()
    assert Sandbox().catalogue
