"""Offline by default; remote service checks require --run-live explicitly."""

import os
from urllib.parse import urlsplit

import matplotlib
import pytest
import requests

matplotlib.use("Agg", force=True)


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        help="Enable explicitly marked live API checks",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-live"):
        for item in items:
            if item.get_closest_marker("live"):
                item.add_marker(
                    pytest.mark.skip(reason="Live API check; opt in with --run-live")
                )


@pytest.fixture(autouse=True)
def network_sandbox(request, monkeypatch):
    """Block external requests even when a developer has production credentials.

    Loopback remains available for the existing HTTP transport/streaming tests.
    """
    if request.node.get_closest_marker("live") and request.config.getoption(
        "--run-live"
    ):
        return
    for key in ("AS_API_KEY", "ATOMSCALE_API_KEY", "AS_API_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    send = requests.sessions.Session.send

    def guarded_send(session, prepared, **kwargs):
        host = urlsplit(prepared.url).hostname
        assert host in {"localhost", "127.0.0.1", "::1"}, (
            f"External HTTP blocked in offline test: {host}"
        )
        return send(session, prepared, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "send", guarded_send)


@pytest.fixture
def sandbox():
    from .sandbox import Sandbox

    return Sandbox()


@pytest.fixture
def result_ids(sandbox):
    return sandbox.ids


@pytest.fixture
def client(sandbox, monkeypatch):
    from atomscale import Client

    client = Client(
        api_key="key_test", endpoint="https://sandbox.invalid/", mute_bars=True
    )
    monkeypatch.setattr(client, "_get", sandbox.get)
    yield client
    client.session.close()


@pytest.fixture
def live_client(request):
    from atomscale import Client

    if not request.node.get_closest_marker("live"):
        pytest.fail("Tests using live_client must be marked live")
    if not request.config.getoption("--run-live"):
        pytest.skip("Live API check; opt in with --run-live")
    endpoint = os.getenv("AS_TEST_API_ENDPOINT")
    key = os.getenv("AS_TEST_API_KEY")
    if not endpoint or not key:
        pytest.fail(
            "Live checks require AS_TEST_API_ENDPOINT and AS_TEST_API_KEY for a dedicated test environment"
        )
    client = Client(api_key=key, endpoint=endpoint, mute_bars=True)
    yield client
    client.session.close()
