"""RHEED configuration patches must preserve explicitly supplied seed settings."""

import json
from copy import deepcopy

import pytest
from requests import PreparedRequest, Response

from atomscale.core import BaseClient


@pytest.mark.parametrize(
    ("method", "settings_key"),
    [("POST", "pipeline_inputs"), ("PATCH", "pipeline_configuration")],
)
@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"seed_frame_count": 4}, id="count-only"),
        pytest.param(
            {"seed_frame_count": 4, "seed_frames": None}, id="explicit-null"
        ),
        pytest.param(
            {"seed_frame_count": 4, "seed_frames": [81, 286, 404]},
            id="explicit-frames",
        ),
    ],
)
def test_rheed_seed_settings_preserve_omission_and_explicit_values(
    monkeypatch, method, settings_key, settings
):
    """Exercise requests' JSON encoding while replacing only the network send.

    A count-only reanalysis lets the backend reselect saved seeds. Explicit null
    clears seeds deliberately, while an explicit list takes precedence over the
    count. Adding or dropping a seed_frames key changes that request's meaning.
    """
    client = BaseClient(api_key="test-key", endpoint="https://example.invalid/")
    sent_bodies = []

    def capture_request(request: PreparedRequest, **kwargs):
        sent_bodies.append(json.loads(request.body))
        response = Response()
        response.status_code = 200
        response._content = b'{"ok": true}'
        return response

    monkeypatch.setattr(client.session, "send", capture_request)
    expected_body = {settings_key: settings}

    response = client._post_or_put(
        method=method,
        sub_url="configuration/rheed/",
        body=deepcopy(expected_body),
    )

    assert response == {"ok": True}
    assert sent_bodies == [expected_body]
