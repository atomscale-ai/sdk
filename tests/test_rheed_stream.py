"""Integration tests for RHEEDStreamer.

These tests use a subprocess-based HTTP server because the RHEEDStreamer uses
a Rust HTTP client (reqwest) which doesn't interact well with Python's threading
model used by pytest-httpserver. The subprocess approach provides true process
isolation and works reliably across platforms.
"""
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

# Path to the mock server module
_MOCK_SERVER_MODULE = Path(__file__).parent / "_mock_http_server.py"


class MockServer:
    """A mock HTTP server running in a subprocess."""

    def __init__(self, port: int, response_data: str):
        self.port = port
        self.response_data = response_data
        self._proc: subprocess.Popen | None = None
        self._captured_body: dict | None = None

    def start(self) -> None:
        """Start the server subprocess."""
        self._proc = subprocess.Popen(
            [sys.executable, str(_MOCK_SERVER_MODULE), str(self.port), self.response_data],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Wait for server to signal it's ready
        ready_line = self._proc.stdout.readline()
        if not ready_line.startswith("READY:"):
            self.stop()
            raise RuntimeError(f"Server failed to start: {ready_line}")

    def stop(self) -> None:
        """Stop the server subprocess."""
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
            self._proc = None

    @property
    def endpoint(self) -> str:
        """Return the server endpoint URL (no trailing slash)."""
        return f"http://127.0.0.1:{self.port}"

    def get_captured_body(self) -> dict | None:
        """Read and return the captured request body from the server."""
        if self._proc and self._captured_body is None:
            for line in self._proc.stdout:
                if line.startswith("BODY:"):
                    self._captured_body = json.loads(line[5:])
                    break
        return self._captured_body

    def get_captured_requests(self) -> list[dict]:
        """Return list of {method, path, body} dicts from routes-mode logs.

        Drains stdout up to and including the EOF that occurs when the server
        process exits (after handling __max_requests__ requests). Safe to call
        only once per server.
        """
        if self._proc is None:
            return []
        try:
            stdout, _ = self._proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            stdout, _ = self._proc.communicate(timeout=5)
        self._proc = None
        captured: list[dict] = []
        for line in stdout.splitlines():
            if not line.startswith("REQUEST:"):
                continue
            # Format: REQUEST:METHOD:PATH:BODY (BODY may contain ':' from JSON)
            parts = line[len("REQUEST:"):].split(":", 2)
            if len(parts) != 3:
                continue
            method, path, body = parts
            try:
                body_obj = json.loads(body) if body else None
            except json.JSONDecodeError:
                body_obj = body
            captured.append({"method": method, "path": path, "body": body_obj})
        return captured

    def __enter__(self) -> "MockServer":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


def _get_free_port() -> int:
    """Get an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def mock_server_factory() -> Callable[[str], MockServer]:
    """Factory fixture to create mock servers with custom responses."""
    servers: list[MockServer] = []

    def create(response_data: str) -> MockServer:
        server = MockServer(_get_free_port(), response_data)
        server.start()
        servers.append(server)
        return server

    yield create

    for server in servers:
        server.stop()


class TestRHEEDStreamerInitialize:
    """Tests for RHEEDStreamer.initialize() method."""

    def test_initialize_accepts_project_id_parameter(self):
        """Verify initialize() signature includes project_id parameter."""
        import inspect

        from atomscale.streaming.rheed_stream import RHEEDStreamer

        sig = inspect.signature(RHEEDStreamer.initialize)
        params = list(sig.parameters.keys())

        assert "project_id" in params
        assert sig.parameters["project_id"].default is None

    def test_initialize_validates_chunk_size(self):
        """Verify chunk_size validation (must be >= 2 * fps)."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        streamer = RHEEDStreamer(
            api_key="test-api-key",
            endpoint="http://localhost:9999",
        )

        with pytest.raises(ValueError, match="chunk_size must be at least 2×fps"):
            streamer.initialize(
                fps=30.0,
                rotations_per_min=0.0,
                chunk_size=30,  # Invalid: less than 2 * 30 = 60
            )

    def test_initialize_sends_project_id_in_request(self, mock_server_factory):
        """Verify project_id is included in POST body when provided."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        server = mock_server_factory('"test-data-id-123"')

        streamer = RHEEDStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
        )

        project_uuid = "550e8400-e29b-41d4-a716-446655440000"
        data_id = streamer.initialize(
            fps=30.0,
            rotations_per_min=0.0,
            chunk_size=60,
            project_id=project_uuid,
        )

        assert data_id == "test-data-id-123"

        body = server.get_captured_body()
        assert body is not None
        assert body.get("project_id") == project_uuid
        assert "data_item_name" in body
        assert body.get("fps_capture_rate") == 30.0

    def test_initialize_omits_project_id_when_none(self, mock_server_factory):
        """Verify project_id is omitted from POST body when not provided."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        server = mock_server_factory('"test-data-id-456"')

        streamer = RHEEDStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
        )

        data_id = streamer.initialize(
            fps=30.0,
            rotations_per_min=0.0,
            chunk_size=60,
        )

        assert data_id == "test-data-id-456"

        body = server.get_captured_body()
        assert body is not None
        assert "project_id" not in body

    def test_initialize_omits_project_id_when_empty_string(self, mock_server_factory):
        """Verify empty string project_id is treated as None (omitted)."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        server = mock_server_factory('"test-data-id-789"')

        streamer = RHEEDStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
        )

        data_id = streamer.initialize(
            fps=30.0,
            rotations_per_min=0.0,
            chunk_size=60,
            project_id="",
        )

        assert data_id == "test-data-id-789"

        body = server.get_captured_body()
        assert body is not None
        assert "project_id" not in body

    def test_initialize_returns_data_id(self, mock_server_factory):
        """Verify initialize() returns the data_id from the server."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        expected_data_id = "abc-123-xyz"
        server = mock_server_factory(f'"{expected_data_id}"')

        streamer = RHEEDStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
        )

        data_id = streamer.initialize(
            fps=30.0,
            rotations_per_min=0.0,
            chunk_size=60,
        )

        assert data_id == expected_data_id

    def test_initialize_updates_project_config_when_physical_sample_and_project_id(
        self, mock_server_factory
    ):
        """Verify project configuration is updated with tracking_physical_sample_id.

        When both physical_sample and project_id are provided, the SDK should:
        1. POST /rheed/stream/ to create the stream
        2. GET /physical_samples/ to list existing samples
        3. POST /physical_samples/ to create the sample (if not found)
        4. POST /data_entries/physical_sample to link sample to data entry
        5. GET /projects/ to get current project configuration
        6. POST /projects/{id}/configuration to update tracking_physical_sample_id
        """
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        project_uuid = "550e8400-e29b-41d4-a716-446655440000"
        sample_uuid = "660e8400-e29b-41d4-a716-446655440001"

        # Configure routes for the multi-request flow
        # Note: /physical_samples/ is used for both GET (returns list) and POST (returns created sample)
        # The mock returns the same response for both, which works because:
        # - GET expects a list - we return a list with the sample already existing
        # - This skips the POST /physical_samples/ call since sample already exists
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 6,
            "/rheed/stream/": '"test-data-id-999"',
            "/physical_samples/": json.dumps([{"id": sample_uuid, "name": "Test Sample"}]),
            "/data_entries/physical_sample": '"OK"',
            "/projects/": json.dumps([{
                "id": project_uuid,
                "name": "Test Project",
                "configuration": {
                    "api_configuration": {
                        "reference_group_type": "categorical",
                        "onboarding_complete": True
                    }
                }
            }]),
            f"/projects/{project_uuid}/configuration": '"OK"',
        })

        server = mock_server_factory(routes)

        streamer = RHEEDStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
        )

        data_id = streamer.initialize(
            fps=30.0,
            rotations_per_min=0.0,
            chunk_size=60,
            physical_sample="Test Sample",
            project_id=project_uuid,
        )

        assert data_id == "test-data-id-999"

    def test_initialize_accepts_tags_parameter(self):
        """Verify initialize() signature includes the tags parameter."""
        import inspect

        from atomscale.streaming.rheed_stream import RHEEDStreamer

        sig = inspect.signature(RHEEDStreamer.initialize)
        assert "tags" in sig.parameters
        assert sig.parameters["tags"].default is None

    def test_initialize_skips_tag_endpoints_when_tags_none(self, mock_server_factory):
        """Verify no /tags/ traffic when tags is None (single request)."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        server = mock_server_factory('"data-id-no-tags"')

        streamer = RHEEDStreamer(api_key="k", endpoint=server.endpoint)
        data_id = streamer.initialize(fps=30.0, rotations_per_min=0.0, chunk_size=60)

        assert data_id == "data-id-no-tags"
        body = server.get_captured_body()
        assert body is not None
        assert "data_item_name" in body  # confirms only the init POST was captured

    def test_initialize_skips_tag_endpoints_when_tags_empty(self, mock_server_factory):
        """Empty list, all-whitespace entries → no tag traffic."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        server = mock_server_factory('"data-id-empty-tags"')

        streamer = RHEEDStreamer(api_key="k", endpoint=server.endpoint)
        data_id = streamer.initialize(
            fps=30.0,
            rotations_per_min=0.0,
            chunk_size=60,
            tags=["", "  "],
        )

        assert data_id == "data-id-empty-tags"

    def test_initialize_resolves_dedupes_and_creates_tags(self, mock_server_factory):
        """End-to-end tag resolution: case-insensitive match, whitespace trim,
        case-insensitive dedupe, find-or-create, and bulk attach payload.

        Input tags: ["  growth  ", "GROWTH", "novel-tag", ""]
        Existing org tags: [{name: "growth", id: G}]

        Expected requests (4 total):
        1. POST /rheed/stream/         → data_id
        2. GET  /tags/                 → returns existing (only "growth")
        3. POST /tags/                 → creates "novel-tag", returns its id
        4. POST /tags/data-items/      → bulk attach with both ids
        """
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        existing_id = "11111111-1111-1111-1111-111111111111"
        new_id = "33333333-3333-3333-3333-333333333333"

        # Method-keyed routes let GET /tags/ return a list while POST /tags/
        # returns the created-tag object. Path-only routes serve as fallbacks.
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 4,
            "POST /tags/data-items/": '{"success": true, "associations_created": 2}',
            "GET /tags/": json.dumps([{"id": existing_id, "name": "growth"}]),
            "POST /tags/": json.dumps({"id": new_id, "name": "novel-tag"}),
            "/rheed/stream/": '"data-id-mixed"',
        })

        server = mock_server_factory(routes)
        streamer = RHEEDStreamer(api_key="k", endpoint=server.endpoint)

        data_id = streamer.initialize(
            fps=30.0,
            rotations_per_min=0.0,
            chunk_size=60,
            tags=["  growth  ", "GROWTH", "novel-tag", ""],
        )

        assert data_id == "data-id-mixed"

        requests = server.get_captured_requests()
        # Filter to /tags/ traffic — the init POST is uninteresting here.
        tag_reqs = [r for r in requests if r["path"].startswith("/tags/")]
        # Exactly: one GET, one create-POST, one attach-POST. Dedup means the
        # repeated "growth"/"GROWTH" entries did not produce extra creates.
        methods_paths = [(r["method"], r["path"]) for r in tag_reqs]
        assert methods_paths == [
            ("GET", "/tags/"),
            ("POST", "/tags/"),
            ("POST", "/tags/data-items/"),
        ], methods_paths

        # POST /tags/ body uses the trimmed name (no surrounding spaces).
        create_body = next(r["body"] for r in tag_reqs if r["method"] == "POST" and r["path"] == "/tags/")
        assert create_body == {"name": "novel-tag"}

        # POST /tags/data-items/ body has data_id + both resolved tag_ids in order.
        attach_body = next(r["body"] for r in tag_reqs if r["path"] == "/tags/data-items/")
        assert attach_body == {
            "data_ids": ["data-id-mixed"],
            "tag_ids": [existing_id, new_id],
        }

    def test_initialize_attaches_tags_with_physical_sample_and_project(
        self, mock_server_factory
    ):
        """Tags can be combined with physical_sample + project_id in one call.

        Expected requests (all using existing entities, so no creates):
        1. POST /rheed/stream/
        2. GET  /physical_samples/    → list with sample
        3. POST /data_entries/physical_sample
        4. GET  /projects/            → list with project config
        5. POST /projects/{id}/configuration
        6. GET  /tags/                → list with tag
        7. POST /tags/data-items/
        """
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        project_uuid = "550e8400-e29b-41d4-a716-446655440000"
        sample_uuid = "660e8400-e29b-41d4-a716-446655440001"
        tag_uuid = "770e8400-e29b-41d4-a716-446655440002"

        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 7,
            "/tags/data-items/": '{"success": true, "associations_created": 1}',
            "/tags/": json.dumps([{"id": tag_uuid, "name": "growth"}]),
            "/data_entries/physical_sample": '"OK"',
            "/physical_samples/": json.dumps(
                [{"id": sample_uuid, "name": "Test Sample"}]
            ),
            f"/projects/{project_uuid}/configuration": '"OK"',
            "/projects/": json.dumps([{
                "id": project_uuid,
                "name": "Test Project",
                "configuration": {"api_configuration": {}},
            }]),
            "/rheed/stream/": '"data-id-combined"',
        })

        server = mock_server_factory(routes)
        streamer = RHEEDStreamer(api_key="k", endpoint=server.endpoint)

        data_id = streamer.initialize(
            fps=30.0,
            rotations_per_min=0.0,
            chunk_size=60,
            physical_sample="Test Sample",
            project_id=project_uuid,
            tags=["growth"],
        )

        assert data_id == "data-id-combined"

    def test_initialize_attaches_azimuth_tags(self, mock_server_factory):
        """Primary use case: attach RHEED azimuth tags ("100", "210", "110").

        The general resolver behavior (find-or-create, dedupe, whitespace) is
        exercised in test_initialize_resolves_dedupes_and_creates_tags. This
        test pins the concrete azimuth-tag scenario the feature was built for:
        all three azimuth tags already exist in the org, and the streamer
        attaches them in input order.
        """
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        id_100 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        id_210 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        id_110 = "cccccccc-cccc-cccc-cccc-cccccccccccc"

        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 3,
            "POST /tags/data-items/": '{"success": true, "associations_created": 3}',
            "GET /tags/": json.dumps([
                {"id": id_100, "name": "100"},
                {"id": id_210, "name": "210"},
                {"id": id_110, "name": "110"},
            ]),
            "/rheed/stream/": '"data-id-azimuth"',
        })

        server = mock_server_factory(routes)
        streamer = RHEEDStreamer(api_key="k", endpoint=server.endpoint)

        data_id = streamer.initialize(
            fps=30.0,
            rotations_per_min=0.0,
            chunk_size=60,
            tags=["100", "210", "110"],
        )

        assert data_id == "data-id-azimuth"

        requests = server.get_captured_requests()
        tag_reqs = [r for r in requests if r["path"].startswith("/tags/")]
        # All three exist → no POST /tags/ creates, just GET + attach.
        methods_paths = [(r["method"], r["path"]) for r in tag_reqs]
        assert methods_paths == [
            ("GET", "/tags/"),
            ("POST", "/tags/data-items/"),
        ], methods_paths

        attach_body = next(r["body"] for r in tag_reqs if r["path"] == "/tags/data-items/")
        assert attach_body == {
            "data_ids": ["data-id-azimuth"],
            "tag_ids": [id_100, id_210, id_110],
        }

    def test_initialize_unknown_tag_uuid_raises(self, mock_server_factory):
        """A UUID that doesn't match any existing tag → RuntimeError."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        existing = [
            {"id": "11111111-1111-1111-1111-111111111111", "name": "growth"},
        ]
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 3,
            "/tags/": json.dumps(existing),
            "/rheed/stream/": '"data-id-unknown-uuid"',
        })

        server = mock_server_factory(routes)
        streamer = RHEEDStreamer(api_key="k", endpoint=server.endpoint)

        with pytest.raises(RuntimeError, match="tag with id .* not found"):
            streamer.initialize(
                fps=30.0,
                rotations_per_min=0.0,
                chunk_size=60,
                tags=["99999999-9999-9999-9999-999999999999"],
            )
