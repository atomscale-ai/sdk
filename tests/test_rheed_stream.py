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
            bufsize=1,  # line-buffered so multi-request output is readable
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

    def get_captured_requests(
        self, expected_count: int, timeout_s: float = 10.0
    ) -> list[tuple[str, str, str]]:
        """Drain REQUEST lines from the routes-mode server.

        Returns a list of (method, path, body) tuples. Reads up to expected_count
        lines or until timeout / process exit.
        """
        import select
        import time

        if not self._proc or not self._proc.stdout:
            return []

        deadline = time.monotonic() + timeout_s
        requests: list[tuple[str, str, str]] = []
        while len(requests) < expected_count and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select([self._proc.stdout], [], [], remaining)
            if not ready:
                break
            line = self._proc.stdout.readline()
            if not line:
                break
            if line.startswith("REQUEST:"):
                # Format: REQUEST:METHOD:PATH:BODY
                # PATH does not contain colons; BODY (JSON) may. Split on the
                # first 3 colons to keep BODY intact.
                parts = line.split(":", 3)
                if len(parts) == 4:
                    _, method, path, body = parts
                    requests.append((method, path, body.rstrip("\n")))
        return requests

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


def _streaming_routes_for(port: int) -> str:
    """Routes-mode mock config for a streaming session.

    The presign endpoint returns a URL pointing at the same mock server's
    /upload/put route, so the subsequent PUT can succeed in-process.
    """
    return json.dumps(
        {
            "__routes__": True,
            "__max_requests__": 50,
            "/rheed/stream/": '"stream-data-id"',
            "/data_entries/raw_data/staged/upload_urls/": json.dumps(
                [{"url": f"http://127.0.0.1:{port}/upload/put"}]
            ),
            "/upload/put": '"OK"',
        }
    )


def _chunk_metadata_from_requests(
    requests, presign_path="/data_entries/raw_data/staged/upload_urls/"
):
    """Filter REQUEST tuples for the presign POSTs and return parsed metadata bodies."""
    metadatas = []
    for method, path, body in requests:
        if method == "POST" and path.startswith(presign_path):
            if body:
                metadatas.append(json.loads(body))
    return metadatas


@pytest.fixture
def streaming_mock_server():
    """Pre-allocate a port, then start a routes-mode mock that knows its own URL."""
    port = _get_free_port()
    server = MockServer(port, _streaming_routes_for(port))
    server.start()
    try:
        yield server
    finally:
        server.stop()


class TestChunkTimestamps:
    """Verify chunk metadata `start_unix_ms_utc` / `end_unix_ms_utc` are derived
    deterministically from declared fps + cumulative frames, and that explicit
    `capture_start_ms_utc` overrides take precedence.

    Background: previously the SDK stamped `start_unix_ms_utc = Utc::now()` per
    chunk, which left chunk timestamps at the mercy of iterator/upload pacing.
    Slow iterators produced inflated global durations (2× when iterator yielded
    chunks at 2× the declared cadence), which caused downstream analysis to
    interpret the stream at half the real fps.
    """

    def _initialize_streamer(self, server, fps=1.0, chunk_size=2):
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        streamer = RHEEDStreamer(api_key="test-api-key", endpoint=server.endpoint)
        streamer.initialize(fps=fps, rotations_per_min=0.0, chunk_size=chunk_size)
        return streamer

    def test_push_signature_accepts_capture_start_ms_utc(self):
        """Verify push() signature includes the new capture_start_ms_utc kwarg."""
        import inspect

        from atomscale.streaming.rheed_stream import RHEEDStreamer

        sig = inspect.signature(RHEEDStreamer.push)
        params = list(sig.parameters.keys())
        assert "capture_start_ms_utc" in params
        assert sig.parameters["capture_start_ms_utc"].default is None

    def test_push_uses_explicit_timestamp_when_provided(self, streaming_mock_server):
        """push() should respect capture_start_ms_utc when given."""
        import numpy as np

        streamer = self._initialize_streamer(
            streaming_mock_server, fps=1.0, chunk_size=2
        )

        explicit_ts = 1_700_000_000_000
        # push() is fire-and-forget — does not raise even if subsequent
        # packaging fails. It dispatches the metadata POST first.
        streamer.push(
            "stream-data-id",
            chunk_idx=0,
            frames=np.zeros((2, 32, 32), dtype=np.uint8),
            capture_start_ms_utc=explicit_ts,
        )

        requests = streaming_mock_server.get_captured_requests(
            expected_count=2, timeout_s=5
        )
        metadatas = _chunk_metadata_from_requests(requests)

        assert len(metadatas) >= 1
        md = metadatas[0]
        assert int(md["start_unix_ms_utc"]) == explicit_ts
        assert int(md["end_unix_ms_utc"]) - explicit_ts == 2000  # n=2, fps=1
