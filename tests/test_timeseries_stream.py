"""Integration tests for TimeseriesStreamer.

These tests use a subprocess-based HTTP server because the TimeseriesStreamer uses
a Rust HTTP client (reqwest) which doesn't interact well with Python's threading
model used by pytest-httpserver. The subprocess approach provides true process
isolation and works reliably across platforms.
"""
import json
import socket
import subprocess
import sys
import time
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
        self._all_requests: list[str] = []

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

    def get_all_requests(self) -> list[str]:
        """Read all captured requests from the server (for routes mode)."""
        if self._proc and not self._all_requests:
            for line in self._proc.stdout:
                if line.startswith("REQUEST:"):
                    self._all_requests.append(line.strip())
        return self._all_requests

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


class TestTimeseriesStreamerInit:
    """Tests for TimeseriesStreamer initialization."""

    def test_can_import_timeseries_streamer(self):
        """Verify TimeseriesStreamer can be imported."""
        from rheed_stream import TimeseriesStreamer

        assert TimeseriesStreamer is not None

    def test_init_with_defaults(self):
        """Verify TimeseriesStreamer can be instantiated with minimal args."""
        from rheed_stream import TimeseriesStreamer

        streamer = TimeseriesStreamer(api_key="test-api-key")
        assert streamer is not None

    def test_init_with_custom_endpoint(self):
        """Verify TimeseriesStreamer accepts custom endpoint."""
        from rheed_stream import TimeseriesStreamer

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint="http://localhost:8080",
        )
        assert streamer is not None

    def test_init_with_custom_points_per_chunk(self):
        """Verify TimeseriesStreamer accepts custom points_per_chunk."""
        from rheed_stream import TimeseriesStreamer

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            points_per_chunk=50,
        )
        assert streamer is not None


class TestTimeseriesStreamerInitialize:
    """Tests for TimeseriesStreamer.initialize() method."""

    def test_initialize_returns_data_id(self, mock_server_factory):
        """Verify initialize() returns data_id from server."""
        from rheed_stream import TimeseriesStreamer

        response = json.dumps({
            "data_id": "test-data-id-123",
            "processed_data_id": "test-processed-id-456",
        })
        server = mock_server_factory(response)

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
            points_per_chunk=100,
        )

        data_id = streamer.initialize(
            stream_name="Test Stream",
            instrument_type="mbe",
        )

        assert data_id == "test-data-id-123"

    def test_initialize_sends_correct_payload(self, mock_server_factory):
        """Verify initialize() sends correct request body."""
        from rheed_stream import TimeseriesStreamer

        response = json.dumps({
            "data_id": "test-data-id",
            "processed_data_id": "test-processed-id",
        })
        server = mock_server_factory(response)

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
            points_per_chunk=50,
        )

        streamer.initialize(
            stream_name="My Stream",
            instrument_type="cvd",
        )

        body = server.get_captured_body()
        assert body is not None
        assert body["stream_name"] == "My Stream"
        assert body["instrument_type"] == "cvd"
        assert body["points_per_chunk"] == 50


class TestTimeseriesStreamerPush:
    """Tests for TimeseriesStreamer.push() method."""

    def test_push_validates_length_mismatch(self):
        """Verify push() raises error when timestamps/values lengths don't match."""
        from rheed_stream import TimeseriesStreamer

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint="http://localhost:9999",
        )

        with pytest.raises(RuntimeError, match="same length"):
            streamer.push(
                data_id="test-data-id",
                chunk_index=0,
                channel_name="temperature",
                timestamps=[0.0, 0.01, 0.02],
                values=[25.0, 25.1],  # Missing one value
            )

    def test_push_sends_correct_payload(self, mock_server_factory):
        """Verify push() sends the correct JSON payload."""
        from rheed_stream import TimeseriesStreamer

        # Configure routes for initialize + chunk
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 2,
            "/instrument-timeseries/initialize": json.dumps({
                "data_id": "test-data-id-123",
                "processed_data_id": "test-processed-id",
            }),
            "/instrument-timeseries/chunk": json.dumps({
                "data_id": "test-data-id-123",
                "channel_name": "temperature",
                "chunk_index": 0,
                "total_points": 3,
            }),
        })
        server = mock_server_factory(routes)

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
            points_per_chunk=100,
        )

        data_id = streamer.initialize()
        streamer.push(
            data_id=data_id,
            chunk_index=0,
            channel_name="temperature",
            timestamps=[0.0, 0.01, 0.02],
            values=[25.0, 25.1, 25.2],
            units="C",
        )

        # Give async task time to complete
        time.sleep(0.5)

        requests = server.get_all_requests()
        assert len(requests) == 2
        # Second request should be the chunk
        assert "/instrument-timeseries/chunk" in requests[1]


class TestTimeseriesStreamerRun:
    """Tests for TimeseriesStreamer.run() method (iterator mode)."""

    def test_run_validates_length_mismatch(self, mock_server_factory):
        """Verify run() raises error when any chunk has mismatched lengths."""
        from rheed_stream import TimeseriesStreamer

        init_response = json.dumps({
            "data_id": "test-data-id",
            "processed_data_id": "test-processed-id",
        })
        server = mock_server_factory(init_response)

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
        )
        data_id = streamer.initialize()

        def bad_data_generator():
            yield ([0.0, 0.01, 0.02], [25.0, 25.1])  # Mismatch

        with pytest.raises(RuntimeError, match="same length"):
            streamer.run(
                data_id=data_id,
                channel_name="temperature",
                data_iter=bad_data_generator(),
            )

    def test_run_streams_from_iterator(self, mock_server_factory):
        """Verify run() streams all chunks from iterator and blocks until complete."""
        from rheed_stream import TimeseriesStreamer

        # Configure routes for initialize + 3 chunks
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 4,
            "/instrument-timeseries/initialize": json.dumps({
                "data_id": "test-data-id",
                "processed_data_id": "test-processed-id",
            }),
            "/instrument-timeseries/chunk": json.dumps({
                "data_id": "test-data-id",
                "channel_name": "temperature",
                "chunk_index": 0,
                "total_points": 3,
            }),
        })
        server = mock_server_factory(routes)

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
            points_per_chunk=100,
        )

        data_id = streamer.initialize()

        def data_generator():
            for i in range(3):
                timestamps = [i * 100 + j * 0.01 for j in range(100)]
                values = [25.0 + j * 0.1 for j in range(100)]
                yield (timestamps, values)

        # run() blocks until all uploads complete
        streamer.run(data_id=data_id, channel_name="temperature", data_iter=data_generator())

        # Should have 4 requests: 1 init + 3 chunks
        requests = server.get_all_requests()
        assert len(requests) == 4
        assert "/instrument-timeseries/initialize" in requests[0]
        # Remaining 3 should be chunks
        chunk_requests = [r for r in requests if "/instrument-timeseries/chunk" in r]
        assert len(chunk_requests) == 3


class TestTimeseriesStreamerPushMulti:
    """Tests for TimeseriesStreamer.push_multi() method."""

    def test_push_multi_validates_length_mismatch(self):
        """Verify push_multi() raises error when any channel has mismatched lengths."""
        from rheed_stream import TimeseriesStreamer

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint="http://localhost:9999",
        )

        with pytest.raises(RuntimeError, match="same length"):
            streamer.push_multi(
                data_id="test-data-id",
                chunk_index=0,
                channels={
                    "temperature": {"timestamps": [0.0, 0.01], "values": [25.0, 25.1]},
                    "pressure": {"timestamps": [0.0, 0.01, 0.02], "values": [1.0, 1.1]},  # Mismatch
                },
            )

    def test_push_multi_with_units(self, mock_server_factory):
        """Verify push_multi() supports units per channel."""
        from rheed_stream import TimeseriesStreamer

        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 3,
            "/instrument-timeseries/initialize": json.dumps({
                "data_id": "test-data-id",
                "processed_data_id": "test-processed-id",
            }),
            "/instrument-timeseries/chunk": json.dumps({
                "data_id": "test-data-id",
                "chunk_index": 0,
                "total_points": 2,
            }),
        })
        server = mock_server_factory(routes)

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
        )
        data_id = streamer.initialize()

        # Should not raise - units are optional per channel
        streamer.push_multi(
            data_id=data_id,
            chunk_index=0,
            channels={
                "temperature": {"timestamps": [0.0, 0.01], "values": [25.0, 25.1], "units": "C"},
                "pressure": {"timestamps": [0.0, 0.01], "values": [1.0, 1.1]},  # No units
            },
        )

        time.sleep(0.5)
        requests = server.get_all_requests()
        assert len(requests) == 3  # 1 init + 2 channels


class TestTimeseriesStreamerIntegration:
    """Integration tests for full streaming workflow."""

    def test_full_streaming_workflow(self, mock_server_factory):
        """Test a complete streaming workflow with initialize and multiple chunks."""
        from rheed_stream import TimeseriesStreamer

        # Configure routes
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 4,
            "/instrument-timeseries/initialize": json.dumps({
                "data_id": "workflow-data-id",
                "processed_data_id": "workflow-processed-id",
            }),
            "/instrument-timeseries/chunk": json.dumps({
                "data_id": "workflow-data-id",
                "channel_name": "temperature",
                "chunk_index": 0,
                "total_points": 100,
            }),
        })
        server = mock_server_factory(routes)

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
            points_per_chunk=100,
        )

        # Initialize
        data_id = streamer.initialize(
            stream_name="Integration Test Stream",
            instrument_type="mbe",
        )
        assert data_id == "workflow-data-id"

        # Push multiple chunks
        for chunk_idx in range(3):
            timestamps = [chunk_idx * 100 + i for i in range(100)]
            values = [25.0 + i * 0.1 for i in range(100)]

            streamer.push(
                data_id=data_id,
                chunk_index=chunk_idx,
                channel_name="temperature",
                timestamps=timestamps,
                values=values,
            )

        # Give async tasks time to complete
        time.sleep(1.5)

        requests = server.get_all_requests()
        # 1 initialize + 3 chunks = 4 requests
        assert len(requests) == 4


class TestTimeseriesStreamerFinalize:
    """Tests for TimeseriesStreamer.finalize() method."""

    def test_finalize_sends_request(self, mock_server_factory):
        """Verify finalize() sends POST to finalize endpoint."""
        from rheed_stream import TimeseriesStreamer

        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 2,
            "/instrument-timeseries/initialize": json.dumps({
                "data_id": "test-data-id",
                "processed_data_id": "test-processed-id",
            }),
            "/instrument-timeseries/test-data-id/finalize": json.dumps({
                "data_id": "test-data-id",
                "processed_data_id": "test-processed-id",
            }),
        })
        server = mock_server_factory(routes)

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint=server.endpoint,
        )

        data_id = streamer.initialize()
        streamer.finalize(data_id)

        requests = server.get_all_requests()
        assert len(requests) == 2
        assert "/finalize" in requests[1]
