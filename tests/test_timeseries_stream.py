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
from collections.abc import Callable
from pathlib import Path

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

    def stop_and_get_requests(self) -> list[str]:
        """Stop the server and return the requests it captured, however many arrived."""
        if self._proc:
            self._proc.terminate()
            out, _ = self._proc.communicate(timeout=5)
            self._all_requests = [
                line.strip() for line in out.splitlines() if line.startswith("REQUEST:")
            ]
            self._proc = None
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
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        assert TimeseriesStreamer is not None

    def test_init_with_defaults(self):
        """Verify TimeseriesStreamer can be instantiated with minimal args."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        streamer = TimeseriesStreamer(api_key="test-api-key")
        assert streamer is not None

    def test_init_with_custom_endpoint(self):
        """Verify TimeseriesStreamer accepts custom endpoint."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            endpoint="http://localhost:8080",
        )
        assert streamer is not None

    def test_init_with_custom_points_per_chunk(self):
        """Verify TimeseriesStreamer accepts custom points_per_chunk."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        streamer = TimeseriesStreamer(
            api_key="test-api-key",
            points_per_chunk=50,
        )
        assert streamer is not None


class TestTimeseriesStreamerInitialize:
    """Tests for TimeseriesStreamer.initialize() method."""

    def test_initialize_returns_data_id(self, mock_server_factory):
        """Verify initialize() returns data_id from server."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

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
        )

        assert data_id == "test-data-id-123"

    def test_initialize_sends_correct_payload(self, mock_server_factory):
        """Verify initialize() sends correct request body."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

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
        )

        body = server.get_captured_body()
        assert body is not None
        assert body["stream_name"] == "My Stream"
        assert body["points_per_chunk"] == 50

    def test_initialize_attaches_tag(self, mock_server_factory):
        """Verify tags wire through TimeseriesStreamer.initialize.

        End-to-end coverage of the shared resolver lives in test_rheed_stream;
        here we just confirm the wiring: an existing tag is looked up and the
        bulk-attach POST is fired with the right ids.
        """
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        tag_id = "770e8400-e29b-41d4-a716-446655440002"
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 3,
            "/tags/data-items/": '{"success": true, "associations_created": 1}',
            "/tags/": json.dumps([{"id": tag_id, "name": "growth"}]),
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "ts-data-id",
                "processed_data_id": "ts-proc-id",
            }),
        })

        server = mock_server_factory(routes)
        streamer = TimeseriesStreamer(
            api_key="k",
            endpoint=server.endpoint,
        )

        data_id = streamer.initialize(
            stream_name="Tagged Stream",
            tags=["GROWTH"],  # case-insensitive match
        )

        assert data_id == "ts-data-id"

        requests = server.get_all_requests()
        # Exactly the three requests we expect, in order.
        assert any("REQUEST:POST:/tool-state/stream/initialize:" in r for r in requests)
        assert any("REQUEST:GET:/tags/:" in r for r in requests)
        attach = next(r for r in requests if "/tags/data-items/" in r)
        # The bulk-attach body must include the resolved tag_id and data_id.
        assert tag_id in attach
        assert "ts-data-id" in attach

    def test_initialize_reuses_tag_created_by_concurrent_stream(self, mock_server_factory):
        """A tag create rejected as a duplicate (another stream just made it) reuses that tag.

        The race itself is covered in test_rheed_stream; this pins the timeseries wiring.
        """
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        winner_id = "44444444-4444-4444-4444-444444444444"
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 5,
            "__body_sequence__": {"GET /tags/": ["[]"]},
            "__status_sequence__": {"POST /tags/": [400]},
            "POST /tags/data-items/": '{"success": true, "associations_created": 1}',
            "GET /tags/": json.dumps([{"id": winner_id, "name": "tool-state"}]),
            "POST /tags/": json.dumps({"detail": "Tag with name 'tool-state' already exists"}),
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "ts-data-id",
                "processed_data_id": "ts-proc-id",
            }),
        })

        server = mock_server_factory(routes)
        streamer = TimeseriesStreamer(api_key="k", endpoint=server.endpoint)

        data_id = streamer.initialize(stream_name="Tool state", tags=["tool-state"])

        assert data_id == "ts-data-id"
        attach = next(r for r in server.stop_and_get_requests() if "/tags/data-items/" in r)
        assert winner_id in attach

    def test_initialize_reuses_sample_named_in_409(self, mock_server_factory):
        """A sample create answered 409 (another stream just made it) reuses that sample.

        The race itself is covered in test_rheed_stream; this pins the timeseries wiring.
        """
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        winner_id = "55555555-5555-5555-5555-555555555555"
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 4,
            "__status_sequence__": {"POST /physical_samples/": [409]},
            "GET /physical_samples/": "[]",
            "POST /physical_samples/": json.dumps({
                "detail": {"message": "taken", "physical_sample_id": winner_id}
            }),
            "POST /data_entries/physical_sample": '"OK"',
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "ts-data-id",
                "processed_data_id": "ts-proc-id",
            }),
        })

        server = mock_server_factory(routes)
        streamer = TimeseriesStreamer(api_key="k", endpoint=server.endpoint)

        data_id = streamer.initialize(stream_name="Tool state", physical_sample="Shared-Run")

        assert data_id == "ts-data-id"
        link = next(
            r for r in server.stop_and_get_requests() if "/data_entries/physical_sample" in r
        )
        assert winner_id in link

    def test_initialize_link_failure_finalizes_stream_and_names_data_id(
        self, mock_server_factory
    ):
        """A tag attach failing after the stream exists finalizes the stream, and the error
        names the data_id instead of leaving a `stream_active` item the caller cannot see."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        stream = json.dumps({"data_id": "ts-data-id", "processed_data_id": "ts-proc-id"})
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 4,
            "__status_sequence__": {"POST /tags/data-items/": [500]},
            "GET /tags/": json.dumps([{"id": "t1", "name": "growth"}]),
            "POST /tags/data-items/": '{"detail": "boom"}',
            "/tool-state/stream/ts-data-id/finalize": stream,
            "/tool-state/stream/initialize": stream,
        })

        server = mock_server_factory(routes)
        streamer = TimeseriesStreamer(api_key="k", endpoint=server.endpoint)

        with pytest.raises(RuntimeError, match="stream ts-data-id was created .* so it was ended"):
            streamer.initialize(stream_name="Tool state", tags=["growth"])

        assert any(
            r.startswith("REQUEST:POST:/tool-state/stream/ts-data-id/finalize:")
            for r in server.stop_and_get_requests()
        )

    def test_initialize_unknown_tag_creates_no_stream(self, mock_server_factory):
        """An unknown tag UUID raises before the stream is created."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 2,
            "GET /tags/": "[]",
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "ts-data-id",
                "processed_data_id": "ts-proc-id",
            }),
        })

        server = mock_server_factory(routes)
        streamer = TimeseriesStreamer(api_key="k", endpoint=server.endpoint)

        with pytest.raises(RuntimeError, match="tag with id .* not found"):
            streamer.initialize(tags=["99999999-9999-9999-9999-999999999999"])

        requests = server.stop_and_get_requests()
        assert not any("/tool-state/stream/initialize" in r for r in requests)


class TestTimeseriesStreamerPush:
    """Tests for TimeseriesStreamer.push() method."""

    def test_push_validates_length_mismatch(self):
        """Verify push() raises error when timestamps/values lengths don't match."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

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
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        # Configure routes for initialize + chunk
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 2,
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "test-data-id-123",
                "processed_data_id": "test-processed-id",
            }),
            "/tool-state/stream/chunk": json.dumps({
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
        assert "/tool-state/stream/chunk" in requests[1]


class TestTimeseriesStreamerRun:
    """Tests for TimeseriesStreamer.run() method (iterator mode)."""

    def test_run_validates_length_mismatch(self, mock_server_factory):
        """Verify run() raises error when any chunk has mismatched lengths."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

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
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        # Configure routes for initialize + 3 chunks
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 4,
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "test-data-id",
                "processed_data_id": "test-processed-id",
            }),
            "/tool-state/stream/chunk": json.dumps({
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
        assert "/tool-state/stream/initialize" in requests[0]
        # Remaining 3 should be chunks
        chunk_requests = [r for r in requests if "/tool-state/stream/chunk" in r]
        assert len(chunk_requests) == 3


class TestTimeseriesStreamerPushMulti:
    """Tests for TimeseriesStreamer.push_multi() method."""

    def test_push_multi_validates_length_mismatch(self):
        """Verify push_multi() raises error when any channel has mismatched lengths."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

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

    def test_push_multi_sends_one_request_per_chunk(self, mock_server_factory):
        """Verify push_multi() sends every channel of a chunk, with units, in one request."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 2,
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "test-data-id",
                "processed_data_id": "test-processed-id",
            }),
            "/tool-state/stream/chunk": json.dumps({
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
        assert len(requests) == 2  # 1 init + 1 chunk carrying both channels
        method, path, body = requests[1].removeprefix("REQUEST:").split(":", 2)
        assert (method, path) == ("POST", "/tool-state/stream/chunk/multi")
        payload = json.loads(body)
        assert payload["data_id"] == "test-data-id"
        assert payload["chunk_index"] == 0
        assert payload["channels"] == {
            "temperature": {"timestamps": [0.0, 0.01], "values": [25.0, 25.1], "units": "C"},
            "pressure": {"timestamps": [0.0, 0.01], "values": [1.0, 1.1]},
        }

    def test_push_multi_retries_server_errors(self, mock_server_factory):
        """A 5xx response is retried with the same body until the chunk lands."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 10,
            "__status_sequence__": {"/tool-state/stream/chunk": [500, 503]},
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "test-data-id",
                "processed_data_id": "test-processed-id",
            }),
            "/tool-state/stream/chunk": json.dumps([]),
        })
        server = mock_server_factory(routes)

        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()
        streamer.push_multi(
            data_id=data_id,
            chunk_index=3,
            channels={"temperature": {"timestamps": [0.0, 0.01], "values": [25.0, 25.1]}},
        )

        time.sleep(3.0)  # backoff: 0.5 s, then 1 s
        chunk_requests = [r for r in server.stop_and_get_requests() if "/chunk/multi" in r]
        assert len(chunk_requests) == 3
        assert len(set(chunk_requests)) == 1

    def test_push_multi_omits_empty_channels(self, mock_server_factory):
        """An empty channel is left out of the batch instead of failing its siblings."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 2,
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "test-data-id",
                "processed_data_id": "test-processed-id",
            }),
            "/tool-state/stream/chunk": json.dumps([]),
        })
        server = mock_server_factory(routes)

        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()
        streamer.push_multi(
            data_id=data_id,
            chunk_index=0,
            channels={
                "temperature": {"timestamps": [0.0, 0.01], "values": [25.0, 25.1]},
                "pressure": {"timestamps": [], "values": []},
            },
        )

        time.sleep(0.5)
        body = server.get_all_requests()[1].removeprefix("REQUEST:").split(":", 2)[2]
        assert list(json.loads(body)["channels"]) == ["temperature"]


class TestTimeseriesStreamerIntegration:
    """Integration tests for full streaming workflow."""

    def test_full_streaming_workflow(self, mock_server_factory):
        """Test a complete streaming workflow with initialize and multiple chunks."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        # Configure routes
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 4,
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "workflow-data-id",
                "processed_data_id": "workflow-processed-id",
            }),
            "/tool-state/stream/chunk": json.dumps({
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
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 2,
            "/tool-state/stream/initialize": json.dumps({
                "data_id": "test-data-id",
                "processed_data_id": "test-processed-id",
            }),
            "/tool-state/stream/test-data-id/finalize": json.dumps({
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

    def test_finalize_waits_for_uploads_in_flight(self, mock_server_factory):
        """The server refuses a chunk that lands after finalize, so finalize waits for it."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        server = mock_server_factory(_routes(3, delays={"/tool-state/stream/chunk": [1.0]}))
        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()
        streamer.push_multi(data_id, 0, {"temperature": {"timestamps": [0.0], "values": [25.0]}})

        report = streamer.finalize(data_id)

        # The mock logs each request as it answers it, so this is the order they completed in.
        assert _paths(server.get_all_requests()) == [
            "/tool-state/stream/initialize",
            "/tool-state/stream/chunk/multi",
            "/tool-state/stream/test-data-id/finalize",
        ]
        assert report == {
            "chunks_uploaded": 1,
            "chunks_failed": 0,
            "chunks_pending": 0,
            "drained": True,
            "lost": [],
        }

    def test_finalize_warns_about_chunks_not_uploaded(self, mock_server_factory):
        """A chunk the server refused is named in one warning; the stream is still finalized."""
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        server = mock_server_factory(_routes(3, statuses={"/tool-state/stream/chunk": [400]}))
        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()
        streamer.push_multi(data_id, 5, {"temperature": {"timestamps": [0.0], "values": [25.0]}})

        with pytest.warns(RuntimeWarning, match="1 chunk.*not uploaded.*chunk 5: HTTP 400"):
            report = streamer.finalize(data_id)

        assert report["chunks_failed"] == 1
        assert _paths(server.get_all_requests())[-1] == "/tool-state/stream/test-data-id/finalize"

    def test_a_failed_finalize_keeps_the_report_for_the_retry(self, mock_server_factory):
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        server = mock_server_factory(
            _routes(4, statuses={"/tool-state/stream/test-data-id/finalize": [503]})
        )
        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()
        streamer.push_multi(data_id, 0, {"temperature": {"timestamps": [0.0], "values": [25.0]}})

        with pytest.raises(RuntimeError, match="503"):
            streamer.finalize(data_id)
        report = streamer.finalize(data_id)

        assert report["chunks_uploaded"] == 1

    def test_finalize_closes_the_stream_when_warnings_are_errors(self, mock_server_factory):
        """The warning about lost chunks is raised only after the stream is finalized."""
        import warnings

        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        server = mock_server_factory(_routes(3, statuses={"/tool-state/stream/chunk": [400]}))
        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()
        streamer.push_multi(data_id, 5, {"temperature": {"timestamps": [0.0], "values": [25.0]}})

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning, match="1 chunk"):
                streamer.finalize(data_id)

        paths = _paths(server.stop_and_get_requests())
        assert paths[-1] == "/tool-state/stream/test-data-id/finalize"


def _routes(max_requests: int, delays=None, statuses=None) -> str:
    """Routes for one stream, `test-data-id`: initialize, chunk uploads and finalize."""
    routes = {
        "__routes__": True,
        "__max_requests__": max_requests,
        "/tool-state/stream/initialize": json.dumps({
            "data_id": "test-data-id",
            "processed_data_id": "test-processed-id",
        }),
        "/tool-state/stream/chunk": json.dumps([]),
        "/tool-state/stream/test-data-id/finalize": json.dumps({
            "data_id": "test-data-id",
            "processed_data_id": "test-processed-id",
        }),
    }
    if delays:
        routes["__delay_sequence__"] = delays
    if statuses:
        routes["__status_sequence__"] = statuses
    return json.dumps(routes)


def _paths(requests: list[str]) -> list[str]:
    return [request.removeprefix("REQUEST:").split(":", 2)[1] for request in requests]


class TestTimeseriesStreamerFlush:
    """flush() waits for the uploads push() and push_multi() started, and says what became of them."""

    def test_flush_waits_for_uploads_in_flight(self, mock_server_factory):
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        server = mock_server_factory(_routes(3, delays={"/tool-state/stream/chunk": [1.0]}))
        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()
        streamer.push_multi(data_id, 0, {"temperature": {"timestamps": [0.0], "values": [25.0]}})
        streamer.push(data_id, 1, "pressure", [0.0], [1.0])

        started = time.monotonic()
        report = streamer.flush(data_id)

        assert time.monotonic() - started >= 0.9
        assert report == {
            "chunks_uploaded": 2,
            "chunks_failed": 0,
            "chunks_pending": 0,
            "drained": True,
            "lost": [],
        }

    def test_flush_that_times_out_reports_chunks_still_uploading(self, mock_server_factory):
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        server = mock_server_factory(_routes(2, delays={"/tool-state/stream/chunk": [1.5]}))
        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()
        streamer.push_multi(data_id, 0, {"temperature": {"timestamps": [0.0], "values": [25.0]}})

        report = streamer.flush(data_id, timeout_s=0.2)
        assert (report["chunks_pending"], report["drained"], report["chunks_uploaded"]) == (1, False, 0)

        # The next flush reports the chunk once it lands.
        report = streamer.flush(data_id)
        assert (report["chunks_pending"], report["chunks_uploaded"]) == (0, 1)

    def test_flush_reports_chunks_the_server_refused(self, mock_server_factory):
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        server = mock_server_factory(_routes(3, statuses={"/tool-state/stream/chunk": [400]}))
        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()
        streamer.push(data_id, 7, "temperature", [0.0], [25.0])
        streamer.push(data_id, 7, "pressure", [0.0], [1.0])

        report = streamer.flush(data_id)

        assert (report["chunks_failed"], report["chunks_uploaded"]) == (1, 1)
        [(chunk_index, reason)] = report["lost"]
        assert chunk_index == 7
        assert "HTTP 400" in reason
        assert reason.startswith(("channel 'temperature'", "channel 'pressure'"))

    def test_flush_with_nothing_pushed_returns_at_once(self):
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint="http://localhost:9999")

        assert streamer.flush("no-such-stream")["drained"] is True

    def test_flush_rejects_a_negative_timeout(self):
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint="http://localhost:9999")

        with pytest.raises(ValueError, match="timeout_s must be non-negative"):
            streamer.flush("test-data-id", timeout_s=-1)


class TestTimeseriesStreamerReleasesGil:
    """Network round trips run with the GIL released, so the caller's other threads keep running."""

    def test_initialize(self, mock_server_factory, gil_released_during):
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        server = mock_server_factory(_routes(1, delays={"/tool-state/stream/initialize": [1.0]}))
        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)

        data_id = gil_released_during(streamer.initialize)

        assert data_id == "test-data-id"

    def test_finalize(self, mock_server_factory, gil_released_during):
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        server = mock_server_factory(
            _routes(2, delays={"/tool-state/stream/test-data-id/finalize": [1.0]})
        )
        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()

        gil_released_during(lambda: streamer.finalize(data_id))

    def test_run(self, mock_server_factory, gil_released_during):
        from atomscale.streaming.rheed_stream import TimeseriesStreamer

        server = mock_server_factory(_routes(2, delays={"/tool-state/stream/chunk": [1.0]}))
        streamer = TimeseriesStreamer(api_key="test-api-key", endpoint=server.endpoint)
        data_id = streamer.initialize()

        gil_released_during(
            lambda: streamer.run(data_id, "temperature", iter([([0.0], [25.0])]))
        )
