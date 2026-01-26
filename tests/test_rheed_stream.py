"""Integration tests for RHEEDStreamer."""
import json
import multiprocessing
import socket
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import pytest


def get_free_port():
    """Get a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures request bodies."""

    captured_bodies = []
    response_data = '"test-data-id"'

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        CaptureHandler.captured_bodies.append(json.loads(body))

        response = CaptureHandler.response_data
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.end_headers()
        self.wfile.write(response.encode())

    def log_message(self, format, *args):
        pass  # Suppress log messages


def run_server(port, response_data, request_count, result_queue):
    """Run HTTP server in a subprocess."""
    CaptureHandler.response_data = response_data
    CaptureHandler.captured_bodies = []

    server = HTTPServer(("127.0.0.1", port), CaptureHandler)
    for _ in range(request_count):
        server.handle_request()

    result_queue.put(CaptureHandler.captured_bodies)
    server.server_close()


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
            endpoint="http://localhost:9999/",
        )

        with pytest.raises(ValueError, match="chunk_size must be at least 2×fps"):
            streamer.initialize(
                fps=30.0,
                rotations_per_min=0.0,
                chunk_size=30,  # Invalid: less than 2 * 30 = 60
            )

    def test_initialize_sends_project_id_in_request(self):
        """Verify project_id is included in POST body when provided."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        port = get_free_port()
        result_queue = multiprocessing.Queue()

        # Start server in subprocess
        server_proc = multiprocessing.Process(
            target=run_server,
            args=(port, '"test-data-id-123"', 1, result_queue),
        )
        server_proc.start()
        time.sleep(0.3)  # Wait for server to start

        try:
            streamer = RHEEDStreamer(
                api_key="test-api-key",
                endpoint=f"http://127.0.0.1:{port}",
            )

            project_uuid = "550e8400-e29b-41d4-a716-446655440000"
            data_id = streamer.initialize(
                fps=30.0,
                rotations_per_min=0.0,
                chunk_size=60,
                project_id=project_uuid,
            )

            assert data_id == "test-data-id-123"

            # Get captured bodies from server process
            captured_bodies = result_queue.get(timeout=5)
            assert len(captured_bodies) == 1
            captured_body = captured_bodies[0]

            assert captured_body.get("project_id") == project_uuid
            assert "data_item_name" in captured_body
            assert captured_body.get("fps_capture_rate") == 30.0
        finally:
            server_proc.join(timeout=5)
            if server_proc.is_alive():
                server_proc.terminate()

    def test_initialize_omits_project_id_when_none(self):
        """Verify project_id is omitted from POST body when not provided."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        port = get_free_port()
        result_queue = multiprocessing.Queue()

        server_proc = multiprocessing.Process(
            target=run_server,
            args=(port, '"test-data-id-456"', 1, result_queue),
        )
        server_proc.start()
        time.sleep(0.3)

        try:
            streamer = RHEEDStreamer(
                api_key="test-api-key",
                endpoint=f"http://127.0.0.1:{port}",
            )

            data_id = streamer.initialize(
                fps=30.0,
                rotations_per_min=0.0,
                chunk_size=60,
            )

            assert data_id == "test-data-id-456"

            captured_bodies = result_queue.get(timeout=5)
            assert len(captured_bodies) == 1
            captured_body = captured_bodies[0]

            assert "project_id" not in captured_body
        finally:
            server_proc.join(timeout=5)
            if server_proc.is_alive():
                server_proc.terminate()

    def test_initialize_omits_project_id_when_empty_string(self):
        """Verify empty string project_id is treated as None (omitted)."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        port = get_free_port()
        result_queue = multiprocessing.Queue()

        server_proc = multiprocessing.Process(
            target=run_server,
            args=(port, '"test-data-id-789"', 1, result_queue),
        )
        server_proc.start()
        time.sleep(0.3)

        try:
            streamer = RHEEDStreamer(
                api_key="test-api-key",
                endpoint=f"http://127.0.0.1:{port}",
            )

            data_id = streamer.initialize(
                fps=30.0,
                rotations_per_min=0.0,
                chunk_size=60,
                project_id="",
            )

            assert data_id == "test-data-id-789"

            captured_bodies = result_queue.get(timeout=5)
            assert len(captured_bodies) == 1
            captured_body = captured_bodies[0]

            assert "project_id" not in captured_body
        finally:
            server_proc.join(timeout=5)
            if server_proc.is_alive():
                server_proc.terminate()

    def test_initialize_returns_data_id(self):
        """Verify initialize() returns the data_id from the server."""
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        port = get_free_port()
        result_queue = multiprocessing.Queue()

        expected_data_id = "abc-123-xyz"
        server_proc = multiprocessing.Process(
            target=run_server,
            args=(port, f'"{expected_data_id}"', 1, result_queue),
        )
        server_proc.start()
        time.sleep(0.3)

        try:
            streamer = RHEEDStreamer(
                api_key="test-api-key",
                endpoint=f"http://127.0.0.1:{port}",
            )

            data_id = streamer.initialize(
                fps=30.0,
                rotations_per_min=0.0,
                chunk_size=60,
            )

            assert data_id == expected_data_id
        finally:
            server_proc.join(timeout=5)
            if server_proc.is_alive():
                server_proc.terminate()
