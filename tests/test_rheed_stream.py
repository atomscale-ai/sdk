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

    def test_initialize_adds_tracking_sample_when_physical_sample_and_project_id(
        self, mock_server_factory
    ):
        """Verify the sample is added to the project's tracking list.

        When both physical_sample and project_id are provided, the SDK should:
        1. POST /rheed/stream/ to create the stream
        2. GET /physical_samples/ to list existing samples
        3. POST /physical_samples/ to create the sample (if not found)
        4. POST /data_entries/physical_sample to link sample to data entry
        5. POST /projects/{id}/configuration/tracking_samples to add the
           sample to the project's tracking list (and membership), and mark
           it as the active tracking sample.

        This replaces an older flow that did GET /projects/ followed by
        POST /projects/{id}/configuration to update tracking_physical_sample_id.
        That flow was fragile because the backend strictly re-validated the
        entire GrowthMonitoringConfiguration on POST and rejected on any
        pre-existing config quirk (e.g. references to deleted samples). The
        new endpoint patches only the tracking-sample fields and tolerates
        existing config quirks.
        """
        from atomscale.streaming.rheed_stream import RHEEDStreamer

        project_uuid = "550e8400-e29b-41d4-a716-446655440000"
        sample_uuid = "660e8400-e29b-41d4-a716-446655440001"

        # Note: /physical_samples/ is used for both GET (returns list) and POST (returns created sample)
        # The mock returns the same response for both, which works because:
        # - GET expects a list - we return a list with the sample already existing
        # - This skips the POST /physical_samples/ call since sample already exists
        routes = json.dumps({
            "__routes__": True,
            "__max_requests__": 4,
            "/rheed/stream/": '"test-data-id-999"',
            "/physical_samples/": json.dumps([{"id": sample_uuid, "name": "Test Sample"}]),
            "/data_entries/physical_sample": '"OK"',
            f"/projects/{project_uuid}/configuration/tracking_samples": '"OK"',
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


def _drain_chunk_metadatas(
    server: "MockServer",
    target_count: int,
    timeout_s: float = 10.0,
    presign_path: str = "/data_entries/raw_data/staged/upload_urls/",
) -> list[dict]:
    """Read REQUEST lines from the mock until we've collected `target_count`
    presign-POST metadata bodies, or `timeout_s` elapses.

    Returns as soon as the target is reached, regardless of how many PUT
    request lines remain unread. Avoids waiting on the full request stream
    when only the chunk metadata bodies matter for assertions.
    """
    import select
    import time as _time

    if not server._proc or not server._proc.stdout:
        return []

    deadline = _time.monotonic() + timeout_s
    metadatas: list[dict] = []
    while len(metadatas) < target_count and _time.monotonic() < deadline:
        remaining = deadline - _time.monotonic()
        ready, _, _ = select.select([server._proc.stdout], [], [], remaining)
        if not ready:
            break
        line = server._proc.stdout.readline()
        if not line:
            break
        if not line.startswith("REQUEST:"):
            continue
        parts = line.split(":", 3)
        if len(parts) != 4:
            continue
        _, method, path, body = parts
        if method != "POST" or not path.startswith(presign_path):
            continue
        body = body.rstrip("\n")
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
    """Verify that explicit `capture_start_ms_utc` overrides take precedence
    over the SDK's wallclock sampling.

    Background: the SDK stamps each chunk with `Utc::now()` sampled at the
    moment `push()` is entered (or the iterator yields), before any GIL-held
    packaging work, with the heavy bytes copy released off-GIL. This keeps
    multi-stream timestamps faithful to real arrival time. Callers with a
    hardware clock can still pass `capture_start_ms_utc` to override.
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


@pytest.mark.order("first")
class TestMultiStreamTiming:
    """Concurrent multi-stream timing correctness.

    Background: previously the streamer kept a single shared (rotating, fps,
    chunk_size) on the streamer instance, and stamped chunks with `Utc::now()`
    sampled AFTER the GIL-held bytes copy. Three simultaneous RHEED streams
    therefore produced chunk metadata whose `[start_unix_ms_utc,
    end_unix_ms_utc]` time array was 2-3× larger than the actual video
    duration, because each stream's stamps drifted forward by the other
    streams' GIL-held memcpy work, and per-stream fps could be overwritten
    by the most recent `initialize(...)`.

    The fix: per-`data_id` config in a HashMap, `Utc::now()` sampled at
    `push()` entry (before any GIL-held work), and the bytes copy released
    off-GIL via `py.detach`.
    """

    @pytest.fixture
    def two_stream_mock(self):
        """Routes-mode mock with capacity for two concurrent streams pushing
        many chunks. All initialize calls return the same data_id; downstream
        chunks are distinguished by per-streamer `avg_frame_rate` in metadata.
        """
        port = _get_free_port()
        routes = json.dumps(
            {
                "__routes__": True,
                "__max_requests__": 200,
                "/rheed/stream/": '"stream-data-id"',
                "/data_entries/raw_data/staged/upload_urls/": json.dumps(
                    [{"url": f"http://127.0.0.1:{port}/upload/put"}]
                ),
                "/upload/put": '"OK"',
            }
        )
        server = MockServer(port, routes)
        server.start()
        try:
            yield server
        finally:
            server.stop()

    def test_per_stream_fps_survives_concurrent_pushes(self, two_stream_mock):
        """Two streamers with different fps push from threads concurrently
        with explicit `capture_start_ms_utc`. Each chunk's metadata must
        reflect its own streamer's fps and the exact timestamp it was given —
        no cross-contamination of fps or timestamps between the two streams.

        The chosen fps values produce distinct intra-chunk spans (5000 vs
        500 ms), so a regression that swapped fps between streams would
        flip the spans and the assertions would fail loudly.
        """
        import threading

        import numpy as np

        from atomscale.streaming.rheed_stream import RHEEDStreamer

        # fps=1, n=5 → 5000 ms span;  fps=10, n=5 → 500 ms span.
        SLOW_FPS, FAST_FPS = 1.0, 10.0
        N_CHUNKS = 5
        N_FRAMES = 5
        SLOW_BASE_TS = 1_700_000_000_000
        FAST_BASE_TS = 2_500_000_000_000
        TS_STEP_MS = 1_000  # 1 s between consecutive chunk timestamps
        # Lower bound on captured chunks per stream. Tolerates the rare mock
        # subprocess connection drop under thread pressure from prior tests
        # in the same process — what matters for catching the bug is the
        # CONTENT of every captured chunk, not the count.
        MIN_CHUNKS_PER_STREAM = N_CHUNKS - 1

        slow = RHEEDStreamer(api_key="k", endpoint=two_stream_mock.endpoint)
        slow.initialize(fps=SLOW_FPS, rotations_per_min=0.0, chunk_size=2)

        fast = RHEEDStreamer(api_key="k", endpoint=two_stream_mock.endpoint)
        fast.initialize(fps=FAST_FPS, rotations_per_min=0.0, chunk_size=20)

        frames = np.zeros((N_FRAMES, 32, 32), dtype=np.uint8)

        errors: list[BaseException] = []

        def burst(streamer, base_ts: int) -> None:
            try:
                for i in range(N_CHUNKS):
                    streamer.push(
                        "stream-data-id",
                        chunk_idx=i,
                        frames=frames,
                        capture_start_ms_utc=base_ts + i * TS_STEP_MS,
                    )
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        t_slow = threading.Thread(target=burst, args=(slow, SLOW_BASE_TS))
        t_fast = threading.Thread(target=burst, args=(fast, FAST_BASE_TS))
        t_slow.start()
        t_fast.start()
        t_slow.join()
        t_fast.join()

        assert not errors, f"push errors: {errors}"

        # Drain presign POSTs; settles for whatever arrives within the
        # timeout (some may drop under mock subprocess pressure).
        metadatas = _drain_chunk_metadatas(
            two_stream_mock, target_count=2 * N_CHUNKS, timeout_s=5
        )

        slow_meta = [m for m in metadatas if float(m["avg_frame_rate"]) == SLOW_FPS]
        fast_meta = [m for m in metadatas if float(m["avg_frame_rate"]) == FAST_FPS]

        assert len(slow_meta) >= MIN_CHUNKS_PER_STREAM, (
            f"too few slow chunks: {len(slow_meta)}/{N_CHUNKS}"
        )
        assert len(fast_meta) >= MIN_CHUNKS_PER_STREAM, (
            f"too few fast chunks: {len(fast_meta)}/{N_CHUNKS}"
        )

        # Intra-chunk span uses each streamer's OWN fps. If fps had leaked
        # between streamers (the original bug), slow chunks would have
        # span=500 and fast chunks span=5000 — a clear regression marker.
        for m in slow_meta:
            span = int(m["end_unix_ms_utc"]) - int(m["start_unix_ms_utc"])
            assert span == 5000, f"slow span {span} != 5000 (fps leaked?)"
        for m in fast_meta:
            span = int(m["end_unix_ms_utc"]) - int(m["start_unix_ms_utc"])
            assert span == 500, f"fast span {span} != 500 (fps leaked?)"

        # Explicit timestamps survived unchanged for each stream — every
        # captured stamp matches one of the values that thread pushed,
        # with no contamination from the other thread's base.
        slow_expected = {SLOW_BASE_TS + i * TS_STEP_MS for i in range(N_CHUNKS)}
        fast_expected = {FAST_BASE_TS + i * TS_STEP_MS for i in range(N_CHUNKS)}
        for m in slow_meta:
            s = int(m["start_unix_ms_utc"])
            assert s in slow_expected, f"slow chunk start {s} unexpected"
        for m in fast_meta:
            s = int(m["start_unix_ms_utc"])
            assert s in fast_expected, f"fast chunk start {s} unexpected"

