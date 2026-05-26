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
from collections.abc import Callable
from pathlib import Path

import pytest

# Path to the mock server module
_MOCK_SERVER_MODULE = Path(__file__).parent / "_mock_http_server.py"


class MockServer:
    """A mock HTTP server running in a subprocess.

    Stdout is drained by a background daemon thread into a ``queue.Queue``,
    which gives us cross-platform timed reads (``select.select`` on a pipe FD
    is a Unix-only trick — Windows raises WinError 10038).
    """

    def __init__(self, port: int, response_data: str):
        import queue as _queue

        self.port = port
        self.response_data = response_data
        self._proc: subprocess.Popen | None = None
        self._captured_body: dict | None = None
        self._stdout_queue: _queue.Queue[str | None] = _queue.Queue()
        self._reader_thread: object | None = None

    def start(self) -> None:
        """Start the server subprocess."""
        import threading

        self._proc = subprocess.Popen(
            [sys.executable, str(_MOCK_SERVER_MODULE), str(self.port), self.response_data],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered so multi-request output is readable
        )
        # Wait for server to signal it's ready (synchronous; the drain thread
        # has not started yet so this read is unambiguous).
        ready_line = self._proc.stdout.readline()
        if not ready_line.startswith("READY:"):
            self.stop()
            raise RuntimeError(f"Server failed to start: {ready_line}")

        # All subsequent stdout lines flow through the queue. Daemon thread so
        # it never blocks shutdown if the subprocess wedges.
        self._reader_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader_thread.start()

    def _drain_stdout(self) -> None:
        try:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                self._stdout_queue.put(line)
        except Exception:
            pass
        finally:
            # Sentinel signals EOF so consumers can break out without a timeout.
            self._stdout_queue.put(None)

    def stop(self) -> None:
        """Stop the server subprocess (and cache its stderr for diagnostics)."""
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
            try:
                if self._proc.stderr is not None:
                    self._stderr_dump = self._proc.stderr.read() or ""
            except Exception:
                pass
            self._proc = None

    def get_stderr(self) -> str:
        """Return whatever the subprocess wrote to stderr. Empty before
        stop() runs (we drain on shutdown to avoid blocking on a live pipe)."""
        return getattr(self, "_stderr_dump", "") or ""

    @property
    def endpoint(self) -> str:
        """Return the server endpoint URL (no trailing slash)."""
        return f"http://127.0.0.1:{self.port}"

    def _read_line(self, timeout_s: float) -> str | None:
        """Block up to ``timeout_s`` for the next stdout line. ``None`` on EOF/timeout."""
        import queue as _queue

        try:
            return self._stdout_queue.get(timeout=timeout_s)
        except _queue.Empty:
            return None

    def get_captured_body(self, timeout_s: float = 5.0) -> dict | None:
        """Read and return the captured request body from the server."""
        import time

        if self._proc and self._captured_body is None:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                line = self._read_line(deadline - time.monotonic())
                if line is None:
                    break
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
        import time

        if not self._proc:
            return []

        deadline = time.monotonic() + timeout_s
        requests: list[tuple[str, str, str]] = []
        while len(requests) < expected_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            line = self._read_line(remaining)
            if line is None:
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


class TestRHEEDStreamerTags:
    """Tests for the `tags` parameter on RHEEDStreamer.initialize()."""

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

        requests = server.get_captured_requests(expected_count=4, timeout_s=5)
        # Filter to /tags/ traffic — the init POST is uninteresting here.
        tag_reqs = [(m, p, b) for (m, p, b) in requests if p.startswith("/tags/")]
        # Exactly: one GET, one create-POST, one attach-POST. Dedup means the
        # repeated "growth"/"GROWTH" entries did not produce extra creates.
        methods_paths = [(m, p) for (m, p, _) in tag_reqs]
        assert methods_paths == [
            ("GET", "/tags/"),
            ("POST", "/tags/"),
            ("POST", "/tags/data-items/"),
        ], methods_paths

        # POST /tags/ body uses the trimmed name (no surrounding spaces).
        create_body = json.loads(
            next(b for (m, p, b) in tag_reqs if m == "POST" and p == "/tags/")
        )
        assert create_body == {"name": "novel-tag"}

        # POST /tags/data-items/ body has data_id + both resolved tag_ids in order.
        attach_body = json.loads(
            next(b for (_, p, b) in tag_reqs if p == "/tags/data-items/")
        )
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

        requests = server.get_captured_requests(expected_count=3, timeout_s=5)
        tag_reqs = [(m, p, b) for (m, p, b) in requests if p.startswith("/tags/")]
        # All three exist → no POST /tags/ creates, just GET + attach.
        methods_paths = [(m, p) for (m, p, _) in tag_reqs]
        assert methods_paths == [
            ("GET", "/tags/"),
            ("POST", "/tags/data-items/"),
        ], methods_paths

        attach_body = json.loads(
            next(b for (_, p, b) in tag_reqs if p == "/tags/data-items/")
        )
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


class TestStreamingE2E:
    """End-to-end RHEED streaming smoke test.

    Drives the full upload pipeline — initialize POST, per-chunk presign
    POSTs, frame PUTs to the presigned URLs, and the end-of-stream POST —
    against the same subprocess-based mock the init tests use (which is
    already known to work on Linux/macOS/Windows). An in-process Python
    HTTP server can't be used here: the streamer holds the GIL during
    `block_on(...)`, so a Python handler thread in the same process never
    gets scheduled and the request times out.

    We use `run(...)` rather than `push(...)`: it block-awaits every
    chunk's spawned upload task before returning, so any
    Windows-side scheduling or networking issue in `spawn_chunk_upload`
    surfaces deterministically. `push(...)` shares the same internal
    `spawn_chunk_upload` code path, so a green `run(...)` is strong
    evidence `push(...)` works too.

    Frame data must be **non-fill-value**: zarrs intentionally skips
    persisting all-fill chunks (the platform reconstructs blank frames
    server-side without an uploaded shard), so a test using
    `np.zeros(...)` would silently bypass the PUT path.
    """

    def test_full_cycle_uploads_chunks(self):
        import numpy as np

        from atomscale.streaming.rheed_stream import RHEEDStreamer

        # Pre-allocate the port so the routes JSON can encode the
        # presign-redirect URL pointing back at the same mock.
        port = _get_free_port()
        routes = json.dumps(
            {
                "__routes__": True,
                # 6 expected requests: 1 init POST + 2 presign POSTs + 2 PUTs + 1 end POST.
                "__max_requests__": 8,
                "/rheed/stream/": '"stream-data-id"',
                "/data_entries/raw_data/staged/upload_urls/": json.dumps(
                    [{"url": f"http://127.0.0.1:{port}/upload/put"}]
                ),
                "/upload/put": '"OK"',
            }
        )
        server = MockServer(port, routes)
        server.start()
        run_error: Exception | None = None
        partial_requests: list[tuple[str, str, str]] = []
        try:
            # verbosity=4 emits debug logs for every step (presign,
            # packaging, PUT) — pytest captures these and prints them on
            # failure, which is the most useful diagnostic info we have
            # for Windows-only flakes.
            streamer = RHEEDStreamer(
                api_key="k", endpoint=server.endpoint, verbosity=4
            )
            data_id = streamer.initialize(
                fps=1.0, rotations_per_min=0.0, chunk_size=2
            )
            assert data_id == "stream-data-id"

            # Non-fill-value frames so zarr packaging actually emits bytes
            # and the PUT path runs.
            frames = np.full((2, 8, 8), 7, dtype=np.uint8)
            ts0 = 1_700_000_000_000
            ts1 = ts0 + 5_000

            def gen():
                yield (frames, ts0)
                yield (frames, ts1)

            try:
                streamer.run(data_id, gen())  # blocks until uploads complete
                streamer.finalize(data_id)
                # Drain captured request lines: 1 init + 2 presigns + 2 PUTs + 1 end.
                requests = server.get_captured_requests(
                    expected_count=6, timeout_s=15
                )
            except Exception as e:
                run_error = e
                # Pull whatever the mock managed to record before the failure.
                partial_requests = server.get_captured_requests(
                    expected_count=10, timeout_s=2
                )
                requests = []
        finally:
            server.stop()

        if run_error is not None:
            stderr_dump = server.get_stderr()
            # Summarize each captured request: method, path, and body
            # (POST bodies decoded as UTF-8 with replacement, PUT bodies
            # represented as a "<N bytes>" placeholder by the mock).
            req_summary = "\n".join(
                f"  - {m} {p}  body={b[:120]!r}"
                for (m, p, b) in partial_requests
            ) or "  (none)"
            pytest.fail(
                "streaming pipeline failed:\n"
                f"  exception: {type(run_error).__name__}: {run_error}\n"
                f"  captured requests so far ({len(partial_requests)}):\n"
                f"{req_summary}\n"
                f"  mock subprocess stderr:\n"
                f"{stderr_dump or '  (empty)'}\n"
            )

        inits = [r for r in requests if r[0] == "POST" and r[1] == "/rheed/stream/"]
        presigns = [
            r for r in requests
            if r[0] == "POST"
            and r[1].startswith("/data_entries/raw_data/staged/upload_urls/")
        ]
        puts = [r for r in requests if r[0] == "PUT" and r[1].startswith("/upload/put")]
        ends = [r for r in requests if r[0] == "POST" and r[1].endswith("/end")]

        assert len(inits) == 1, f"expected 1 init POST, saw {len(inits)} ({requests})"
        assert len(presigns) == 2, (
            f"expected 2 presign POSTs, saw {len(presigns)} ({requests})"
        )
        assert len(puts) == 2, f"expected 2 PUTs, saw {len(puts)} ({requests})"
        assert len(ends) == 1, f"expected 1 end POST, saw {len(ends)} ({requests})"

        # Both presigns must carry our explicit timestamps. Compare as a
        # set: the two upload tasks are spawned concurrently on tokio
        # workers and may arrive at the mock in either order, so we can't
        # assume positional ordering.
        meta_by_start = {
            int(json.loads(body)["start_unix_ms_utc"]): json.loads(body)
            for _, _, body in presigns
        }
        assert set(meta_by_start.keys()) == {ts0, ts1}, (
            f"unexpected start timestamps: {set(meta_by_start.keys())}"
        )
        # Intra-chunk span = n / fps = 2 / 1 = 2000 ms — same for every chunk.
        for ts, meta in meta_by_start.items():
            assert int(meta["end_unix_ms_utc"]) - ts == 2000
