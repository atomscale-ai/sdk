"""Streams are stamped in AtomCloud reference UTC, like the gateway: the lab clock is measured
with an NTP-style four-timestamp exchange against GET /gateway/time and every timestamp is
corrected by that offset.

The mock answers /gateway/time with a fixed reference time ahead of this computer's clock, so the
measured offset is that skew less the time the test took to get there.
"""
import json
import time

import numpy as np
import pytest

from atomscale.streaming.rheed_stream import RHEEDStreamer, TimeseriesStreamer

from .test_rheed_stream import MockServer, _get_free_port, _logged_requests

SKEW_MS = 5_000
TOLERANCE_MS = 150  # round-trip noise on loopback; the mock's fixed reference time ages, see below


@pytest.fixture(autouse=True)
def clock_sync_on(monkeypatch):
    monkeypatch.delenv("ATOMSCALE_SKIP_CLOCK_SYNC", raising=False)


def _server(extra_routes: dict, skew_ms: int | None = SKEW_MS) -> MockServer:
    port = _get_free_port()
    routes = {
        "__routes__": True,
        "__max_requests__": 60,
        "/rheed/stream/": '"stream-data-id"',
        "/tool-state/stream/initialize": json.dumps(
            {"data_id": "ts-data-id", "processed_data_id": "ts-processed-id"}
        ),
        "/data_entries/raw_data/staged/upload_urls/": json.dumps(
            [{"url": f"http://127.0.0.1:{port}/upload/put"}]
        ),
        "/upload/put": '"OK"',
        **extra_routes,
    }
    if skew_ms is not None:
        # The mock's reference time is fixed, so it ages while the server starts and the test runs:
        # the measured offset is the skew less the time elapsed since this call (`server.t_ref_ms`).
        reference = time.time() * 1000 + skew_ms
        routes["/gateway/time"] = json.dumps(
            {"receivedAtUnixMs": reference, "transmittedAtUnixMs": reference}
        )
    server = MockServer(port, json.dumps(routes))
    server.t_ref_ms = time.time() * 1000
    server.start()
    return server


def _expected_offset_bounds(server, before_ms: float, after_ms: float) -> tuple[float, float]:
    """The offset a measurement taken between `before_ms` and `after_ms` must report."""
    skew_at = lambda now_ms: SKEW_MS - (now_ms - server.t_ref_ms)  # noqa: E731
    return skew_at(after_ms) - TOLERANCE_MS, skew_at(before_ms) + TOLERANCE_MS


def _presigned_starts(requests) -> list[tuple[int, int]]:
    return [
        (int(json.loads(b)["start_unix_ms_utc"]), int(json.loads(b)["end_unix_ms_utc"]))
        for m, p, b in requests
        if p.startswith("/data_entries/raw_data/staged/")
    ]


class TestRHEEDClockCorrection:
    def test_chunk_stamps_carry_the_measured_offset(self):
        server = _server({})
        try:
            before = time.time() * 1000
            streamer = RHEEDStreamer(api_key="k", endpoint=server.endpoint)
            data_id = streamer.initialize(fps=1.0, rotations_per_min=0.0, chunk_size=2)
            after = time.time() * 1000
            status = streamer.clock_status()
            host_capture = int(time.time() * 1000) - 10_000
            frames = np.full((2, 8, 8), 7, dtype=np.uint8)
            streamer.push(data_id, 0, frames, capture_start_ms_utc=host_capture)
            streamer.finalize(data_id)
            requests = _logged_requests(server)
        finally:
            server.stop()
        assert status is not None
        low, high = _expected_offset_bounds(server, before, after)
        assert low < status["offset_ms"] < high
        assert status["uncertainty_ms"] < TOLERANCE_MS
        ((start, end),) = _presigned_starts(requests)
        assert abs(start - (host_capture + status["offset_ms"])) < 2  # the offset slews, never jumps
        assert end - start == 2_000

    def test_default_capture_time_is_corrected_too(self):
        server = _server({})
        try:
            streamer = RHEEDStreamer(api_key="k", endpoint=server.endpoint)
            data_id = streamer.initialize(fps=1.0, rotations_per_min=0.0, chunk_size=2)
            offset = streamer.clock_status()["offset_ms"]
            before = int(time.time() * 1000)
            streamer.push(data_id, 0, np.full((2, 8, 8), 7, dtype=np.uint8))
            after = int(time.time() * 1000)
            streamer.finalize(data_id)
            requests = _logged_requests(server)
        finally:
            server.stop()
        ((_, end),) = _presigned_starts(requests)
        # The chunk had just finished capturing, so it ends at the push, in reference time.
        assert before + offset - 2 < end < after + offset + 2

    def test_an_unreachable_reference_clock_streams_uncorrected(self):
        server = _server({}, skew_ms=None)  # /gateway/time is a 404
        try:
            streamer = RHEEDStreamer(api_key="k", endpoint=server.endpoint)
            data_id = streamer.initialize(fps=1.0, rotations_per_min=0.0, chunk_size=2)
            host_capture = 1_700_000_000_000
            streamer.push(data_id, 0, np.full((2, 8, 8), 7, dtype=np.uint8), capture_start_ms_utc=host_capture)
            streamer.finalize(data_id)
            requests = _logged_requests(server)
            status = streamer.clock_status()
        finally:
            server.stop()
        assert status is None
        assert _presigned_starts(requests) == [(host_capture, host_capture + 2_000)]

    def test_the_skip_switch_sends_host_time(self, monkeypatch):
        monkeypatch.setenv("ATOMSCALE_SKIP_CLOCK_SYNC", "1")
        server = _server({})
        try:
            streamer = RHEEDStreamer(api_key="k", endpoint=server.endpoint)
            data_id = streamer.initialize(fps=1.0, rotations_per_min=0.0, chunk_size=2)
            streamer.push(data_id, 0, np.full((2, 8, 8), 7, dtype=np.uint8), capture_start_ms_utc=1_700_000_000_000)
            streamer.finalize(data_id)
            requests = _logged_requests(server)
        finally:
            server.stop()
        assert streamer.clock_status() is None
        assert not any(p == "/gateway/time" for _, p, _ in requests)
        assert _presigned_starts(requests) == [(1_700_000_000_000, 1_700_000_002_000)]


class TestTimeseriesClockCorrection:
    def test_points_carry_the_same_offset(self):
        server = _server({"/tool-state/stream/chunk/multi": '"OK"', "/tool-state/stream/ts-data-id/finalize": '"OK"'})
        try:
            before = time.time() * 1000
            streamer = TimeseriesStreamer(api_key="k", endpoint=server.endpoint)
            data_id = streamer.initialize(stream_name="tool")
            after = time.time() * 1000
            offset_s = streamer.clock_status()["offset_ms"] / 1000  # before a later burst sample ages it
            host_times = [1_700_000_000.0, 1_700_000_001.0]
            streamer.push_multi(data_id, 0, {"temp": {"timestamps": host_times, "values": [1.0, 2.0]}})
            time.sleep(1.0)
            requests = _logged_requests(server)
        finally:
            server.stop()
        sent = next(json.loads(b) for _, p, b in requests if p.startswith("/tool-state/stream/chunk/multi"))
        points = sent["channels"]["temp"]["timestamps"]
        low, high = _expected_offset_bounds(server, before, after)
        assert low / 1000 < offset_s < high / 1000
        assert points == pytest.approx([t + offset_s for t in host_times], abs=0.01)
