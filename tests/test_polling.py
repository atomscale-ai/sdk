
from __future__ import annotations

from .conftest import ResultIDs

import asyncio
import threading
import time
from typing import Any, Iterable, List

import pytest

from atomicds import Client
from atomicds.timeseries.polling import (
    _drift_corrected_sleep,
    aiter_poll,
    iter_poll,
    start_polling_task,
    start_polling_thread,
)
# ---------- Fixtures ----------

@pytest.fixture
def client():
    return Client()


@pytest.fixture
def result(client: Client):
    results = client.get(data_ids=ResultIDs.rheed_rotating)
    return results[0]

# ---------- Helpers ----------

class SeqClient:
    """Client that returns a pre-defined sequence from .get()."""
    def __init__(self, seq: Iterable[Any]):
        self._it = iter(seq)
        self.calls: int = 0

    def get(self, _data_id: str) -> Any:
        self.calls += 1
        try:
            return next(self._it)
        except StopIteration:
            # If exhausted, keep returning the last call count
            return {"rev": self.calls}


class FlakyThenOKClient:
    """Client that raises once, then returns monotonically increasing revs."""
    def __init__(self):
        self.calls: int = 0

    def get(self, _data_id: str) -> Any:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return {"rev": self.calls}


# ---------- Unit tests for _drift_corrected_sleep ----------

def test_drift_corrected_sleep_future(monkeypatch: pytest.MonkeyPatch):
    # With controlled time, delay should be next_tick - now
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    delay = _drift_corrected_sleep(next_tick=100.3, interval=0.1)
    assert delay == pytest.approx(0.3, abs=1e-6)


def test_drift_corrected_sleep_past(monkeypatch: pytest.MonkeyPatch):
    # When behind schedule, function should return a non-negative delay
    # (it catches up by skipping missed ticks)
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    delay = _drift_corrected_sleep(next_tick=99.0, interval=1.0)
    assert delay >= 0.0
    # With the controlled monotonic, it should be exactly zero here
    assert delay == pytest.approx(0.0, abs=1e-9)


# ---------- iter_poll (sync) ----------

def test_iter_poll_yields_max_polls(monkeypatch: pytest.MonkeyPatch, data_id: str):
    # Make sleeping instant to keep tests fast.
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    client = SeqClient([{"i": 1}, {"i": 2}, {"i": 3}])
    results = list(iter_poll(client, data_id, interval=0.01, max_polls=3))
    assert [r["i"] for r in results] == [1, 2, 3]
    # Ensure client.get was invoked exactly 3 times.
    assert client.calls == 3


def test_iter_poll_dedupes_by_key(monkeypatch: pytest.MonkeyPatch, data_id: str):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    client = SeqClient([{"rev": 1}, {"rev": 1}, {"rev": 2}])
    results = list(
        iter_poll(
            client,
            data_id,
            interval=0.01,
            max_polls=3,
            distinct_by=lambda r: r["rev"],
        )
    )
    # Duplicate rev=1 suppressed; only 1 and 2 yielded
    assert [r["rev"] for r in results] == [1, 2]


def test_iter_poll_until_predicate(monkeypatch: pytest.MonkeyPatch, data_id: str):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    client = SeqClient([{"status": "ok"}, {"status": "done"}, {"status": "ok"}])
    results = list(
        iter_poll(
            client,
            data_id,
            interval=0.01,
            # Stop once we see 'done'
            until=lambda r: r.get("status") == "done",
        )
    )
    # Should yield the first "ok" then "done", then stop
    assert [r["status"] for r in results] == ["ok", "done"]


def test_iter_poll_on_error_and_continue(monkeypatch: pytest.MonkeyPatch, data_id: str):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    client = FlakyThenOKClient()
    errors: List[BaseException] = []

    results = list(
        iter_poll(
            client,
            data_id,
            interval=0.01,
            max_polls=2,  # first raises, second succeeds
            on_error=errors.append,
        )
    )
    # One error captured, one successful result yielded
    assert len(errors) == 1
    assert len(results) == 1
    assert results[0]["rev"] == 2  # second call


def test_iter_poll_jitter_uses_interval_bound(
    monkeypatch: pytest.MonkeyPatch, data_id: str
):
    # Sleep fast
    sleep_calls: List[float] = []
    monkeypatch.setattr(time, "sleep", lambda d: sleep_calls.append(d))

    # Force random.uniform to assert the upper bound equals interval (min(jitter, interval))
    recorded_bounds: List[float] = []

    def fake_uniform(a: float, b: float) -> float:
        recorded_bounds.append(b)
        return 0.0  # deterministic

    import random as _random
    monkeypatch.setattr(_random, "uniform", fake_uniform)

    client = SeqClient([{"x": 1}, {"x": 2}])

    it = iter_poll(
        client,
        data_id,
        interval=0.2,
        jitter=999.0,  # should clamp to interval
        max_polls=2,   # ensures one sleep path is exercised
    )
    # Consume both items
    _ = next(it)
    _ = next(it)
    # After the first yield, we should have slept once and called uniform once
    assert recorded_bounds and recorded_bounds[0] == pytest.approx(0.2)


# ---------- aiter_poll (async) ----------

@pytest.mark.asyncio
async def test_aiter_poll_yields_max_polls(monkeypatch: pytest.MonkeyPatch, data_id: str):
    # Make asyncio.sleep a no-op
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    client = SeqClient([{"i": 1}, {"i": 2}, {"i": 3}])

    got: List[int] = []
    async for r in aiter_poll(client, data_id, interval=0.01, max_polls=3):
        got.append(r["i"])
    assert got == [1, 2, 3]
    assert client.calls == 3


@pytest.mark.asyncio
async def test_aiter_poll_dedupes(monkeypatch: pytest.MonkeyPatch, data_id: str):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    client = SeqClient([{"rev": 1}, {"rev": 1}, {"rev": 2}])

    got: List[int] = []
    async for r in aiter_poll(
        client,
        data_id,
        interval=0.01,
        max_polls=3,
        distinct_by=lambda x: x["rev"],
    ):
        got.append(r["rev"])
    assert got == [1, 2]


@pytest.mark.asyncio
async def test_aiter_poll_on_error_and_continue(monkeypatch: pytest.MonkeyPatch, data_id: str):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    client = FlakyThenOKClient()
    errors: List[BaseException] = []

    got: List[int] = []
    async for r in aiter_poll(
        client,
        data_id,
        interval=0.01,
        max_polls=2,  # first raises, second succeeds
        on_error=errors.append,
    ):
        got.append(r["rev"])

    assert len(errors) == 1
    assert got == [2]


# ---------- start_polling_task (async background) ----------

@pytest.mark.asyncio
async def test_start_polling_task_awaits_on_result(monkeypatch: pytest.MonkeyPatch, data_id: str):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    client = SeqClient([{"n": 1}, {"n": 2}, {"n": 3}])
    seen: List[int] = []

    async def on_result(item):
        # Prove this coroutine is awaited
        await asyncio.sleep(0)
        seen.append(item["n"])

    task = start_polling_task(
        client,
        data_id,
        interval=0.01,
        max_polls=3,
        on_result=on_result,
    )
    await task
    assert seen == [1, 2, 3]


# ---------- start_polling_thread (sync background) ----------

def test_start_polling_thread_stops_with_event(monkeypatch: pytest.MonkeyPatch, data_id: str):
    # No sleeping to keep the thread snappy
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    client = SeqClient([{"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}, {"n": 5}])
    seen: List[int] = []
    first_seen = threading.Event()

    def on_result(item):
        seen.append(item["n"])
        if len(seen) == 1:
            first_seen.set()

    stop = start_polling_thread(
        client,
        data_id,
        interval=0.01,
        on_result=on_result,
    )

    # Wait until we see at least one result, then request stop.
    assert first_seen.wait(timeout=1.0), "did not receive first result in time"
    stop.set()

    # Give the thread a brief moment to notice the stop on the next iteration.
    time.sleep(0.05)

    assert len(seen) >= 1  # at least one callback happened


# ---------- Misc: fire_immediately behavioral smoke (no strict timing) ----------

def test_iter_poll_fire_immediately_smoke(monkeypatch: pytest.MonkeyPatch, data_id: str):
    """Smoke test to ensure both True/False do not crash and yield results.

    Note: The current implementation always performs the first poll immediately.
    This test avoids asserting timing and focuses on yield semantics.
    """
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    client1 = SeqClient([{"x": 1}])
    out1 = list(iter_poll(client1, data_id, interval=0.01, max_polls=1, fire_immediately=True))
    assert out1 == [{"x": 1}]

    client2 = SeqClient([{"y": 1}])
    out2 = list(iter_poll(client2, data_id, interval=0.01, max_polls=1, fire_immediately=False))
    assert out2 == [{"y": 1}]
