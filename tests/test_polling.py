from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Iterable, List

import pytest


from .conftest import ResultIDs
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
def data_id() -> str:
    # Take the first ID from the rotating demo set
    return ResultIDs.rheed_rotating[0]


@pytest.fixture
def result(client: Client):
    # Example "real-ish" payload you can reuse in tests
    results = client.get(data_ids=ResultIDs.rheed_rotating)
    return results[0]


# ---------- Test helpers (mock providers) ----------


class SeqProvider:
    """Provider that yields a predefined sequence via fetch_raw()."""

    def __init__(self, seq: Iterable[Any]):
        self._it = iter(seq)
        self.calls = 0

    def fetch_raw(self, _client: Client, _data_id: str) -> Any:
        self.calls += 1
        try:
            return next(self._it)
        except StopIteration:
            return {"rev": self.calls}  # continue returning a stable value

    def to_dataframe(self, raw: Any) -> Any:
        # In tests we just pass-through; in prod this is a DataFrame typically.
        return raw


class FlakyThenOKProvider:
    """Provider that raises once, then returns monotonically increasing revs."""

    def __init__(self):
        self.calls = 0

    def fetch_raw(self, _client: Client, _data_id: str) -> Any:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return {"rev": self.calls}

    def to_dataframe(self, raw: Any) -> Any:
        return raw


# ---------- Unit tests for _drift_corrected_sleep ----------


def test_drift_corrected_sleep_future(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    delay = _drift_corrected_sleep(next_tick=100.3, interval=0.1)
    assert delay == pytest.approx(0.3, abs=1e-6)


def test_drift_corrected_sleep_past(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    delay = _drift_corrected_sleep(next_tick=99.0, interval=1.0)
    assert delay == pytest.approx(0.0, abs=1e-9)


# ---------- iter_poll (sync) ----------


def test_iter_poll_yields_max_polls(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    provider = SeqProvider([{"i": 1}, {"i": 2}, {"i": 3}])
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

    results = list(iter_poll(client, data_id, interval=0.01, max_polls=3))
    assert [r["i"] for r in results] == [1, 2, 3]
    assert provider.calls == 3


def test_iter_poll_dedupes_by_key(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    provider = SeqProvider([{"rev": 1}, {"rev": 1}, {"rev": 2}])
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

    results = list(
        iter_poll(
            client,
            data_id,
            interval=0.01,
            max_polls=3,
            distinct_by=lambda r: r["rev"],
        )
    )
    assert [r["rev"] for r in results] == [1, 2]


def test_iter_poll_until_predicate(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    provider = SeqProvider([{"status": "ok"}, {"status": "done"}, {"status": "ok"}])
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

    results = list(
        iter_poll(
            client,
            data_id,
            interval=0.01,
            until=lambda r: r.get("status") == "done",
        )
    )
    assert [r["status"] for r in results] == ["ok", "done"]


def test_iter_poll_on_error_and_continue(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    provider = FlakyThenOKProvider()
    errors: List[BaseException] = []
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

    results = list(
        iter_poll(
            client,
            data_id,
            interval=0.01,
            max_polls=2,  # first raises, second succeeds
            on_error=errors.append,
        )
    )
    assert len(errors) == 1
    assert len(results) == 1 and results[0]["rev"] == 2


def test_iter_poll_jitter_uses_interval_bound(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    sleep_calls: List[float] = []
    monkeypatch.setattr(time, "sleep", lambda d: sleep_calls.append(d))

    recorded_bounds: List[float] = []

    def fake_uniform(a: float, b: float) -> float:
        recorded_bounds.append(b)
        return 0.0

    import random as _random

    monkeypatch.setattr(_random, "uniform", fake_uniform)

    provider = SeqProvider([{"x": 1}, {"x": 2}])
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

    it = iter_poll(
        client,
        data_id,
        interval=0.2,
        jitter=999.0,
        max_polls=2,  # clamp jitter to interval
    )
    next(it)
    next(it)  # consume
    assert recorded_bounds and recorded_bounds[0] == pytest.approx(0.2)


def test_iter_poll_with_fixture_result_payload(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str, result: Any
):
    """Ensure we can carry real-ish payloads through the provider path."""
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    provider = SeqProvider(
        [{"rev": 1, "payload": result}, {"rev": 2, "payload": result}]
    )
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

    out = list(
        iter_poll(
            client, data_id, interval=0.01, max_polls=2, distinct_by=lambda r: r["rev"]
        )
    )
    assert [o["rev"] for o in out] == [1, 2]
    # payload passed through untouched
    assert out[0]["payload"] is result


# ---------- aiter_poll (async) ----------


@pytest.mark.asyncio
async def test_aiter_poll_yields_max_polls(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    provider = SeqProvider([{"i": 1}, {"i": 2}, {"i": 3}])
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

    got: List[int] = []
    async for r in aiter_poll(client, data_id, interval=0.01, max_polls=3):
        got.append(r["i"])
    assert got == [1, 2, 3]
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_aiter_poll_dedupes(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    provider = SeqProvider([{"rev": 1}, {"rev": 1}, {"rev": 2}])
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

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
async def test_aiter_poll_on_error_and_continue(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    provider = FlakyThenOKProvider()
    errors: List[BaseException] = []
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

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
async def test_start_polling_task_awaits_on_result(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    provider = SeqProvider([{"n": 1}, {"n": 2}, {"n": 3}])
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

    seen: List[int] = []

    async def on_result(item):
        await asyncio.sleep(0)  # prove await happens
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


def test_start_polling_thread_stops_with_event(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    provider = SeqProvider([{"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}, {"n": 5}])
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider
    )

    seen: List[int] = []
    first_seen = threading.Event()

    def on_result(item):
        seen.append(item["n"])
        if len(seen) == 1:
            first_seen.set()

    stop = start_polling_thread(client, data_id, interval=0.01, on_result=on_result)
    assert first_seen.wait(timeout=1.0), "did not receive first result in time"
    stop.set()
    time.sleep(0.05)  # allow thread to exit
    assert len(seen) >= 1


# ---------- Misc: fire_immediately behavioral smoke ----------


def test_iter_poll_fire_immediately_smoke(
    monkeypatch: pytest.MonkeyPatch, client: Client, data_id: str
):
    """Smoke test to ensure both True/False do not crash and yield results."""
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    provider1 = SeqProvider([{"x": 1}])
    provider2 = SeqProvider([{"y": 1}])

    # True
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider1
    )
    out1 = list(
        iter_poll(client, data_id, interval=0.01, max_polls=1, fire_immediately=True)
    )
    assert out1 == [{"x": 1}]

    # False
    monkeypatch.setattr(
        "atomicds.timeseries.polling.get_provider", lambda name: provider2
    )
    out2 = list(
        iter_poll(client, data_id, interval=0.01, max_polls=1, fire_immediately=False)
    )
    assert out2 == [{"y": 1}]
