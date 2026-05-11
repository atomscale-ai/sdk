from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterable
from typing import Any

import pandas as pd
import pytest

from atomscale import Client
from atomscale.similarity.polling import (
    aiter_poll_trajectory,
    iter_poll_trajectory,
    start_polling_trajectory_task,
    start_polling_trajectory_thread,
)
from atomscale.timeseries.polling import (
    _drift_corrected_sleep,
    aiter_poll,
    iter_poll,
    start_polling_task,
    start_polling_thread,
)

from .conftest import ResultIDs

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
        "atomscale.timeseries.polling.get_provider", lambda name: provider
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
        "atomscale.timeseries.polling.get_provider", lambda name: provider
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
        "atomscale.timeseries.polling.get_provider", lambda name: provider
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
    errors: list[BaseException] = []
    monkeypatch.setattr(
        "atomscale.timeseries.polling.get_provider", lambda name: provider
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
    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda d: sleep_calls.append(d))

    recorded_bounds: list[float] = []

    def fake_uniform(a: float, b: float) -> float:
        recorded_bounds.append(b)
        return 0.0

    import random as _random

    monkeypatch.setattr(_random, "uniform", fake_uniform)

    provider = SeqProvider([{"x": 1}, {"x": 2}])
    monkeypatch.setattr(
        "atomscale.timeseries.polling.get_provider", lambda name: provider
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
        "atomscale.timeseries.polling.get_provider", lambda name: provider
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
        "atomscale.timeseries.polling.get_provider", lambda name: provider
    )

    got: list[int] = []
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
        "atomscale.timeseries.polling.get_provider", lambda name: provider
    )

    got: list[int] = []
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
    errors: list[BaseException] = []
    monkeypatch.setattr(
        "atomscale.timeseries.polling.get_provider", lambda name: provider
    )

    got: list[int] = []
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
        "atomscale.timeseries.polling.get_provider", lambda name: provider
    )

    seen: list[int] = []

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
        "atomscale.timeseries.polling.get_provider", lambda name: provider
    )

    seen: list[int] = []
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
        "atomscale.timeseries.polling.get_provider", lambda name: provider1
    )
    out1 = list(
        iter_poll(client, data_id, interval=0.01, max_polls=1, fire_immediately=True)
    )
    assert out1 == [{"x": 1}]

    # False
    monkeypatch.setattr(
        "atomscale.timeseries.polling.get_provider", lambda name: provider2
    )
    out2 = list(
        iter_poll(client, data_id, interval=0.01, max_polls=1, fire_immediately=False)
    )
    assert out2 == [{"y": 1}]


# ---------- Trajectory Polling Helpers ----------


class TrajectorySeqProvider:
    """Provider that yields trajectory-like DataFrames with Active column."""

    def __init__(self, active_sequence: Iterable[bool]):
        self._active_list = list(active_sequence)
        self.calls = 0

    def fetch_raw(self, _client: Client, _source_id: str, **kwargs) -> Any:
        self.calls += 1
        idx = min(self.calls - 1, len(self._active_list) - 1)
        active = self._active_list[idx]
        return {"Active": active, "call": self.calls}

    def to_dataframe(self, raw: Any) -> pd.DataFrame:
        # Return a DataFrame with Active column (matching real provider output)
        return pd.DataFrame({"Active": [raw["Active"]], "call": [raw["call"]]})


class FlakyTrajectoryProvider:
    """Trajectory provider that raises once, then returns data."""

    def __init__(self):
        self.calls = 0

    def fetch_raw(self, _client: Client, _source_id: str, **kwargs) -> Any:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("trajectory fetch failed")
        return {"Active": True, "call": self.calls}

    def to_dataframe(self, raw: Any) -> pd.DataFrame:
        return pd.DataFrame({"Active": [raw["Active"]], "call": [raw["call"]]})


@pytest.fixture
def source_id() -> str:
    return "test-source-id-123"


# ---------- iter_poll_trajectory (sync) ----------


def test_iter_poll_trajectory_yields_max_polls(
    monkeypatch: pytest.MonkeyPatch, client: Client, source_id: str
):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    provider = TrajectorySeqProvider([True, True, True])
    monkeypatch.setattr(
        "atomscale.similarity.polling.get_provider", lambda name: provider
    )

    results = list(iter_poll_trajectory(client, source_id, interval=0.01, max_polls=3))
    assert len(results) == 3
    assert provider.calls == 3


def test_iter_poll_trajectory_default_until_stops_when_inactive(
    monkeypatch: pytest.MonkeyPatch, client: Client, source_id: str
):
    """Default until predicate stops when Active is False."""
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    # Active=True, True, False -> should stop after 3rd
    provider = TrajectorySeqProvider([True, True, False])
    monkeypatch.setattr(
        "atomscale.similarity.polling.get_provider", lambda name: provider
    )

    results = list(iter_poll_trajectory(client, source_id, interval=0.01))
    assert len(results) == 3
    assert results[-1]["Active"].iloc[0] == False


def test_iter_poll_trajectory_custom_until(
    monkeypatch: pytest.MonkeyPatch, client: Client, source_id: str
):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    provider = TrajectorySeqProvider([True, True, True, True])
    monkeypatch.setattr(
        "atomscale.similarity.polling.get_provider", lambda name: provider
    )

    results = list(
        iter_poll_trajectory(
            client,
            source_id,
            interval=0.01,
            until=lambda r: r["call"].iloc[0] >= 2,  # Stop after 2nd call
        )
    )
    assert len(results) == 2


def test_iter_poll_trajectory_dedupes_by_key(
    monkeypatch: pytest.MonkeyPatch, client: Client, source_id: str
):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    # All return same Active value, so dedupe by Active should yield only first
    provider = TrajectorySeqProvider([True, True, True])
    monkeypatch.setattr(
        "atomscale.similarity.polling.get_provider", lambda name: provider
    )

    results = list(
        iter_poll_trajectory(
            client,
            source_id,
            interval=0.01,
            max_polls=3,
            distinct_by=lambda r: r["Active"].iloc[0],
        )
    )
    # Only first unique Active value yielded
    assert len(results) == 1
    assert results[0]["Active"].iloc[0] == True


def test_iter_poll_trajectory_on_error_and_continue(
    monkeypatch: pytest.MonkeyPatch, client: Client, source_id: str
):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    provider = FlakyTrajectoryProvider()
    errors: list[BaseException] = []
    monkeypatch.setattr(
        "atomscale.similarity.polling.get_provider", lambda name: provider
    )

    results = list(
        iter_poll_trajectory(
            client,
            source_id,
            interval=0.01,
            max_polls=2,
            on_error=errors.append,
        )
    )
    assert len(errors) == 1
    assert len(results) == 1
    assert results[0]["call"].iloc[0] == 2


# ---------- aiter_poll_trajectory (async) ----------


@pytest.mark.asyncio
async def test_aiter_poll_trajectory_yields_max_polls(
    monkeypatch: pytest.MonkeyPatch, client: Client, source_id: str
):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    provider = TrajectorySeqProvider([True, True, True])
    monkeypatch.setattr(
        "atomscale.similarity.polling.get_provider", lambda name: provider
    )

    got: list[Any] = []
    async for r in aiter_poll_trajectory(client, source_id, interval=0.01, max_polls=3):
        got.append(r)
    assert len(got) == 3
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_aiter_poll_trajectory_default_until(
    monkeypatch: pytest.MonkeyPatch, client: Client, source_id: str
):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    provider = TrajectorySeqProvider([True, True, False])
    monkeypatch.setattr(
        "atomscale.similarity.polling.get_provider", lambda name: provider
    )

    got: list[Any] = []
    async for r in aiter_poll_trajectory(client, source_id, interval=0.01):
        got.append(r)
    assert len(got) == 3
    assert got[-1]["Active"].iloc[0] == False


@pytest.mark.asyncio
async def test_aiter_poll_trajectory_on_error_and_continue(
    monkeypatch: pytest.MonkeyPatch, client: Client, source_id: str
):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    provider = FlakyTrajectoryProvider()
    errors: list[BaseException] = []
    monkeypatch.setattr(
        "atomscale.similarity.polling.get_provider", lambda name: provider
    )

    got: list[Any] = []
    async for r in aiter_poll_trajectory(
        client,
        source_id,
        interval=0.01,
        max_polls=2,
        on_error=errors.append,
    ):
        got.append(r)
    assert len(errors) == 1
    assert len(got) == 1


# ---------- start_polling_trajectory_task (async background) ----------


@pytest.mark.asyncio
async def test_start_polling_trajectory_task_awaits_on_result(
    monkeypatch: pytest.MonkeyPatch, client: Client, source_id: str
):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    provider = TrajectorySeqProvider([True, True, True])
    monkeypatch.setattr(
        "atomscale.similarity.polling.get_provider", lambda name: provider
    )

    seen: list[int] = []

    async def on_result(item):
        await asyncio.sleep(0)
        seen.append(item["call"].iloc[0])

    task = start_polling_trajectory_task(
        client,
        source_id,
        interval=0.01,
        max_polls=3,
        on_result=on_result,
    )
    await task
    assert seen == [1, 2, 3]


# ---------- start_polling_trajectory_thread (sync background) ----------


def test_start_polling_trajectory_thread_stops_with_event(
    monkeypatch: pytest.MonkeyPatch, client: Client, source_id: str
):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    provider = TrajectorySeqProvider([True, True, True, True, True])
    monkeypatch.setattr(
        "atomscale.similarity.polling.get_provider", lambda name: provider
    )

    seen: list[int] = []
    first_seen = threading.Event()

    def on_result(item):
        seen.append(item["call"].iloc[0])
        if len(seen) == 1:
            first_seen.set()

    stop = start_polling_trajectory_thread(
        client, source_id, interval=0.01, on_result=on_result
    )
    assert first_seen.wait(timeout=1.0), "did not receive first result in time"
    stop.set()
    time.sleep(0.05)
    assert len(seen) >= 1
