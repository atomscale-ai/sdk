# polling.py
from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from pandas import DataFrame

from atomicds.timeseries.registry import get_provider

Result = DataFrame
DistinctFn = Callable[[Result], Any]
Predicate = Callable[[Result], bool]
ErrorHandler = Callable[[BaseException], None]


def _fetch_result(client, data_id: str, last_n: int | None) -> Result:
    """Build a result via provider -> fetch_raw -> to_dataframe.

    Args:
        client: API client instance passed to the provider.
        data_id: Identifier of the resource to fetch.
        last_n: Last number of entries to poll for

    Returns:
        Any: The provider-converted result (typically a pandas.DataFrame).
    """
    provider = get_provider("rheed")
    kwargs = {"last_n": last_n} if last_n is not None else {}
    raw = provider.fetch_raw(client, data_id, **kwargs)
    return provider.to_dataframe(raw)


def _drift_corrected_sleep(next_tick: float, interval: float) -> float:
    """Compute a non-negative sleep delay to keep a fixed polling cadence.

    This helper calculates how long to sleep to reach `next_tick`, and if
    the caller is behind schedule, it computes a catch-up strategy that
    preserves the target cadence (i.e., it skips missed ticks rather than
    accumulating delay).

    Args:
        next_tick: Target monotonic time (seconds) for the next poll.
        interval: Desired polling interval in seconds.

    Returns:
        float: Non-negative delay (seconds) to sleep before the next poll.
    """
    now = time.monotonic()
    delay = next_tick - now
    if delay < 0:
        missed = int((-delay) // interval)
        next_tick += missed * interval
        delay = next_tick - time.monotonic()
    return max(0.0, delay)


def iter_poll(
    client,
    data_id: str,
    *,
    interval: float = 1.0,
    last_n: int | None = None,
    distinct_by: DistinctFn | None = None,
    until: Predicate | None = None,
    max_polls: int | None = None,
    fire_immediately: bool = True,
    jitter: float = 0.0,
    on_error: ErrorHandler | None = None,
) -> Iterator[Result]:
    """
    Yield time series results at a fixed cadence.

    Supports deduplication (via a key extractor), stop conditions,
    bounded polling, optional jitter, and non-fatal error handling.

    Args:
        client: API client instance forwarded to the provider.
        data_id: Identifier to fetch data for.
        last_n: Last number of time series data points to poll. None is all.
        interval: Seconds between polls. Defaults to 1.0.
        distinct_by: Optional function mapping a result to a hashable key for
            deduping. If provided, only results with a new key are yielded.
        until: Optional predicate; stop when it returns True for a result.
        max_polls: Optional maximum number of polls before stopping.
        fire_immediately: If True, perform the first poll immediately; otherwise
            wait one interval before the first poll. Defaults to True.
        jitter: Optional random delay (0..jitter) added to each sleep to avoid
            thundering herds. Clamped at `interval`. Defaults to 0.0.
        on_error: Optional error handler called with the raised exception when a
            poll fails. Errors are swallowed so polling continues.

    Yields:
        Any: Each (optionally deduped) time series data frame result.

    Notes:
        - Uses drift-corrected scheduling to maintain the requested cadence
          even if individual polls are slow.
        - Stops when `until` is satisfied or `max_polls` is reached (if set).
    """
    last_key = object()
    polls = 0
    next_tick = time.monotonic()

    if not fire_immediately:
        next_tick += interval

    while True:
        polls += 1
        try:
            result = _fetch_result(client, data_id, last_n)
        except BaseException as exc:
            if on_error:
                on_error(exc)
        else:
            key = distinct_by(result) if distinct_by else object()
            if distinct_by is None or key != last_key:
                last_key = key
                yield result
                if until and until(result):
                    return
        if max_polls and polls >= max_polls:
            return

        # timing
        next_tick += interval
        delay = _drift_corrected_sleep(next_tick, interval)
        if jitter:
            delay += random.uniform(0, max(0.0, min(jitter, interval)))
        time.sleep(delay)


async def aiter_poll(
    client,
    data_id: str,
    *,
    interval: float = 1.0,
    last_n: int | None = None,
    distinct_by: DistinctFn | None = None,
    until: Predicate | None = None,
    max_polls: int | None = None,
    fire_immediately: bool = True,
    jitter: float = 0.0,
    on_error: ErrorHandler | None = None,
) -> AsyncIterator[Result]:
    """
    Asynchronously yield time series results without blocking the loop.

    Uses the the same semantics as `iter_poll`.

    Args:
        client: API client instance forwarded to the provider.
        data_id: Identifier to fetch data for.
        interval: Seconds between polls. Defaults to 1.0.
        last_n: Last number of time series data points to poll. None is all.
        distinct_by: Optional function mapping a result to a hashable key for
            deduping. If provided, only results with a new key are yielded.
        until: Optional predicate; stop when it returns True for a result.
        max_polls: Optional maximum number of polls before stopping.
        fire_immediately: If True, perform the first poll immediately; otherwise
            wait one interval before the first poll. Defaults to True.
        jitter: Optional random delay (0..jitter) added to each sleep to avoid
            thundering herds. Clamped at `interval`. Defaults to 0.0.
        on_error: Optional error handler called with the raised exception when a
            poll fails. Errors are swallowed so polling continues.

    Yields:
        Any: Each (optionally deduped) time series data frame result.

    Notes:
        - Uses `asyncio.to_thread` so provider calls never block the event loop.
        - Drift-corrected scheduling preserves cadence even with slow polls.
        - Stops when `until` is satisfied or `max_polls` is reached (if set).
    """
    loop = asyncio.get_running_loop()
    last_key = object()
    polls = 0
    next_tick = loop.time()

    if not fire_immediately:
        next_tick += interval

    while True:
        polls += 1
        try:
            result = await asyncio.to_thread(_fetch_result, client, data_id, last_n)
        except BaseException as exc:
            if on_error:
                on_error(exc)
        else:
            key = distinct_by(result) if distinct_by else object()
            if distinct_by is None or key != last_key:
                last_key = key
                yield result
                if until and until(result):
                    return
        if max_polls and polls >= max_polls:
            return

        # timing
        next_tick += interval
        delay = next_tick - loop.time()
        if delay < 0:
            missed = int((-delay) // interval) + 1
            next_tick += missed * interval
            delay = next_tick - loop.time()
        if jitter:
            delay += random.uniform(0, max(0.0, min(jitter, interval)))
        await asyncio.sleep(delay)


def start_polling_thread(
    client,
    data_id: str,
    *,
    interval: float = 1.0,
    last_n: int | None = None,
    on_result: Callable[[Result], None],
    **kwargs,
) -> threading.Event:
    """Start the sync poller in a background thread and stream results to a callback.

    Wraps `iter_poll` in a daemon thread and invokes `on_result(result)` for
    each yielded item. Returns a `threading.Event` that can be set to stop
    polling gracefully.

    Args:
        client: API client instance forwarded to the provider.
        data_id: Identifier to fetch data for.
        interval: Seconds between polls. Defaults to 1.0.
        last_n: Last number of time series data points to poll for. None is all.
        on_result: Callback invoked with each yielded result.
        **kwargs: Additional keyword arguments forwarded to `iter_poll`
            (e.g., `distinct_by`, `until`, `max_polls`, `fire_immediately`,
            `jitter`, `on_error`).

    Returns:
        threading.Event: Event that, when set, requests the polling thread to stop.
    """
    stop = threading.Event()

    def _runner():
        for res in iter_poll(
            client, data_id, interval=interval, last_n=last_n, **kwargs
        ):
            if stop.is_set():
                break
            on_result(res)

    t = threading.Thread(target=_runner, name=f"poll:{data_id}", daemon=True)
    t.start()
    return stop


def start_polling_task(
    client,
    data_id: str,
    *,
    interval: float = 1.0,
    last_n: int | None = None,
    on_result: Callable[[Result], Any] | None = None,
    **kwargs,
) -> asyncio.Task[None]:
    """Start the async poller as an `asyncio.Task` and stream results to a callback.

    Wraps `aiter_poll` in a background Task. If `on_result` returns a coroutine,
    it will be awaited before the next iteration.

    Args:
        client: API client instance forwarded to the provider.
        data_id: Identifier to fetch data for.
        interval: Seconds between polls. Defaults to 1.0.
        last_n: Last number of time series data points to poll for. None is all.
        on_result: Optional callback invoked with each yielded result. If it
            returns a coroutine, it will be awaited.
        **kwargs: Additional keyword arguments forwarded to `aiter_poll`
            (e.g., `distinct_by`, `until`, `max_polls`, `fire_immediately`,
            `jitter`, `on_error`).

    Returns:
        asyncio.Task[None]: A created and started Task. Cancel it to stop polling.

    Raises:
        RuntimeError: If no running event loop is available when called.
    """

    async def _runner():
        async for res in aiter_poll(
            client, data_id, interval=interval, last_n=last_n, **kwargs
        ):
            if on_result is None:
                continue
            maybe = on_result(res)
            if asyncio.iscoroutine(maybe):
                await maybe

    return asyncio.create_task(_runner(), name=f"poll:{data_id}")
