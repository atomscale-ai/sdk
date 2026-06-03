"""Polling utilities for similarity trajectory data."""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from pandas import DataFrame

from atomscale.timeseries._polling_utils import drift_corrected_sleep
from atomscale.timeseries.registry import get_provider

Result = DataFrame
DistinctFn = Callable[[Result], Any]
Predicate = Callable[[Result], bool]
ErrorHandler = Callable[[BaseException], None]

_DEFAULT_TRAJECTORY_WORKFLOW = "rheed_stationary"


def _fetch_trajectory_result(
    client, source_id: str, last_n: int | None, *, display: bool = True
) -> Result:
    """Fetch trajectory data via SimilarityTrajectoryProvider.

    Args:
        client: API client instance passed to the provider.
        source_id: The data_id or physical_sample_id to fetch trajectory for.
        last_n: Last number of entries to poll for.
        display: If True (default), DataFrame columns carry user-facing display
            labels; if False they keep their raw snake_case API names.

    Returns:
        DataFrame: Trajectory data with multi-index ["Reference ID", "Time"]
        (or ["reference_id", "real_time_seconds"] when ``display`` is False).
    """
    provider = get_provider("similarity_trajectory")
    kwargs: dict[str, Any] = {"workflow": _DEFAULT_TRAJECTORY_WORKFLOW}
    if last_n is not None:
        kwargs["last_n"] = last_n
    raw = provider.fetch_raw(client, source_id, **kwargs)
    return provider.to_dataframe(raw, display=display)


def _default_trajectory_until(df: Result) -> bool:
    """Default stop condition: stop when trajectory has data and is no longer active.

    Reads the active flag under either its display label (``Active``) or its
    raw API name (``is_active``) so the predicate works regardless of the
    ``display`` mode the poller was started with.
    """
    if len(df) == 0:
        return False
    active_col = "Active" if "Active" in df.columns else "is_active"
    return not df[active_col].any()


def iter_poll_trajectory(
    client,
    source_id: str,
    *,
    interval: float = 1.0,
    last_n: int | None = None,
    display: bool = True,
    distinct_by: DistinctFn | None = None,
    until: Predicate | None = None,
    max_polls: int | None = None,
    fire_immediately: bool = True,
    jitter: float = 0.0,
    on_error: ErrorHandler | None = None,
) -> Iterator[Result]:
    """Synchronously poll similarity trajectory data, yielding DataFrames.

    Args:
        client: API client instance forwarded to the provider.
        source_id: The data_id or physical_sample_id to poll trajectory for.
        interval: Seconds between polls. Defaults to 1.0.
        last_n: Last number of trajectory data points to poll. None is all.
        display: If True (default), DataFrame columns use user-facing display
            labels; if False they keep raw snake_case API names.
        distinct_by: Optional function mapping a result to a hashable key for
            deduping. If provided, only results with a new key are yielded.
        until: Optional predicate; stop when it returns True for a result.
            If None, defaults to stopping when no trajectory is active
            (i.e., ``not df["Active"].any()``).
        max_polls: Optional maximum number of polls before stopping.
        fire_immediately: If True, perform the first poll immediately; otherwise
            wait one interval before the first poll. Defaults to True.
        jitter: Optional random delay (0..jitter) added to each sleep to avoid
            thundering herds. Clamped at `interval`. Defaults to 0.0.
        on_error: Optional error handler called with the raised exception when a
            poll fails. Errors are swallowed so polling continues.

    Yields:
        DataFrame: Trajectory data with multi-index ["Reference ID", "Time"].

    Notes:
        - Uses drift-corrected scheduling to maintain the requested cadence
          even if individual polls are slow.
        - Stops when `until` is satisfied or `max_polls` is reached (if set).
    """
    effective_until = until if until is not None else _default_trajectory_until
    last_key = object()
    polls = 0
    next_tick = time.monotonic()

    if not fire_immediately:
        next_tick += interval

    while True:
        polls += 1
        try:
            result = _fetch_trajectory_result(
                client, source_id, last_n, display=display
            )
        except BaseException as exc:
            if on_error:
                on_error(exc)
        else:
            key = distinct_by(result) if distinct_by else object()
            if distinct_by is None or key != last_key:
                last_key = key
                yield result
                if effective_until(result):
                    return
        if max_polls and polls >= max_polls:
            return

        next_tick += interval
        delay = drift_corrected_sleep(next_tick, interval)
        if jitter:
            delay += random.uniform(0, max(0.0, min(jitter, interval)))
        time.sleep(delay)


async def aiter_poll_trajectory(
    client,
    source_id: str,
    *,
    interval: float = 1.0,
    last_n: int | None = None,
    display: bool = True,
    distinct_by: DistinctFn | None = None,
    until: Predicate | None = None,
    max_polls: int | None = None,
    fire_immediately: bool = True,
    jitter: float = 0.0,
    on_error: ErrorHandler | None = None,
) -> AsyncIterator[Result]:
    """Asynchronously poll similarity trajectory data without blocking the loop.

    Uses the same semantics as `iter_poll_trajectory`.

    Args:
        client: API client instance forwarded to the provider.
        source_id: The data_id or physical_sample_id to poll trajectory for.
        interval: Seconds between polls. Defaults to 1.0.
        last_n: Last number of trajectory data points to poll. None is all.
        display: If True (default), DataFrame columns use user-facing display
            labels; if False they keep raw snake_case API names.
        distinct_by: Optional function mapping a result to a hashable key for
            deduping. If provided, only results with a new key are yielded.
        until: Optional predicate; stop when it returns True for a result.
            If None, defaults to stopping when no trajectory is active
            (i.e., ``not df["Active"].any()``).
        max_polls: Optional maximum number of polls before stopping.
        fire_immediately: If True, perform the first poll immediately; otherwise
            wait one interval before the first poll. Defaults to True.
        jitter: Optional random delay (0..jitter) added to each sleep to avoid
            thundering herds. Clamped at `interval`. Defaults to 0.0.
        on_error: Optional error handler called with the raised exception when a
            poll fails. Errors are swallowed so polling continues.

    Yields:
        DataFrame: Trajectory data with multi-index ["Reference ID", "Time"].

    Notes:
        - Uses `asyncio.to_thread` so provider calls never block the event loop.
        - Drift-corrected scheduling preserves cadence even with slow polls.
        - Stops when `until` is satisfied or `max_polls` is reached (if set).
    """
    effective_until = until if until is not None else _default_trajectory_until
    loop = asyncio.get_running_loop()
    last_key = object()
    polls = 0
    next_tick = loop.time()

    if not fire_immediately:
        next_tick += interval

    while True:
        polls += 1
        try:
            result = await asyncio.to_thread(
                _fetch_trajectory_result, client, source_id, last_n, display=display
            )
        except BaseException as exc:
            if on_error:
                on_error(exc)
        else:
            key = distinct_by(result) if distinct_by else object()
            if distinct_by is None or key != last_key:
                last_key = key
                yield result
                if effective_until(result):
                    return
        if max_polls and polls >= max_polls:
            return

        next_tick += interval
        delay = next_tick - loop.time()
        if delay < 0:
            missed = int((-delay) // interval) + 1
            next_tick += missed * interval
            delay = next_tick - loop.time()
        if jitter:
            delay += random.uniform(0, max(0.0, min(jitter, interval)))
        await asyncio.sleep(delay)


def start_polling_trajectory_thread(
    client,
    source_id: str,
    *,
    interval: float = 1.0,
    last_n: int | None = None,
    on_result: Callable[[Result], None],
    **kwargs,
) -> threading.Event:
    """Start polling trajectory data in a background daemon thread.

    Wraps `iter_poll_trajectory` in a daemon thread and invokes
    `on_result(result)` for each yielded item. Returns a `threading.Event`
    that can be set to stop polling gracefully.

    Args:
        client: API client instance forwarded to the provider.
        source_id: The data_id or physical_sample_id to poll trajectory for.
        interval: Seconds between polls. Defaults to 1.0.
        last_n: Last number of trajectory data points to poll for. None is all.
        on_result: Callback invoked with each yielded result.
        **kwargs: Additional keyword arguments forwarded to `iter_poll_trajectory`
            (e.g., `display`, `distinct_by`, `until`, `max_polls`,
            `fire_immediately`, `jitter`, `on_error`).

    Returns:
        threading.Event: Event that, when set, requests the polling thread to stop.
    """
    stop = threading.Event()

    def _runner():
        for res in iter_poll_trajectory(
            client, source_id, interval=interval, last_n=last_n, **kwargs
        ):
            if stop.is_set():
                break
            on_result(res)

    t = threading.Thread(
        target=_runner, name=f"poll_trajectory:{source_id}", daemon=True
    )
    t.start()
    return stop


def start_polling_trajectory_task(
    client,
    source_id: str,
    *,
    interval: float = 1.0,
    last_n: int | None = None,
    on_result: Callable[[Result], Any] | None = None,
    **kwargs,
) -> asyncio.Task[None]:
    """Start polling trajectory data as an `asyncio.Task`.

    Wraps `aiter_poll_trajectory` in a background Task. If `on_result` returns
    a coroutine, it will be awaited before the next iteration.

    Args:
        client: API client instance forwarded to the provider.
        source_id: The data_id or physical_sample_id to poll trajectory for.
        interval: Seconds between polls. Defaults to 1.0.
        last_n: Last number of trajectory data points to poll for. None is all.
        on_result: Optional callback invoked with each yielded result. If it
            returns a coroutine, it will be awaited.
        **kwargs: Additional keyword arguments forwarded to `aiter_poll_trajectory`
            (e.g., `display`, `distinct_by`, `until`, `max_polls`,
            `fire_immediately`, `jitter`, `on_error`).

    Returns:
        asyncio.Task[None]: A created and started Task. Cancel it to stop polling.

    Raises:
        RuntimeError: If no running event loop is available when called.
    """

    async def _runner():
        async for res in aiter_poll_trajectory(
            client, source_id, interval=interval, last_n=last_n, **kwargs
        ):
            if on_result is None:
                continue
            maybe = on_result(res)
            if asyncio.iscoroutine(maybe):
                await maybe

    return asyncio.create_task(_runner(), name=f"poll_trajectory:{source_id}")
