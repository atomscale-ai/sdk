"""Shared polling utilities."""
from __future__ import annotations

import time


def drift_corrected_sleep(next_tick: float, interval: float) -> float:
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
