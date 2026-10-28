pytest_plugins = ["pytest_order"]

import os

# Streamers measure the clock against AtomCloud's /gateway/time on creation. The mock servers the
# suite uses count and order requests, so switch that off; tests/test_clock_sync.py turns it on.
os.environ.setdefault("ATOMSCALE_SKIP_CLOCK_SYNC", "1")

# Force the headless backend so plot tests never touch Tk/Cocoa on CI runners
import matplotlib

matplotlib.use("Agg", force=True)

# Minimal async test support without external plugins
# If pytest-asyncio is not installed, this hook will execute async tests
# by running the coroutine with asyncio.run(). This allows @pytest.mark.asyncio
# tests (and any async def test) to work in environments without extra deps.
import asyncio
import inspect


def pytest_pyfunc_call(pyfuncitem):
    test_func = pyfuncitem.obj
    # Only handle ourselves if no async plugin is available
    has_asyncio_plugin = pyfuncitem.config.pluginmanager.hasplugin("pytest_asyncio")
    if inspect.iscoroutinefunction(test_func) and not has_asyncio_plugin:
        asyncio.run(test_func(**pyfuncitem.funcargs))
        return True
    # return None to let pytest handle non-async tests as usual


import pytest


@pytest.fixture(autouse=True)
def _isolated_rheed_spool(tmp_path, monkeypatch):
    """Give every test its own RHEED spool directory.

    By default a streamer spools under the user cache directory, keyed by data_id
    and locked per stream. The mock-server tests reuse data_ids such as
    ``stream-data-id``, so a streamer an earlier test left alive (e.g. through an
    exception held in a reference cycle) kept that stream's lock, and the next
    test's initialize() failed with "another streamer is already spooling".
    It also left test streams in the real cache for later runs to resume.
    """
    monkeypatch.setenv("ATOMSCALE_SPOOL_DIR", str(tmp_path / "rheed-spool"))


@pytest.fixture
def python_ticks_during():
    """Run a call and count how often another Python thread ran while it blocked.

    A call that holds the GIL while it waits on the network starves every other Python
    thread, so the count stays near zero; one that releases it lets the thread tick every
    few milliseconds. Returns ``(ticks, result)``.
    """
    import threading
    import time

    def measure(call):
        ticks = 0
        stop = threading.Event()

        def tick():
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                time.sleep(0.005)

        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        try:
            result = call()
        finally:
            stop.set()
            thread.join()
        return ticks, result

    return measure


class ResultIDs:
    rheed_image = ""
    rheed_stationary = ""
    rheed_rotating = ""
    xps = ""
    optical = ""
    tool_state = ""
    recipe = ""
    photoluminescence = ""
    raman = ""
    changepoint = ""
    similarity_workflow = "rheed_stationary"
    similarity_source_id = "bb3494b1-b5fb-4f3e-ac50-e4024f8aacf5"
