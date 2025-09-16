pytest_plugins = ["pytest_order"]

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


class ResultIDs:
    rheed_image = ""
    rheed_stationary = ""
    rheed_rotating = ""
    xps = ""
