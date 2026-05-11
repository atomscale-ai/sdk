import importlib
import sys
import warnings


def test_atomicds_alias_warns_and_maps_to_atomscale():
    """The legacy namespace should warn and forward to atomscale modules."""

    sys.modules.pop("atomicds", None)
    sys.modules.pop("atomicds.core", None)
    sys.modules.pop("atomicds.timeseries.polling", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy_root = importlib.import_module("atomicds")

    assert any(
        issubclass(w.category, DeprecationWarning)
        and "deprecated" in str(w.message).lower()
        for w in caught
    )

    from atomicds import Client as LegacyClient
    from atomscale import Client

    assert LegacyClient is Client
    assert legacy_root.__version__ == importlib.import_module("atomscale").__version__

    from atomicds.core import BaseClient

    from atomscale.core import BaseClient as NewBaseClient

    assert BaseClient is NewBaseClient

    from atomicds.timeseries.polling import iter_poll

    from atomscale.timeseries.polling import iter_poll as new_iter_poll

    assert iter_poll is new_iter_poll
