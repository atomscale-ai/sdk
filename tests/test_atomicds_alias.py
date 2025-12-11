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

    from atomscale import Client  # noqa: WPS433
    from atomicds import Client as LegacyClient  # noqa: WPS433

    assert LegacyClient is Client
    assert legacy_root.__version__ == importlib.import_module("atomscale").__version__

    from atomicds.core import BaseClient  # noqa: WPS433
    from atomscale.core import BaseClient as NewBaseClient  # noqa: WPS433

    assert BaseClient is NewBaseClient

    from atomicds.timeseries.polling import iter_poll  # noqa: WPS433
    from atomscale.timeseries.polling import iter_poll as new_iter_poll  # noqa: WPS433

    assert iter_poll is new_iter_poll
