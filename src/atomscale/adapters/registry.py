"""Registry of available ingestion adapters.

Adding an instrument = implement :class:`~atomscale.adapters.base.Adapter` and
register it here. The host and GUI then pick it up automatically.
"""

from __future__ import annotations

from atomscale.adapters.base import Adapter
from atomscale.adapters.filmsense.adapter import FilmSenseAdapter

_ADAPTERS: dict[str, Adapter] = {
    adapter.id: adapter
    for adapter in (FilmSenseAdapter(),)
}


def available() -> dict[str, Adapter]:
    """Return a copy of the id → adapter mapping."""
    return dict(_ADAPTERS)


def get(adapter_id: str) -> Adapter:
    """Look up an adapter by id, or raise ``KeyError`` with the known ids."""
    try:
        return _ADAPTERS[adapter_id]
    except KeyError:
        msg = f"unknown adapter id {adapter_id!r} (available: {sorted(_ADAPTERS)})"
        raise KeyError(msg) from None
