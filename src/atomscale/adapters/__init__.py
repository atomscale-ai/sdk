"""Vendor-specific adapters that bridge instrument APIs to atomscale ingestion.

The generic adapter host (``python -m atomscale.adapters``) discovers adapters
registered in :mod:`atomscale.adapters.registry`, exposes their config schemas,
launches them, and monitors their JSON-line status streams. Implement
:class:`Adapter` and register it to add a new instrument.
"""

from atomscale.adapters.base import Adapter, StatusEmitter

__all__ = ["Adapter", "StatusEmitter"]
