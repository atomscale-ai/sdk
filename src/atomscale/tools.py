"""Tools: the instruments AtomCloud records data from, formerly called growth instruments."""

from __future__ import annotations

import warnings
from typing import Any, Literal

ToolType = Literal[
    # Process tools, which make or change a film.
    "mbe",
    "pvd",
    "sputter",
    "cvd",
    "ald",
    "pld",
    "etch",
    "anneal",
    "lithography",
    # Measurement instruments, named for the data type they produce.
    "rheed",
    "arpes",
    "xps",
    "sem",
    "optical",
    "raman",
    "photoluminescence",
    "ellipsometry",
    "sims",
    "oes",
    "xrd",
    "xrr",
    "afm",
    "uv_vis",
    "other",
]

#: The key each field of a tool had before instruments were renamed tools.
LEGACY_TOOL_KEYS = {
    "synth_source_id": "id",
    "source_name": "name",
    "synth_source_type": "tool_type",
    "source_makes_id": "make_id",
    "source_metadata": "tool_metadata",
    "source_model": "model",
    "source_manufacturer": "manufacturer",
}


def _warn_legacy_key(key: str, stacklevel: int) -> str:
    new = LEGACY_TOOL_KEYS[key]
    warnings.warn(
        f"The tool key {key!r} is deprecated; use {new!r}.",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )
    return new


class ToolRecord(dict):
    """A tool as AtomCloud returns it.

    Its keys are the current ones (``id``, ``name``, ``tool_type``, ...). Reading a
    key it had before instruments were renamed tools (``synth_source_id``,
    ``source_name``, ...) still works, with a :class:`DeprecationWarning`.
    """

    def __missing__(self, key: Any) -> Any:
        if key not in LEGACY_TOOL_KEYS or LEGACY_TOOL_KEYS[key] not in self:
            raise KeyError(key)
        return self[_warn_legacy_key(key, stacklevel=2)]

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self and key in LEGACY_TOOL_KEYS:
            if LEGACY_TOOL_KEYS[key] not in self:
                return default
            key = _warn_legacy_key(key, stacklevel=2)
        return super().get(key, default)


def as_tool(raw: Any) -> Any:
    """A tool from either API shape as a :class:`ToolRecord`; anything else as given.

    A response from before the rename carries the UUID as ``synth_source_id`` and
    an integer ``id``; the UUID becomes ``id``.
    """
    if not isinstance(raw, dict):
        return raw
    tool = ToolRecord((k, v) for k, v in raw.items() if k not in LEGACY_TOOL_KEYS)
    for old, new in LEGACY_TOOL_KEYS.items():
        if old in raw:
            tool[new] = raw[old]
    return tool
