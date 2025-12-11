"""Backward-compatibility shim for the renamed :mod:`atomscale` package."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_DEPRECATION_MESSAGE = (
    "atomicds is deprecated; install and import atomscale instead. "
    "Support for the atomicds alias will be removed in a future release."
)

# Warn as soon as the legacy namespace is imported.
warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

# Load the real package and mirror its attributes.
_atomscale = importlib.import_module("atomscale")
__all__ = getattr(_atomscale, "__all__", [])  # noqa: PLE0605
__path__ = getattr(_atomscale, "__path__", [])
__version__ = getattr(_atomscale, "__version__", None)

if _atomscale.__spec__:
    __spec__ = importlib.util.spec_from_loader(  # type: ignore[assignment]
        __name__,
        loader=None,
        origin=_atomscale.__spec__.origin,
        is_package=True,
    )


def __getattr__(name: str):
    """Delegate attribute lookups to the real package or its submodules."""
    try:
        return getattr(_atomscale, name)
    except AttributeError:
        return importlib.import_module(f"{_atomscale.__name__}.{name}")


def __dir__() -> list[str]:
    return sorted(set(dir(_atomscale)))


class _AtomicdsAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Redirect ``atomicds.*`` imports to ``atomscale.*``."""

    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ARG002
        if not fullname.startswith("atomicds."):
            return None

        target_name = fullname.replace("atomicds", "atomscale", 1)
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None

        is_package = target_spec.submodule_search_locations is not None
        return importlib.util.spec_from_loader(
            fullname, self, origin=target_spec.origin, is_package=is_package
        )

    def create_module(
        self,
        spec,  # noqa: ARG002
    ):  # pragma: no cover - default module creation is fine
        return None

    def exec_module(self, module):
        target_name = module.__spec__.name.replace("atomicds", "atomscale", 1)  # type: ignore[union-attr]
        target_module = importlib.import_module(target_name)
        sys.modules[module.__spec__.name] = target_module  # type: ignore[union-attr]


# Install the finder so any atomicds.* import falls back to atomscale.*.
if not any(isinstance(f, _AtomicdsAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AtomicdsAliasFinder())

# Alias already-imported atomscale submodules to the legacy namespace.
for _mod_name, _mod in list(sys.modules.items()):
    if _mod_name.startswith("atomscale."):
        sys.modules.setdefault(_mod_name.replace("atomscale", "atomicds", 1), _mod)
