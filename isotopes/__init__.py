import warnings
from typing import Any

from .core import _DEPRECATED_ALIASES, Element

__all__ = ["Element"]


def __getattr__(name: str) -> Any:
    # Deprecated Spanish names (`Elemento`, `elemento`) kept for old notebooks.
    if name in _DEPRECATED_ALIASES:
        new = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"`{name}` is deprecated; use `{new}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
