from typing import Any

from . import (
    data,
    dataio,
    experiment,
    isotopes,
    models,
    numerics,
    physics,
    plotting,
    utils,
)
from . import _aliases

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "data",
    "dataio",
    "experiment",
    "isotopes",
    "models",
    "numerics",
    "physics",
    "plotting",
    "utils",
]

# Deprecated names `toscana.math` / `toscana.io` (renamed to avoid clashing
# with the standard library); see _aliases.py.
_aliases.install()


def __getattr__(name: str) -> Any:
    if name in _aliases.RENAMED:
        new = _aliases.RENAMED[name]
        _aliases.warn_renamed(f"{__name__}.{name}", f"{__name__}.{new}", stacklevel=3)
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
