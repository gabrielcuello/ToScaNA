"""
Deprecated import names kept for existing notebooks.

The subpackages `math` and `io` were renamed to `numerics` and `dataio` so
that no folder in the project shadows the standard-library modules `math`
and `io`. This module makes the old dotted names still importable:

    import toscana.math.fourier            -> toscana.numerics.fourier
    from toscana.io.loading import read_3col -> toscana.dataio.loading

with a DeprecationWarning. No file or folder named `math` or `io` exists, so
the standard library can never be shadowed.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from types import ModuleType
from typing import Sequence

RENAMED = {"math": "numerics", "io": "dataio"}


def _new_name(fullname: str) -> str | None:
    for old, new in RENAMED.items():
        prefix = f"{__package__}.{old}"
        if fullname == prefix or fullname.startswith(prefix + "."):
            return f"{__package__}.{new}" + fullname[len(prefix):]
    return None


def warn_renamed(old: str, new: str, stacklevel: int = 3) -> None:
    warnings.warn(
        f"`{old}` is deprecated; use `{new}` instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: ModuleType) -> None:
        self._target = target

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return self._target

    def exec_module(self, module: ModuleType) -> None:
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        new = _new_name(fullname)
        if new is None:
            return None
        warn_renamed(fullname, new, stacklevel=4)
        module = importlib.import_module(new)
        return importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(module), is_package=hasattr(module, "__path__")
        )


def install() -> None:
    # Must come first: otherwise, once `toscana.math` resolves to the
    # `numerics` package, the regular path finder would load
    # `toscana.math.fourier` from numerics/ as a SECOND copy of the module.
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
