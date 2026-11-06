"""
Deprecated import names kept for existing notebooks.

The subpackages `math` and `io` were renamed to `numerics` and `dataio` so
that no folder in the project shadows the standard-library modules `math`
and `io`. This module makes the old dotted names still importable:

    import toscana.math.fourier            -> toscana.numerics.fourier
    from toscana.io.loading import read_3col -> toscana.dataio.loading

with a DeprecationWarning. No file or folder named `math` or `io` exists, so
the standard library can never be shadowed.

How it works: `install` puts an `_AliasFinder` at the FRONT of
``sys.meta_path``. For any module name starting with ``toscana.math`` or
``toscana.io`` it imports the corresponding ``toscana.numerics...`` /
``toscana.dataio...`` module normally and returns a module spec whose loader
(`_AliasLoader`) hands back that SAME module object, so the old and new
names refer to one module (no second copy, no re-execution). The attribute
form ``toscana.math`` is handled by ``__getattr__`` in the package
``__init__``. Background: audit 2026-11, C21 (the folders ``math/`` and
``io/`` shadowed the standard library when Python was started in the
project directory). Remove this module once all notebooks use the new names.

Public names
------------
RENAMED
    ``{"math": "numerics", "io": "dataio"}``, old -> new subpackage name.
warn_renamed(old, new, stacklevel=3)
    Emit the DeprecationWarning.
install()
    Install the finder (idempotent). Called by ``toscana/__init__.py``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from types import ModuleType
from typing import Sequence

# Old subpackage name -> new subpackage name.
RENAMED = {"math": "numerics", "io": "dataio"}


def _new_name(fullname: str) -> str | None:
    """
    Translate a deprecated dotted module name to the new one.

    Parameters
    ----------
    fullname : str
        Fully qualified module name, e.g. ``"toscana.math.fourier"``.

    Returns
    -------
    str or None
        ``"toscana.numerics.fourier"`` for the example; the same for
        ``toscana.io`` -> ``toscana.dataio``. None if `fullname` is neither
        ``toscana.<old>`` nor starts with ``toscana.<old>.`` (so
        ``toscana.mathx`` is not matched). The package name is taken from
        ``__package__``.
    """
    for old, new in RENAMED.items():
        prefix = f"{__package__}.{old}"
        if fullname == prefix or fullname.startswith(prefix + "."):
            return f"{__package__}.{new}" + fullname[len(prefix):]
    return None


def warn_renamed(old: str, new: str, stacklevel: int = 3) -> None:
    """
    Emit a DeprecationWarning "`<old>` is deprecated; use `<new>` instead.".

    Parameters
    ----------
    old, new : str
        Deprecated and replacement names, as shown in the message.
    stacklevel : int, optional
        Passed to ``warnings.warn`` so that the warning points to the user's
        code. Default 3 (caller of the caller).
    """
    warnings.warn(
        f"`{old}` is deprecated; use `{new}` instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


class _AliasLoader(importlib.abc.Loader):
    """
    Loader that returns an already imported module instead of creating one.

    Parameters
    ----------
    target : module
        The module (e.g. ``toscana.numerics.fourier``) to return.
    """

    def __init__(self, target: ModuleType) -> None:
        """Store the target module."""
        self._target = target

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        """Return the target module itself (no new module object)."""
        return self._target

    def exec_module(self, module: ModuleType) -> None:
        """Do nothing: the target module has already been executed."""
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    """
    Meta-path finder for the deprecated names ``toscana.math...`` and
    ``toscana.io...``.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """
        Return a spec aliasing `fullname` to its new module, or None.

        Parameters
        ----------
        fullname : str
            Module being imported.
        path, target
            Part of the finder protocol; unused.

        Returns
        -------
        importlib.machinery.ModuleSpec or None
            None if `fullname` is not a deprecated name (the normal finders
            then handle it). Otherwise: emits the DeprecationWarning
            (stacklevel 4), imports the new module with
            ``importlib.import_module`` and returns a spec named `fullname`
            with an `_AliasLoader` for that module, marked as a package if
            the module has ``__path__``.

        Raises
        ------
        ImportError
            If the new module does not exist (``ModuleNotFoundError`` from
            ``import_module``), e.g. ``toscana.math.nonexistent``.
        """
        new = _new_name(fullname)
        if new is None:
            return None
        warn_renamed(fullname, new, stacklevel=4)
        module = importlib.import_module(new)
        return importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(module), is_package=hasattr(module, "__path__")
        )


def install() -> None:
    """
    Insert an `_AliasFinder` at the front of ``sys.meta_path``.

    Idempotent: does nothing if one is already installed. Called once by
    ``toscana/__init__.py`` at import time.
    """
    # Must come first: otherwise, once `toscana.math` resolves to the
    # `numerics` package, the regular path finder would load
    # `toscana.math.fourier` from numerics/ as a SECOND copy of the module.
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
