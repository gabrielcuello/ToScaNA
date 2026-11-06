"""
Nuclear data table and the `Element` class.

Public names
------------
Element
    Nuclear data (masses, scattering lengths in fm, cross sections in barns)
    of an element (``"Ni"``) or isotope (``"Li6"``, ``"H2"``), from
    `toscana.isotopes.database.ISOTOPE_DICT`. See `toscana.isotopes.core`.
Elemento, elemento
    Deprecated aliases of `Element`; accessing them emits a
    ``DeprecationWarning``. Not in ``__all__``.

The table itself and its column names and units are in
`toscana.isotopes.database` (``ISOTOPE_DICT``, ``NOMBRES``, ``UNITS``); the
validators and `getFree` are in `toscana.isotopes.core`.
"""

import warnings
from typing import Any

from .core import _DEPRECATED_ALIASES, Element

__all__ = ["Element"]


def __getattr__(name: str) -> Any:
    """
    Resolve ``toscana.isotopes.Elemento`` / ``elemento`` (PEP 562).

    Parameters
    ----------
    name : str
        Attribute requested from the package.

    Returns
    -------
    type
        `Element`, after a ``DeprecationWarning`` "`<name>` is deprecated;
        use `Element` instead.".

    Raises
    ------
    AttributeError
        For any other unknown name.
    """
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
