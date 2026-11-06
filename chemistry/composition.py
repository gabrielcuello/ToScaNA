"""
Back-compatibility module: re-exports composition helpers.

The functions were moved to their canonical locations; this module only
imports them so that ``from toscana.chemistry.composition import ...`` keeps
working. The objects are the same (no wrappers).

Public names
------------
AtomicAvg, getConcentrations, getNofAtoms
    From `toscana.physics.properties`.
extractAttr, extractDict
    From `toscana.utils.dicts`.
"""

from __future__ import annotations

# Back-compat wrapper: symbols were moved to their canonical locations.
from ..physics.properties import AtomicAvg, getConcentrations, getNofAtoms
from ..utils.dicts import extractAttr, extractDict

__all__ = [
    "AtomicAvg",
    "extractAttr",
    "extractDict",
    "getConcentrations",
    "getNofAtoms",
]
