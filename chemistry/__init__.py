"""
Compatibility package: composition helpers at their old import location.

Not imported by ``import toscana``; use ``import toscana.chemistry``.
New code should import from the canonical modules.

Public names
------------
AtomicAvg
    Concentration-weighted average Σ c_i x_i (from
    `toscana.physics.properties`).
getConcentrations
    Atomic concentrations n_i / Σ n_j (from `toscana.physics.properties`).
getNofAtoms
    Number of atoms Σ n_i in a formula unit (from `toscana.physics.properties`).
extractAttr, extractDict
    Dictionary helpers (from `toscana.utils.dicts`).
"""

from .composition import (
    AtomicAvg,
    extractAttr,
    extractDict,
    getConcentrations,
    getNofAtoms,
)

__all__ = [
    "AtomicAvg",
    "extractAttr",
    "extractDict",
    "getConcentrations",
    "getNofAtoms",
]
