"""
Complete reduction workflows built on ``toscana.dataio``.

Position in the processing chain
--------------------------------
Top level of the D4 regrouping: raw numors -> corrected cells -> regrouped
diffractogram in 2θ -> diffractogram on a Q grid, ready for CORRECT. It is
normally launched by the ``<run>`` line of a d4creg parameter file read by
``toscana.dataio.parameters.readParam``.

Public functions
----------------
d4creg
    The full regrouping of a list of numors described by ``runInfo``; writes the
    ``.reg``, ``.adat`` and ``.qdat`` files and renames the log file.
run_correction, read_regrouped, structure_factor, save_structure_factor, build_state
    (module ``correction``) From the regrouped ``.adat``/``.qdat`` files and a
    ``Measurement`` to the corrected cross section and to S(Q), ready for
    ``sineFT``: ToScaNA's Paalman & Pings chain with MCAMS factors (default,
    any set of concentric cylinders, absorber) or CORRECT 2.24 (pycorrect).
mcams_correct, without_container, atoms_in_beam
    (module ``correction``) Single-scattering intensities of sample and
    vanadium before normalisation (multiple scattering additive, proportional
    or none), and the numbers of atoms in the beam, for a normalisation by a
    fitted vanadium.

Conventions
-----------
Angles in degrees, Q in 1/Å, wavelength in Å. Output files are written in the
current working directory; the log is ``../logfiles/d4creg.log``.
"""

from .correction import (
    MATERIALS,
    CorrectionResult,
    atoms_in_beam,
    build_state,
    material_properties,
    mcams_correct,
    read_regrouped,
    run_correction,
    save_structure_factor,
    structure_factor,
    without_container,
)
from .d4creg import d4creg

__all__ = [
    "MATERIALS",
    "CorrectionResult",
    "atoms_in_beam",
    "build_state",
    "d4creg",
    "material_properties",
    "mcams_correct",
    "read_regrouped",
    "run_correction",
    "save_structure_factor",
    "structure_factor",
    "without_container",
]
