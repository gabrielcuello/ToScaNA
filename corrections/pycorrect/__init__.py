"""
pycorrect -- Python translation of CORRECT 2.24 (M.A. Howe, R.L. McGreevy,
P. Zetterstrom), the neutron diffraction data-correction program.

Main entry points
-----------------
Correct               command interpreter (same commands as the Fortran)
CorrectState          all parameters (Fortran COMMON blocks)
atcors                Paalman & Pings factors, cylinder (+ can, furnace)
plate                 Paalman & Pings factors, flat plate
mscat                 Blech & Averbach multiple-scattering ratio
correct_2axis, correct_tof   full correction chains
"""
from .state import CorrectState, Species, Spectrum
from .ndplib import atcors, mscat, mulscs, Collimator
from .routines import plate, polyfit
from .correct import Correct, read_atom_data
from .correct_2axis import correct_2axis, absor, placzek
from .correct_tof import correct_tof

__all__ = ["CorrectState", "Species", "Spectrum", "atcors", "mscat", "mulscs",
           "Collimator", "plate", "polyfit", "Correct", "read_atom_data",
           "correct_2axis", "absor", "placzek", "correct_tof"]
