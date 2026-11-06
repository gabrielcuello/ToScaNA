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

Conventions
-----------
Angles in degrees, Q in 1/Å, wavelength in Å. Output files are written in the
current working directory; the log is ``../logfiles/d4creg.log``.
"""

from .d4creg import d4creg

__all__ = ["d4creg"]
