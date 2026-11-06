"""
Experiment, beam, container, vanadium and sample bookkeeping.

Two ways of describing a measurement are offered:

- the ``set*Info`` functions of `toscana.experiment.info`, which take keyword
  arguments, print a summary and return a dictionary each;
- the class `toscana.experiment.measurement.Measurement`, built from the
  dictionary of `toscana.dataio.running_params.getRunningParams`, which also
  reads the data files.

Public names
------------
Measurement
    Everything about one sample measurement from a parameter file.
setExpInfo
    Experiment identification dictionary.
setBeamInfo
    Beam dictionary (wavelength, zero angle, slits, flags, size).
setCanInfo
    Container dictionary (material, shape, diameters, wall thickness).
setBinInfo
    Angular, Q and r grids and their numbers of points.
setNumorInfo
    Numor ranges of each measured object.
setVanaInfo
    Vanadium standard dictionary.
setSampleInfo
    Sample dictionary (densities, averaged and free cross sections, ratios
    to vanadium).

Units: mm, cm³, g, g/cm³, atoms/Å³, barns/atom, fm, degrees, Å, 1/Å.
The constants ``VANADIUM_DENSITY_GCC``, ``VANADIUM_MOLAR_MASS`` and
``VANADIUM_ATOMIC_DENSITY`` are in `toscana.experiment.info` (not re-exported).
"""

from .info import (
    setBeamInfo,
    setBinInfo,
    setCanInfo,
    setExpInfo,
    setNumorInfo,
    setSampleInfo,
    setVanaInfo,
)
from .measurement import Measurement

__all__ = [
    "Measurement",
    "setBeamInfo",
    "setBinInfo",
    "setCanInfo",
    "setExpInfo",
    "setNumorInfo",
    "setSampleInfo",
    "setVanaInfo",
]
