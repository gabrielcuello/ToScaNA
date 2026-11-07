"""
state.py  --  Python translation of  correct_224.inc  +  BLOCK DATA
=====================================================================

In Fortran all the program parameters live in COMMON blocks declared in the
include file correct_224.inc, and their default values are set in the
BLOCK DATA unit of correct_224.f.  In Python they are attributes of one
object, `CorrectState`, built with the standard library `dataclasses`.

    @dataclass  : decorator that writes __init__/__repr__ automatically from
                  the annotated class attributes.
    field(default_factory=...) : default for mutable attributes (lists), so
                  that each instance gets its own list.

COMMON block  ->  attribute(s)
-------------------------------
/files/   file(12)          -> files: dict with readable keys (see FILE_KEYS)
/both1/   fu_*, ca_*, va_*, sa_*, radii, species arrays, flags
/both2/   instrument, inst_type, geometry, det_type, new_title,
          vanadium_smoothing
/both3/   placzek_correction, omegav
/tof1/,/tof2/  path1, path2, wav_min, wav_max, monitor_spectrum, xtype
/axis1/,/axis2/ wavelength, angle, lowang, low_*, calib_u/v/w, xouttype
/coll1/../coll4/ -> collimator  (a ndplib.Collimator object)

Species: the Fortran keeps parallel arrays frac(10), atwgt(10), scs(10),
acs(10), atsym(10) and a counter nspecies.  Here: a list of `Species`.

Spectra: block(60), angle(60), path1(60), path2(60), det_type(60) and
nspectra.  Here: a list of `Spectrum`.

New (not in Fortran)
--------------------
faithful : bool, default False.
    False -> corrected physics (recommended).
    True  -> reproduce the known bugs of CORRECT 2.24 so that numbers can be
             compared with old outputs.  The list of affected places is in
             README.md, section "Known bugs of CORRECT 2.24".
overwrite : bool, default False.  The PC Fortran stops if the output file
    exists; set True to overwrite it instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .ndplib import Collimator

# Fortran file(i) index  ->  key used in Python
FILE_KEYS = {
    1: "furnace", 2: "container", 3: "sample", 4: "vanadium",
    5: "bkg_furnace", 6: "bkg_container", 7: "bkg_sample", 8: "bkg_vanadium",
    9: "output", 10: "cs_file", 11: "cadmium", 12: "second",
}
# background key belonging to each measured body (Fortran: file(isam+4))
BKG_OF = {"furnace": "bkg_furnace", "container": "bkg_container",
          "sample": "bkg_sample", "vanadium": "bkg_vanadium"}


@dataclass
class Species:
    """One atomic species of the sample.  Fortran: frac(i), atwgt(i), scs(i),
    acs(i), atsym(i) (cross sections in barns, acs at 2200 m/s = 1.798 A)."""
    symbol: str
    fraction: float
    weight: float
    scatt_cs: float
    abs_cs: float


@dataclass
class Spectrum:
    """One spectrum (detector group / block) to correct.
    Fortran: block(i), angle(i), path1(i), path2(i), det_type(i)."""
    block: int
    angle: float = 0.0          # TOF only: scattering angle [deg]
    path1: float = 10.0         # TOF only: incident flight path [m]
    path2: float = 1.043        # TOF only: scattered flight path [m]
    det_type: str = "HE_GAS"    # TOF only: 'HE_GAS' or 'SCINTILLATION'


@dataclass
class CorrectState:
    # ---- /files/ ----------------------------------------------------------
    files: dict = field(default_factory=lambda: {k: "" for k in FILE_KEYS.values()})
    # ---- furnace, container, vanadium: barns, atoms/A^3 ------------------
    fu_abs_cs: float = 5.08
    fu_scat_cs: float = 4.95
    fu_rho: float = 0.0722
    ca_abs_cs: float = 5.08
    ca_scat_cs: float = 4.95
    ca_rho: float = 0.0722
    va_abs_cs: float = 5.08
    va_scat_cs: float = 4.95
    va_rho: float = 0.0722
    # ---- sample ---------------------------------------------------------
    sa_rho: float = 0.0
    sa_temp: float = 300.0
    sa_fullness: float = 1.0
    ca_multiplier: float = 1.0
    va_multiplier: float = 1.0
    # ---- beam and geometry (cm, deg) -------------------------------------
    beam_height: float = 3.0
    beam_width: float = 1.5
    sample_angle: float = 0.0
    van_angle: float = 0.0
    va_r: float = 0.0
    ca_r1: float = 0.0          # sample radius (= container inner radius)
    ca_r2: float = 0.0          # container outer radius
    fu_r1: float = 0.0
    fu_r2: float = 0.0
    geometry: str = "CYLINDRICAL"
    # ---- composition and spectra -----------------------------------------
    species: List[Species] = field(default_factory=list)
    spectra: List[Spectrum] = field(default_factory=list)
    # ---- flags ----------------------------------------------------------
    va_furnace: bool = False
    va_container: bool = False
    write_second_file: bool = False
    write_low_angle: bool = False
    write_calibration: bool = False
    write_absorption: bool = False
    write_ms: bool = False
    write_placzek: bool = False
    # ---- instrument -----------------------------------------------------
    instrument: str = "D4"
    inst_type: str = "2-AXIS"
    placzek_correction: str = "NONE"
    omegav: float = 0.0
    new_title: str = ""
    vanadium_smoothing: int = 0
    # ---- TOF ------------------------------------------------------------
    monitor_spectrum: int = 0
    wav_min: float = 0.1
    wav_max: float = 3.0
    xtype: str = "WAVELENGTH"
    # ---- 2-axis ---------------------------------------------------------
    wavelength: float = 0.7
    lowang: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    low_ba: bool = False
    low_fu: bool = False
    low_ca: bool = False
    low_va: bool = False
    xouttype: str = "Q"
    calib_ur: float = 2.595388
    calib_vr: float = -2.206898
    calib_wr: float = 0.709038
    # ---- collimator ------------------------------------------------------
    collimator: Collimator = field(default_factory=Collimator)
    # ---- Python additions -------------------------------------------------
    faithful: bool = False
    overwrite: bool = False
    data_format: str = "TEXT"    # input format: 'TEXT' or 'DATA'

    # convenience ----------------------------------------------------------
    def has(self, key: str) -> bool:
        """True if file `key` was given (Fortran:  file(i).ne.' ')."""
        return bool(self.files.get(key, "").strip())
