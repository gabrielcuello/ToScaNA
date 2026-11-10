"""
Raw D4 data, detector corrections, regrouping and file input/output.

Position in the processing chain
--------------------------------
``dataio`` covers the steps between the raw numors written by the D4
diffractometer (ILL) and the files that are given to the external program CORRECT,
plus the readers and writers used by the later analysis (S(Q) -> G(r))::

    parameter file  --readParam-->  runInfo  --<run>-->  workflows.d4creg.d4creg
    raw numors (.nxs / ILL ASCII)            acquisition.load_nxs / readD4numorASCII
      -> dead time, efficiency, normalisation (corrections.*, acquisition.getOneNumor)
      -> stack of numors                      acquisition.getNumors
      -> regrouping on a 2θ grid              d4creg_outputs.getDiffA
      -> <file>.reg, <file>.adat              d4creg_outputs.saveDiffAngle
      -> Q grid (numerics.binning.rebin)  ->  <file>.qdat  d4creg_outputs.saveDiffQ
      -> CORRECT command file                 correct.saveCORRECT / generate_com
    ... S(Q) -> G(r), g(r), RDF, T(r), N(r)   saving.saveRSCF / saveCorrelations

Public functions (re-exported here)
-----------------------------------
acquisition
    printRates        log counts and counting rates (per s and per MW).
    getRefLines       locate the R/A/F/S marker lines of an ILL ASCII numor.
    readD4numorASCII  read one numor in ILL ASCII format.
    load_nxs          read one numor in NeXus format (h5py).
    getOneNumor       read one numor and apply all the per-cell corrections.
    getNumors         read and stack all the numors of a run.
corrections
    getDTC_mcounts    dead-time correction of the monitor counts.
    getDTC_dcounts    dead-time correction of the detector counts, cell by cell.
    getEff            read the relative efficiency of every cell.
    getDec            read the angular shift of every detector bank.
    getErrorsNumor    Poisson errors √N of raw counts.
    getNormalFactors  monitor and time normalisers with their errors.
    normalise         normalise counts and errors to monitor or time.
    getAngles         2θ angle and identifier of every cell.
d4creg_outputs
    getDiffA          regroup all cells of all numors on a regular 2θ grid.
    saveDiffAngle     write <file>.adat (CORRECT) and <file>.reg (D4) files.
    saveDiffQ         write <file>.qdat (CORRECT, Q scale).
loading
    read_xye          read a 2/3-column ASCII file into three 1-D arrays.
    read_3col         read a 2/3-column ASCII file into one (n, 3) array.
numors
    getNumorFiles     expand "123456" / "123456-123460" into a list of numors.
parameters
    readParam         read a d4creg parameter file into runInfo (may run d4creg).
    regroup           readParam with a short printout (for notebooks).
running_params
    getRunningParams  read the parameter file of the data-treatment notebooks.
saving
    saveFile_xye, saveFile_3col   write 3-column (x, y, e) files.
    saveCorrelations  write r, G, g, RDF, T, N from separate arrays.
    saveRSCF          write the output of numerics.fourier.sineFT (6/11 columns).
    save_columns      write named 1-D arrays as a table with a header.
    saveOneNumor      write the corrected cells of one numor (<numor>.cdat).
correct
    generate_com      fill a CORRECT template file.
    saveCORRECT       write a CORRECT command file from an experiment object.

Conventions
-----------
- Angles in degrees (2θ), wavelength in Å, Q in 1/Å, times in s, dead times in s.
- D4 geometry: 9 detector banks of 64 cells each; a bank covers 8° (key
  ``runInfo["angular_range_bank"]``), i.e. 0.125° per cell; banks are 15° apart.
- Numors are 6-digit strings ("387229").
- ``runInfo`` is the dictionary created by ``parameters.readParam`` and passed to
  every reduction function; ``head`` (or ``metadata``) is the per-numor metadata
  dictionary returned by the readers.
- Logs are appended to ``../logfiles/d4creg.log`` (relative to the current working
  directory, see ``_logfiles.d4creg_log_path``). Output data files are written in
  the current working directory, with the extensions of ``runInfo["ext"]``.
- Import order matters: ``parameters`` imports ``workflows.d4creg``, which imports
  ``dataio``; this cycle works because ``parameters`` is imported after the other
  submodules here.
"""

from .acquisition import (
    getNumors,
    getOneNumor,
    getRefLines,
    load_nxs,
    printRates,
    readD4numorASCII,
)
from .correct import generate_com, saveCORRECT
from .corrections import (
    getAngles,
    getDec,
    getDTC_dcounts,
    getDTC_mcounts,
    getEff,
    getErrorsNumor,
    getNormalFactors,
    normalise,
)
from .d4creg_outputs import getDiffA, saveDiffAngle, saveDiffQ
from .loading import read_3col, read_xye
from .numors import getNumorFiles
from .parameters import readParam, regroup
from .running_params import getRunningParams
from .saving import (
    saveCorrelations,
    saveFile_3col,
    saveFile_xye,
    save_columns,
    saveOneNumor,
    saveRSCF,
)

__all__ = [
    "generate_com",
    "getAngles",
    "getDec",
    "getDTC_dcounts",
    "getDTC_mcounts",
    "getEff",
    "getErrorsNumor",
    "getNormalFactors",
    "getNumors",
    "getOneNumor",
    "getRefLines",
    "getNumorFiles",
    "getRunningParams",
    "getDiffA",
    "load_nxs",
    "normalise",
    "printRates",
    "readD4numorASCII",
    "read_3col",
    "read_xye",
    "readParam",
    "regroup",
    "saveCORRECT",
    "saveCorrelations",
    "saveDiffAngle",
    "saveDiffQ",
    "saveFile_3col",
    "saveFile_xye",
    "saveOneNumor",
    "saveRSCF",
    "save_columns",
]
