"""
The d4creg regrouping workflow for D4 numors.

Position in the processing chain
--------------------------------
Top level of the regrouping, normally launched by a ``<run>`` line of the d4creg
parameter file (``dataio.parameters.readParam``)::

    getNumors -> getDiffA -> saveDiffAngle (.reg, .adat) [+ makePlotsA]
      -> 2θ to Q -> numerics.binning.rebin onto runInfo["q_scale"]
      -> masking of non-positive points -> saveDiffQ (.qdat) [+ makePlotsQ]
      -> rename the log file

The ``.adat`` / ``.qdat`` files are the input of the external program CORRECT.

Functions
---------
d4creg
    Run the complete regrouping described by ``runInfo``.

Conventions
-----------
Angles in degrees (zero angle subtracted), Q in 1/Å, wavelength in Å. Output files
``runInfo["file"] + ext`` in the current working directory; log in
``../logfiles/`` relative to the current working directory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from toscana.dataio.acquisition import getNumors
from toscana.dataio.d4creg_outputs import getDiffA, saveDiffAngle, saveDiffQ
from toscana.numerics.binning import rebin
from toscana.physics.conversions import ang2q
from toscana.plotting.workflows import makePlotsA, makePlotsQ
from toscana.utils.dates import getDate


def d4creg(runInfo: dict[str, Any] | Mapping[str, Any]) -> None:  # noqa: N802 (legacy API)
    """
    End-to-end D4 reduction workflow (legacy behavior).

    Steps:

    1. ``runInfo["runDate"] = utils.dates.getDate()`` (" Dayname DD/MM/YYYY
       HH:MM:SS").
    2. ``numor, head = dataio.acquisition.getNumors(runInfo)``: read and correct
       all numors; ``runInfo["nbrNumors"] = len(head)``.
    3. Build the equivalent d4creg command line
       ``d4creg -x <amin> <amax> <step> -z <twotheta0> -w <wavelength>
       -o <file>.reg <first numor> <last numor>`` (first and last numor READ,
       i.e. ``head[0]["numor"]`` and ``head[-1]["numor"]``), store it in
       ``runInfo["d4creg_cl"]`` and print it together with the efficiency file,
       the shift file, the normalisation and the number of numors.
    4. ``diffA = getDiffA(numor, head, runInfo)``: regroup on the 2θ grid.
    5. ``saveDiffAngle``: write ``<file>.adat`` and ``<file>.reg``; if
       ``int(runInfo.get("plotDiff", 0)) == 1``, ``plotting.workflows.makePlotsA``.
    6. Conversion to Q of the total diffractogram ``diffA[0]``: an (n_angles, 3)
       array (Q = 4π/λ · sin(2θ/2) computed from the lower bin edges, intensity,
       error; NaN entries are kept and skipped by ``rebin``).
    7. ``numerics.binning.rebin`` onto the Q grid ``runInfo["q_scale"]`` (bins
       centred at qmin + k·qstep), with the angular width of each point equal to
       the actual step of the regrouped grid, ``diffA[0, 0, 1] - diffA[0, 0, 0]``
       (or the requested step if the grid has a single point). Each point covers
       ΔQ = (2π/λ)·cos θ·Δ(2θ)·π/180. Empty Q bins come out of ``rebin`` with
       intensity 0.
    8. Masking: for every Q point with intensity <= 0 (including the empty bins),
       that point AND the previous one are set to NaN in all three columns.
       Points that are already NaN are not tested (NaN <= 0 is False).
    9. ``saveDiffQ``: write ``<file>.qdat`` (only positive, non-NaN points); if
       ``plotDiff == 1``, ``makePlotsQ``.
    10. Rename ``../logfiles/d4creg.log`` to
        ``../logfiles/<runInfo["logfile"]>.log``.

    Parameters
    ----------
    runInfo : dict or mapping
        Run parameters from ``dataio.parameters.readParam``. Keys read:
        ``angular_scale``, ``twotheta0``, ``wavelength``, ``file``, ``efffile``,
        ``decfile``, ``normalisation``, ``plotDiff`` (optional, default 0),
        ``q_scale``, ``logfile``, plus all the keys read by ``getNumors``,
        ``getOneNumor``, ``getDiffA``, ``saveDiffAngle`` and ``saveDiffQ``.
        Keys WRITTEN: ``runDate``, ``nbrNumors``, ``d4creg_cl`` (and
        ``writeNumor`` = 1 by ``getNumors`` for a single numor). If ``runInfo``
        is a dict, it is modified in place; any other mapping is copied first,
        so the caller's object is not modified.

    Returns
    -------
    None
        Results are written to files.

    Raises
    ------
    ValueError, KeyError, FileNotFoundError, ImportError
        Propagated from the reading and correction of the numors
        (``getNumors``), e.g. angle out of tolerance, dead-time factor not
        positive, missing raw-data file, missing ``runInfo`` key.
    OSError
        If an output file cannot be written.

    Notes
    -----
    Files written (current working directory unless ``runInfo["file"]``
    contains a path): ``<file><ext[1]>`` (.adat), ``<file><ext[0]>`` (.reg),
    ``<file><ext[2]>`` (.qdat), and one ``<numor><ext[3]>`` (.cdat) per numor if
    ``writeNumor == 1``. The log ``../logfiles/d4creg.log`` is appended to, then
    renamed. With the default ``logfile = "d4creg.log"`` the new name is
    ``../logfiles/d4creg.log.log`` (the extension is added twice; open item).
    A missing log file only prints a message; other rename errors are printed,
    not raised. On POSIX an existing file with the new name is replaced.

    Prints the command line, the run summary, the file names written, a
    diagnostic line from ``rebin`` and the result of the rename.

    Changes from the legacy code (audit 2026-11):

    - C9: the angular width used for the Q rebinning is the actual step of the
      regrouped grid (``getDiffA`` may enlarge the requested step to the cell
      width), not a hard-coded 0.125°.
    - C8: when the first Q point (i = 0) is masked, the "previous" index ``-1``
      used to wipe the LAST Q point; the previous point is now masked only for
      i > 0.

    This is a parity-first copy of `legacy/lib/toscana.py::d4creg`.

    See Also
    --------
    toscana.dataio.parameters.readParam : builds ``runInfo`` and calls this.
    toscana.dataio.d4creg_outputs.getDiffA : the regrouping.
    toscana.numerics.binning.rebin : the Q rebinning.
    """
    if not isinstance(runInfo, dict):
        # Legacy mutates runInfo; require a dict to preserve semantics.
        runInfo = dict(runInfo)

    runInfo["runDate"] = getDate()

    numor, head = getNumors(runInfo)
    runInfo["nbrNumors"] = len(head)

    d4creg_cl = "d4creg"
    d4creg_cl += " -x " + str(runInfo["angular_scale"][0]) + " "
    d4creg_cl += str(runInfo["angular_scale"][1]) + " "
    d4creg_cl += str(runInfo["angular_scale"][2])
    d4creg_cl += " -z " + str(runInfo["twotheta0"])
    d4creg_cl += " -w " + str(runInfo["wavelength"])
    d4creg_cl += " -o " + str(runInfo["file"]) + ".reg "
    d4creg_cl += str(head[0]["numor"]) + " " + str(head[-1]["numor"])
    runInfo["d4creg_cl"] = d4creg_cl

    print()
    print("Equivalent d4creg command line")
    print(d4creg_cl)
    print()
    print(" - Efficiency: ", runInfo["efffile"])
    print(" - Det shifts: ", runInfo["decfile"])
    if runInfo["normalisation"][0] == "monitor":
        print(
            " - Normalisation to {1} of {0} counts".format(
                runInfo["normalisation"][0], runInfo["normalisation"][2]
            )
        )
    else:
        print(
            " - Normalisation to {1} s of counting {0}".format(
                runInfo["normalisation"][0], runInfo["normalisation"][1]
            )
        )
    print(" - Number of numors: ", runInfo["nbrNumors"])

    diffA = getDiffA(numor, head, runInfo)

    print()
    print("Output:")
    saveDiffAngle(diffA, head, runInfo)

    if int(runInfo.get("plotDiff", 0)) == 1:
        makePlotsA(runInfo, diffA, head, numor)

    qmin = float(runInfo["q_scale"][0])
    qmax = float(runInfo["q_scale"][1])
    qstep = float(runInfo["q_scale"][2])

    dataQ = np.zeros((diffA.shape[2], 3), dtype=np.float64)
    ang = diffA[0, 0, :]
    dataQ[:, 0] = ang2q(ang, wlength=float(runInfo["wavelength"]))
    for i in range(diffA.shape[2]):
        for j in range(1, 3):
            dataQ[i, j] = diffA[0, j, i]

    # Bug fix vs legacy: the angular width of each regrouped point is the actual
    # step of the regrouped grid (getDiffA may enlarge the requested step to the
    # cell width), not a hard-coded 0.125 deg.
    if diffA.shape[2] > 1:
        angular_width = float(diffA[0, 0, 1] - diffA[0, 0, 0])
    else:
        angular_width = float(runInfo["angular_scale"][2])
    diffQbin = rebin(
        angular_width, float(runInfo["wavelength"]), dataQ, qmin, qmax, qstep
    )

    # Mask non-positive points (empty Q bins come out of rebin as 0) together
    # with the previous point.
    for i in range(len(diffQbin)):
        if diffQbin[i, 1] <= 0.0:
            for j in range(3):
                diffQbin[i, j] = np.nan
                # Bug fix vs legacy: at i = 0, `i - 1` is -1 and wiped the
                # LAST Q point.
                if i > 0:
                    diffQbin[i - 1, j] = np.nan

    saveDiffQ(diffQbin, head, runInfo)

    if int(runInfo.get("plotDiff", 0)) == 1:
        makePlotsQ(runInfo, diffQbin, head, numor)

    current_file_path = "../logfiles/d4creg.log"
    new_file_name = "../logfiles/" + str(runInfo["logfile"]) + ".log"
    try:
        os.rename(current_file_path, new_file_name)
        print(f"File '{current_file_path}' renamed to '{new_file_name}' successfully.")
    except FileNotFoundError:
        print(f"File '{current_file_path}' not found.")
    except Exception as e:  # pragma: no cover
        print(f"An error occurred: {str(e)}")
    return
